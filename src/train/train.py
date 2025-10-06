from typing import Optional
import torch
import ujson
from pathlib import Path
from src.models.conversations import CONV_TEMPLATES, CONV_MAPPING, DEFAULT_IMAGE_TOKEN
import src.custom_utils as utils
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection
)
from src.train.args import CustomTrainingArguments, DataArguments, ModelArguments, postprocess_args
from src.train.trainers import CustomTrainer
from argparse import Namespace
from src.models.mllm import MLLMConfig, MLLMForIWM, MLLMForCausalLM
from src.train.collators import MaskCollator, SupervisedCollator
from PIL import Image
import webdataset as wds
from braceexpand import braceexpand
from random import shuffle

logger = utils.get_logger()


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
                    

class LlavaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        conv_template: CONV_TEMPLATES = CONV_TEMPLATES.PLAIN,
        instruction_tuning: bool = False,
        stage: int = 0,
        iwm_captions: bool = False,
        prompt_max_length: Optional[int] = None,
        image_aspect_ratio: Optional[str] = None,
        iwm_txt_cond: bool = False
    ):
        super().__init__()
        self.data_path = data_path
        self.stage = stage
        self.image_folder = Path(image_folder)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_template = CONV_TEMPLATES(conv_template)
        self.instruction_tuning = instruction_tuning
        self.img_placeholder = Image.new('RGB', (400, 400))
        self.iwm_captions = iwm_captions
        self.prompt_max_length = prompt_max_length
        self.image_aspect_ratio = image_aspect_ratio
        self.iwm_txt_cond = iwm_txt_cond

        with open(data_path, 'r') as f:
            self.data = ujson.load(f)
            n_samples = utils.is_debug_n_dataset_samples()
            if n_samples:
                self.data = self.data[:n_samples]
                logger.info(
                    f"Using only {len(self.data)} dataset samples for debugging.")
        logger.info(f'Loaded {len(self.data)} samples from {data_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        img_err = False
        if sample.get('image', None):
            try:
                image_path = self.image_folder.joinpath(sample['image'])
                image = Image.open(image_path).convert('RGB')
                image_mask = [1]
                if self.image_aspect_ratio == 'pad':
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            except FileNotFoundError as e:
                logger.warning(e)
                image = self.img_placeholder
                image_mask = [0]
                img_err = True
        else:
            image = self.img_placeholder
            image_mask = [0]

        input_ids = None
        attention_mask = None
        labels = None
        tok_kwargs = {}

        if self.prompt_max_length is not None:
            tok_kwargs['max_length'] = self.prompt_max_length
            tok_kwargs['truncation'] = True

        if self.stage >= 2:
            if self.conv_template == CONV_TEMPLATES.PLAIN:
                text = DEFAULT_IMAGE_TOKEN + sample['conversations'][1]['value']
                inputs_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, **tok_kwargs)
                input_ids = inputs_text.input_ids[0]
                attention_mask = inputs_text.attention_mask[0]
                image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
                labels = input_ids.clone()
                labels[labels == image_token_id] = -100
            else:
                conv = CONV_MAPPING[self.conv_template].new_empty()
                conv.add_message(dict(role='system'))
                # Ensure exactly one image token is present in the prompt.
                img_tok_seen = False
                for turn in sample['conversations']:
                    if turn['from'] == 'human':
                        role = 'user'
                        if DEFAULT_IMAGE_TOKEN in turn['value']:
                            turn['value'] = turn['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                            turn['value'] = DEFAULT_IMAGE_TOKEN + '\n' + turn['value']
                            turn['value'] = turn['value'].strip()                            
                            img_tok_seen = True
                        elif not img_tok_seen:
                            # No image token provided by the dataset; prefix it on the first human turn.
                            turn['value'] = (DEFAULT_IMAGE_TOKEN + '\n' + turn['value']).strip()
                            img_tok_seen = True
                    elif turn['from'] == 'gpt':
                        role = 'assistant'
                    conv.add_message(dict(role=role, content=turn['value']))

                text = conv.get_prompt()
                input_ids = self.tokenizer(text, return_tensors="pt", **tok_kwargs).input_ids[0]
                attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype)

                if img_err:
                    labels = torch.full((len(input_ids),), fill_value=-100, dtype=input_ids.dtype)
                else:
                    tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids)
                    labels = conv.get_labels(input_ids, tokenized_text)

        elif self.conv_template == CONV_TEMPLATES.PLAIN:
            if self.iwm_txt_cond:
                text = sample['conversations'][1]['value'] + DEFAULT_IMAGE_TOKEN
            elif self.iwm_captions:
                    text = DEFAULT_IMAGE_TOKEN + sample['conversations'][1]['value']
            else:
                text = DEFAULT_IMAGE_TOKEN

        else:
            raise ValueError()

        return dict(
            text=text, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            image=image,
            image_mask=image_mask
        )


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = []
    blacklist = ['projector', 'vision_model', 'lm_head']
    for name, module in model.named_modules():
        if any(k in name for k in blacklist):
            continue
        if isinstance(module, cls):
            lora_module_names.append(name)

    return list(lora_module_names)


def build_model_commons(args: ModelArguments) -> Namespace:
    if args.iwm_loss:
        model_cls = MLLMForIWM
    else:
        model_cls = MLLMForCausalLM
 
    if args.model_name:
        config = None
        if utils.is_debug_one_layer_mode():
            logger.info('Using a one-layer language model for debugging')
            config = MLLMConfig.from_pretrained(args.model_name)
            config.text_config.num_hidden_layers = 1
        lm_kwargs = {}
        if args.attn_implementation:
            lm_kwargs['attn_implementation'] = args.attn_implementation
        if args.stage == 2:
            model = model_cls.from_pretrained_MLLMForIWM(name_or_path=args.model_name, lm_kwargs=lm_kwargs)
        elif args.iwm_loss:
            model = model_cls.from_pretrained_MLLMForCausalLM(name_or_path=args.model_name, lm_kwargs=lm_kwargs)
        else:
            model = model_cls.from_pretrained(args.model_name, config=config, lm_kwargs=lm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        text_config = AutoConfig.from_pretrained(args.language_model_name)
        if utils.is_debug_one_layer_mode():
            text_config.num_hidden_layers = 1
            logger.info('Using a one-layer language model for debugging')
        logger.info(f"Attention implementation: {args.attn_implementation}")
        language_model = AutoModelForCausalLM.from_pretrained(
            args.language_model_name,
            config=text_config,
            attn_implementation=args.attn_implementation
        )

        vision_cls = AutoModel
        if 'openai/clip' in args.vision_model_name:
            vision_cls = CLIPVisionModel
            if args.iwm_tgt_vision_model_proj_head:
                vision_cls = CLIPVisionModelWithProjection
        vision_model = vision_cls.from_pretrained(args.vision_model_name)

        config = MLLMConfig(
            text_config=text_config,
            vision_config=vision_model.config,
            vision_layer_idx=args.vision_layer_idx,
            iwm_tgt_vision_layer_idx=args.iwm_tgt_vision_layer_idx,
            iwm_tgt_vision_model_proj_head=args.iwm_tgt_vision_model_proj_head,
            iwm_tgt_proj_output_size=args.iwm_tgt_proj_output_size,
            iwm_full_img_on_encoder=args.iwm_full_img_on_encoder,
            projector_type=args.projector_type,
            projector_bias=args.projector_bias,
            full_mask_image_tokens=args.full_mask_image_tokens
        )
        model = model_cls(config=config, language_model=language_model, vision_model=vision_model)
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_config.name_or_path, local_files_only=True)

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            ret = tokenizer.add_special_tokens(dict(pad_token=tokenizer.unk_token))
        else:
            tokenizer.pad_token = tokenizer.eos_token
            ret = tokenizer.add_special_tokens(dict(pad_token=tokenizer.eos_token))
        assert ret == 0, "Failed to add pad token to tokenizer"

    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_config.name_or_path)
    model.init_vision_tokenizer(tokenizer)

    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.contiguous()

    return Namespace(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor
    )


MIN_IMG_SIZE = 16
def wds_map_fn(sample):
    image = sample['jpg'].convert('RGB')
    if image.height < MIN_IMG_SIZE or image.width < MIN_IMG_SIZE:
        raise ValueError(f"Image too small: {image.height}x{image.width}, expected at least {MIN_IMG_SIZE}x{MIN_IMG_SIZE}")

    return dict(
        text=DEFAULT_IMAGE_TOKEN, 
        image=image,
        image_mask=1
    )


def build_data_commons(
    args: DataArguments,
    image_processor,
    tokenizer,
    vision_config,
    training_args: Optional[CustomTrainingArguments] = None,
) -> Namespace:
    if args.train_wds_path:
        urls = list(braceexpand(args.train_wds_path))
        shuffle(urls)
        global_batch_size_no_acc = training_args.per_device_train_batch_size * utils.get_world_size()
        shuffle_buffer_size = args.wds_buffer_size if args.wds_buffer_size else global_batch_size_no_acc * 10
        logger.info(f"Using WebDataset with {len(urls)} shards, buffer size: {shuffle_buffer_size}")
        train_dataset = wds.DataPipeline(
            wds.ResampledShards(urls),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(shuffle_buffer_size),
            wds.decode('pil', handler=wds.warn_and_continue),
            wds.map(wds_map_fn, handler=wds.warn_and_continue),
            wds.batched(training_args.per_device_train_batch_size)
        )
    else:
        train_dataset = LlavaDataset(args.train_data_path, args.train_image_folder, tokenizer=tokenizer, image_processor=image_processor,
                                    conv_template=args.conv_template, instruction_tuning=args.instruction_tuning,
                                    iwm_captions=args.iwm_captions, prompt_max_length=args.prompt_max_length,
                                    image_aspect_ratio=args.image_aspect_ratio, stage=args.stage, iwm_txt_cond=args.iwm_txt_cond)
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = LlavaDataset(args.eval_data_path, args.eval_image_folder, tokenizer=tokenizer)

    if args.instruction_tuning and not args.iwm_loss:
        data_collator = SupervisedCollator(image_processor, tokenizer)
    else:
        def get_input_size():
            if hasattr(image_processor, 'crop_size'):
                # CLIP, DiNO
                return image_processor.crop_size['height'], image_processor.crop_size['width']
            elif hasattr(image_processor, 'size'):
                # I-JEPA, SigLIP
                return image_processor.size['height'], image_processor.size['width']
            
        data_collator = MaskCollator(
            image_processor=image_processor,
            tokenizer=tokenizer,
            return_labels=args.iwm_captions and not args.iwm_txt_cond,
            input_size=get_input_size(),
            patch_size=vision_config.patch_size,
            enc_mask_scale=args.ijepa_enc_mask_scale,
            pred_mask_scale=args.ijepa_pred_mask_scale,
            aspect_ratio=args.ijepa_aspect_ratio,
            nenc=args.ijepa_num_enc_masks,
            npred=args.ijepa_num_pred_masks,
            min_keep=args.ijepa_min_keep,
            allow_overlap=False,
            iwm_full_img_on_encoder=args.iwm_full_img_on_encoder
        )

    return Namespace(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


if __name__ == '__main__':
    parser = HfArgumentParser((CustomTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = postprocess_args(*parser.parse_args_into_dataclasses())
    data_args.iwm_full_img_on_encoder = model_args.iwm_full_img_on_encoder

    local_rank = utils.get_local_rank()
    torch.cuda.set_device(local_rank)
    print(f"Rank [{utils.get_rank()}]: set device ID {local_rank}")

    model_commons = build_model_commons(model_args)

    if training_args.gradient_checkpointing:
        model_commons.model.gradient_checkpointing_enable()

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model_commons.model.get_input_embeddings(
        ).register_forward_hook(make_inputs_require_grad)

    data_commons = build_data_commons(
        data_args,
        image_processor=model_commons.image_processor,
        tokenizer=model_commons.tokenizer,
        vision_config=model_commons.model.config.vision_config,
        training_args=training_args
    )

    model_commons.model.get_vision_model().requires_grad_(False)
    stage: int = model_args.stage
    if stage != 3:
        model_commons.model.get_language_model().requires_grad_(False)

    trainer = CustomTrainer(
        model=model_commons.model,
        args=training_args,
        train_dataset=data_commons.train_dataset,
        eval_dataset=data_commons.eval_dataset,
        data_collator=data_commons.data_collator,
        processing_class=model_commons.tokenizer,
        data_args=data_args
    )

    if training_args.resume_from_checkpoint and data_args.train_wds_path is not None:
        training_args.ignore_data_skip = True
        logger.info(f"Setting ignore_data_skip to True for WebDataset resuming")

    if stage == 1:
        """
            In the first stage, we train the adapter with SmoothL1 only. (IWM loss)
        """
        model_commons.model.set_losses({"SmoothL1": True})
        logger.info(f"Training - First Stage")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info('Training - First Stage - Finished')

    elif stage == 2 or stage >= 4:
        """
            In the second stage, we train the adapter with CrossEntropy only. (LLaVA 1^ stage)
        """
        if model_args.iwm_loss:
            model_commons.model.set_losses({"CrossEntropy": True})
        logger.info("Training - Second Stage")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info('Training - Second Stage - Finished')

    elif stage == 3:
        """
            In the third stage, we train the adapter+LLM with both SmoothL1 and CrossEntropy losses. (IWM + LLaVA 2^ stage)
            Another variation: train the adapter+LLM with only CrossEntropy loss. (LLaVA 2^ stage)
        """
        if model_args.iwm_loss:
            model_commons.model.set_losses({
                "SmoothL1": False,
                "CrossEntropy": True,
            })
        logger.info(f"Training - Third Stage")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.info('Training - Third Stage - Finished')
    
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")