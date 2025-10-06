from src.models import MLLMForCausalLM, DEFAULT_IMAGE_TOKEN, CONV_MAPPING, CONV_TEMPLATES
from typing import List, Tuple
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval import utils as lmms_utils
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig
from PIL import Image
import src.custom_utils as utils

from loguru import logger as eval_logger

from src.train.train import expand2square


def pad_image_list(image_list, img_size):
    lens = [len(x) for x in image_list]
    if any(lens):
        maxlen = max(lens)
        for i in range(len(lens)):
            imgs = image_list[i]
            if len(imgs) < maxlen:
                imgs.extend(Image.new('RGB', (img_size, img_size), color='black') for _ in range(maxlen - len(imgs)))


DTYPE_MAPPING = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


@register_model('mllm')
class MLLM(lmms):
    def __init__(
        self,
        name_or_path: str,
        batch_size: int = 1,
        device: str = "cuda",
        conv_mode: CONV_TEMPLATES = CONV_TEMPLATES.PHI4,
        **kwargs
    ):
        super().__init__()

        dtype = DTYPE_MAPPING[kwargs.get('dtype', 'float16')]
        max_length = kwargs.pop('max_length', None)

        attn_implementation = kwargs.get('attn_implementation')
        if attn_implementation is None:
            cfg = AutoConfig.from_pretrained(name_or_path)
            attn_implementation = cfg.text_config._attn_implementation if cfg.text_config._attn_implementation else 'eager'
        
        lm_kwargs = dict(
            torch_dtype=dtype,
            attn_implementation=attn_implementation
        )

        self.image_aspect_ratio = kwargs.get('image_aspect_ratio')

        self.model = MLLMForCausalLM.from_pretrained(name_or_path, device_map=device, lm_kwargs=lm_kwargs)
        self.model.requires_grad_(False)
        self.model.eval()
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if max_length is not None:
            self.tokenizer.model_max_length = max_length
        self.model.init_vision_tokenizer(self.tokenizer, resize_embeds=False)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config.vision_config.name_or_path)
        self.conv_mode = conv_mode
        if 'phi4' in conv_mode.lower():
            self.model.generation_config.eos_token_id = [199_999, 200_020]

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        ...

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = lmms_utils.Collator([reg.args for reg in requests], lambda x: x, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            # TODO: handle tasks without images
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # Set default values for until and max_new_tokens
            until.insert(0, CONV_MAPPING[self.conv_mode].seps['assistant'].strip())

            prompts = []
            images = []
            for visual, context in zip(batched_visuals, contexts):
                if len(visual) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context

                    if self.image_aspect_ratio == 'pad':
                        images.append([expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in visual])
                    else:
                        images.append(visual)
                else:
                    question = context
                    images.append([])

                # conv = [dict(role='user', content=[dict(type='text', text=question)])]
                # prompts.append(self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
                conv = CONV_MAPPING[self.conv_mode].new_empty()
                conv.add_sys_prompt()
                conv.add_message(dict(role='user', content=question))
                prompts.append(conv.get_prompt(add_generation_prompt=True))
            
            txt_inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            pad_image_list(images, utils.get_image_size_from_image_processor(self.image_processor))
            # workaround: maybe an issue with the version of some libraries?
            if isinstance(images, list) and isinstance(images[0], list):
                assert self.batch_size == 1
                images = images[0]
            pixel_values = self.image_processor(images, return_tensors='pt').to(self.device).pixel_values

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                do_sample = True if gen_kwargs["temperature"] > 0 else False
                temperature = gen_kwargs["temperature"] if gen_kwargs["temperature"] else 1.0
                cont = self.model.generate(
                    **txt_inputs,
                    pixel_values=pixel_values,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=True,
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=False)
                parsed_text = []
                for txt_out in text_outputs:
                    parsed = ''
                    for u in until:
                        txt_splits = txt_out.split(u)
                        if len(txt_splits) > 1:
                            parsed = txt_splits[0]
                            break
                    parsed_text.append(parsed)
                text_outputs = parsed_text
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LMM
            # if self.truncate_context:
            #     cont_toks = cont_toks[input_ids.shape[1] :]
            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            # if self.truncate_context:
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for MLLMs")        