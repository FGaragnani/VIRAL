from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel, AutoConfig, CLIPVisionModelWithProjection, CLIPVisionModel
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from .configuration_mllm import MLLMConfig, ProjectorType
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.custom_utils as utils
from torch import Tensor
from src.models.llava_conversations import VICUNA_V1_JINJA_TEMPLATE
import torch.distributed as dist
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput

logger = utils.get_logger()


def build_projector(projector_type, **kwargs):
    if projector_type == ProjectorType.LINEAR:
        return nn.Linear(
            in_features=kwargs['input_size'],
            out_features=kwargs['output_size'],
            bias=kwargs['bias']
        )
    elif projector_type == ProjectorType.MLP:
        return nn.Sequential(
            nn.Linear(kwargs['input_size'], kwargs['output_size'], bias=kwargs['bias']),
            nn.GELU(),
            nn.Linear(kwargs['output_size'], kwargs['output_size'], bias=kwargs['bias'])
        )
    else:
        raise ValueError(f'Projector type {projector_type} not supported')


class MLLMPreTrainedModel(PreTrainedModel):
    config_class = MLLMConfig
    base_model_prefix = 'mllm'
    _tied_weights_keys = ["language_model.lm_head.weight"]

    def __init__(
        self,
        config: MLLMConfig,
        language_model: Optional[AutoModelForCausalLM] = None,
        vision_model: Optional[PreTrainedModel] = None,
        projector: Optional[nn.Module] = None,
        lm_kwargs: Dict = {},
        **kwargs
    ):
        if language_model is None:
            attn_implementation = lm_kwargs.pop('attn_implementation', None)
            if attn_implementation is None:
                attn_implementation = config.text_config._attn_implementation if config.text_config._attn_implementation else 'eager'
            if utils.is_debug_one_layer_mode():
                config.text_config.num_hidden_layers = 1
                logger.debug(
                    f"Debug mode: forcing text_config.num_hidden_layers to {config.text_config.num_hidden_layers}")
            language_model = AutoModelForCausalLM.from_config(
                config.text_config, attn_implementation=attn_implementation, **lm_kwargs)
            logger.info(f"attn_implementation: {attn_implementation}")
        config.text_config = language_model.config

        if vision_model is None:
            vision_model = AutoModel.from_config(config.vision_config)
        else:
            config.vision_config = vision_model.config

        super().__init__(config, **kwargs)

        self.language_model = language_model
        self.vision_model = vision_model

        if projector is None:
            projector = build_projector(
                config.projector_type,
                input_size=config.projector_input_size,
                output_size=config.text_config.hidden_size,
                bias=config.projector_bias
            )
        self.projector = projector.to(self.language_model.dtype)

        self.tie_weights()

    def get_projector(self):
        return self.projector

    def get_vision_model(self):
        return self.vision_model

    def get_language_model(self) -> AutoModelForCausalLM:
        return self.language_model

    def get_lm_head(self):
        return self.language_model.lm_head

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.language_model.lm_head

    def init_vision_tokenizer(self, tokenizer, resize_embeds: bool = False):
        ret = tokenizer.add_tokens(
            [self.config.image_token], special_tokens=True)
        assert ret in [0, 1]
        if ret:
            image_token_id = tokenizer.convert_tokens_to_ids(
                self.config.image_token)
            assert self.config.image_token_id in [None, image_token_id]
            self.config.image_token_id = image_token_id
            logger.info(
                f'Added special token {self.config.image_token} to the tokenizer')
        else:
            logger.info(
                f'Special token {self.config.image_token} already exists in the tokenizer')

        if tokenizer.chat_template is None:
            tokenizer.chat_template = VICUNA_V1_JINJA_TEMPLATE
            test_prompt = tokenizer.apply_chat_template([
                dict(role='user', content=[
                    dict(type='text', text='Briefly describe the image.'),
                    dict(type='image')
                ]),
                dict(role='assistant',
                     content=dict(type='text', text='The image shows a ...')),
                dict(role='user', content=[
                    dict(type='text', text='Please repeat it in Italian.')
                ])
            ], add_generation_prompt=True, tokenize=False)
            logger.info(
                f"No chat_template found in the tokenizer. Defaulting to `VICUNA_V1_JINJA_TEMPLATE`. "
                "\n***** CHAT EXAMPLE *****\n"
                f"{test_prompt}"
                "************************\n"
            )

        if resize_embeds:
            lm = self.get_language_model()
            if lm.get_input_embeddings().weight.size(0) < len(tokenizer):
                mean_resizing = False
                lm.resize_token_embeddings(
                    len(tokenizer), pad_to_multiple_of=64, mean_resizing=mean_resizing)
                logger.info('Resized the language model token embeddings')

    def visual_model_forward(self, pixel_values):
        # TODO: use different layers from the visual encoder, keep the CLS, etc.
        if self.config.vision_layer_idx == -1:
            ret = self.vision_model(
                pixel_values=pixel_values).last_hidden_state[:, 1:]
        else:
            ret = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True).hidden_states[self.config.vision_layer_idx][:, 1:]
        return ret.to(self.dtype)

    def get_visual_embeds(self, pixel_values):
        return self.projector(self.visual_model_forward(pixel_values))

    def prepare_causal_mask_with_full_mask_on_image_tokens(
        self,
        attention_mask_2d=None,
        visual_position_ids=None,
        has_image=None
    ) -> Union[Tensor, None]:
        attn_mask_4d = None
        if has_image:
            N, L = attention_mask_2d.shape
            dtype, device = self.dtype, self.device

            # start with a causal mask
            attn_mask_4d = torch.tril(torch.ones(
                (L, L), dtype=dtype, device=device)).unsqueeze(0).expand(N, -1, -1).clone()

            # add full mask among visual tokens
            for i, imgs_pos in enumerate(visual_position_ids):
                for img_pos in imgs_pos:
                    attn_mask_4d[i, img_pos[0]:img_pos[1],
                                 img_pos[0]:img_pos[1]] = 1

            # add padding and make 4d
            attn_mask_4d = (
                attn_mask_4d * attention_mask_2d.unsqueeze(1)).unsqueeze(1)

            attn_mask_4d = attn_mask_4d.masked_fill(
                attn_mask_4d == 0, float('-inf'))
            # from PIL import Image
            # from pathlib import Path
            # for i in range(attn_mask_4d.size(0)):
            #     show_inv_attn_mask = ((1 - attn_mask_4d[i, 0]) * 255).to(torch.uint8).cpu().numpy()
            #     show_inv_attn_mask = Image.fromarray(show_inv_attn_mask)
            #     p = Path('.scratch').joinpath('attn_mask_check')
            #     p.mkdir(parents=True, exist_ok=True)
            #     show_inv_attn_mask.save(p.joinpath(f"attn_mask_{i}.png"))
        else:
            attn_mask_4d = attention_mask_2d
        return attn_mask_4d

    def embed_tokens(
        self,
        batch_input_ids: Tensor,
        batch_attention_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        batch_visual_embeds: Optional[Tensor] = None,
        batch_position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, List[List[List[int]]]]:
        E = self.get_language_model().model.embed_tokens
        if batch_visual_embeds is None:
            return (
                E(batch_input_ids),
                batch_attention_mask,
                batch_labels,
                batch_position_ids,
                [[] for _ in batch_input_ids]
            )

        IMG_TOK = self.config.image_token_id
        has_labels = batch_labels is not None
        all_input_ids = []
        all_attention_mask = []
        all_labels = [] if has_labels else None
        all_visual_position_ids = []
        toks_dtype, device = batch_input_ids.dtype, batch_input_ids.device

        if batch_input_ids.size(0) < batch_visual_embeds.size(0):
            batch_visual_embeds = batch_visual_embeds.view(batch_input_ids.size(0), -1, *batch_visual_embeds.shape[1:])
        if batch_visual_embeds.ndim == 3:
            # (N, C, L) -> (N, ImgsPerSample, C, L)
            batch_visual_embeds = batch_visual_embeds.unsqueeze(1)
        n_vembeds = batch_visual_embeds.size(2)

        # first pass - get visual token positions and replace image tokens with fuzzy ones, e.g. 0
        for i, input_ids in enumerate(batch_input_ids):
            curr_toks = []
            curr_vis_pos = []
            for j, tok in enumerate(input_ids):
                if tok == IMG_TOK:
                    curr_vis_pos.append([j, -1])
                    curr_toks.append(torch.full(
                        (1,), fill_value=0, dtype=toks_dtype, device=device))
                else:
                    curr_toks.append(input_ids[j:j + 1])
            all_input_ids.append(torch.cat(curr_toks))
            all_visual_position_ids.append(curr_vis_pos)

        # embed txt tokens
        all_input_ids = torch.stack(all_input_ids)
        all_inputs_embeds = E(all_input_ids)

        # second pass -> add visual embeds, handle mask and labels, add last position of each image sequence
        all_mm_inputs_embeds = []
        all_attention_mask = []
        all_labels = []

        for i, vis_pos in enumerate(all_visual_position_ids):
            chunked_embeds = []
            chunked_attn_mask = []
            chunked_labels = []
            txt_start = 0
            n_img = 0
            for vp in vis_pos:
                chunked_embeds.extend([
                    all_inputs_embeds[i, txt_start:vp[0]],
                    batch_visual_embeds[i, n_img],
                ])
                chunked_attn_mask.extend([
                    batch_attention_mask[i, txt_start:vp[0]],
                    torch.ones((n_vembeds,), dtype=toks_dtype, device=device)
                ])
                if has_labels:
                    chunked_labels.extend([
                        batch_labels[i, txt_start:vp[0]],
                        torch.full((n_vembeds,), fill_value=-100,
                                   dtype=toks_dtype, device=device)
                    ])
                txt_start = vp[0] + 1
                vp[0] += n_img * n_vembeds
                vp[1] = vp[0] + n_vembeds
                n_img += 1
            if txt_start < len(batch_attention_mask[i]):
                last_no_pad_mask = batch_attention_mask[i, txt_start:].nonzero()
                # if training with iwm with text conditioning, there is nothing after the visual embeds
                # and what comes before has already been added
                if last_no_pad_mask.numel():
                    txt_end = txt_start + last_no_pad_mask[-1].item() + 1
                    chunked_embeds.append(all_inputs_embeds[i, txt_start:txt_end])
                    chunked_attn_mask.append(
                        batch_attention_mask[i, txt_start:txt_end])
                    if has_labels:
                        chunked_labels.append(batch_labels[i, txt_start:txt_end])
                    if not n_img and self.training:
                        # ensure that fuzzy batch_visual_embeds are accessed by this rank
                        # to avoid NCCL mismatch collective errors and hangs
                        chunked_embeds.append(batch_visual_embeds[i, 0])
                        chunked_attn_mask.append(torch.zeros(
                            (n_vembeds,), dtype=toks_dtype, device=device))
                        if has_labels:
                            chunked_labels.append(torch.full(
                                (n_vembeds,), fill_value=-100, dtype=toks_dtype, device=device))

            all_mm_inputs_embeds.append(torch.cat(chunked_embeds))
            all_attention_mask.append(torch.cat(chunked_attn_mask))
            if has_labels:
                all_labels.append(torch.cat(chunked_labels))

        all_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            all_mm_inputs_embeds, True, 0.)
        del all_mm_inputs_embeds
        all_attention_mask = torch.nn.utils.rnn.pad_sequence(
            all_attention_mask, True, 0)
        if has_labels:
            all_labels = torch.nn.utils.rnn.pad_sequence(
                all_labels, True, -100)
        else:
            all_labels = None

        # TODO: check if position_ids are needed

        if self.config.full_mask_image_tokens:
            all_attention_mask = self.prepare_causal_mask_with_full_mask_on_image_tokens(
                all_attention_mask, all_visual_position_ids, has_image=True)

        return all_inputs_embeds, all_attention_mask, all_labels, batch_position_ids, all_visual_position_ids

    def embed_tokens_with_pixel_values(
        self,
        batch_input_ids: Tensor,
        batch_attention_mask: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        batch_pixel_values: Optional[Tensor] = None,
        batch_position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[List[List[int]]]]:
        batch_visual_embeds = None
        if batch_pixel_values is not None:
            batch_visual_embeds = self.get_visual_embeds(batch_pixel_values)
        return self.embed_tokens(batch_input_ids, batch_attention_mask, batch_labels, batch_visual_embeds, batch_position_ids)


@dataclass
class IWMOutput(ModelOutput):
    loss: Optional[Tensor] = None


class MLLMForIWM(MLLMPreTrainedModel):
    supports_gradient_checkpointing = True
    required_losses = ["CrossEntropy", "SmoothL1"]

    def __init__(
        self,
        config: MLLMConfig,
        language_model: Optional[AutoModelForCausalLM] = None,
        vision_model: Optional[PreTrainedModel] = None,
        projector: Optional[nn.Module] = None,
        losses_functions_to_apply: Optional[Dict[str, bool]] = None,
        lm_kwargs: Dict = {},
        masked_lm: bool = True,
        **kwargs
    ):
        super().__init__(config, language_model=language_model, vision_model=vision_model, projector=projector, lm_kwargs=lm_kwargs, **kwargs)

        self.z = nn.Parameter(torch.empty(
            (1, 1, self.config.text_config.hidden_size), dtype=self.dtype))
        nn.init.normal_(
            self.z, mean=0, std=self.config.text_config.initializer_range)
        tgt_proj_out_size = self.config.iwm_tgt_proj_output_size
        if not tgt_proj_out_size:
            tgt_proj_out_size = self.config.vision_config.hidden_size
        self.target_proj = nn.Linear(self.config.hidden_size, tgt_proj_out_size, bias=False)
        if losses_functions_to_apply is None:
            losses_functions_to_apply = {loss: True for loss in self.required_losses}
        self.losses_functions_to_apply = losses_functions_to_apply
        self.num_losses = sum([1 for v in self.losses_functions_to_apply.values() if v])
        self.masked_lm = masked_lm

    @classmethod
    def from_pretrained_MLLMForCausalLM(
        cls,
        name_or_path: str,
        **kwargs
    ):
        causal_model = MLLMForCausalLM.from_pretrained(name_or_path)
        vision_model = causal_model.get_vision_model()

        if isinstance(vision_model, CLIPVisionModel):
            vision_model = CLIPVisionModelWithProjection._from_config(vision_model.config)
            vision_model.load_state_dict(vision_model.state_dict(), strict=False)

        return cls(
            causal_model.config,
            language_model=causal_model.get_language_model(),
            vision_model=vision_model,
            projector=causal_model.get_projector(),
            masked_lm=False,
            **kwargs
        )


    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values_enc: Optional[Tensor] = None,
        pixel_values_pred: Optional[Tensor] = None,
        mask_idxs_enc: Optional[Tensor] = None,
        mask_idxs_pred: Optional[Tensor] = None
    ) -> Union[IWMOutput]:
        
        torch.cuda.empty_cache()
        bsz = input_ids.size(0) if input_ids is not None else pixel_values_pred.size(0)
        n_pred_masks = mask_idxs_pred.size(1)
        lm = self.get_language_model()
        vision_model = self.get_vision_model().to(memory_format=torch.channels_last)
        pixel_values_pred = pixel_values_pred.to(memory_format=torch.channels_last)

        if self.config.iwm_tgt_vision_layer_idx == -1:
            vision_embeds_pred = vision_model(pixel_values_pred).last_hidden_state[:, 1:]
        else:
            vision_embeds_pred = vision_model(pixel_values_pred, output_hidden_states=True).hidden_states[self.config.iwm_tgt_vision_layer_idx][:, 1:]

        if self.config.iwm_full_img_on_encoder:
            enc_mask = torch.zeros(bsz, vision_embeds_pred.size(1), device=self.device, dtype=vision_embeds_pred.dtype)
            enc_mask.scatter_(1, mask_idxs_enc, 1)
            vision_embeds_enc = self.projector(vision_embeds_pred * enc_mask.unsqueeze(-1))
        else:
            vision_embeds_enc = self.get_visual_embeds(pixel_values_enc)

        if self.config.iwm_tgt_vision_model_proj_head:
            vision_embeds_pred_proj = vision_model.visual_projection(vision_embeds_pred)
        else:
            vision_embeds_pred_proj = vision_embeds_pred

        H = self.config.hidden_size
        vision_embeds_enc_exp = vision_embeds_enc.repeat_interleave(n_pred_masks, dim=0)  # (N*n_pred_masks, P, H)
        mask_idxs_pred_exp = mask_idxs_pred.flatten(0, 1)  # (N*n_pred_masks, CP)
        z = self.z.expand(*mask_idxs_pred_exp.shape, -1).to(vision_embeds_enc_exp.dtype)  # (N*n_pred_masks, CP, H)
        vision_embeds_enc_exp = vision_embeds_enc_exp.scatter_(1, mask_idxs_pred_exp.unsqueeze(2).expand(-1, -1, H), z)

        input_ids_exp = input_ids.repeat_interleave(n_pred_masks, 0)
        attention_mask_exp = attention_mask.repeat_interleave(n_pred_masks, 0)
        inputs_embeds_exp, attention_mask_exp, _, _, visual_position_ids_exp = self.embed_tokens(input_ids_exp, attention_mask_exp, None, vision_embeds_enc_exp)
        last_hidden_state_exp = lm.model(inputs_embeds=inputs_embeds_exp, attention_mask=attention_mask_exp).last_hidden_state

        visual_position_ids_exp = torch.tensor(visual_position_ids_exp, dtype=torch.long, device=self.device).squeeze(1)
        img_start_idx_exp = visual_position_ids_exp[:, 0:1, None]

        vision_embeds_pred_proj_for_lm = self.projector(vision_embeds_pred)  # always project to LM hidden size
        inputs_embeds_ce, attention_mask_ce, _, _, visual_position_ids_ce = self.embed_tokens(input_ids, attention_mask, None, vision_embeds_pred_proj_for_lm)
        last_hidden_state_ce = lm.model(inputs_embeds=inputs_embeds_ce, attention_mask=attention_mask_ce).last_hidden_state
        visual_position_ids_ce = torch.tensor(visual_position_ids_ce, dtype=torch.long, device=self.device).squeeze(1)

        loss = 0

        if self.losses_functions_to_apply.get("SmoothL1", True):
            mask_idxs_pred_exp_unsq = mask_idxs_pred_exp.unsqueeze(2)
            preds = last_hidden_state_exp.gather(1, (mask_idxs_pred_exp_unsq + img_start_idx_exp).expand(-1, -1, H))
            preds = self.target_proj(preds)
            targets = vision_embeds_pred_proj.repeat_interleave(n_pred_masks, 0).gather(1, mask_idxs_pred_exp_unsq.expand(-1, -1, preds.size(-1)))
            loss += F.smooth_l1_loss(preds.flatten(0, 1), targets.flatten(0, 1))

        if labels is not None and self.losses_functions_to_apply.get("CrossEntropy", True):
            img_start_idx, img_end_idx = visual_position_ids_ce[0].tolist()
            labels_ce = labels[labels != self.config.image_token_id].reshape(bsz, -1)[:, 1:]
            logits_ce = lm.lm_head(torch.cat([last_hidden_state_ce[:, :img_start_idx], last_hidden_state_ce[:, img_end_idx:]], dim=1))[:, :-1]
            loss += F.cross_entropy(logits_ce.flatten(0, 1), labels_ce.flatten(), ignore_index=-100)

        loss /= self.num_losses
        return IWMOutput(loss=loss)
    
    def set_losses(self, losses_functions_to_apply: Dict[str, bool]):
        """
        Set the losses to apply during training.
        
        :param losses_functions_to_apply: A dictionary where keys are loss function names and values are booleans indicating whether to apply the loss.
            Supported keys are "CrossEntropy" and "SmoothL1".
        """
        self.losses_functions_to_apply = losses_functions_to_apply
        for loss in self.required_losses:
            if loss not in self.losses_functions_to_apply:
                self.losses_functions_to_apply[loss] = False
        self.num_losses = sum([1 for v in self.losses_functions_to_apply.values() if v])



class MLLMForCausalLM(MLLMPreTrainedModel, GenerationMixin):
    supports_gradient_checkpointing = True

    @classmethod
    def from_pretrained_MLLMForIWM(
        cls,
        name_or_path: str,
        pretrained_kwargs: Dict = {},
        **kwargs
    ):
        model = MLLMForIWM.from_pretrained(name_or_path, **pretrained_kwargs)
        return cls(
            model.config,
            language_model=model.get_language_model(),
            vision_model=model.get_vision_model(),
            projector=model.get_projector(),
            **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache,
                                        List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds, attention_mask, labels, position_ids, _ = self.embed_tokens_with_pixel_values(
                input_ids, attention_mask, labels, pixel_values, position_ids)
            input_ids = None
        
        # assuming the LLM uses RoPE, we do not need, `position_ids` for most cases.
        # If we do not set it to None, then there will be problem in `prepare_inputs_for_generation` if using full attention mask on image tokens
        position_ids = None

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        putback_att_mask = False
        attention_mask = model_kwargs.pop('attention_mask', None)
        if attention_mask is None or attention_mask.ndim == 2:
            model_kwargs['attention_mask'] = attention_mask
        elif attention_mask.ndim == 4:
            putback_att_mask = True
            N, _, L = attention_mask.shape[:3]
            dtype, device = attention_mask.dtype, attention_mask.device
            attention_mask = torch.ones((N, L + 1), dtype=dtype, device=device)

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens)

        if putback_att_mask:
            model_kwargs['attention_mask'] = attention_mask
        return model_kwargs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if "inputs_embeds" in kwargs:
            raise NotImplementedError(
                "`inputs_embeds` is not supported in generation")
        if 'position_ids' in kwargs:
            raise NotImplementedError(
                "`position_ids` is not supported in generation")

        inputs_embeds, attention_mask, _, position_ids, __ = self.embed_tokens_with_pixel_values(
            batch_input_ids=input_ids, batch_attention_mask=attention_mask, batch_pixel_values=pixel_values)

        return super().generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs
        )


__all__ = [
    "MLLMPreTrainedModel",
    "MLLMForIWM",
    "IWMOutput",
    "MLLMForCausalLM",
]
