from typing import Dict, Optional, Union
from transformers import PretrainedConfig, AutoConfig
from enum import StrEnum
from ..conversations import DEFAULT_IMAGE_TOKEN


class ProjectorType(StrEnum):
    LINEAR = 'linear'
    MLP = 'mlp'


class MLLMConfig(PretrainedConfig):
    model_type = 'mllm'
    is_composition = True
    image_token = DEFAULT_IMAGE_TOKEN

    def __init__(
        self,
        text_config: Optional[Union[PretrainedConfig, Dict]] = None,
        vision_config: Optional[Union[PretrainedConfig, Dict]] = None,
        projector_type: Optional[ProjectorType] = ProjectorType.LINEAR,
        projector_input_size: Optional[int] = None,
        projector_output_size: Optional[int] = None,
        projector_tie_weights: Optional[bool] = True,
        projector_bias: Optional[bool] = False,
        image_token_id: Optional[int] = None,
        full_mask_image_tokens: bool = True,
        vision_layer_idx: int = -2,
        iwm_tgt_vision_layer_idx: int = -1,
        iwm_tgt_vision_model_proj_head: bool = False,
        iwm_tgt_proj_output_size: int = 0,
        iwm_full_img_on_encoder: bool = False,
        **kwargs
    ):
        assert full_mask_image_tokens is not None
        super().__init__(**kwargs)

        if isinstance(text_config, PretrainedConfig) or text_config is None:
            self.text_config = text_config
        else:
            self.text_config = AutoConfig.for_model(text_config.pop('model_type'), **text_config)

        if isinstance(vision_config, PretrainedConfig) or vision_config is None:
            self.vision_config = vision_config
        else:
            self.vision_config = AutoConfig.for_model(vision_config.pop('model_type'), **vision_config)
        
        self.projector_type = projector_type
        self.projector_input_size = projector_input_size
        self.projector_output_size = projector_output_size
        self.projector_tie_weights = projector_tie_weights
        self.projector_bias = projector_bias
        self.image_token_id = image_token_id
        self.full_mask_image_tokens = full_mask_image_tokens
        self.vision_layer_idx = vision_layer_idx
        self.iwm_tgt_vision_layer_idx = iwm_tgt_vision_layer_idx
        self.iwm_tgt_vision_model_proj_head = iwm_tgt_vision_model_proj_head
        self.iwm_tgt_proj_output_size = iwm_tgt_proj_output_size
        self.iwm_full_img_on_encoder = iwm_full_img_on_encoder

        self.maybe_init_defaults()

    def maybe_init_defaults(self):
        if self.projector_input_size is None and self.vision_config is not None:
            self.projector_input_size = self.vision_config.hidden_size
        if self.projector_output_size is None and self.text_config is not None:
            self.projector_output_size = self.text_config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size
    

__all__ = [
    "ProjectorType",
    "MLLMConfig"
]