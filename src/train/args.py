from dataclasses import dataclass, field
from typing import Optional, List
from src.models.conversations import CONV_TEMPLATES
from transformers import TrainingArguments
from src.models import ProjectorType
import src.custom_utils as utils

logger = utils.get_logger()

@dataclass
class CustomTrainingArguments(TrainingArguments):
    resume_from_last_checkpoint: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.resume_from_last_checkpoint and self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = True
            logger.info(f"Setting `resume_from_checkpoint` to True because `resume_from_last_checkpoint` is set to True and no checkpoint was specified in `resume_from_checkpoint`.")


@dataclass
class DataArguments:
    train_data_path: Optional[str] = None
    train_image_folder: Optional[str] = None
    train_wds_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    eval_image_folder: Optional[str] = None
    conv_template: str = CONV_TEMPLATES.PLAIN
    prompt_max_length: Optional[int] = None
    image_aspect_ratio: Optional['str'] = None
    iwm_captions: bool = False
    iwm_txt_cond: bool = False

    wds_buffer_size: Optional[int] = None
    
    # IJEPA
    ijepa_aspect_ratio: List[float] = field(default_factory=lambda: [0.75, 1.5])
    ijepa_enc_mask_scale: List[float] = field(default_factory=lambda: [0.85, 1.0])
    ijepa_min_keep: int = 10
    ijepa_num_enc_masks: int = 1
    ijepa_num_pred_masks: int = 4
    ijepa_pred_mask_scale: List[float] = field(default_factory=lambda: [0.15, 0.2])    


@dataclass
class ModelArguments:
    iwm_loss: bool = False
    instruction_tuning: bool = False
    train_proj_only: bool = False
    attn_implementation: str = 'sdpa'
    stage: int = 0

    # from checkpoint
    model_name: Optional[str] = None

    # from scratch
    language_model_name: Optional[str] = None
    vision_model_name: Optional[str] = None
    vision_layer_idx: int = -2
    iwm_tgt_vision_layer_idx: int = -1
    iwm_tgt_vision_model_proj_head: bool = False
    iwm_tgt_proj_output_size: int = 0
    iwm_full_img_on_encoder: bool = True
    projector_type: Optional[ProjectorType] = ProjectorType.LINEAR
    projector_bias: bool = True
    projector_input_size: Optional[int] = None
    projector_output_size: Optional[int] = None
    projector_tie_weights: Optional[bool] = True
    full_mask_image_tokens: Optional[bool] = None


def postprocess_args(training_args: TrainingArguments, model_args: ModelArguments, data_args: DataArguments):
    training_args.iwm_loss = model_args.iwm_loss
    training_args.instruction_tuning = model_args.instruction_tuning
    training_args.train_proj_only = model_args.train_proj_only
    training_args.stage = model_args.stage

    data_args.iwm_loss = model_args.iwm_loss
    data_args.stage = model_args.stage
    data_args.instruction_tuning = model_args.instruction_tuning    

    return training_args, model_args, data_args