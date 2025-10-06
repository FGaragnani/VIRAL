from transformers import logging
import os
import torch.distributed as dist

logging.set_verbosity_info()
logging.enable_explicit_format()


def get_logger():
    return logging.get_logger("transformers")


def is_debug():
    return int(os.getenv('DEBUG', 0))


def is_debug_one_layer_mode():
    return int(os.getenv('DEBUG_ONE_LAYER_MODE', 0))


def is_debug_skip_pca_train():
    return int(os.getenv('DEBUG_SKIP_PCA_TRAIN', 0))


def is_debug_n_dataset_samples():
    return int(os.getenv('DEBUG_N_DATASET_SAMPLES', 0))


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    return int(os.getenv('LOCAL_RANK'))


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_image_size_from_image_processor(image_processor):
    if hasattr(image_processor, 'crop_size'):
        # CLIP, DiNO
        return image_processor.crop_size['height'], image_processor.crop_size['width']
    elif hasattr(image_processor, 'size'):
        # I-JEPA, SigLIP
        return image_processor.size['height'], image_processor.size['width']