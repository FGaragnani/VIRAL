import copy
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model. Error: %s" % e)

@register_model("viral")
class VIRAL(lmms):
    """
    VIRAL wrapper model that loads an underlying LLaVA-style model using
    `load_pretrained_model` and implements the lmms interface.
    """

    def __init__(
        self,
        name_or_path: str = "./checkpoints/viral-7b",
        base: str = "lmsys/vicuna-7b-v1.5",
        device: str = "cuda:0",
        batch_size: int = 1,
        model_name: Optional[str] = None,
        device_map: str = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        attn_implementation: Optional[str] = None,
        image_aspect_ratio: Optional[float] = None,
        use_cache: bool = True,
        tie_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        model_name = model_name if model_name is not None else get_model_name_from_path(name_or_path)

        # prepare kwargs for loader
        loader_kwargs = {}
        if isinstance(dtype, str):
            if dtype != "auto":
                loader_kwargs["torch_dtype"] = getattr(torch, dtype)
        elif dtype is not None:
            loader_kwargs["torch_dtype"] = dtype

        if attn_implementation is not None:
            loader_kwargs["attn_implementation"] = attn_implementation


        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            name_or_path, base, model_name, device_map=self.device_map, **loader_kwargs
        )

        self._config = getattr(self._model, "config", None)
        # Ensure image processor is always set
        if self._image_processor is None:
            try:
                from transformers import AutoImageProcessor
                # Try multiple fallback strategies to find vision processor
                vision_paths = []
                
                # 1. Check model config for vision tower path
                if self._config is not None:
                    if hasattr(self._config, "vision_tower") and self._config.vision_tower:
                        vision_paths.append(self._config.vision_tower)
                    if hasattr(self._config, "mm_vision_tower") and self._config.mm_vision_tower:
                        vision_paths.append(self._config.mm_vision_tower)
                    if hasattr(self._config, "vision_config") and hasattr(self._config.vision_config, "name_or_path"):
                        vision_paths.append(self._config.vision_config.name_or_path)
                
                # 2. Common vision model defaults for LLaVA-style models
                vision_paths.extend([
                    "openai/clip-vit-large-patch14-336",
                    "openai/clip-vit-large-patch14",
                    model_name or name_or_path
                ])
                
                # Try each path until one works
                for vision_path in vision_paths:
                    try:
                        eval_logger.debug(f"VIRAL: Trying to load image processor from {vision_path}")
                        self._image_processor = AutoImageProcessor.from_pretrained(vision_path)
                        eval_logger.info(f"VIRAL: Successfully loaded image processor from {vision_path}")
                        break
                    except Exception as e:
                        eval_logger.debug(f"VIRAL: Failed to load image processor from {vision_path}: {e}")
                        continue
                
                if self._image_processor is None:
                    eval_logger.error(f"VIRAL: Could not load image processor from any source. Tried: {vision_paths}")
                    
            except Exception as e:
                eval_logger.warning(f"VIRAL: Could not automatically load image processor: {e}")
                self._image_processor = None

        # optional user-specified image aspect ratio
        if image_aspect_ratio is not None and self._config is not None:
            try:
                self._config.image_aspect_ratio = image_aspect_ratio
            except Exception:
                pass
        self.model.eval()
        if tie_weights:
            try:
                self.model.tie_weights()
            except Exception:
                pass

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        # default conversation template name used by other models
        self.conv_template = "vicuna_v1"

        # accelerator/device placement
        if accelerator.num_processes > 1:
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)

            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._rank = 0
            self._world_size = 1
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except Exception:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def _get_doc(self, task: str, split: str, doc_id: int):
        """Defensive lookup for a document in self.task_dict.

        Tries direct lookup first, then attempts fuzzy matching on keys
        (prefix/substring) to help when task naming differs between
        Task.config.task and the keys stored by the evaluator.
        Returns the doc or None if not found.
        """
        # If task_dict is missing or empty, try to lazily populate it from
        # the tasks module. This is a best-effort fallback for cases where the
        # evaluator did not set lm.task_dict (e.g., custom calling flows).
        if not hasattr(self, "task_dict") or not self.task_dict:
            try:
                from lmms_eval.tasks import get_task_dict

                # attempt to fetch only the requested task mapping
                task_map = get_task_dict([task], None)
                if task_map:
                    self.task_dict = task_map
            except Exception:
                eval_logger.debug("VIRAL._get_doc: no lm.task_dict available on model")
                return None

        # direct lookup
        try:
            task_map = self.task_dict.get(task)
            if task_map and split in task_map:
                docs = task_map[split]
                return docs[doc_id]
        except Exception:
            # fallthrough to fuzzy matching
            pass

        # fuzzy match: look for keys that match or contain the requested task
        for key, task_map in self.task_dict.items():
            try:
                if not isinstance(key, str):
                    continue
                if key == task or key.startswith(task) or (task in key):
                    if split in task_map:
                        docs = task_map[split]
                        return docs[doc_id]
            except Exception:
                continue

        eval_logger.warning(f"VIRAL._get_doc: couldn't find doc for task={task}, split={split}, doc_id={doc_id}. Available task_dict keys: {list(self.task_dict.keys())}")
        return None

    def _ensure_tensor(self, x):
        """Ensure x is a torch tensor on the model device.

        tokenizer_image_token sometimes returns a tensor or a list; this helper
        normalizes common cases so callers can safely call .unsqueeze/.to.
        """
        if isinstance(x, list):
            # If it's a list of tensors, try to stack if shapes agree, otherwise return list
            if len(x) == 0:
                return None
            if all(hasattr(el, "unsqueeze") for el in x):
                try:
                    return torch.stack(x, dim=0).to(self.device)
                except Exception:
                    return x
            else:
                return x
        else:
            # assume tensor-like
            try:
                return x.to(self.device)
            except Exception:
                return x

    def _ensure_image_processor(self) -> bool:
        """Attempt to lazily populate self._image_processor if missing.

        Returns True if an image processor is available after this call.
        """
        if getattr(self, "_image_processor", None) is not None:
            return True
        try:
            # try to create an AutoImageProcessor from model config if possible
            from transformers import AutoImageProcessor

            cfg = getattr(self, "_config", None)
            if cfg is not None and hasattr(cfg, "vision_config") and getattr(cfg.vision_config, "name_or_path", None):
                try:
                    self._image_processor = AutoImageProcessor.from_pretrained(cfg.vision_config.name_or_path)
                    return True
                except Exception:
                    pass
        except Exception:
            # transformers not available or other issue; fall through
            pass
        return False

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Basic implementation similar to Llava/LlavaHf
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # resolve document defensively
            doc = self._get_doc(task, split, doc_id)
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(doc) if doc is not None else ""
            visuals = [doc_to_visual(doc) if doc is not None else None]
            visuals = self.flatten(visuals)

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts
            if visuals and DEFAULT_IMAGE_TOKEN not in prompts_input:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + prompts_input

            if "llama_3" in getattr(self, "conv_template", ""):
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[getattr(self, "conv_template", "vicuna_v1")].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id_raw = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            contxt_id = self._ensure_tensor(contxt_id_raw)
            if hasattr(contxt_id, "unsqueeze"):
                contxt_id = contxt_id.unsqueeze(0)
            conv.messages[1][1] = continuation
            prompt = conv.get_prompt()
            input_ids_raw = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = self._ensure_tensor(input_ids_raw)
            if hasattr(input_ids, "unsqueeze"):
                input_ids = input_ids.unsqueeze(0)
            labels = input_ids.clone()
            labels[0, : contxt_id.shape[1]] = -100
            if visuals:
                if not self._ensure_image_processor():
                    eval_logger.warning(f"VIRAL.loglikelihood: no image_processor available; skipping image processing for task={task}, doc_id={doc_id}")
                    image = None
                else:
                    image = process_images(visuals, self._image_processor, self._config)
                if isinstance(image, list):
                    try:
                        image = [
                            _image.to(dtype=torch.float16, device=self.device) if hasattr(_image, "to") else _image
                            for _image in image
                        ]
                    except Exception:
                        pass
                else:
                    if hasattr(image, "to"):
                        image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True)
            loss = outputs.get("loss")
            logits = outputs.get("logits")
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            # resolve docs defensively for the batch
            batched_visuals = []
            for ids in doc_id:
                doc = self._get_doc(task, split, ids)
                if doc is None:
                    eval_logger.warning(f"VIRAL.generate_until: No doc found for task={task}, split={split}, doc_id={ids}. Skipping this item.")
                    batched_visuals.append(None)
                else:
                    try:
                        batched_visuals.append(doc_to_visual[0](doc))
                    except Exception as e:
                        eval_logger.warning(f"VIRAL.generate_until: doc_to_visual failed for task={task}, split={split}, doc_id={ids}: {e}")
                        batched_visuals.append(None)
            flattened_visuals = self.flatten(batched_visuals)
            gen_kwargs = all_gen_kwargs[0]

            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]

            if flattened_visuals:
                eval_logger.debug(f"VIRAL.generate_until: Processing {len(flattened_visuals)} images for task={task}")
                if not self._ensure_image_processor():
                    eval_logger.warning(f"VIRAL.generate_until: no image_processor available; skipping image processing for task={task}")
                    image_tensor = None
                else:
                    eval_logger.debug(f"VIRAL.generate_until: Image processor available, calling process_images")
                    image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                    eval_logger.debug(f"VIRAL.generate_until: process_images returned: {type(image_tensor)}, shape: {getattr(image_tensor, 'shape', 'N/A') if hasattr(image_tensor, 'shape') else len(image_tensor) if isinstance(image_tensor, list) else 'N/A'}")
                # Robustly handle image_tensor type
                if isinstance(image_tensor, list):
                    # Remove None or empty images
                    image_tensor = [_img for _img in image_tensor if _img is not None and hasattr(_img, 'to')]
                    if len(image_tensor) == 0:
                        image_tensor = None
                    else:
                        try:
                            image_tensor = torch.stack([_img.to(dtype=torch.float16, device=self.device) for _img in image_tensor])
                        except Exception as e:
                            eval_logger.error(f"VIRAL.generate_until: Could not stack or move image_tensor to device: {e}")
                            image_tensor = None
                elif image_tensor is not None and hasattr(image_tensor, "to"):
                    try:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    except Exception as e:
                        eval_logger.error(f"VIRAL.generate_until: Could not move image_tensor to device: {e}")
                        image_tensor = None
                else:
                    image_tensor = None
            else:
                image_tensor = None

            question_input = []
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                if "llama_3" in getattr(self, "conv_template", ""):
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[getattr(self, "conv_template", "vicuna_v1")].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))] if flattened_visuals else []
            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            try:
                # VIRAL models don't accept images in generate(), but we can try different approaches
                generate_kwargs: dict = dict(
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=pad_token_ids,
                    attention_mask=attention_masks,
                )
                
                # Try different generation strategies based on whether we have images
                if image_tensor is not None:
                    # Strategy 1: Try to use generate with input_embeds if available
                    try:
                        # Get input embeddings that include image features
                        with torch.inference_mode():
                            model_inputs = self.model.prepare_inputs_for_generation(
                                input_ids, images=image_tensor, **generate_kwargs
                            )
                        cont = self.model.generate(**model_inputs)
                        eval_logger.debug("VIRAL.generate_until: Used prepare_inputs_for_generation with images")
                    except Exception as e1:
                        eval_logger.debug(f"VIRAL.generate_until: prepare_inputs_for_generation failed: {e1}")
                        # Strategy 2: Try standard generate without images (images might be embedded in input_ids)
                        try:
                            cont = self.model.generate(input_ids, **generate_kwargs)
                            eval_logger.warning("VIRAL.generate_until: Generated without passing images explicitly - images may be embedded in tokens")
                        except Exception as e2:
                            eval_logger.error(f"VIRAL.generate_until: All generation strategies failed: {e1}, {e2}")
                            raise e2
                else:
                    # No images, standard generation
                    cont = self.model.generate(input_ids, **generate_kwargs)
                    
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                text_outputs = [""] * len(question_input)

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for VIRAL")