import copy
import os
import json
import inspect
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

        # Determine model_name; if it's not clearly a LLaVA model but the checkpoint looks multimodal,
        # force a name that contains "llava" so the builder loads the correct subclass (matches training code).
        inferred_name = model_name if model_name is not None else get_model_name_from_path(name_or_path)
        try:
            ckpt_dir = os.path.abspath(name_or_path)
            cfg_path = os.path.join(ckpt_dir, "config.json")
            cfg_llava = False
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_json = json.load(f)
                mt = str(cfg_json.get("model_type", ""))
                cfg_llava = ("llava" in mt.lower()) or bool(cfg_json.get("mm_vision_tower")) or bool(cfg_json.get("vision_tower"))
            mm_proj_exists = os.path.isfile(os.path.join(ckpt_dir, "mm_projector.bin"))
            if ("llava" not in inferred_name.lower()) and (cfg_llava or mm_proj_exists):
                eval_logger.debug(f"VIRAL: Overriding model_name to include 'llava' based on checkpoint contents: {inferred_name} -> llava-{inferred_name}")
                inferred_name = f"llava-{inferred_name}"
        except Exception as _e:
            eval_logger.debug(f"VIRAL: Could not infer LLaVA nature from checkpoint: {_e}")
        model_name = inferred_name

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
        # Introspect whether this model's generate() accepts images (as in Llava* classes)
        try:
            sig = inspect.signature(self.model.generate)
            self._accepts_image_generate = ("images" in sig.parameters)
        except Exception:
            self._accepts_image_generate = False
        eval_logger.debug(
            f"VIRAL: Model class={self.model.__class__.__name__}, accepts images in generate: {self._accepts_image_generate}"
        )
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

    def _vision_device_dtype(self):
        """Retrieve the device and dtype used by the model's vision tower.

        Returns a tuple (device, dtype). Falls back to (self.device, model dtype or torch.float16).
        """
        try:
            vt = self.model.get_vision_tower() if hasattr(self.model, 'get_vision_tower') else None
            if vt is not None:
                try:
                    p = next(vt.parameters())
                    return p.device, p.dtype
                except Exception:
                    # try attribute-based
                    dev = getattr(vt, 'device', self.device)
                    dt = getattr(vt, 'dtype', getattr(self.model, 'dtype', torch.float16))
                    return dev, dt
        except Exception:
            pass
        return self.device, getattr(self.model, 'dtype', torch.float16)

    def _should_debug(self, task: str, split: str, doc_id: int, gen_kwargs: dict) -> bool:
        """Determine whether to emit deep debug logs for a specific request.

        Priority:
        - gen_kwargs['debug'] truthy enables for this request
        - Environment VIRAL_DEBUG enables globally with optional filters:
          VIRAL_DEBUG_TASK, VIRAL_DEBUG_SPLIT, VIRAL_DEBUG_DOC_ID
        """
        try:
            if isinstance(gen_kwargs, dict) and gen_kwargs.get("debug", False):
                return True
        except Exception:
            pass
        try:
            if str(os.getenv("VIRAL_DEBUG", "")).strip() not in ("", "0", "false", "False", "no"):
                want_task = os.getenv("VIRAL_DEBUG_TASK")
                want_split = os.getenv("VIRAL_DEBUG_SPLIT")
                want_doc = os.getenv("VIRAL_DEBUG_DOC_ID")
                if want_task and str(task) != str(want_task):
                    return False
                if want_split and str(split) != str(want_split):
                    return False
                if want_doc and str(doc_id) != str(want_doc):
                    return False
                return True
        except Exception:
            pass
        return False

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Basic implementation for Simple model inputs
        res: List[Tuple[float, bool]] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for reg in requests:
            # Unpack with resilience
            tup = reg.args if isinstance(reg.args, (list, tuple)) else (reg.args,)
            if len(tup) == 6:
                contexts, doc_to_target, doc_to_visual, doc_id, task, split = tup  # type: ignore[misc]
            else:
                contexts = tup[0] if len(tup) > 0 else ""
                doc_to_target = tup[1] if len(tup) > 1 else (lambda _doc: "")
                doc_to_visual = tup[2] if len(tup) > 2 else (lambda _doc: None)
                doc_id = tup[3] if len(tup) > 3 else -1
                task = tup[4] if len(tup) > 4 else ""
                split = tup[5] if len(tup) > 5 else ""

            # Resolve doc, continuation, visuals
            doc = self._get_doc(task, split, doc_id)
            continuation = doc_to_target if isinstance(doc_to_target, str) else (doc_to_target(doc) if doc is not None else "")
            try:
                visuals = doc_to_visual(doc) if doc is not None else None
            except Exception:
                visuals = None

            # Build prompt with optional image tokens
            prompt_text = contexts[0] if isinstance(contexts, list) else contexts
            if visuals is not None and DEFAULT_IMAGE_TOKEN not in prompt_text:
                num_imgs = len(visuals) if isinstance(visuals, list) else 1
                image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_imgs)
                prompt_text = f"{image_tokens}\n{prompt_text}"

            if "llama_3" in getattr(self, "conv_template", ""):
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[getattr(self, "conv_template", "vicuna_v1")].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            ctx_prompt = conv.get_prompt()

            # Tokenize context only
            ctx_ids_raw = tokenizer_image_token(ctx_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            if isinstance(ctx_ids_raw, list):
                try:
                    parts = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in ctx_ids_raw]
                    ctx_ids = torch.cat(parts, dim=0)
                except Exception:
                    first = ctx_ids_raw[0]
                    ctx_ids = first if isinstance(first, torch.Tensor) else torch.tensor(first)
            else:
                ctx_ids = ctx_ids_raw
            if ctx_ids.dim() == 1:
                ctx_ids = ctx_ids.unsqueeze(0)
            ctx_ids = ctx_ids.to(self.device)

            # Tokenize context + continuation
            conv.messages[1][1] = continuation
            full_prompt = conv.get_prompt()
            inp_ids_raw = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            if isinstance(inp_ids_raw, list):
                try:
                    parts = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in inp_ids_raw]
                    input_ids = torch.cat(parts, dim=0)
                except Exception:
                    first = inp_ids_raw[0]
                    input_ids = first if isinstance(first, torch.Tensor) else torch.tensor(first)
            else:
                input_ids = inp_ids_raw
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)

            # Build labels to mask out context
            labels = input_ids.clone()
            ctx_len = int(ctx_ids.shape[1])
            labels[0, :ctx_len] = -100

            # Process visuals
            images_arg = None
            if visuals is not None:
                if self._ensure_image_processor():
                    try:
                        processed = process_images(visuals if isinstance(visuals, list) else [visuals], self._image_processor, self._config)
                        vis_device, vis_dtype = self._vision_device_dtype()
                        if isinstance(processed, list):
                            images_arg = []
                            for _img in processed:
                                if _img is None:
                                    continue
                                images_arg.append(_img.to(dtype=vis_dtype, device=vis_device))
                            if len(images_arg) == 0:
                                images_arg = None
                        else:
                            images_arg = processed.to(dtype=vis_dtype, device=vis_device)
                    except Exception:
                        images_arg = None

            # Forward to compute loss and greedy match
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=images_arg, use_cache=True)

            loss = float(outputs.loss.item()) if hasattr(outputs, "loss") else float(outputs.get("loss").item())  # type: ignore[call-arg]
            logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")  # type: ignore[call-arg]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, ctx_len:]
            greedy_slice = greedy_tokens[:, ctx_len : input_ids.shape[1]]
            max_equal = bool((greedy_slice == cont_toks).all().item())
            res.append((loss, max_equal))
            pbar.update(1)
        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate outputs for Simple Model (Legacy) requests.

        Each Instance.args is expected to be a 6-tuple:
        (contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split)
        """
        results: List[str] = []

        # simple progress bar over individual requests for robustness
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for reg in requests:
            # Robustly unpack tuple according to Simple model contract
            tup = reg.args if isinstance(reg.args, (list, tuple)) else (reg.args,)
            if len(tup) == 6:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = tup  # type: ignore[misc]
            else:
                # Fallback for unexpected shapes; try to unpack common subset
                contexts = tup[0] if len(tup) > 0 else ""
                all_gen_kwargs = tup[1] if len(tup) > 1 else {}
                doc_to_visual = tup[2] if len(tup) > 2 else (lambda _doc: None)
                doc_id = tup[3] if len(tup) > 3 else -1
                task = tup[4] if len(tup) > 4 else ""
                split = tup[5] if len(tup) > 5 else ""

            # Resolve doc and visuals (Simple model criterion)
            doc = self._get_doc(task, split, doc_id)
            try:
                visuals = doc_to_visual(doc) if doc is not None else None
            except Exception as e:
                eval_logger.warning(f"VIRAL.generate_until: doc_to_visual failed for task={task}, split={split}, doc_id={doc_id}: {e}")
                visuals = None

            # Normalize visuals: drop None items and collapse empty lists to None
            if isinstance(visuals, list):
                visuals = [v for v in visuals if v is not None]
                if len(visuals) == 0:
                    visuals = None

            # Build the prompt: prepend image token(s) if visuals exist and token not already present
            context_str = contexts
            if visuals is not None and DEFAULT_IMAGE_TOKEN not in context_str:
                num_imgs = len(visuals) if isinstance(visuals, list) else 1
                image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_imgs)
                context_str = f"{image_tokens}\n{context_str}"

            # Wrap with conversation template
            if "llama_3" in getattr(self, "conv_template", ""):
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[getattr(self, "conv_template", "vicuna_v1")].copy()
            conv.append_message(conv.roles[0], context_str)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Optional deep debug: show prompt and visuals metadata
            debug_this = self._should_debug(task, split, doc_id, all_gen_kwargs if isinstance(all_gen_kwargs, dict) else {})
            if debug_this:
                try:
                    vis_info = None
                    if visuals is None:
                        vis_info = "None"
                    elif isinstance(visuals, list):
                        vis_info = [getattr(v, 'size', None) for v in visuals]
                    else:
                        vis_info = getattr(visuals, 'size', None)
                    eval_logger.debug(
                        f"VIRAL DEBUG: task={task} split={split} doc_id={doc_id}\n"
                        f"- prompt (first 400 chars) => {prompt[:400]!r}\n"
                        f"- visuals => {vis_info}\n"
                        f"- template => {getattr(self, 'conv_template', 'vicuna_v1')}"
                    )
                except Exception:
                    pass

            # Tokenize with support for image token placeholders
            ids_raw = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            # Fallback if tokenization returns None/empty tensor
            needs_fallback = False
            if ids_raw is None:
                needs_fallback = True
            elif isinstance(ids_raw, list) and len(ids_raw) == 0:
                needs_fallback = True
            elif hasattr(ids_raw, "numel"):
                try:
                    if ids_raw.numel() == 0:  # type: ignore[attr-defined]
                        needs_fallback = True
                except Exception:
                    pass
            if needs_fallback:
                try:
                    ids_raw = self.tokenizer(prompt, return_tensors="pt").input_ids
                except Exception:
                    ids_raw = torch.tensor([[self.eot_token_id]], dtype=torch.long)
            # Normalize to a single 2D tensor on device
            if isinstance(ids_raw, list):
                try:
                    parts = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in ids_raw]
                    input_ids = torch.cat(parts, dim=0)
                except Exception:
                    # best-effort: take first element
                    first = ids_raw[0]
                    input_ids = first if isinstance(first, torch.Tensor) else torch.tensor(first)
            else:
                input_ids = ids_raw
            if hasattr(input_ids, "dim") and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            # Sanitize token IDs to avoid out-of-range indices in embeddings
            try:
                vocab_size = getattr(self.tokenizer, 'vocab_size', None)
                unk_id = getattr(self.tokenizer, 'unk_token_id', None)
                if unk_id is None:
                    unk_id = getattr(self.tokenizer, 'eos_token_id', 0)
                if isinstance(input_ids, torch.Tensor) and vocab_size is not None:
                    ids_flat = input_ids.view(-1)
                    bad_neg_mask = (ids_flat < 0) & (ids_flat != IMAGE_TOKEN_INDEX)
                    oor_mask = ids_flat >= vocab_size
                    fix_count = int(bad_neg_mask.sum().item() + oor_mask.sum().item())
                    if fix_count > 0:
                        ids_flat[bad_neg_mask] = int(unk_id)
                        ids_flat[oor_mask] = int(unk_id)
                        input_ids = ids_flat.view_as(input_ids)
                        eval_logger.warning(
                            f"VIRAL: sanitized {fix_count} invalid token ids (replaced with unk_id={unk_id})."
                        )
            except Exception:
                pass
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(self.device)
            # Ensure 2D tensor
            if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
                try:
                    input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                except Exception:
                    input_ids = torch.tensor([[self.eot_token_id]], dtype=torch.long, device=self.device)

            if debug_this:
                try:
                    # Count image tokens in tokenized input
                    num_img_tokens = int((input_ids == IMAGE_TOKEN_INDEX).sum().item()) if isinstance(input_ids, torch.Tensor) else 0
                    # Token ID range validation (excluding IMAGE_TOKEN_INDEX)
                    ids_flat = input_ids.view(-1)
                    non_img_mask = ids_flat != IMAGE_TOKEN_INDEX
                    safe_ids = ids_flat[non_img_mask]
                    vocab_size = getattr(self.tokenizer, 'vocab_size', None)
                    neg_count = int((safe_ids < 0).sum().item()) if safe_ids.numel() > 0 else 0
                    oor_count = None
                    min_id = int(safe_ids.min().item()) if safe_ids.numel() > 0 else None
                    max_id = int(safe_ids.max().item()) if safe_ids.numel() > 0 else None
                    if vocab_size is not None and safe_ids.numel() > 0:
                        oor_count = int((safe_ids >= vocab_size).sum().item())
                    eval_logger.debug(
                        f"VIRAL DEBUG: tokenized input shape={tuple(input_ids.shape)} | IMAGE_TOKEN_INDEX occurrences={num_img_tokens} | "
                        f"min_id={min_id} max_id={max_id} vocab_size={vocab_size} neg_non_img={neg_count} out_of_range={oor_count}"
                    )
                except Exception:
                    pass

            # Prepare image tensor if any
            images_arg = None
            image_sizes = None
            if visuals is not None:
                if not self._ensure_image_processor():
                    eval_logger.warning("VIRAL.generate_until: no image_processor available; generating without images.")
                else:
                    try:
                        # Build a consistent visuals list for sizes
                        visuals_list = visuals if isinstance(visuals, list) else [visuals]
                        image_sizes = [v.size for v in visuals_list if hasattr(v, 'size')]
                        processed = process_images(visuals_list, self._image_processor, self._config)
                        # Normalize to list of tensors on the correct device/dtype
                        vis_device, vis_dtype = self._vision_device_dtype()
                        if isinstance(processed, list):
                            images_arg = []
                            for _img in processed:
                                if _img is None:
                                    continue
                                images_arg.append(_img.to(dtype=vis_dtype, device=vis_device))
                            if len(images_arg) == 0:
                                images_arg = None
                        else:
                            images_arg = processed.to(dtype=vis_dtype, device=vis_device)
                    except Exception as e:
                        eval_logger.warning(f"VIRAL.generate_until: image processing failed; continuing without images. Error: {e}")
                        images_arg = None
                        image_sizes = None

            if debug_this:
                try:
                    if images_arg is None:
                        eval_logger.debug("VIRAL DEBUG: images_arg=None (will run text-only or model fallback)")
                    elif isinstance(images_arg, list):
                        shapes = [tuple(t.shape) for t in images_arg]
                        dtypes = [str(t.dtype) for t in images_arg]
                        devices = [str(t.device) for t in images_arg]
                        eval_logger.debug(f"VIRAL DEBUG: images_arg=list count={len(images_arg)} shapes={shapes} dtypes={dtypes} devices={devices}")
                    else:
                        eval_logger.debug(f"VIRAL DEBUG: images_arg=tensor shape={tuple(images_arg.shape)} dtype={images_arg.dtype} device={images_arg.device}")
                except Exception:
                    pass

            # Generation parameters
            gen_kwargs = dict(all_gen_kwargs) if isinstance(all_gen_kwargs, dict) else {}
            # Stopping sequences
            until = gen_kwargs.pop("until", None)
            if until is None:
                until = [self.tok_decode(self.eot_token_id)]
            elif isinstance(until, str):
                until = [until]

            # Defaults
            temperature = gen_kwargs.pop("temperature", 0)
            top_p = gen_kwargs.pop("top_p", None)
            num_beams = gen_kwargs.pop("num_beams", 1)
            max_new_tokens = gen_kwargs.pop("max_new_tokens", 1024)

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            # Let the model construct attention mask internally for maximum compatibility
            attention_mask = None

            # Compose stopping criteria from strings
            input_len = int(input_ids.shape[1]) if hasattr(input_ids, 'shape') else int(len(input_ids[0]))
            # For maximum robustness across HF versions, skip custom stopping_criteria
            stopping_criteria = None

            # Run generation
            try:
                try:
                    eval_logger.debug(
                        f"VIRAL.generate_until: input_ids shape={tuple(input_ids.shape) if hasattr(input_ids,'shape') else 'N/A'}, "
                        f"images_arg={'None' if images_arg is None else ('list['+str(len(images_arg))+']' if isinstance(images_arg,list) else tuple(images_arg.shape))}, "
                        f"image_sizes={'None' if image_sizes is None else len(image_sizes)}, "
                        f"merge={getattr(self._config,'mm_patch_merge_type', 'flat')}, ar={getattr(self._config,'image_aspect_ratio', None)}"
                    )
                    # Additional dtype debug for multimodal path
                    try:
                        vt = self.model.get_vision_tower() if hasattr(self.model, 'get_vision_tower') else None
                        vt_param = (next(vt.parameters()) if vt is not None else None)
                        vt_dtype = (vt_param.dtype if vt_param is not None else None)
                        vt_device = (vt_param.device if vt_param is not None else None)
                    except Exception:
                        vt_dtype = None
                        vt_device = None
                    try:
                        proj = self.model.get_model().mm_projector if hasattr(self.model, 'get_model') else None
                        proj_dtype = (next(proj.parameters()).dtype if proj is not None else None)
                    except Exception:
                        proj_dtype = None
                    try:
                        img_dtype = None
                        if images_arg is not None:
                            if isinstance(images_arg, list) and len(images_arg) > 0 and hasattr(images_arg[0], 'dtype'):
                                img_dtype = images_arg[0].dtype
                            elif hasattr(images_arg, 'dtype'):
                                img_dtype = images_arg.dtype
                    except Exception:
                        img_dtype = None
                    try:
                        model_device = next(self.model.parameters()).device
                    except Exception:
                        model_device = None
                    eval_logger.debug(
                        f"VIRAL.generate_until dtypes: model={getattr(self.model, 'dtype', None)}, vision={vt_dtype}, projector={proj_dtype}, images={img_dtype}; "
                        f"devices: model={model_device}, vision={vt_device}"
                    )
                except Exception:
                    pass
                with torch.inference_mode():
                    do_sample = True if temperature and float(temperature) > 0 else False
                    generate_common = dict(
                        do_sample=do_sample,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        use_cache=self.use_cache,
                        pad_token_id=pad_token_id,
                        attention_mask=attention_mask,
                    )
                    if do_sample:
                        if temperature is not None:
                            generate_common['temperature'] = float(temperature)
                        if top_p is not None:
                            generate_common['top_p'] = float(top_p)

                    if getattr(self, "_accepts_image_generate", False) and images_arg is not None:
                        # Only pass image_sizes when required by config
                        mm_merge = getattr(self._config, 'mm_patch_merge_type', 'flat')
                        aspect = getattr(self._config, 'image_aspect_ratio', None)
                        pass_sizes = (aspect == 'anyres') or (isinstance(mm_merge, str) and mm_merge.startswith('spatial'))
                        # Determine whether to pass inputs or input_ids based on generate signature
                        try:
                            gen_sig = inspect.signature(self.model.generate)
                            use_inputs_kw = 'inputs' in gen_sig.parameters
                        except Exception:
                            use_inputs_kw = True
                        gen_call_common = dict(images=images_arg, **generate_common)
                        if pass_sizes and image_sizes is not None:
                            gen_call_common['image_sizes'] = image_sizes
                        try:
                            if use_inputs_kw:
                                output_ids = self.model.generate(inputs=input_ids, **gen_call_common)
                            else:
                                output_ids = self.model.generate(input_ids=input_ids, **gen_call_common)
                        except Exception as gen_e:
                            # Fallback: retry generation without images (text-only) to avoid total failure
                            eval_logger.warning(f"VIRAL.generate_until: multimodal generate failed; retrying text-only. Error: {gen_e}")
                            try:
                                # Re-evaluate signature in case different path is used without images
                                try:
                                    gen_sig_fb = inspect.signature(self.model.generate)
                                    use_inputs_kw_fb = 'inputs' in gen_sig_fb.parameters
                                except Exception:
                                    use_inputs_kw_fb = True
                                if use_inputs_kw_fb:
                                    output_ids = self.model.generate(inputs=input_ids, **generate_common)
                                else:
                                    output_ids = self.model.generate(input_ids=input_ids, **generate_common)
                            except Exception as gen_e2:
                                # Re-raise the original exception context if fallback also fails
                                raise gen_e2
                    else:
                        if images_arg is not None and not getattr(self, "_accepts_image_generate", False):
                            eval_logger.warning("VIRAL.generate_until: Model.generate() does not accept 'images'; generating without images.")
                        # Determine whether to pass inputs or input_ids based on generate signature (LLaVA override uses `inputs`)
                        try:
                            gen_sig = inspect.signature(self.model.generate)
                            use_inputs_kw = 'inputs' in gen_sig.parameters
                        except Exception:
                            use_inputs_kw = True
                        if use_inputs_kw:
                            output_ids = self.model.generate(inputs=input_ids, **generate_common)
                        else:
                            output_ids = self.model.generate(input_ids=input_ids, **generate_common)

                # HF may return a GenerateOutput struct; extract sequences if present
                if hasattr(output_ids, "sequences"):
                    output_tensor = output_ids.sequences
                else:
                    output_tensor = output_ids
                # Slice to only new tokens and decode
                new_tokens = output_tensor[0, input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Final defensive truncation by stopping strings
                if until:
                    cut_idx = len(text)
                    for s in until:
                        if not s:
                            continue
                        pos = text.find(s)
                        if pos != -1:
                            cut_idx = min(cut_idx, pos)
                    text = text[:cut_idx]

            except Exception as e:
                eval_logger.error(f"VIRAL.generate_until: generation error for task={task}, doc_id={doc_id}: {e}")
                text = ""

            results.append(text)

            # Cache hook (best-effort)
            try:
                if hasattr(self, "cache_hook") and getattr(self, "cache_hook") is not None:
                    self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), [text])
            except Exception:
                pass

            pbar.update(1)

        pbar.close()
        return results

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for VIRAL")