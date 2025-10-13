import copy
import os
import json
import inspect
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union, Any, Dict, cast


import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
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
    def _check_llava_imports(self):
        # Check that all required LLaVA symbols are available, else raise ImportError
        required = [
            'DEFAULT_IMAGE_TOKEN', 'IMAGE_TOKEN_INDEX', 'conv_templates',
            'get_model_name_from_path', 'process_images', 'tokenizer_image_token', 'load_pretrained_model'
        ]
        missing = [s for s in required if s not in globals()]
        if missing:
            raise ImportError(f"LLaVA is not installed or missing required symbols: {missing}. Please install LLaVA to use this model.")

    @property
    def rank(self):
        return getattr(self, '_rank', 0)

    @property
    def world_size(self):
        return getattr(self, '_world_size', 1)
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
        image_aspect_ratio: Optional[str] = None,
        use_cache: bool = True,
        tie_weights: bool = True,
        force_full_gpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self._check_llava_imports()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1:
            # Respect explicit device_map if provided
            if device_map == "auto":
                self._device = torch.device(device)
                # If requested, force a single-device map to avoid CPU shards
                if force_full_gpu and str(self._device).startswith("cuda"):
                    self.device_map = {"": str(self._device)}
                else:
                    self.device_map = device_map
            else:
                self._device = torch.device(device)
                self.device_map = device_map

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
            name_or_path, base, model_name, device_map=self.device_map, **loader_kwargs  # type: ignore[arg-type]
        )

        self._config = getattr(self._model, "config", None)
        # --- Sanity diagnostics for tokenizer/model compatibility ---
        try:
            tok_len = None
            try:
                tok_len = int(len(self._tokenizer))  # includes added tokens
            except Exception:
                tok_len = int(getattr(self._tokenizer, 'vocab_size', 0))
            emb_rows = None
            emb = None
            try:
                emb = self._model.get_input_embeddings()
            except Exception:
                pass
            if emb is None:
                try:
                    base_model = getattr(self._model, 'model', None)
                    emb = getattr(base_model, 'embed_tokens', None)
                except Exception:
                    emb = None
            if emb is not None and hasattr(emb, 'weight'):
                try:
                    emb_rows = int(getattr(emb.weight, 'shape', [0, 0])[0])
                except Exception:
                    emb_rows = None
            cfg_vocab = None
            try:
                cfg_vocab = int(getattr(self._config, 'vocab_size', 0)) if self._config is not None else None
            except Exception:
                cfg_vocab = None
            # Report mismatches that often cause garbage generations
            if tok_len and emb_rows and emb_rows != tok_len:
                eval_logger.warning(
                    f"VIRAL: tokenizer/model vocab mismatch: len(tokenizer)={tok_len} vs input_embedding_rows={emb_rows}."\
                    "\nThis is a common source of gibberish outputs. Ensure the tokenizer and base model match, "\
                    "and that resize_token_embeddings(len(tokenizer)) has been applied."
                )
                # Optional auto-fix via env flag
                try:
                    if str(os.getenv('VIRAL_AUTO_RESIZE_EMB', '')).lower() in ("1", "true", "yes"):
                        self._model.resize_token_embeddings(tok_len, pad_to_multiple_of=None)  # type: ignore[attr-defined]
                        eval_logger.info(f"VIRAL: auto-resized input embeddings to {tok_len} (VIRAL_AUTO_RESIZE_EMB)")
                        # refresh emb_rows for downstream logic
                        try:
                            emb = self._model.get_input_embeddings()
                            emb_rows = int(getattr(emb.weight, 'shape', [0, 0])[0]) if emb is not None else emb_rows
                        except Exception:
                            pass
                except Exception as e:
                    eval_logger.debug(f"VIRAL: auto-resize embeddings failed/unsupported: {e}")
            if cfg_vocab and tok_len and cfg_vocab != tok_len:
                eval_logger.debug(
                    f"VIRAL: config.vocab_size={cfg_vocab} differs from len(tokenizer)={tok_len}; this can be OK if embeddings were resized."
                )
            # Basic special token checks
            try:
                if getattr(self._tokenizer, 'pad_token_id', None) is None:
                    # Some Vicuna/LLaMA tokenizers lack pad; set to eos to keep generation sane
                    self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[attr-defined]
                    eval_logger.info("VIRAL: tokenizer had no pad_token; set pad_token=eos_token for stability.")
            except Exception:
                pass
            # Log vision tower source for awareness
            try:
                vt_name = None
                if self._config is not None:
                    vt_name = getattr(self._config, 'mm_vision_tower', None) or getattr(self._config, 'vision_tower', None)
                if vt_name:
                    eval_logger.info(f"VIRAL: vision tower => {vt_name}")
            except Exception:
                pass
        except Exception:
            pass
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

        # optional user-specified image aspect ratio (should be string: e.g., "pad", "square", "resize")
        if image_aspect_ratio is not None and self._config is not None:
            try:
                valid_aspects = {"pad", "square", "resize"}
                if isinstance(image_aspect_ratio, str) and (image_aspect_ratio in valid_aspects):
                    self._config.image_aspect_ratio = image_aspect_ratio
                else:
                    eval_logger.warning(f"VIRAL: image_aspect_ratio '{image_aspect_ratio}' is not a recognized string value; expected one of {valid_aspects}.")
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

        # Optionally co-locate the vision tower and projector on the same CUDA device/dtype as the base model
        try:
            if force_full_gpu and str(self._device).startswith("cuda"):
                model_device = next(self.model.parameters()).device
                model_dtype = getattr(self.model, "dtype", None)
                # Move vision tower
                vt = self.model.get_vision_tower() if hasattr(self.model, 'get_vision_tower') else None
                if vt is not None:
                    try:
                        if model_dtype is None:
                            try:
                                model_dtype = next(self.model.get_model().parameters()).dtype
                            except Exception:
                                model_dtype = torch.float16
                        vt.to(device=model_device, dtype=model_dtype)
                        eval_logger.info(f"VIRAL: moved vision tower to {model_device} ({model_dtype})")
                    except Exception as e:
                        eval_logger.warning(f"VIRAL: could not move vision tower to {model_device}: {e}")
                # Move projector
                try:
                    mm_proj = self.model.get_model().mm_projector if hasattr(self.model, 'get_model') else None
                    if mm_proj is not None:
                        mm_proj.to(device=model_device, dtype=(model_dtype or torch.float16))
                        eval_logger.info(f"VIRAL: moved mm_projector to {model_device} ({model_dtype})")
                except Exception as e:
                    eval_logger.debug(f"VIRAL: projector move skipped/failed: {e}")
        except Exception:
            pass

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        # Default to LLaVA template if available (better alignment for multimodal models), otherwise Vicuna
        try:
            self.conv_template = "llava_v1" if ("llava_v1" in conv_templates) else "vicuna_v1"
        except Exception:
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
        padding_side = getattr(self.tokenizer, 'padding_side', 'right')
        if padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if padding_side == "left":
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
            # If multiple towers are returned, pick the first for placement info
            if isinstance(vt, (list, tuple)) and len(vt) > 0:
                vt0 = vt[0]
            else:
                vt0 = vt
            if vt0 is not None:
                try:
                    if not hasattr(vt0, 'parameters'):
                        raise AttributeError('vision tower has no parameters attribute')
                    vt0_any = cast(Any, vt0)
                    p = next(vt0_any.parameters())
                    return p.device, p.dtype
                except Exception:
                    # try attribute-based
                    dev = getattr(vt0, 'device', self.device)
                    dt = getattr(vt0, 'dtype', getattr(self.model, 'dtype', torch.float16))
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
            # Ensure a trailing space after the assistant tag to avoid subword-start artifacts (e.g., 's' instead of 'Yes')
            if not prompt.endswith(" "):
                prompt = prompt + " "

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
            # Normalize to a single 2D tensor (keep on CPU for now)
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
            # Prepare image tensor if any (needed before deciding on IMAGE_TOKEN_INDEX handling)
            images_arg = None
            image_sizes = None
            if visuals is not None:
                if not self._ensure_image_processor():
                    eval_logger.warning("VIRAL.generate_until: no image_processor available; generating without images.")
                else:
                    try:
                        # Build a consistent visuals list for sizes
                        visuals_list = visuals if isinstance(visuals, list) else [visuals]
                        # LLaVA expects (height, width); PIL provides (width, height)
                        image_sizes = [
                            (int(getattr(v, 'size')[1]), int(getattr(v, 'size')[0]))
                            for v in visuals_list if hasattr(v, 'size') and getattr(v, 'size') is not None
                        ]
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

            # Sanitize token IDs to avoid out-of-range indices in embeddings
            try:
                vocab_size = getattr(self.tokenizer, 'vocab_size', None)
                unk_id = getattr(self.tokenizer, 'unk_token_id', None)
                if unk_id is None:
                    unk_id = getattr(self.tokenizer, 'eos_token_id', 0)
                if isinstance(input_ids, torch.Tensor):
                    # Clone before mutating
                    input_ids = input_ids.clone()
                    ids_flat = input_ids.view(-1)
                    # Replace invalid negatives and out-of-range ids
                    if vocab_size is not None:
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
                    # If image tokens exist but images won't be used, neutralize IMAGE_TOKEN_INDEX
                    accepts_img = getattr(self, "_accepts_image_generate", False)
                    will_use_images = accepts_img and (images_arg is not None)
                    if not will_use_images:
                        img_mask = ids_flat == IMAGE_TOKEN_INDEX
                        img_count = int(img_mask.sum().item())
                        if img_count > 0:
                            ids_flat[img_mask] = int(unk_id)
                            input_ids = ids_flat.view_as(input_ids)
                            eval_logger.warning(
                                f"VIRAL: replaced {img_count} IMAGE_TOKEN_INDEX with unk_id={unk_id} because images are not used in generate()."
                            )
            except Exception:
                pass

            if debug_this:
                try:
                    num_img_tokens = int((input_ids == IMAGE_TOKEN_INDEX).sum().item()) if isinstance(input_ids, torch.Tensor) else 0
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
                        f"VIRAL DEBUG: tokenized input (pre-device) shape={tuple(input_ids.shape)} | IMAGE_TOKEN_INDEX occurrences={num_img_tokens} | "
                        f"min_id={min_id} max_id={max_id} vocab_size={vocab_size} neg_non_img={neg_count} out_of_range={oor_count}"
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

            # (images_arg already prepared above)

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
            # Stopping sequences (string-based, post-decode fallback)
            until = gen_kwargs.pop("until", None)
            if until is None:
                until = []
            elif isinstance(until, str):
                until = [until]
            # Drop empty/whitespace stop strings to avoid truncating output to empty
            if isinstance(until, list):
                until = [s for s in until if isinstance(s, str) and s.strip() != ""]

            # Defaults
            temperature = gen_kwargs.pop("temperature", 0)
            top_p = gen_kwargs.pop("top_p", None)
            num_beams = gen_kwargs.pop("num_beams", 1)
            max_new_tokens = gen_kwargs.pop("max_new_tokens", 1024)
            min_new_tokens = gen_kwargs.pop("min_new_tokens", None)

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if debug_this:
                try:
                    eval_logger.debug(
                        f"VIRAL DEBUG: special token ids => pad={pad_token_id}, eos={getattr(self.tokenizer,'eos_token_id',None)}, bos={getattr(self.tokenizer,'bos_token_id',None)}"
                    )
                except Exception:
                    pass
            # Provide an attention mask only for pure text generation; for multimodal paths, let the model compute it
            attention_mask = None
            if images_arg is None:
                try:
                    if isinstance(input_ids, torch.Tensor):
                        attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
                except Exception:
                    attention_mask = None

            # Compose stopping criteria from strings
            input_len = int(input_ids.shape[1]) if hasattr(input_ids, 'shape') else int(len(input_ids[0]))
            # Use LLaVA-style keyword stopping on token-level (prefer sep2 and role markers; avoid single-space sep)
            stopping_criteria = None
            try:
                from transformers import StoppingCriteriaList
                from llava.mm_utils import KeywordsStoppingCriteria
                keywords: List[str] = []
                sep2 = getattr(conv, 'sep2', None)
                sep = getattr(conv, 'sep', None)
                if isinstance(sep2, str) and sep2.strip() != "":
                    keywords.append(sep2)
                # Only use sep if it's not trivial whitespace
                if isinstance(sep, str) and len(sep.strip()) > 1:
                    keywords.append(sep)
                # Add role markers as universal guards
                keywords.extend(["USER:", "ASSISTANT:"])
                if len(keywords) > 0:
                    stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)])
            except Exception:
                stopping_criteria = None

            # Run generation
            try:
                # By default, assume the returned sequences include the prompt tokens
                # (this is true when passing input_ids/inputs). When using inputs_embeds,
                # HF returns only the newly generated tokens, so we must NOT slice them off.
                prefix_len = input_len
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
                            if isinstance(images_arg, list) and len(images_arg) > 0:
                                img_dtype = getattr(images_arg[0], 'dtype', None)
                            else:
                                img_dtype = getattr(images_arg, 'dtype', None)
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
                    generate_common: Dict[str, Any] = dict(
                        do_sample=do_sample,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        # Optional fields below are added conditionally
                        use_cache=self.use_cache,
                        pad_token_id=pad_token_id,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                    )
                    if debug_this:
                        generate_common['output_scores'] = True
                    # Only include attention_mask for text-only runs
                    if attention_mask is not None and images_arg is None:
                        generate_common['attention_mask'] = attention_mask
                    # Add optional knobs only if explicitly provided
                    if min_new_tokens is not None:
                        try:
                            generate_common['min_new_tokens'] = int(min_new_tokens)
                        except Exception:
                            pass
                    rp = gen_kwargs.pop("repetition_penalty", None)
                    if rp is not None:
                        try:
                            generate_common['repetition_penalty'] = float(rp)
                        except Exception:
                            pass
                    nrep = gen_kwargs.pop("no_repeat_ngram_size", None)
                    if nrep is not None:
                        try:
                            generate_common['no_repeat_ngram_size'] = int(nrep)
                        except Exception:
                            pass
                    # Add BOS/EOS if available to stabilize decoding
                    eos_id = getattr(self.tokenizer, 'eos_token_id', None)
                    bos_id = getattr(self.tokenizer, 'bos_token_id', None)
                    if eos_id is not None:
                        generate_common['eos_token_id'] = int(eos_id)
                    if bos_id is not None:
                        generate_common['bos_token_id'] = int(bos_id)
                    if do_sample:
                        if temperature is not None:
                            generate_common['temperature'] = float(temperature)
                        if top_p is not None:
                            generate_common['top_p'] = float(top_p)

                    if getattr(self, "_accepts_image_generate", False) and images_arg is not None:
                        # Preferred: let the model handle images directly via its custom generate()
                        try:
                            if debug_this:
                                eval_logger.debug("VIRAL DEBUG: generation branch = multimodal-direct-generate")
                            prefix_len = input_len
                            # Some LLaVA versions use `inputs` instead of `input_ids` in generate
                            try:
                                gen_sig2 = inspect.signature(self.model.generate)
                                use_inputs_kw2 = 'inputs' in gen_sig2.parameters
                            except Exception:
                                use_inputs_kw2 = True
                            # Ensure images are a list of [3,H,W] tensors and image_sizes align
                            images_list = None
                            sizes_list = None
                            try:
                                if isinstance(images_arg, list):
                                    images_list = images_arg
                                elif isinstance(images_arg, torch.Tensor):
                                    if images_arg.dim() == 4 and images_arg.shape[0] == 1:
                                        images_list = [images_arg[0]]
                                    elif images_arg.dim() == 3:
                                        images_list = [images_arg]
                                if isinstance(image_sizes, list):
                                    sizes_list = image_sizes
                                elif image_sizes is not None:
                                    sizes_list = [image_sizes]
                            except Exception:
                                images_list = None
                                sizes_list = None
                            images_kw = images_list if images_list is not None else images_arg
                            sizes_kw = sizes_list if sizes_list is not None else image_sizes
                            if use_inputs_kw2:
                                output_ids = self.model.generate(inputs=input_ids, images=images_kw, image_sizes=sizes_kw, **generate_common)
                            else:
                                output_ids = self.model.generate(input_ids=input_ids, images=images_kw, image_sizes=sizes_kw, **generate_common)
                        except Exception as gen_e:
                            # Fallback: precompute inputs_embeds if direct images path fails
                            try:
                                if debug_this:
                                    eval_logger.debug("VIRAL DEBUG: generation branch = multimodal-inputs_embeds-fallback")
                                out = self.model.prepare_inputs_labels_for_multimodal(
                                    input_ids,
                                    position_ids=None,
                                    attention_mask=None,
                                    past_key_values=None,
                                    labels=None,
                                    images=images_arg,
                                    image_sizes=image_sizes,
                                )
                                if out is None or not isinstance(out, tuple) or len(out) != 6:
                                    raise RuntimeError("prepare_inputs_labels_for_multimodal returned invalid output")
                                (
                                    new_input_ids,
                                    new_position_ids,
                                    new_attention_mask,
                                    _pkv,
                                    inputs_embeds,
                                    _labels,
                                ) = out
                                # Ensure we have a proper attention mask for inputs_embeds
                                attn_for_embeds = new_attention_mask
                                if attn_for_embeds is None and isinstance(inputs_embeds, torch.Tensor):
                                    attn_for_embeds = torch.ones(
                                        (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                                        dtype=torch.long,
                                        device=inputs_embeds.device,
                                    )
                                gen_clean = {k: v for k, v in generate_common.items() if k not in ("attention_mask", "position_ids")}
                                # When using inputs_embeds, returned sequences contain ONLY newly generated tokens
                                prefix_len = 0
                                output_ids = self.model.generate(
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attn_for_embeds,
                                    position_ids=new_position_ids,
                                    **gen_clean,
                                )
                            except Exception as prep_e:
                                # Fallback: retry generation without images (text-only) to avoid total failure
                                try:
                                    eval_logger.exception(
                                        f"VIRAL.generate_until: multimodal generate failed; retrying text-only. Error: {gen_e} | prep_fallback_error: {prep_e}"
                                    )
                                except Exception:
                                    eval_logger.warning(
                                        f"VIRAL.generate_until: multimodal generate failed; retrying text-only. Error: {gen_e} | prep_fallback_error: {prep_e}"
                                    )
                                if debug_this:
                                    eval_logger.debug("VIRAL DEBUG: generation branch = text-only-fallback-after-mm-failure")
                                    try:
                                        eval_logger.debug(f"VIRAL DEBUG: generate_common on fallback keys={sorted(list(generate_common.keys()))}")
                                    except Exception:
                                        pass
                                try:
                                    unk_id = getattr(self.tokenizer, 'unk_token_id', None)
                                    if unk_id is None:
                                        unk_id = getattr(self.tokenizer, 'eos_token_id', 0)
                                    if isinstance(input_ids, torch.Tensor):
                                        mask_img = (input_ids == IMAGE_TOKEN_INDEX)
                                        count_img = int(mask_img.sum().item())
                                        if count_img > 0:
                                            input_ids = input_ids.masked_fill(mask_img, int(unk_id))
                                            eval_logger.warning(
                                                f"VIRAL: neutralized {count_img} IMAGE_TOKEN_INDEX for text-only retry (unk_id={unk_id})."
                                            )
                                except Exception:
                                    pass
                                # Avoid passing attention_mask/position_ids at all for fallback; let model infer
                                prefix_len = input_len  # reset: now using input_ids path again
                                output_ids = self.model.generate(inputs=input_ids, **generate_common)
                    else:
                        if images_arg is not None and not getattr(self, "_accepts_image_generate", False):
                            eval_logger.warning("VIRAL.generate_until: Model.generate() does not accept 'images'; generating without images.")
                        # Determine whether to pass inputs or input_ids based on generate signature (LLaVA override uses `inputs`)
                        try:
                            gen_sig = inspect.signature(self.model.generate)
                            use_inputs_kw = 'inputs' in gen_sig.parameters
                        except Exception:
                            use_inputs_kw = True
                        if debug_this:
                            branch = 'text-only-inputs' if use_inputs_kw else 'text-only-input_ids'
                            eval_logger.debug(f"VIRAL DEBUG: generation branch = {branch}")
                        if use_inputs_kw:
                            prefix_len = input_len
                            output_ids = self.model.generate(inputs=input_ids, **generate_common)
                        else:
                            prefix_len = input_len
                            output_ids = self.model.generate(input_ids=input_ids, **generate_common)

                # HF may return a GenerateOutput struct; extract sequences if present
                # Prefer sequences when return_dict_in_generate=True; otherwise assume tensor
                output_tensor = getattr(output_ids, 'sequences', output_ids)
                # Slice to only new tokens using the known prefix_len for the chosen generate path
                try:
                    if hasattr(output_tensor, 'shape') and output_tensor.shape[1] > prefix_len:
                        new_tokens = output_tensor[0, prefix_len:]
                    else:
                        new_tokens = output_tensor[0]
                except Exception:
                    # Defensive fallback
                    new_tokens = output_tensor[0]
                if debug_this:
                    try:
                        preview_tokens = new_tokens.tolist() if hasattr(new_tokens, 'tolist') else list(new_tokens)
                        eval_logger.debug(f"VIRAL DEBUG: new_tokens length={len(preview_tokens)} head={preview_tokens[:16]}")
                        # Decode head tokens to strings for sanity
                        head_decode = []
                        for tid in preview_tokens[:8]:
                            try:
                                head_decode.append(self.tokenizer.decode([int(tid)], skip_special_tokens=False))
                            except Exception:
                                head_decode.append(str(tid))
                        eval_logger.debug(f"VIRAL DEBUG: new_tokens head decoded={head_decode}")
                    except Exception:
                        pass
                # If we requested scores, show top-5 for the first step
                try:
                    if debug_this and hasattr(output_ids, 'scores') and output_ids.scores:
                        import math
                        first_step = output_ids.scores[0]
                        probs = torch.softmax(first_step, dim=-1)
                        topk = torch.topk(probs, k=5, dim=-1)
                        ids = topk.indices[0].tolist() if probs.dim() == 2 else topk.indices.tolist()
                        vals = topk.values[0].tolist() if probs.dim() == 2 else topk.values.tolist()
                        decs = []
                        for t in ids:
                            try:
                                decs.append(self.tokenizer.decode([int(t)], skip_special_tokens=False))
                            except Exception:
                                decs.append(str(t))
                        eval_logger.debug(
                            f"VIRAL DEBUG: step0 top5 ids={ids} probs={[round(v,4) for v in vals]} decs={decs}"
                        )
                except Exception:
                    pass
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Final defensive truncation by stopping strings
                if until:
                    cut_idx = len(text)
                    stop_matched = None
                    for s in until:
                        if not s:
                            continue
                        pos = text.find(s)
                        if pos != -1:
                            cut_idx = min(cut_idx, pos)
                            stop_matched = s
                    text = text[:cut_idx]
                    if debug_this:
                        eval_logger.debug(f"VIRAL DEBUG: applied post-decode stop '{stop_matched}' -> cut to {len(text)} chars")

                # Optional: log generation output for debugging/inspection (logger only, no file writes)
                try:
                    want_log = bool(debug_this) or bool((gen_kwargs.get("log_output", False) if isinstance(gen_kwargs, dict) else False))
                    if want_log:
                        preview = text if len(text) <= 800 else (text[:800] + "…[truncated]")
                        eval_logger.info(
                            f"VIRAL OUTPUT: task={task} split={split} doc_id={doc_id} | len={len(text)}\n" +
                            f"— preview —\n{preview}"
                        )
                except Exception:
                    pass

            except Exception as e:
                try:
                    eval_logger.exception(
                        f"VIRAL.generate_until: generation error for task={task}, doc_id={doc_id}: {e}"
                    )
                except Exception:
                    eval_logger.error(
                        f"VIRAL.generate_until: generation error for task={task}, doc_id={doc_id}: {e}"
                    )
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