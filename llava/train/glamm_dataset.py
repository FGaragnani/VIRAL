import torch
import copy
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Sequence, Callable

# Imports to align with LLaVA preprocessing/tokenization
from llava import conversation as conversation_lib
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import tokenizer_image_token

class GranDDataset(Dataset):
    """
    Dataset for GLAMM training using GranD annotations.

    flatten_patches=True (default), __len__ equals the number of valid patches
      across all images, and each __getitem__ returns one patch + tokenized text
      ready for training, with keys: input_ids, labels, image. (if False won't work)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        tokenizer=None,
        data_args=None,
        image_processor=None,
        patch_size: Tuple[int, int] = (224, 224),
        check_area: float = 0.05,
        flatten_patches: bool = True,
        prompt_template: Optional[str] = None,
        label_joiner: str = ", ",
    ):
        self.image_dir = image_dir
        self.patch_size = patch_size if patch_size is not None else (224, 224)
        self.transform = self._resize_and_pad
        self.annotation_dir = annotation_dir
        self.annotation_files: List[str] = [
            os.path.join(annotation_dir, f)
            for f in os.listdir(annotation_dir)
            if f.lower().endswith(".json")
        ]
        self.check_area_fn: Callable[[float, float], bool] = (lambda img_size, patch_area: (patch_area / img_size) > check_area)

        # Tokenizer and data-related config
        self.tokenizer = tokenizer
        self.data_args = data_args
        # Prefer explicit image_processor arg, otherwise from data_args if provided
        self.image_processor = image_processor if image_processor is not None else (
            getattr(data_args, "image_processor", None) if data_args is not None else None
        )
        self.flatten_patches = flatten_patches
        self.prompt_template = prompt_template or "Describe the object(s) present in this region."
        self.label_joiner = label_joiner
        self._flat_index: list = []
        if self.flatten_patches:
            self._build_flat_index()

    def _resize_and_pad(self, img: Image.Image) -> torch.Tensor:
        """Resize PIL Image and pad to exact `patch_size`, centering the content.

        This uses PIL.ImageOps.pad which preserves aspect ratio and centers the
        resized image in the target canvas. It fixes issues where crops taken at
        image borders weren't padded in a centered way.

        Returns a tensor (C,H,W).
        """
        target_w, target_h = self.patch_size

        try:
            pil = ImageOps.pad(img, (target_w, target_h), color=0, centering=(0.5, 0.5))
        except TypeError:
            # for older Pillow versions 
            w, h = img.size
            if w <= 0 or h <= 0:
                raise ValueError("Image has zero or negative dimension")
            # scale so longer side == max(target_w, target_h)
            target_long = max(target_w, target_h)
            scale = target_long / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h))
            pad_w = max(0, target_w - new_w)
            pad_h = max(0, target_h - new_h)
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            if left or right or top or bottom:
                pil = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
            else:
                pil = img

        return F.to_tensor(pil)

    def _tokenize_fn(self, strings: Sequence[str]) -> Dict[str, List]:
        assert self.tokenizer is not None, "Tokenizer must be provided to GranDDataset to produce input_ids/labels."
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [t.input_ids[0] for t in tokenized_list]
        input_ids_lens = labels_lens = [
            t.input_ids.ne(self.tokenizer.pad_token_id).sum().item() for t in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    @staticmethod
    def _mask_targets(target, tokenized_lens, speakers):
        cur_idx = tokenized_lens[0]
        tokenized_lens = tokenized_lens[1:]
        target[:cur_idx] = IGNORE_INDEX
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker == "human":
                # Mask the human turn (plus the role prefix) in the targets
                target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len

    def _add_speaker_and_signal(self, header, source, get_conversation=True):
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = header
        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() == "human":
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() == "gpt":
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                from_str = "unknown"
            sentence["value"] = (
                BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
            )
            if get_conversation:
                conversation += sentence["value"]
        conversation += BEGIN_SIGNAL
        return conversation

    def _preprocess_multimodal(self, sources: Sequence[Sequence[Dict]]) -> Sequence[Sequence[Dict]]:
        # Minimal version of preprocess_multimodal to insert image tokens and optional start/end wrappers
        is_multimodal = True  # by construction for this dataset
        if not is_multimodal:
            return sources

        use_im_start_end = getattr(self.data_args, "mm_use_im_start_end", False) if self.data_args is not None else False
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence["value"] = sentence["value"].replace(
                            DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                        )
                replace_token = DEFAULT_IMAGE_TOKEN
                if use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return sources

    def _preprocess(self, sources: Sequence[Sequence[Dict]], has_image: bool = True):
        # Generic preprocess mirroring the default path in train.py
        tokenizer = self.tokenizer
        conversations = []
        for source in sources:
            header = f"{conversation_lib.default_conversation.system}\n\n"
            conversation = self._add_speaker_and_signal(header, source)
            conversations.append(conversation)

        def get_tokenize_len(prompts):
            return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

        if has_image:
            input_ids = [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations
            ]
        else:
            conversations_tokenized = self._tokenize_fn(conversations)
            input_ids = conversations_tokenized["input_ids"]

        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            if has_image:
                tokenized_lens = get_tokenize_len([f"{conversation_lib.default_conversation.system}\n\n"] + [s["value"] for s in source])
            else:
                tokenized_lens = self._tokenize_fn(
                    [f"{conversation_lib.default_conversation.system}\n\n"] + [s["value"] for s in source]
                )["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            self._mask_targets(target, tokenized_lens, speakers)

        return dict(input_ids=input_ids, labels=targets)

    def _preprocess_patch_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert a PIL patch to model-ready tensor using image_processor if available.

        Falls back to the dataset's own resize+pad transform.
        """
        if self.image_processor is None:
            return self._resize_and_pad(pil_img)

        # Optional square padding (image_aspect_ratio="pad")
        image_aspect_ratio = getattr(self.data_args, "image_aspect_ratio", "square") if self.data_args else "square"
        if image_aspect_ratio == "pad":
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
            mean = getattr(self.image_processor, "image_mean", [0.0, 0.0, 0.0])
            bg = tuple(int(x * 255) for x in mean)
            pil_img = expand2square(pil_img, bg)

        tensor = self.image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"][0]
        return tensor

    def _build_conversation_for_patch(self, labels) -> Sequence[Dict[str, str]]:
        """Create a simple two-turn conversation for a patch.

        TODO: Replace the instructions.
        """
        # Normalize labels to a string
        if isinstance(labels, (list, tuple)):
            label_text = self.label_joiner.join([str(x) for x in labels])
        else:
            label_text = str(labels)

        instruction = self.prompt_template

        return [
            {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"},
            {"from": "gpt", "value": label_text},
        ]

    def _load_region_by_entry(self, entry: Dict) -> Tuple[Tuple[int, int, int, int], List[str] | str]:
        """Given a flat-index entry, load bbox and labels from its annotation JSON.

        Returns (bbox, labels). bbox is (l,t,r,b) as ints. labels is list or string.
        """
        ann_path: str = entry["ann_path"]
        image_name: str = entry["image_name"]
        src: str = entry["src"]
        region_idx: int = entry["region_idx"]

        with open(ann_path, "r", encoding="utf-8") as f:
            ann_file = json.load(f)
        ann = ann_file.get(image_name, {})
        regions = ann.get(src, [])
        if not isinstance(regions, list) or not (0 <= region_idx < len(regions)):
            raise RuntimeError(f"Region not found: {ann_path}:{image_name}:{src}[{region_idx}]")
        region = regions[region_idx]
        bbox = region.get("bbox")
        labels = region.get("labels")
        if bbox is None or labels is None:
            raise RuntimeError(f"Missing bbox/labels in {ann_path}:{image_name}:{src}[{region_idx}]")
        l, t, r, b = map(int, bbox)
        return (l, t, r, b), labels

    def _iter_regions(self, ann: dict):
        """Yield (bbox, labels) tuples from both objects and floating_objects."""
        for key in ("objects", "floating_objects"):
            regions = ann.get(key, [])
            if not isinstance(regions, list):
                continue
            for region in regions:
                if not isinstance(region, dict):
                    continue
                bbox = region.get("bbox")
                labels = region.get("labels")
                if bbox is None or labels is None:
                    continue
                # Ensure labels is a list
                if not isinstance(labels, list):
                    labels = [labels]
                yield bbox, labels

    def _get_captions(self, ann: dict) -> list:
        """Extract captions from annotation, preferring dense_caption over short_captions."""
        captions = []
        
        # dense_caption
        if "dense_caption" in ann:
            dense = ann["dense_caption"]
            if isinstance(dense, dict) and "caption" in dense:
                caption_text = dense["caption"]
                if caption_text and isinstance(caption_text, str):
                    captions.append(caption_text)
            elif isinstance(dense, str) and dense:
                captions.append(dense)
        
        # short_captions
        if not captions and "short_captions" in ann:
            short = ann["short_captions"]
            if isinstance(short, list):
                for cap_obj in short:
                    if isinstance(cap_obj, dict) and "caption" in cap_obj:
                        caption_text = cap_obj["caption"]
                        if caption_text and isinstance(caption_text, str):
                            captions.append(caption_text)
                    elif isinstance(cap_obj, str) and cap_obj:
                        captions.append(cap_obj)
        
        # simple captions
        if not captions and "captions" in ann:
            old_caps = ann["captions"]
            if isinstance(old_caps, list):
                captions.extend([c for c in old_caps if c and isinstance(c, str)])
            elif isinstance(old_caps, str) and old_caps:
                captions.append(old_caps)

        return captions

    def _build_flat_index(self):
        """Build list of valid patch entries.
        """
        self._flat_index = []
        for ann_idx, ann_path in enumerate(self.annotation_files):
            # Stream-load JSON, don't keep beyond this scope
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_file = json.load(f)
            except Exception:
                continue
            if not isinstance(ann_file, dict) or len(ann_file) == 0:
                continue
            image_name = list(ann_file.keys())[0]
            ann = ann_file.get(image_name, {})
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(image_path):
                continue
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                continue
            img_area = width * height

            # Collect valid regions
            for src in ("objects", "floating_objects"):
                regions = ann.get(src, [])
                if not isinstance(regions, list):
                    continue
                for region_idx, region in enumerate(regions):
                    if not isinstance(region, dict):
                        continue
                    bbox = region.get("bbox")
                    if bbox is None:
                        continue
                    try:
                        left, upper, right, lower = map(int, bbox)
                    except Exception:
                        continue
                    patch_area = max(0, right - left) * max(0, lower - upper)
                    if patch_area <= 0:
                        continue
                    if not self.check_area_fn(img_area, patch_area):
                        continue
                    self._flat_index.append(
                        {
                            "ann_path": ann_path,
                            "image_name": image_name,
                            "src": src,
                            "region_idx": region_idx,
                        }
                    )

    def __len__(self):
        if self.flatten_patches:
            return len(self._flat_index)
        # In non-flattened mode, report number of annotation files
        return len(self.annotation_files)
    
    def _extract_patch(self, img: Image.Image, bbox) -> Image.Image:
        left, upper, right, lower = bbox
        img_crop = img.crop((left, upper, right, lower))
        return img_crop

    def __getitem__(self, idx):
        if self.flatten_patches:
            # Return a single patch + tokenized text
            entry = self._flat_index[idx]
            image_name = entry["image_name"]
            bbox, labels = self._load_region_by_entry(entry)

            image_path = os.path.join(self.image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            img_crop = self._extract_patch(image, bbox)

            image_tensor = self._preprocess_patch_image(img_crop)

            # Build per-patch conversation and tokenize
            conversations = [self._build_conversation_for_patch(labels)]
            conversations = self._preprocess_multimodal(conversations)
            data_dict = self._preprocess(conversations, has_image=True)

            # Prepare output dict similar to LazySupervisedDataset single item
            out = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                image=image_tensor,
            )
            # Optional flags
            out["is_coco"] = False
            return out
        
        # Non-flattened path (returns multiple patches per item). TODO
        else:
            raise NotImplementedError("Non-flattened patch mode not yet implemented.")