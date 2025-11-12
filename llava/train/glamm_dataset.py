import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import numpy as np
import json
import os
from typing import List, Tuple, Optional

class GranDDataset(Dataset):
    """
        Dataset for GLAMM training using GranD annotations.
        Each item returns image patches and their corresponding captions.
    """
    def __init__(self, image_dir, annotation_dir, 
                 patch_size=(224,224), 
                 check_area: float = 0.05):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = self._resize_and_pad
        self.annotations = [json.load(open(os.path.join(annotation_dir, f))) for f in os.listdir(annotation_dir)]
        self.check_area_fn = (lambda img_size, patch_size: (patch_size/img_size) > check_area)

    def _resize_and_pad(self, img) -> torch.Tensor:
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
            # for older Pillow versions without color argument
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

    def __len__(self):
        return len(self.annotations)
    
    def _extract_patch(self, img: Image.Image, img_area, bbox) -> Optional[Image.Image]:
        left, upper, right, lower = bbox
        # check if patch is large enough
        if self.check_area_fn(img_area, (right - left)*(lower - upper)) == False:
            return None
        img_crop = img.crop((left, upper, right, lower))
        return img_crop

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_name = list(ann.keys())[0]
        ann = ann[image_name]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        img_area = image.width * image.height
        patches = []
        captions = []

        for region in ann["objects"]:
            if "bbox" not in region or "labels" not in region:
                continue

            img_crop = self._extract_patch(image, img_area, region["bbox"])
            if img_crop is None:
                continue

            img_t: torch.Tensor = self.transform(img_crop)
            patches.append(img_t)
            captions.append(region["labels"])

        for region in ann["floating_objects"]:
            if "bbox" not in region or "labels" not in region:
                continue

            img_crop = self._extract_patch(image, img_area, region["bbox"])
            if img_crop is None:
                continue

            img_t: torch.Tensor = self.transform(img_crop)
            patches.append(img_t)
            captions.append(region["labels"])

        if len(patches)==0:
            # fallback: whole image
            img_t = self.transform(image)
            patches = [img_t]
            captions = [ann["captions"] if "captions" in ann else []]

        patches = torch.stack(patches)

        return {
            "image_id": image_name.split(".")[0],
            "patches": patches,
            "labels": captions,
            "caption": ann.get("captions", []),
        }
