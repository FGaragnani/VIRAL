# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from src.custom_utils import get_logger, is_debug

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

_GLOBAL_SEED = 0
logger = get_logger()


class MaskCollator(object):

    def __init__(
        self,
        image_processor,
        tokenizer,
        return_labels=False,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        iwm_full_img_on_encoder=False
    ):
        super(MaskCollator, self).__init__()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.return_labels = return_labels
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        # minimum number of patches to keep
        self.min_keep = min_keep
        # whether to allow overlap b/w enc and pred masks
        self.allow_overlap = allow_overlap
        # collator is shared across worker processes
        self._itr_counter = Value('i', -1)
        self.iwm_full_img_on_encoder = iwm_full_img_on_encoder

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(
            (self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        if isinstance(batch, dict):
            # webdataset returns a dict rather than a list of dicts
            batch = [dict((k, batch[k][i]) for k in batch) for i in range(len(next(iter(batch.values()))))]
        B = len(batch)

        # collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for i in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # collated_masks_pred = [[cm[:min_keep_pred]
        #                         for cm in cm_list] for cm_list in collated_masks_pred]
        # collated_masks_pred = torch.utils.data.default_collate(
        #     collated_masks_pred)
        # collated_masks_enc = [[cm[:min_keep_enc]
        #                        for cm in cm_list] for cm_list in collated_masks_enc]
        # collated_masks_enc = torch.utils.data.default_collate(
        #     collated_masks_enc)
        # --
        collated_masks_pred = torch.stack(
            [torch.stack([y[:min_keep_pred] for y in x]) for x in collated_masks_pred])
        assert self.nenc == 1, "Not implemented for nenc > 1"
        collated_masks_enc = torch.stack(
            [x[0][:min_keep_enc] for x in collated_masks_enc])

        # We use frozen visual encoders that have been trained with fixed-sized images.
        # So, rather than feeding only the context patches to the context encoder,
        # we feed the entire image to the context encoder,
        # and we mask out all but the context patches
        def rescale_and_normalize(x):
            return self.image_processor.normalize(
                self.image_processor.rescale(
                    x,
                    scale=self.image_processor.rescale_factor
                ),
                mean=self.image_processor.image_mean,
                std=self.image_processor.image_std
            )

        pixel_values_enc = []
        pixel_values_pred = []
        texts = []
        for i, sample in enumerate(batch):
            img = sample['image']
            pixel_values_pred_unnorm = self.image_processor(
                img, do_rescale=False, do_normalize=False).pixel_values[0]
            c, h, w = pixel_values_pred_unnorm.shape
            patches_per_side = pixel_values_pred_unnorm.shape[1] // self.patch_size
            n_patches = patches_per_side ** 2

            patchified_image = pixel_values_pred_unnorm.reshape(
                c, patches_per_side, self.patch_size, patches_per_side, self.patch_size)

            if not self.iwm_full_img_on_encoder:
                patchified_image = patchified_image.swapaxes(2, 3).reshape(
                    c, n_patches, self.patch_size, self.patch_size)
                enc_mask = collated_masks_enc[i]
                enc_masked_img = np.zeros_like(patchified_image)
                enc_masked_img[:, enc_mask] = patchified_image[:, enc_mask]
                enc_masked_img = enc_masked_img.reshape(
                    c, patches_per_side, patches_per_side, self.patch_size, self.patch_size)
                enc_masked_img = enc_masked_img.swapaxes(2, 3).reshape(c, h, w)
                pixel_values_enc.append(rescale_and_normalize(enc_masked_img))

                if is_debug():
                    show = Image.fromarray(enc_masked_img.transpose(1, 2, 0))
                    ...

            pixel_values_pred.append(
                rescale_and_normalize(pixel_values_pred_unnorm))
            texts.append(sample['text'])

        if self.iwm_full_img_on_encoder:
            pixel_values_enc = None
        else:
            pixel_values_enc = torch.from_numpy(np.stack(pixel_values_enc))
        pixel_values_pred = torch.from_numpy(np.stack(pixel_values_pred))

        if is_debug():
            assert (pixel_values_pred == self.image_processor(
                [x['image'] for x in batch], return_tensors='pt').pixel_values).all()
            # save_masks_preview(pixel_values_pred, collated_masks_enc, collated_masks_pred, self.patch_size)

        text_inputs = self.tokenizer(texts, padding=True, return_tensors='pt', add_special_tokens=False)
        if self.return_labels:
            labels = text_inputs['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            text_inputs['labels'] = labels

        return dict(
            **text_inputs,
            pixel_values_enc=pixel_values_enc,
            pixel_values_pred=pixel_values_pred,
            mask_idxs_enc=collated_masks_enc,
            mask_idxs_pred=collated_masks_pred
        )


def save_masks_preview(collated_batch, collated_masks_enc, collated_masks_pred, patch_size):
    from pathlib import Path

    def make_viewable_for_PIL(x):
        x = ((x - x.min()) / (x.max() - x.min()) * 255).to(torch.uint8)
        return x.numpy().swapaxes(0, 1).swapaxes(1, 2)
    for BIDX in range(len(collated_batch)):
        img = collated_batch[BIDX].to(torch.float)
        h, w = img.shape[-2:]
        patchified_img = F.unfold(img, patch_size, stride=patch_size)

        enc_mask = collated_masks_enc[BIDX]
        enc_masked_img = torch.zeros_like(patchified_img)
        enc_masked_img[:, enc_mask] = patchified_img[:, enc_mask]
        enc_masked_img = F.fold(enc_masked_img, (h, w),
                                patch_size, stride=patch_size)
        enc_masked_img = Image.fromarray(make_viewable_for_PIL(enc_masked_img))
        P = Path(
            "src/train/__pycache__/preview/PIL").joinpath(f"sample_{BIDX}")
        P.mkdir(parents=True, exist_ok=True)
        enc_masked_img.save(P.joinpath("enc_masked_img.png"))

        for i, pred_mask in enumerate(collated_masks_pred[BIDX]):
            pred_mask_img = torch.zeros_like(patchified_img)
            pred_mask_img[:, pred_mask] = patchified_img[:, pred_mask]
            pred_mask_img = F.fold(pred_mask_img, (h, w),
                                   patch_size, stride=patch_size)
            pred_mask_img = Image.fromarray(
                make_viewable_for_PIL(pred_mask_img))
            pred_mask_img.save(P.joinpath(f"pred_mask_img_{i}.png"))


class SupervisedCollator:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []
        pixel_values = []
        image_mask = []
        texts = []
        for sample in batch:
            input_ids.append(sample['input_ids'])
            attention_mask.append(sample['attention_mask'])
            labels.append(sample['labels'])
            texts.append(sample['text'])
            pixel_values.append(sample['image'])
            image_mask.append(sample['image_mask'])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100)
        pixel_values = self.image_processor(
            pixel_values, return_tensors='pt').pixel_values

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
        )
