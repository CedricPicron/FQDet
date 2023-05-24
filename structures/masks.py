"""
Collection of utility functions related to masks.
"""

import numpy as np
import torch
from pycocotools import mask as coco_mask
import torchvision.transforms.functional as F


def mask_inv_transform(in_masks, images, batch_ids):
    """
    Transforms the given masks back to the original image space as encoded by the given Images structure.

    Args:
        in_masks (BoolTensor): Input masks of shape [num_masks, in_height, in_width].
        images (Images): Images structure [batch_size] containing batched images with their entire transform history.
        batch_ids (LongTensor): Tensor containing the batch indices of each input mask of shape [num_masks].

    Returns:
        out_masks_list (List): List of size [num_masks] with output masks of shape [out_height, out_width].

    Raises:
        ValueError: Error when one of the transforms has an unknown transform type.
    """

    # Resize input masks to image size after transforms
    iW, iH = images.size()
    in_masks = F.resize(in_masks, size=(iH, iW))

    # Get list with output masks
    num_masks = len(in_masks)
    mask_ids = torch.arange(num_masks, device=in_masks.device)
    out_masks_list = [None for _ in range(num_masks)]

    for i, transforms_i in enumerate(images.transforms):
        batch_mask = batch_ids == i
        mask_ids_i = mask_ids[batch_mask]
        out_masks_i = in_masks[batch_mask]

        for transform in reversed(transforms_i):
            transform_type = transform[0]

            if transform_type == 'crop':
                crop_deltas = transform[2]
                out_masks_i = F.pad(out_masks_i, crop_deltas)

            elif transform_type == 'hflip':
                out_masks_i = F.hflip(out_masks_i)

            elif transform_type == 'pad':
                (left, top, right, bottom) = transform[2]
                (width, height) = (right-left, bottom-top)
                out_masks_i = F.crop(out_masks_i, top, left, height, width)

            elif transform_type == 'resize':
                (width_ratio, height_ratio) = transform[1]
                (old_height, old_width) = out_masks_i.size()[1:]

                new_height = int(old_height / height_ratio)
                new_width = int(old_width / width_ratio)
                new_size = (new_height, new_width)

                if len(out_masks_i) > 0:
                    out_masks_i = F.resize(out_masks_i, new_size)
                else:
                    out_masks_i = out_masks_i.new_zeros([0, *new_size])

            else:
                error_msg = f"Unknown transform type (got '{transform_type}')."
                raise ValueError(error_msg)

        for j, mask_id in enumerate(mask_ids_i.tolist()):
            out_masks_list[mask_id] = out_masks_i[j]

    return out_masks_list


def mask_to_rle(masks):
    """
    Function encoding the dense input masks using run-length encoding (RLE).

    Args:
        masks (List): List of size [num_masks] containing the dense input masks of shape [height, width].

    Returns:
        rles (List): List of size [num_masks] containing the RLEs of the dense input masks.
    """

    # Get RLEs of dense input masks
    rles = [coco_mask.encode(np.array(mask[:, :, None].cpu(), order='F', dtype='uint8'))[0] for mask in masks]

    # Alter counts entry of RLEs
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    return rles
