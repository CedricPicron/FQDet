"""
Data loading and data structure utilities.
"""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from structures.boxes import Boxes
from structures.images import Images


def collate_fn(batch):
    """
    The collate function used during data loading.

    Args:
        batch (List): List of size [batch_size] containing tuples of:
            - image (Images): structure containing the image tensor after data augmentation;
            - tgt_dict (Dict): target dictionary potentially containing following keys (empty when no annotations):
                - labels (LongTensor): tensor of shape [num_targets] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets];
                - masks (BoolTensor): segmentation masks of shape [num_targets, iH, iW].

    Returns:
        images (Images): New Images structure containing the concatenated Images structures across batch entries.

        tgt_dict (Dict): New target dictionary potentially containing following keys (empty when no annotations):
            - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
            - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
            - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
            - masks (ByteTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].
    """

    # Get batch images and target dictionaries
    images, tgt_dicts = list(zip(*batch))

    # Concatenate images across batch entries
    images = Images.cat(images)

    # Return if target dictionaries are empty
    if not tgt_dicts[0]:
        tgt_dict = {}
        return images, tgt_dict

    # Concatenate target labels and target boxes across batch entries
    tgt_labels = torch.cat([tgt_dict['labels'] for tgt_dict in tgt_dicts])
    tgt_boxes = Boxes.cat([tgt_dict['boxes'] for tgt_dict in tgt_dicts], offset_batch_ids=True)

    # Compute cumulative target sizes of batch entries
    tgt_sizes = [0] + [len(tgt_dict['labels']) for tgt_dict in tgt_dicts]
    tgt_sizes = torch.tensor(tgt_sizes, dtype=torch.int64)
    tgt_sizes = torch.cumsum(tgt_sizes, dim=0)

    # Place target labels, boxes and sizes in new target dictionary
    tgt_dict = {'labels': tgt_labels, 'boxes': tgt_boxes, 'sizes': tgt_sizes}

    # Concatenate masks if provided and add to target dictionary
    if 'masks' in tgt_dicts[0]:
        max_iW, max_iH = images.size(mode='with_padding')
        masks = [old_tgt_dict['masks'] for old_tgt_dict in tgt_dicts]
        padded_masks = [F.pad(mask, (0, max_iW-mask.shape[-1], 0, max_iH-mask.shape[-2])) for mask in masks]
        tgt_dict['masks'] = torch.cat(padded_masks, dim=0)

    return images, tgt_dict


class SubsetSampler(Sampler):
    """
    Class implementing the SubsetSampler sampler.

    Attributes:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        subset_size (int): Positive integer containing the subset size.
        subset_offset (int): Positive integer containing the subset offset.
    """

    def __init__(self, dataset, subset_size, subset_offset=0, random_offset=False):
        """
        Initializes the SubsetSampler sampler.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to sample from.
            subset_size (int): Positive integer specifying the desired subset size.
            subset_offset (int): Positive integer specifying the desired subset offset (default=0).
            random_offset (bool): Boolean indicating whether to generate a random subset offset (default=False).
        """

        self.dataset = dataset
        self.subset_size = min(subset_size, len(dataset))

        max_offset = len(dataset) - self.subset_size
        self.subset_offset = random.randint(0, max_offset) if random_offset else min(subset_offset, max_offset)

    def __iter__(self):
        """
        Implements the __iter__ method of the SubsetSampler sampler.

        Returns:
            An iterator containing the subset indices.
        """

        return iter(range(self.subset_offset, self.subset_offset+self.subset_size))

    def __len__(self):
        """
        Implements the __len__ method of the SubsetSampler sampler.

        Returns:
            An integer containing the subset size.
        """

        return self.subset_size
