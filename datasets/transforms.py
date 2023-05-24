"""
Transforms on both image and corresponding target dictionaries.
"""
import math
import random

import torch.nn.functional as F
from torchvision.ops.misc import interpolate
import torchvision.transforms as T


# 1. Transforms based on cropping
def crop(image, tgt_dict, crop_region):
    """
    Function cropping the input image w.r.t. the given crop region. The input target dictionary is updated accordingly.

    Args:
        image (Images): Images structure with PIL Image to be cropped.

        tgt_dict (Dict): Target dictionary corresponding to the image, potentially containing following keys:
            - labels (LongTensor): tensor of shape [num_targets] containing the class indices;
            - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets];
            - masks (BoolTensor): segmentation masks of shape [num_targets, height, width].

        crop_region (Tuple): Tuple delineating the cropped region in (left, top, right, bottom) format.

    Returns:
        image (Images): Updated Images structure with cropped PIL Image.
        tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
    """

    # Crop the image w.r.t. the given crop region
    image = image.crop(crop_region)

    # Update bounding boxes and identify targets no longer in the image
    if 'boxes' in tgt_dict:
        tgt_dict['boxes'], well_defined = tgt_dict['boxes'].crop(crop_region)

    # Update segmentation masks and identify targets no longer in the image (if not already done)
    if 'masks' in tgt_dict:
        left, top, right, bottom = crop_region
        tgt_dict['masks'] = tgt_dict['masks'][:, top:bottom, left:right]
        well_defined = tgt_dict['masks'].flatten(1).any(1) if 'boxes' not in tgt_dict else well_defined

    # Remove targets that no longer appear in the cropped image
    if 'boxes' in tgt_dict or 'masks' in tgt_dict:
        for key in tgt_dict.keys():
            if key in ['labels', 'boxes', 'masks']:
                tgt_dict[key] = tgt_dict[key][well_defined]

    return image, tgt_dict


class CenterCrop(object):
    """
    Class implementing the CenterCrop transform.

    Attributes:
        crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
    """

    def __init__(self, crop_size):
        """
        Initializes the CenterCrop transform.

        Args:
            crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
        """

        self.crop_size = crop_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method corresponding to the CenterCrop transform.

        Args:
            image (Images): Images structure with PIL Image to be cropped.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with cropped PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
        """

        crop_height, crop_width = self.size
        crop_left = int(round((image.images.width - crop_width) / 2.))
        crop_top = int(round((image.images.height - crop_height) / 2.))
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height

        crop_region = (crop_left, crop_top, crop_right, crop_bottom)
        image, tgt_dict = crop(image, tgt_dict, crop_region)

        return image, tgt_dict


class RandomCrop(object):
    """
    Class implementing the RandomCrop transform.

    Attributes:
        crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
    """

    def __init__(self, crop_size):
        """
        Initializes the RandomCrop transform.

        Args:
            crop_size (Tuple): Tuple of size [2] containing the crop height and crop width respectively.
        """

        self.crop_size = crop_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method corresponding to the RandomCrop transform.

        Args:
            image (Images): Images structure with PIL Image to be cropped.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with cropped PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
        """

        crop_top, crop_left, crop_height, crop_width = T.RandomCrop.get_params(image.images, self.crop_size)
        crop_right, crop_bottom = (crop_left + crop_width, crop_top + crop_height)

        crop_region = (crop_left, crop_top, crop_right, crop_bottom)
        image, tgt_dict = crop(image, tgt_dict, crop_region)

        return image, tgt_dict


class RandomSizeCrop(object):
    """
    Class implementing the RandomSizeCrop transform.

    Attributes:
        min_crop_size (int): Integer containing the minimum crop height and crop width.
        max_crop_size (int): Integer containing the maximum crop height and crop width.
    """

    def __init__(self, min_crop_size, max_crop_size):
        """
        Initializes the RandomSizeCrop transform.

        Args:
            min_crop_size (int): Integer containing the minimum crop height and crop width.
            max_crop_size (int): Integer containing the maximum crop height and crop width.
        """

        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method corresponding to the RandomSizeCrop transform.

        Args:
            image (Images): Images structure with PIL Image to be cropped.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with cropped PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new cropped image.
        """

        crop_width = random.randint(self.min_crop_size, min(image.images.width, self.max_crop_size))
        crop_height = random.randint(self.min_crop_size, min(image.images.height, self.max_crop_size))

        crop_top, crop_left, crop_height, crop_width = T.RandomCrop.get_params(image.images, [crop_height, crop_width])
        crop_right, crop_bottom = (crop_left + crop_width, crop_top + crop_height)

        crop_region = (crop_left, crop_top, crop_right, crop_bottom)
        image, tgt_dict = crop(image, tgt_dict, crop_region)

        return image, tgt_dict


# 2. Transforms based on horizontal flipping
def hflip(image, tgt_dict):
    """
    Function flipping the input image horizontally and updating the target dictionary accordingly.

    Args:
        image (Images): Images structure with PIL Image to be flipped horizontally.

        tgt_dict (Dict): Target dictionary corresponding to the image, potentially containing following keys:
            - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets];
            - masks (BoolTensor): segmentation masks of shape [num_targets, height, width].

    Returns:
        image (Images): Updated Images structure with horizontally flipped PIL Image.
        tgt_dict (Dict): Updated target dictionary corresponding to the new flipped image.
    """

    # Flip the image horizontally
    image = image.hflip()

    # Update the bounding boxes
    if 'boxes' in tgt_dict:
        tgt_dict['boxes'] = tgt_dict['boxes'].hflip(image.images.width)

    # Update the segmentation masks
    if 'masks' in tgt_dict:
        tgt_dict['masks'] = tgt_dict['masks'].flip([-1])

    return image, tgt_dict


class RandomHorizontalFlip(object):
    """
    Class implementing the RandomHorizontalFlip transform.

    Attributes:
        flip_prob (float): Value between 0 and 1 determining the flip probability.
    """

    def __init__(self, flip_prob=0.5):
        """
        Initializes the RandomHorizontalFlip transform.

        Args:
            flip_prob (float): Value between 0 and 1 determining the flip probability.
        """

        self.flip_prob = flip_prob

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomHorizontalFlip transform.

        Args:
            image (Images): Images structure with PIL Image to (potentially) be flipped horizontally.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Images structure with potential update due to horizontal flipping.
            tgt_dict (Dict): The (potentially updated) target dictionary corresponding to the returned image.
        """

        if random.random() < self.flip_prob:
            image, tgt_dict = hflip(image, tgt_dict)

        return image, tgt_dict


# 3. Transforms based on padding
def pad(image, tgt_dict, padding):
    """
    Function padding the input image w.r.t. the given padding vector. The target dictionary is updated accordingly.

    Args:
        image (Images): Images structure with PIL Image to be padded.

        tgt_dict (Dict): Target dictionary corresponding to the image, potentially containing following key:
            - masks (BoolTensor): segmentation masks of shape [num_targets, height, width].

        padding (Tuple): Padding vector of size [4] with padding values in (left, top, right, bottom) format.

    Returns:
        image (Images): Updated Images structure with padded PIL Image.
        tgt_dict (Dict): Updated target dictionary corresponding to the new padded image.
    """

    # Pad the image as specified by the padding vector
    image = image.pad(padding)

    # Update the bounding boxes
    if 'boxes' in tgt_dict:
        tgt_dict['boxes'] = tgt_dict['boxes'].pad(padding)

    # Update the segmentation masks
    if 'masks' in tgt_dict:
        mask_padding = (padding[0], padding[2], padding[1], padding[3])
        tgt_dict['masks'] = F.pad(tgt_dict['masks'], mask_padding)

    return image, tgt_dict


class RandomPad(object):
    """
    Class implementing the RandomPad transform.

    It randomly pads the right and bottom of images with 'max_pad' attribute as maximum padding value.

    Attributes:
        max_pad (int): Integer containing the maximum value for the right and bottom padding values.
    """

    def __init__(self, max_pad):
        """
        Initializes the RandomPad transform.

        Args:
            max_pad (int): Integer containing the maximum value for the right and bottom padding values.
        """

        self.max_pad = max_pad

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomPad transform.

        Args:
            image (Images): Images structure with PIL Image to be padded.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with padded PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new padded image.
        """

        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)

        padding = (0, 0, pad_x, pad_y)
        image, tgt_dict = pad(image, tgt_dict, padding)

        return image, tgt_dict


# 4. Transforms based on resizing
def resize(image, tgt_dict, size, max_size=None):
    """
    Function resizing the input image. The input target dictionary is updated accordingly.

    Args:
        image (Images): Images structure with PIL Image to be resized.

        tgt_dict (Dict): Target dictionary corresponding to the image, potentially containing following keys:
            - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets];
            - masks (BoolTensor): segmentation masks of shape [num_targets, height, width].

        size: It can be one of following two possibilities:
            1) size (int): integer containing the minimum size (width or height) to resize to;
            2) size (Tuple): tuple of size [2] containing the image size in (width, height) format to resize to.

        max_size (int): Overrules above size whenever this value is exceeded in width or height (default=None).

    Returns:
        image (Images): Updated Images structure with resized PIL Image.
        tgt_dict (Dict): Updated target dictionary corresponding to the new resized image.
    """

    # Resize the image as specified by the size and max_size arguments
    image, size, resize_ratio = image.resize(size, max_size)

    # Update the bounding boxes
    if 'boxes' in tgt_dict:
        tgt_dict['boxes'] = tgt_dict['boxes'].resize(resize_ratio)

    # Update the segmentation masks
    if 'masks' in tgt_dict:
        kwargs = {'size': (size[1], size[0]), 'mode': 'nearest', 'align_corners': None}
        tgt_dict['masks'] = interpolate(tgt_dict['masks'][:, None].float(), **kwargs)[:, 0] > 0.5

    return image, tgt_dict


class RandomCropResize(object):
    """
    Class implementing the RandomCropResize transform.

    Attributes:
        min_crop_factor (float): Value containing the minumum crop factor.
        max_crop_factor (float): Value containing the maximum crop factor.
    """

    def __init__(self, min_crop_factor, max_crop_factor=1.0):
        """
        Initializes the RandomCropResize transform.

        Args:
            min_crop_factor (float): Value containing the minumum crop factor.
            max_crop_factor (float): Value containing the maximum crop factor (default=1.0).
        """

        assert max_crop_factor <= 1
        assert min_crop_factor <= max_crop_factor

        self.min_crop_factor = min_crop_factor
        self.max_crop_factor = max_crop_factor

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomCropResize transform.

        Args:
            image (Images): Images structure with PIL Image to be resized.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with resized PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new resized image.
        """

        low = math.log2(self.min_crop_factor)
        high = math.log2(self.max_crop_factor)
        crop_factor = 2**random.uniform(low, high)

        iW, iH = image.size()
        crop_size = (int(crop_factor * iH), int(crop_factor * iW))

        crop_top, crop_left, crop_height, crop_width = T.RandomCrop.get_params(image.images, crop_size)
        crop_right, crop_bottom = (crop_left + crop_width, crop_top + crop_height)

        crop_region = (crop_left, crop_top, crop_right, crop_bottom)
        image, tgt_dict = crop(image, tgt_dict, crop_region)
        image, tgt_dict = resize(image, tgt_dict, (iW, iH))

        return image, tgt_dict


class RandomRescale(object):
    """
    Class implementing the RandomRescale transform.

    Attributes:
        min_scale_factor (float): Value containing the minumum scale factor.
        max_scale_factor (float): Value containing the maximum scale factor.
    """

    def __init__(self, min_scale_factor, max_scale_factor):
        """
        Initializes the RandomRescale transform.

        Args:
            min_scale_factor (float): Value containing the minumum scale factor.
            max_scale_factor (float): Value containing the maximum scale factor.
        """

        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomRescale transform.

        Args:
            image (Images): Images structure with PIL Image to be resized.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with resized PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new resized image.
        """

        low = math.log2(self.min_scale_factor)
        high = math.log2(self.max_scale_factor)
        scale_factor = 2**random.uniform(low, high)

        size = int(scale_factor * min(image.size()))
        image, tgt_dict = resize(image, tgt_dict, size)

        return image, tgt_dict


class RandomResize(object):
    """
    Class implementing the RandomResize transform.

    Attributes:
        sizes (List or Tuple): Collection of minimum size integers or size tuples in (width, height) format.
        max_size (int): Overrules size whenever this value is exceeded in width or height (default=None).
    """

    def __init__(self, sizes, max_size=None):
        """
        Initializes the RandomResize transform.

        Args:
            sizes (List or Tuple): Collection of minimum size integers or size tuples in (width, height) format.
            max_size (int): Overrules size whenever this value is exceeded in width or height (default=None).
        """

        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomResize transform.

        Args:
            image (Images): Images structure with PIL Image to be resized.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with resized PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new resized image.
        """

        size = random.choice(self.sizes)
        image, tgt_dict = resize(image, tgt_dict, size, self.max_size)

        return image, tgt_dict


# 5. Miscellaneous transforms
class RandomSelect(object):
    """
    Class implementing the RandomSelect transform.

    It Randomly selects between one of two transforms.

    Attributes:
        t1 (object): Transform object chosen with probability 'prob'.
        t2 (object): Transform object chosen with probability '1-prob'.
        prob (float): Probability (between 0 and 1) of choosing transforms1.
    """

    def __init__(self, t1, t2, prob=0.5):
        """
        Initializes the RandomSelect transform.

        Args:
            t1 (object): Transform object chosen with probability 'prob'.
            t2 (object): Transform object chosen with probability '1-prob'.
            prob (float): Probability (between 0 and 1) of choosing transforms1 (default=0.5).
        """

        self.t1 = t1
        self.t2 = t2
        self.prob = prob

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the RandomSelect transform.

        Args:
            image (Images): Images structure with PIL Image to be transformed.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with transformed PIL Image.
            tgt_dict (Dict): Updated target dictionary corresponding to the new transformed image.
        """

        image, tgt_dict = self.t1(image, tgt_dict) if random.random() < self.prob else self.t2(image, tgt_dict)

        return image, tgt_dict


class ToTensor(object):
    """
    Class implementing the ToTensor transform.
    """

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the ToTensor transform.

        Args:
            image (Images): Images structure with PIL Image to be converted to tensor.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with image tensor instead of PIL Image.
            tgt_dict (Dict): Unchanged target dictionary corresponding to the image.
        """

        image = image.to_tensor()

        return image, tgt_dict


class Compose(object):
    """
    Class implementing the Compose transform.

    Attributes:
        transforms (List): List of transform objects of size [num_transforms].
    """

    def __init__(self, transforms):
        """
        Initializes the Compose transform.

        Args:
            transforms (List): List of transform objects of size [num_transforms].
        """

        self.transforms = transforms

    def __call__(self, image, tgt_dict):
        """
        The __call__ method of the Compose transform.

        Args:
            image (Images): Images structure with PIL Image to be transformed.
            tgt_dict (Dict): Target dictionary corresponding to the input image.

        Returns:
            image (Images): Updated Images structure with transformed image.
            tgt_dict (Dict): Updated target dictionary corresponding to the transformed image.
        """

        for transform in self.transforms:
            image, tgt_dict = transform(image, tgt_dict)

        return image, tgt_dict

    def __repr__(self):
        """
        Implements the __repr__ method of the Compose transform.

        Returns:
            compose_string (str): String with information about the different sub-transforms of this transform.
        """

        compose_string = "Compose transform with:"
        for i, transform in enumerate(self.transforms):
            compose_string += f"\n    {i}: {transform}"

        return compose_string


# 6. Functions getting training and evaluation transforms
def get_train_transforms(transforms_type='coco_multi_scale'):
    """
    Function getting a list containing the specified training transforms.

    Args:
        transforms_type (str): String containing the training transforms type (default='coco_multi_scale').

    Returns:
        transforms (List): List [num_train_transforms] containing training transforms.

    Raises:
        ValueError: Error when an unknown training transforms type is provided.
    """

    # Get training transforms
    if transforms_type in ('coco_single_scale', 'objects365_single_scale'):
        hflip = RandomHorizontalFlip()
        resize = RandomResize([800], max_size=1333)
        to_tensor = ToTensor()
        transforms = [Compose([hflip, resize, to_tensor])]

    elif transforms_type in ('coco_multi_scale', 'objects365_multi_scale'):
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        default_resize = RandomResize(scales, max_size=1333)

        crop = Compose([RandomResize([400, 500, 600]), RandomSizeCrop(384, 600)])
        cropped_resize = Compose([crop, default_resize])

        hflip = RandomHorizontalFlip()
        resize = RandomSelect(default_resize, cropped_resize)
        to_tensor = ToTensor()
        transforms = [Compose([hflip, resize, to_tensor])]

    elif transforms_type == 'cityscapes_single_scale':
        hflip = RandomHorizontalFlip()
        resize = RandomResize([1024], max_size=2048)
        to_tensor = ToTensor()
        transforms = [Compose([hflip, resize, to_tensor])]

    elif transforms_type == 'cityscapes_multi_scale':
        shrink = RandomRescale(0.25, 1.0)
        crop_expand = RandomCropResize(0.25, 1.0)

        hflip = RandomHorizontalFlip()
        resize = RandomSelect(shrink, crop_expand)
        to_tensor = ToTensor()
        transforms = [Compose([hflip, resize, to_tensor])]

    else:
        error_msg = f"Unknown training transforms type (got '{transforms_type}')."
        raise ValueError(error_msg)

    return transforms


def get_eval_transforms(transforms_type='coco_single_scale'):
    """
    Function getting a list containing the specified evaluation transforms.

    Args:
        transform_type (str): String containing the evaluation transforms type (default='coco_single_scale').

    Returns:
        transforms (List): List [num_eval_transforms] containing evaluation transforms.

    Raises:
        ValueError: Error when an unknown evaluation transforms type is provided.
    """

    # Get evaluation transforms
    if transforms_type in ('coco_single_scale', 'objects365_single_scale'):
        resize = RandomResize([800], max_size=1333)
        to_tensor = ToTensor()
        transforms = [Compose([resize, to_tensor])]

    elif transforms_type in ('coco_single_scale_flip', 'objects365_single_scale_flip'):
        hflip = RandomHorizontalFlip(flip_prob=1.0)
        resize = RandomResize([800], max_size=1333)
        to_tensor = ToTensor()
        transforms = [Compose([resize, to_tensor]), Compose([hflip, resize, to_tensor])]

    elif transforms_type in ('coco_multi_scale', 'objects365_multi_scale'):
        scales = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
        resizes = [RandomResize([scale]) for scale in scales]
        to_tensor = ToTensor()
        transforms = [Compose([resize, to_tensor]) for resize in resizes]

    elif transforms_type in ('coco_multi_scale_flip', 'objects365_multi_scale_flip'):
        hflip = RandomHorizontalFlip(flip_prob=1.0)
        scales = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
        resizes = [RandomResize([scale]) for scale in scales]
        to_tensor = ToTensor()

        transforms = [Compose([resize, to_tensor]) for resize in resizes]
        transforms.extend([Compose([hflip, resize, to_tensor]) for resize in resizes])

    elif transforms_type == 'cityscapes_single_scale':
        resize = RandomResize([1024], max_size=2048)
        to_tensor = ToTensor()
        transforms = [Compose([resize, to_tensor])]

    elif transforms_type == 'cityscapes_single_scale_flip':
        hflip = RandomHorizontalFlip(flip_prob=1.0)
        resize = RandomResize([1024], max_size=2048)
        to_tensor = ToTensor()
        transforms = [Compose([resize, to_tensor]), Compose([hflip, resize, to_tensor])]

    else:
        error_msg = f"Unknown evaluation transforms type (got '{transforms_type}')."
        raise ValueError(error_msg)

    return transforms
