"""
Images structure.
"""
from copy import deepcopy

import PIL
import torch
import torchvision.transforms.functional as F


class Images(object):
    """
    Class implementing the Images structure.

    Attributes:
        images: It is one of following two possibilities:
            1) images (PIL.Image.Image): Single image in PIL format;
            2) images (FloatTensor): Image tensor of shape [batch_size, num_channels, iH, iW].

        image_ids (List): List of size [num_images] containing the image id corresponding to each image.
        masks (BoolTensor): Tensor of shape [batch_size, iH, iW] indicating active pixels (as opposed to padding).
        transforms (List): List of size [num_images] with transforms corresponding to each image.
    """

    def __init__(self, images, image_ids=None, masks=None, transforms=None):
        """
        Initializes the Images structure.

        Args:
            images: We support following two possibilities:
                1) images (PIL.Image.Image): Single image in PIL format;
                2) images (FloatTensor): Image tensor of shape [batch_size, num_channels, iH, iW].

            image_ids: Optional input supporting following two possibilities:
                1) image_ids (int): Integer containing the dataset image id (works only with single image input);
                2) image_ids (List): List of size [batch_size] containing the image ids of each of the input images.

            masks (BoolTensor): Optional tensor of shape [batch_size, iH, iW] indicating active pixels.
            transforms (List): Optional list of size [num_images] with transforms corresponding to each image.
        """

        # Set images, image ids and transforms attributes
        self.images = images
        self.image_ids = image_ids if isinstance(image_ids, list) else [image_ids] * len(self)
        self.transforms = transforms if transforms is not None else [[] for _ in range(len(self))]

        # Set masks attribute
        img_width, img_height = self.size()
        img_shape = (len(self), img_height, img_width)

        if masks is not None:
            check = img_shape == masks.shape
            assert check, f"The input images and masks have inconsistent shapes (got {img_shape} and {masks.shape})."
            self.masks = masks.to(images.device)

        else:
            masks = torch.ones(img_shape, dtype=torch.bool)
            self.masks = masks.to(images.device) if isinstance(images, torch.Tensor) else masks

    def __getitem__(self, key):
        """
        Implements the __getitem__ method of the Images structure.

        Args:
            key: Object of any type also supported by Tensor determining the selected images.

        Returns:
            item (Images): New Images structure containing the selected images.
        """

        images_tensor = self.images[key][None, :] if isinstance(key, int) else self.images[key]
        masks = self.masks[key][None, :] if isinstance(key, int) else self.masks[key]

        image_ids = [self.image_ids[key]] if isinstance(key, int) else self.image_ids[key]
        transforms = [self.transforms[key]] if isinstance(key, int) else self.transforms[key]
        item = Images(images_tensor, image_ids, masks, transforms)

        return item

    def __len__(self):
        """
        Implements the __len__ method of the Images structure.

        It is measured as the number of images within the structure.

        Returns:
            num_images (int): Number of images within the structure.
        """

        if isinstance(self.images, PIL.Image.Image):
            num_images = 1

        elif isinstance(self.images, torch.Tensor):
            num_images = len(self.images)

        return num_images

    def __repr__(self):
        """
        Implements the __repr__ method of the Images structure.

        Returns:
            images_string (str): String containing information about the Images structure.
        """

        images_string = "Images structure:\n"
        images_string += f"   Size: {len(self)}\n"
        images_string += f"   Image ids: {self.image_ids}\n"
        images_string += f"   Transforms: {self.transforms}\n"

        images_string += f"   Images content: {self.images}\n"
        images_string += f"   Masks content: {self.masks}"

        return images_string

    @staticmethod
    def cat(images_list):
        """
        Concatenate list of Images structures into single Images structure.

        Args:
            image_list (List): List of size [num_structures] with Image structures to be concatenated.

        Returns:
            cat_images (Images): Images structure containing the concatenated input Images structures.
        """

        # Check whether all Images structures have same number of channels
        channels_set = {s.images.shape[1] for s in images_list}
        assert len(channels_set) == 1, f"All images should have same number of channels (got {channels_set})."
        channels = channels_set.pop()

        # Check whether all Images structures reside on the same device
        device_set = {s.images.device for s in images_list}
        assert len(device_set) == 1, f"All images should reside on the same device (got {device_set})."
        device = device_set.pop()

        # Get list of image ids and list of transforms
        image_ids = []
        transforms = []

        for structure in images_list:
            image_ids.extend(structure.image_ids)
            transforms.extend(structure.transforms)

        # Initialize padded tensor of concatenated images and corresponding masks
        sizes = [0] + [len(structure.images) for structure in images_list]
        cum_sizes = torch.cumsum(torch.tensor(sizes), dim=0)

        batch_size = cum_sizes[-1]
        max_iH, max_iW = torch.tensor([list(s.images.shape[2:]) for s in images_list]).max(dim=0)[0].tolist()

        images_tensor = torch.zeros(batch_size, channels, max_iH, max_iW, dtype=torch.float, device=device)
        masks_tensor = torch.zeros(batch_size, max_iH, max_iW, dtype=torch.bool, device=device)

        # Fill the padded images and masks, and update list of transforms
        for i0, i1, structure in zip(cum_sizes[:-1], cum_sizes[1:], images_list):
            iH, iW = structure.images.shape[2:]
            images_tensor[i0:i1, :, :iH, :iW].copy_(structure.images)
            masks_tensor[i0:i1, :iH, :iW] = structure.masks

            padding_transform = ('pad', (0, 0, max_iW-iW, max_iH-iH), (0, 0, iW, iH))
            [transforms[i].append(padding_transform) for i in range(i0, i1)]

        # Get concatenated Images structure
        cat_images = Images(images_tensor, image_ids, masks_tensor, transforms)

        return cat_images

    def clone(self):
        """
        Clones the Images structure into a new Images structure.

        Returns:
            cloned_images (Images): Cloned Images structure.
        """

        args = (self.images.clone(), deepcopy(self.image_ids), self.masks.clone(), deepcopy(self.transforms))
        cloned_images = Images(*args)

        return cloned_images

    def crop(self, crop_region):
        """
        Crops the images w.r.t. the given crop region.

        Args:
            crop_region (Tuple): Tuple delineating the cropped region in (left, top, right, bottom) format.

        Returns:
            self (Images): Updated Images structure with cropped images.
        """

        # Get original image size
        img_width, img_height = self.size()

        # Crop the images masks w.r.t. the given crop region
        left, top, right, bottom = crop_region
        width, height = (right-left, bottom-top)

        self.images = F.crop(self.images, top, left, height, width)
        self.masks = F.crop(self.masks, top, left, height, width)

        # Get crop deltas
        crop_deltas = (left, top, img_width-right, img_height-bottom)

        # Add crop operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(('crop', crop_region, crop_deltas))

        return self

    def hflip(self):
        """
        Flips the images horizontally.

        Returns:
            self (Images): Updated Images structure with horizontally flipped images.
        """

        # Horizontally flip the images and masks
        self.images = F.hflip(self.images)
        self.masks = F.hflip(self.masks)

        # Add horizontal flip operation to list of transforms
        img_width, _ = self.size()
        for img_transforms in self.transforms:
            img_transforms.append(('hflip', img_width))

        return self

    def hflipped(self):
        """
        Checks whether the images within the Images structure were flipped horizontally.

        Returns:
            hflipped (List): List of size [num_images] with booleans indicating horizontally flipped images.
        """

        # Get booleans indicating horizontally flipped images
        hflipped = []

        for transforms in self.transforms:
            hflipped_i = False

            for transform in transforms:
                if transform[0] == 'hflip':
                    hflipped_i = not hflipped_i

            hflipped.append(hflipped_i)

        return hflipped

    def normalize(self, mean, std, inplace=False):
        """
        Method normalizing the channels of the images tensor at non-padded positions.

        Args:
            mean (FloatTensor): Tensor of shape [num_channels] containing the normalization means.
            std (FloatTensor): Tensor of shape [num_channels] containing the normalization standard deviations.
            inplace (bool): Boolean indicating whether to perform operation in place or not (default=False).

        Returns:
            images (FloatTensor): Images tensor normalized at non-padded positions.
        """

        images = self.images.clone() if not inplace else self.images
        images_view = images.permute(0, 2, 3, 1)
        images_view[self.masks, :] = images_view[self.masks, :].sub_(mean).div_(std)

        return images

    def pad(self, padding):
        """
        Pads the images according to the given padding vector.

        Args:
            padding (Tuple): Padding vector of size [4] with padding values in (left, top, right, bottom) format.

        Returns:
            self (Images): Updated Images structure with padded images.
        """

        # Pad the images and masks according to the given padding vector
        self.images = F.pad(padding)
        self.masks = F.pad(padding)

        # Get image region
        left, top, right, bottom = padding
        img_width, img_height = self.size()
        img_region = (left, top, img_width-right, img_height-bottom)

        # Add padding operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(('pad', padding, img_region))

        return self

    def resize(self, size, max_size=None):
        """
        Resizes the images according to the given resize specifications.

        Args:
            size: It can be one of following two possibilities:
                1) size (int): integer containing the minimum size (width or height) to resize to;
                2) size (Tuple): tuple of size [2] containing the image size in (width, height) format to resize to.

            max_size (int): Overrules above size whenever this value is exceded in width or height (defaults to None).

        Returns:
            self (Images): Updated Images structure with resized images.
            size (Tuple): Tuple of size [2] containing the image size after resizing in (width, height) format.
            resize_ratio (Tuple): Tuple of size [2] containing the resize ratio as (width_ratio, height_ratio).
        """

        # Get original image size
        image_size = self.size()

        # Get new image size in (width, height) format
        if isinstance(size, int):
            size = tuple(size*x/min(image_size) for x in image_size)

        # Overrule with max size if given and necessary
        if max_size is not None:
            if max_size < max(size):
                size = tuple(max_size*x/max(size) for x in size)

        # Resize the images and masks
        size = tuple(int(x) for x in size)
        self.images = F.resize(self.images, (size[1], size[0]))
        self.masks = F.resize(self.masks, (size[1], size[0]))

        # Get the resize ratio
        resize_ratio = tuple(x/y for x, y in zip(size, image_size))

        # Add resize operation to list of transforms
        for img_transforms in self.transforms:
            img_transforms.append(('resize', resize_ratio))

        return self, size, resize_ratio

    def resize_ratios(self):
        """
        Gets the resize ratio of images within the Images structure after their last crop.

        Returns:
            resize_ratios (List): List of size [num_images] containing the (width, height) resize ratios.
        """

        # Get the desired resize ratios
        resize_ratios = []

        for transforms in self.transforms:
            resize_ratio = (1.0, 1.0)

            for transform in reversed(transforms):
                if transform[0] == 'crop':
                    break

                elif transform[0] == 'resize':
                    new_width_ratio = resize_ratio[0] * transform[1][0]
                    new_height_ratio = resize_ratio[1] * transform[1][1]
                    resize_ratio = (new_width_ratio, new_height_ratio)

            resize_ratios.append(resize_ratio)

        return resize_ratios

    def size(self, mode='with_padding'):
        """
        Gets the image size of images within the Images structure.

        The mode determines the type of requested image size, chosen from:
            1) 'with_padding': the final image size with the padding used for batching;
            2) 'without_padding': the final image size but without the padding used for batching;
            3) 'original': the original image size before applying any transformation.

        Args:
            mode (str): String containing type of requested image size (default='with_padding').

        Returns:
            If 'mode' is 'with_padding':
                img_size (Tuple): Tuple of size [2] containing the (width, height) image size with padding.

            If 'mode' is 'without_padding':
                img_sizes (List): List of size [num_images] containing the (width, height) image sizes without padding.

            If 'mode' is 'original':
                img_sizes (List): List of size [num_images] containing the original (width, height) image sizes.

        Raises:
            ValueError: Error when invalid mode is provided.
        """

        # Check mode
        if mode not in ('with_padding', 'without_padding', 'original'):
            error_msg = f"Invalid mode '{mode}' to compute image sizes within Images structure."
            raise ValueError(error_msg)

        # Get image size with padding
        if isinstance(self.images, PIL.Image.Image):
            img_size = self.images.size

        elif isinstance(self.images, torch.Tensor):
            iH, iW = self.images.shape[2:]
            img_size = (iW, iH)

        # Return image size with padding if requested
        if mode == 'with_padding':
            return img_size

        # Get image sizes without padding or original images sizes
        img_sizes = [list(img_size) for _ in range(len(self))]

        for i, transforms in enumerate(self.transforms):
            for transform in reversed(transforms):
                if transform[0] == 'pad':
                    padding = transform[1]
                    img_sizes[i][0] = img_sizes[i][0] - padding[0] - padding[2]
                    img_sizes[i][1] = img_sizes[i][1] - padding[1] - padding[3]

                elif mode == 'without_padding':
                    break

                elif transform[0] == 'crop':
                    crop_deltas = transform[2]
                    img_sizes[i][0] = img_sizes[i][0] + crop_deltas[0] + crop_deltas[2]
                    img_sizes[i][1] = img_sizes[i][1] + crop_deltas[1] + crop_deltas[3]

                elif transform[0] == 'resize':
                    resize_ratio = transform[1]
                    img_sizes[i][0] = int(img_sizes[i][0] / resize_ratio[0])
                    img_sizes[i][1] = int(img_sizes[i][1] / resize_ratio[1])

        img_sizes = [tuple(img_size) for img_size in img_sizes]

        return img_sizes

    def to(self, *args, **kwargs):
        """
        Performs type and/or device conversion for the tensors within the Images structure.

        Returns:
            self (Images): Updated Images structure with converted tensors.
        """

        self.images = self.images.to(*args, **kwargs)
        self.masks = self.masks.to(*args, **kwargs)

        return self

    def to_tensor(self):
        """
        Converts PIL Image to tensor with batch size 1.

        Returns:
            self (Images): Updated Images structure with PIL Image converted to tensor.
        """

        if isinstance(self.images, PIL.Image.Image):
            self.images = F.to_tensor(self.images).unsqueeze(0)

        return self
