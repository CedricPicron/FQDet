"""
Boxes structure and bounding box utilities.
"""
import math

import torch

from utils.distributed import get_world_size, is_dist_avail_and_initialized


class Boxes(object):
    """
    Class implementing the Boxes structure.

    Attributes:
        boxes (FloatTensor): Tensor of axis-aligned bounding boxes of shape [num_boxes, 4].
        format (str): String containing the format in which the bounding boxes are expressed.
        normalized (str): String indicating whether and w.r.t. what boxes are normalized.
        batch_ids (LongTensor): Tensor containing the batch indices of shape [num_boxes].
    """

    def __init__(self, boxes, format, normalized='false', batch_ids=None):
        """
        Initializes the Boxes structure.

        Args:
            boxes (FloatTensor): Tensor of axis-aligned bounding boxes of shape [num_boxes, 4].
            format (str): String containing the format in which the bounding boxes are expressed.
            normalized (str): String indicating whether and w.r.t. what boxes are normalized (default='false').
            batch_ids (LongTensor): Tensor containing the batch indices of shape [num_boxes] (default=None).
        """

        # Check whether input boxes tensor has valid shape
        check = boxes.dim() == 2 and boxes.shape[-1] == 4
        assert_msg = f"Tensor has incorrect shape {boxes.shape} to become Boxes structure."
        assert check, assert_msg

        # Set boxes, format and normalized attributes
        self.boxes = boxes
        self.format = format
        self.normalized = normalized

        # Set batch_ids attribute
        if batch_ids is None:
            self.batch_ids = torch.zeros(len(boxes), dtype=torch.int64, device=boxes.device)
        else:
            self.batch_ids = batch_ids.to(device=boxes.device)

    def __getitem__(self, key):
        """
        Implements the __getitem__ method of the Boxes structure.

        Args:
            key (Any): Object of any type also supported by Tensor determining the selected bounding boxes.

        Returns:
            item (Boxes): New Boxes structure containing the selected bounding boxes.
        """

        boxes_tensor = self.boxes[None, key] if isinstance(key, int) else self.boxes[key]
        batch_ids = self.batch_ids[None, key] if isinstance(key, int) else self.batch_ids[key]
        item = Boxes(boxes_tensor, self.format, self.normalized, batch_ids=batch_ids)

        return item

    def __len__(self):
        """
        Implements the __len__ method of the Boxes structure.

        Returns:
            num_boxes (int): Number of boxes within the Boxes structure.
        """

        num_boxes = len(self.boxes)

        return num_boxes

    def __repr__(self):
        """
        Implements the __repr__ method of the Boxes structure.

        Returns:
            boxes_string (str): String containing information about the Boxes structure.
        """

        boxes_string = "Boxes structure:\n"
        boxes_string += f"   Size: {len(self)}\n"
        boxes_string += f"   Format: {self.format}\n"
        boxes_string += f"   Normalized: {self.normalized}\n"
        boxes_string += f"   Batch indices: {self.batch_ids}\n"
        boxes_string += f"   Content: {self.boxes}"

        return boxes_string

    def __setitem__(self, key, item):
        """
        Implements the __setitem__ method of the Boxes structure.

        Args:
            key (Any): Object of any type also supported by Tensor determining the bounding boxes to be set.
            item (Boxes): Boxes structure containing the bounding boxes to be set.
        """

        # Check whether formats are consistent
        check = self.format == item.format
        assert_msg = f"Inconsistent formats between self '{self.format}' and item '{item.format}'."
        assert check, assert_msg

        # Check whether normalized attributes are consistent
        check = self.normalized == item.normalized
        assert_msg = f"Inconsistent normalizations between self '{self.normalized}' and item '{item.normalized}'."
        assert check, assert_msg

        # Set boxes and batch_ids attributes
        self.boxes[key] = item.boxes
        self.batch_ids[key] = item.batch_ids

    def area(self):
        """
        Computes the area of each of the bounding boxes within the Boxes structure.

        Returns:
            areas (FloatTensor): Tensor of shape [num_boxes] containing the area of each of the boxes.
        """

        if self.format in ['cxcywh', 'xywh']:
            areas = self.boxes[:, 2] * self.boxes[:, 3]

        elif self.format == 'xyxy':
            areas = (self.boxes[:, 2] - self.boxes[:, 0]) * (self.boxes[:, 3] - self.boxes[:, 1])

        return areas

    @staticmethod
    def cat(boxes_list, offset_batch_ids=False):
        """
        Concatenate list of Boxes structures into single Boxes structure.

        Args:
            boxes_list (List): List of size [num_structures] with Boxes structures to be concatenated.
            offset_batch_ids (bool): Boolean indicating whether to offset batch indices (default=False).

        Returns:
            cat_boxes (Boxes): Boxes structure containing the concatenated input Boxes structures.
        """

        # Check whehter all Boxes structures have the same format
        format_set = {boxes.format for boxes in boxes_list}
        assert_msg = f"All Boxes structures must have the same format (got {format_set})."
        assert len(format_set) == 1, assert_msg

        # Check whehter all Boxes structures have the same normalized attribute
        normalized_set = {boxes.normalized for boxes in boxes_list}
        assert_msg = f"All Boxes structures must have the same normalized attribute (got {normalized_set})."
        assert len(normalized_set) == 1, assert_msg

        # Check whether all Boxes structures reside on the same device
        device_set = {s.boxes.device for s in boxes_list}
        assert len(device_set) == 1, f"All boxes should reside on the same device (got {device_set})."

        # Concatenate Boxes structures into single Boxes structure
        boxes_tensor = torch.cat([structure.boxes for structure in boxes_list])

        batch_ids_list = []
        batch_id_offset = 0

        for structure in boxes_list:
            batch_ids_list.append(structure.batch_ids + batch_id_offset)

            if offset_batch_ids:
                assert (structure.batch_ids == 0).all().item()
                batch_id_offset += 1

        batch_ids = torch.cat(batch_ids_list, dim=0)
        cat_boxes = Boxes(boxes_tensor, format_set.pop(), normalized_set.pop(), batch_ids=batch_ids)

        return cat_boxes

    def clip(self, clip_region, eps=0.0):
        """
        Clips bounding boxes to be within the given clip region.

        Args:
            clip_region (Tuple): Following two input formats are supported:
                1) tuple of size [2] containing the (right, bottom) clip boundaries;
                2) tuple of size [4] containing the (left, top, right, bottom) clip boundaries.

            eps (float): Value altering some boundaries to avoid ill-defined boxes after clipping (default=0.0).

        Returns:
            self (Boxes): Updated Boxes structure with clipped bounding boxes.
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after clipping.
        """

        # Set left and top boundaries to zero if not specified
        if len(clip_region) == 2:
            clip_region = (0, 0, *clip_region)

        # Convert boxes if needed
        in_format = self.format
        if in_format != 'xyxy':
            self.to_format('xyxy')

        # Clip boxes according to its boundaries
        self.boxes[:, 0].clamp_(min=clip_region[0], max=clip_region[2]-eps)
        self.boxes[:, 1].clamp_(min=clip_region[1], max=clip_region[3]-eps)
        self.boxes[:, 2].clamp_(min=clip_region[0]+eps, max=clip_region[2])
        self.boxes[:, 3].clamp_(min=clip_region[1]+eps, max=clip_region[3])

        # Convert boxes back if needed
        if in_format != 'xyxy':
            self.to_format(in_format)

        # Find out which boxes are well-defined after clipping
        well_defined = self.well_defined()

        return self, well_defined

    def clone(self):
        """
        Clones the Boxes structure into a new Boxes structure.

        Returns:
            cloned_boxes (Boxes): Cloned Boxes structure.
        """

        cloned_boxes = Boxes(self.boxes.clone(), self.format, self.normalized, self.batch_ids.clone())

        return cloned_boxes

    def crop(self, crop_region):
        """
        Updates the bounding boxes w.r.t. the given crop operation.

        Args:
            crop_region (Tuple): Tuple delineating the cropped region in (left, top, right, bottom) format.

        Returns:
            self (Boxes): Boxes structure updated w.r.t. the crop operation.
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after cropping.
        """

        # Update bounding boxes w.r.t. the crop operation
        left, top, right, bottom = crop_region

        if self.format in ['cxcywh', 'xywh']:
            self.boxes[:, :2] = self.boxes[:, :2] - torch.tensor([left, top]).to(self.boxes)

        elif self.format == 'xyxy':
            self.boxes = self.boxes - torch.tensor([left, top, left, top]).to(self.boxes)

        # Clip boxes and find out which ones are well-defined
        self, well_defined = self.clip((right-left, bottom-top))

        return self, well_defined

    def detach(self):
        """
        Detaches the boxes tensor of the Boxes structure from the current computation graph.

        Returns:
            self (Boxes): Updated Boxes structure with the boxes tensor detached from the current computation graph.
        """

        self.boxes = self.boxes.detach()

        return self

    def dist_len(self):
        """
        Gets the average number of boxes across Boxes structures from different processes.

        Returns:
            num_boxes (float): Average number of boxes across Boxes structures from different processes.
        """

        # Get number of boxes for structure from this process
        num_boxes = len(self)

        # Return if not in distributed mode
        if not is_dist_avail_and_initialized():
            return num_boxes

        # Get average number of boxes across structures from different processes
        num_boxes = torch.tensor([num_boxes], dtype=torch.float, device=self.boxes.device)
        torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes/get_world_size(), min=1).item()

        return num_boxes

    def expand(self, expand_size):
        """
        Expands the Boxes structure to contain multiple consecutive views of each bounding box.

        Args:
            expand_size (int): Integer containing the expanded size (i.e. the number of views per bounding box).

        Returns:
            self (Boxes): Updated Boxes structure containing multiple consecutive views of each bounding box.
        """

        # Expand Boxes structure
        self.boxes = self.boxes[:, None, :].expand(-1, expand_size, -1).flatten(0, 1)
        self.batch_ids = self.batch_ids[:, None].expand(-1, expand_size).flatten()

        return self

    def fmt_cxcywh_to_xywh(self):
        """
        Function transforming boxes from (center_x, center_y, width, height) to (left, top, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, width, height) format.
        """

        cx, cy, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([cx-0.5*w, cy-0.5*h, w, h], dim=-1)
        self.format = 'xywh'

        return self

    def fmt_cxcywh_to_xyxy(self):
        """
        Function transforming boxes from (center_x, center_y, width, height) to (left, top, right, bottom) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, right, bottom) format.
        """

        cx, cy, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], dim=-1)
        self.format = 'xyxy'

        return self

    def fmt_xywh_to_cxcywh(self):
        """
        Function transforming boxes from (left, top, width, height) to (center_x, center_y, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (center_x, center_y, width, height) format.
        """

        x0, y0, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0+w/2, y0+h/2, w, h], dim=-1)
        self.format = 'cxcywh'

        return self

    def fmt_xywh_to_xyxy(self):
        """
        Function transforming boxes from (left, top, width, height) to (left, top, right, bottom) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, right, bottom) format.
        """

        x0, y0, w, h = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0, y0, x0+w, y0+h], dim=-1)
        self.format = 'xyxy'

        return self

    def fmt_xyxy_to_cxcywh(self):
        """
        Function transforming boxes from (left, top, right, bottom) to (center_x, center_y, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (center_x, center_y, width, height) format.
        """

        x0, y0, x1, y1 = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0], dim=-1)
        self.format = 'cxcywh'

        return self

    def fmt_xyxy_to_xywh(self):
        """
        Function transforming boxes from (left, top, right, bottom) to (left, top, width, height) format.

        Returns:
            self (Boxes): Updated Boxes structure in the (left, top, width, height) format.
        """

        x0, y0, x1, y1 = self.boxes.unbind(dim=-1)
        self.boxes = torch.stack([x0, y0, x1-x0, y1-y0], dim=-1)
        self.format = 'xywh'

        return self

    def hflip(self, image_width):
        """
        Flips the bounding boxes horizontally.

        Args:
            image_width (int): Integer containing the image width.

        Returns:
            self (Boxes): Updated Boxes structure with horizontally flipped bounding boxes.
        """

        if self.format == 'cxcywh':
            self.boxes[:, 0] = image_width - self.boxes[:, 0]

        elif self.format == 'xywh':
            self.boxes[:, 0] = image_width - (self.boxes[:, 0] + self.boxes[:, 2])

        elif self.format == 'xyxy':
            self.boxes[:, [0, 2]] = image_width - self.boxes[:, [2, 0]]

        return self

    def normalize(self, images, with_padding=True):
        """
        Normalizes bounding boxes w.r.t. the image sizes from the given Images structure.

        It is the inverse operation of 'to_img_scale'.

        Args:
            images (Images): Images structure containing batched images with their entire transform history.
            with_padding (bool): Whether to normalize w.r.t the image sizes with padding or not (default=True).

        Returns:
            self (Boxes): Updated Boxes structure with normalized bounding boxes.
        """

        # Normalize bounding box coordinates if not yet normalized
        if self.normalized == 'false':

            # Get image sizes in (width, height) format
            mode = 'with_padding' if with_padding else 'without_padding'
            img_sizes = images.size(mode=mode)
            img_sizes = [img_sizes for _ in range(len(images))] if with_padding else img_sizes

            # Normalize bounding box coordinates w.r.t. the image sizes
            scales = torch.tensor([[*img_size, *img_size] for img_size in img_sizes]).to(self.boxes)
            self.boxes = self.boxes / scales[self.batch_ids]
            self.normalized = 'img_with_padding' if with_padding else 'img_without_padding'

        return self

    def pad(self, padding):
        """
        Updates the bounding boxes w.r.t. the given padding operation.

        Args:
            padding (Tuple): Padding vector of size [4] with padding values in (left, top, right, bottom) format.

        Returns:
            self (Boxes): Boxes structure updated w.r.t. the padding operation.
        """

        # Update bounding boxes w.r.t. the padding operation
        left, top = padding[:2]

        if self.format in ['cxcywh', 'xywh']:
            self.boxes[:, :2] = self.boxes[:, :2] + torch.tensor([left, top]).to(self.boxes)

        elif self.format == 'xyxy':
            self.boxes = self.boxes + torch.tensor([left, top, left, top]).to(self.boxes)

        return self

    def resize(self, resize_ratio):
        """
        Resizes the bounding boxes by the given resize ratio.

        Args:
            resize_ratio (Tuple): Tuple of size [2] containing the resize ratio as (width_ratio, height_ratio).

        Returns:
            self (Boxes): Boxes structure updated w.r.t. the resize operation.
        """

        resize_ratio = torch.tensor([*resize_ratio, *resize_ratio]).to(self.boxes)
        self.boxes = resize_ratio * self.boxes

        return self

    def split(self, split_size_or_sections):
        """
        Splits the bounding boxes into chunks of bounding boxes.

        Args:
            split_size_or_sections (int or List):
                1) integer containing the (maximum) size of each chunk;
                2) list of size [num_chunks] with integers containing the sizes of each chunk.

        Returns:
            boxes_list (List): List with chunks of bounding boxes of size [num_chunks].
        """

        boxes_list = self.boxes.split(split_size_or_sections, dim=0)
        batch_ids_list = self.batch_ids.split(split_size_or_sections, dim=0)

        zip_obj = zip(boxes_list, batch_ids_list)
        boxes_list = [Boxes(boxes, self.format, self.normalized, batch_ids=batch_ids) for boxes, batch_ids in zip_obj]

        return boxes_list

    def to(self, *args, **kwargs):
        """
        Performs type and/or device conversion for the tensors within the Boxes structure.

        Returns:
            self (Boxes): Updated Boxes structure with converted tensors.
        """

        self.boxes = self.boxes.to(*args, **kwargs)
        self.batch_ids = self.batch_ids.to(*args, **kwargs)

        return self

    def to_format(self, format):
        """
        Changes the bounding boxes of the Boxes structure to the given format.

        Args:
            format (str): String containing the bounding box format to convert to.

        Returns:
            self (Boxes): Boxes structure with bounding boxes in the specified format.

        Raises:
            ValueError: Raised when given format results in unknown format conversion.
        """

        if self.format != format:
            method_name = f'fmt_{self.format}_to_{format}'

            if hasattr(self, method_name):
                self = getattr(self, method_name)()
            else:
                raise ValueError(f"Unknown format conversion {method_name}.")

        return self

    def to_img_scale(self, images):
        """
        Scales normalized bounding boxes w.r.t. the image sizes from the given Images structure.

        It is the inverse operation of 'normalize'.

        Args:
            images (Images): Images structure containing batched images with their entire transform history.

        Returns:
            self (Boxes): Updated Boxes structure with bounding boxes resized to image scale.
        """

        # Scale if boxes are normalized w.r.t. image sizes
        if self.normalized in ['img_with_padding', 'img_without_padding']:

            # Get image sizes in (width, height) format
            img_sizes = images.size(mode=self.normalized[4:])
            with_padding = self.normalized == 'img_with_padding'
            img_sizes = [img_sizes for _ in range(len(images))] if with_padding else img_sizes

            # Scale bounding box coordinates w.r.t. the image sizes
            scales = torch.tensor([[*img_size, *img_size] for img_size in img_sizes]).to(self.boxes)
            self.boxes = self.boxes * scales[self.batch_ids]
            self.normalized = 'false'

        return self

    def transform(self, images, inverse=False):
        """
        Applies transforms (recorded by the given Images structure) to each of the bounding boxes.

        Args:
            images (Images): Images structure containing batched images with their entire transform history.
            inverse (bool): Boolean indicating whether inverse transformation should be applied or not.

        Returns:
            self (Boxes): Boxes structure updated by the transforms from the Images structure.
            well_defined (BoolTensor): Tensor of shape [num_boxes] indicating well-defined boxes after transformation.
        """

        # Scale if boxes are normalized
        self = self.to_img_scale(images)

        # Apply transforms
        for i, transforms in enumerate(images.transforms):
            mask = self.batch_ids == i
            boxes = self[mask]

            if inverse:
                range_obj = range(len(transforms)-1, -1, -1)
            else:
                range_obj = range(len(transforms))

            for j in range_obj:
                transform = transforms[j]

                if transform[0] == 'crop':
                    if not inverse:
                        boxes, _ = boxes.crop(transform[1])
                    else:
                        boxes = boxes.pad(transform[2])

                elif transform[0] == 'hflip':
                    boxes = boxes.hflip(transform[1])

                elif transform[0] == 'pad':
                    if not inverse:
                        boxes = boxes.pad(transform[1])
                    else:
                        boxes, _ = boxes.crop(transform[2])

                elif transform[0] == 'resize':
                    resize_ratio = tuple(1/x for x in transform[1]) if inverse else transform[1]
                    boxes = boxes.resize(resize_ratio)

            self[mask] = boxes

        # Get well-defined boxes after transformation
        well_defined = self.well_defined()

        return self, well_defined

    def well_defined(self):
        """
        Finds well-defined boxes, i.e. boxes with (stricly) positive width and height.

        Returns:
            well_defined (BoolTensor): Boolean tensor of shape [num_boxes] indicating which boxes are well-defined.
        """

        if self.format in ['cxcywh', 'xywh']:
            well_defined = (self.boxes[:, 2] > 0) & (self.boxes[:, 3] > 0)

        elif self.format == 'xyxy':
            well_defined = (self.boxes[:, 2] > self.boxes[:, 0]) & (self.boxes[:, 3] > self.boxes[:, 1])

        return well_defined


def box_giou(boxes1, boxes2, images=None):
    """
    Function computing the 2D GIoU's between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].
        images (Images): Images structure containing the batched images of size [batch_size] (default=None).

    Returns:
        gious (FloatTensor): The 2D GIoU's between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all().item(), "The boxes1 input contains degenerate boxes."
    assert boxes2.well_defined().all().item(), "The boxes2 input contains degenerate boxes."

    # Make sure normalized attributes are consistent
    if boxes1.normalized != boxes2.normalized:
        assert images is not None, "Inconsistent normalizations between Boxes inputs and no images input was provided."
        boxes1 = boxes1.to_img_scale(images)
        boxes2 = boxes2.to_img_scale(images)

    # Compute 2D IoU's and union areas
    ious, unions = box_iou(boxes1, boxes2, return_unions=True)

    # Convert bounding boxes to (left, top, right, bottom) format and get box tensors
    boxes1 = boxes1.clone().to_format('xyxy').boxes
    boxes2 = boxes2.clone().to_format('xyxy').boxes

    # Compute areas of smallest axis-aligned box containing the union
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    areas = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute GIoU's based on 2D IoU's, union area's and axis-aligned union areas
    gious = ious - (areas - unions) / areas

    return gious


def box_intersection(boxes1, boxes2, images=None, shard_size=int(1e7)):
    """
    Function computing the intersection areas between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].
        images (Images): Images structure containing the batched images of size [batch_size] (default=None).
        shard_size (int): Integer containing the maximum number of elements per shard (default=1e7).

    Returns:
        inters (FloatTensor): The intersection areas between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all().item(), "The boxes1 input contains degenerate boxes."
    assert boxes2.well_defined().all().item(), "The boxes2 input contains degenerate boxes."

    # Make sure normalized attributes are consistent
    if boxes1.normalized != boxes2.normalized:
        assert images is not None, "Inconsistent normalizations between Boxes inputs and no images input was provided."
        boxes1 = boxes1.to_img_scale(images)
        boxes2 = boxes2.to_img_scale(images)

    # Convert bounding boxes to (left, top, right, bottom) format and get box tensors
    boxes1 = boxes1.clone().to_format('xyxy').boxes
    boxes2 = boxes2.clone().to_format('xyxy').boxes

    # Get intersection areas
    N = len(boxes1)
    M = len(boxes2)

    if N*M <= shard_size:
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (rb - lt).clamp(min=0).prod(dim=2)

    else:
        tensor_kwargs = {'dtype': boxes1.dtype, 'device': boxes1.device}
        inters = torch.empty(N, M, **tensor_kwargs)

        if N >= M:
            sM = M
            sN = shard_size // sM
            num_steps = int(math.ceil(N / sN))
            shard_n = True

        else:
            sN = N
            sM = shard_size // sN
            num_steps = int(math.ceil(M / sM))
            shard_n = False

        lt = torch.empty(sN, sM, 2, **tensor_kwargs)
        rb = torch.empty(sN, sM, 2, **tensor_kwargs)

        for i in range(num_steps):
            glob_n = slice(i*sN, min((i+1)*sN, N)) if shard_n else slice(None)
            glob_m = slice(i*sM, min((i+1)*sM, M)) if not shard_n else slice(None)

            loc_n = slice(min((i+1)*sN, N) - i*sN) if shard_n else slice(None)
            loc_m = slice(min((i+1)*sM, M) - i*sM) if not shard_n else slice(None)

            lt[loc_n, loc_m] = torch.max(boxes1[glob_n, None, :2], boxes2[glob_m, :2])
            rb[loc_n, loc_m] = torch.min(boxes1[glob_n, None, 2:], boxes2[glob_m, 2:])
            inters[glob_n, glob_m] = (rb[loc_n, loc_m] - lt[loc_n, loc_m]).clamp(min=0).prod(dim=2)

    return inters


def box_iou(boxes1, boxes2, images=None, return_inters=False, return_unions=False):
    """
    Function computing the 2D IoU's and union areas between every pair of boxes from two Boxes structures.

    Args:
        boxes1 (Boxes): First Boxes structure of axis-aligned bounding boxes of size [N].
        boxes2 (Boxes): Second Boxes structure of axis-aligned bounding boxes of size [M].
        images (Images): Images structure containing the batched images of size [batch_size] (default=None).
        return_inters (bool): Whether to return areas of the box intersections or not (default=False).
        return_unions (bool): Whether to return areas of the box unions or not (default=False).

    Returns:
        ious (FloatTensor): The 2D IoU's between every pair of boxes of shape [N, M].

        If 'return_inters' is True:
            inters (FloatTensor): The areas of the box intersections between every pair of boxes of shape [N, M].

        If 'return_unions' is True:
            unions (FloatTensor): The areas of the box unions between every pair of boxes of shape [N, M].
    """

    # Check for degenerate boxes (i.e. boxes with non-positive width or height)
    assert boxes1.well_defined().all().item(), "The boxes1 input contains degenerate boxes."
    assert boxes2.well_defined().all().item(), "The boxes2 input contains degenerate boxes."

    # Make sure normalized attributes are consistent
    if boxes1.normalized != boxes2.normalized:
        assert images is not None, "Inconsistent normalizations between Boxes inputs and no images input was provided."
        boxes1 = boxes1.to_img_scale(images)
        boxes2 = boxes2.to_img_scale(images)

    # Compute bounding box areas
    areas1 = boxes1.area()
    areas2 = boxes2.area()

    # Get intersection areas
    inters = box_intersection(boxes1, boxes2)

    # Get union areas and 2D IoU's
    unions = areas1[:, None] + areas2 - inters
    ious = inters / unions

    # Return IoU's and intersection/union areas if requested
    if not return_inters and not return_unions:
        return ious
    elif return_inters and not return_unions:
        return ious, inters
    elif not return_inters and return_unions:
        return ious, unions
    else:
        return ious, inters, unions


def mask_to_box(masks, batch_ids=None):
    """
    Function converting binary masks into smallest axis-aligned bounding boxes containing the masks.

    Args:
        masks (BoolTensor): Tensor containing the binary masks of shape [num_masks, H, W].
        batch_ids (LongTensor): Tensor containing the batch indices of shape [num_masks] (default=None).

    Returns:
        boxes (Boxes): Boxes structure with smallest bounding boxes containing the masks of size [num_masks].
    """

    # Get device and shape of masks
    device = masks.device
    H, W = masks.size()[1:]

    # Get left indices
    left_ids = torch.arange(W, 0, -1, device=device)
    left_ids = left_ids[None, None, :] * masks
    left_ids = W - left_ids.flatten(1, 2).amax(dim=1)

    # Get top indices
    top_ids = torch.arange(H, 0, -1, device=device)
    top_ids = top_ids[None, :, None] * masks
    top_ids = H - top_ids.flatten(1, 2).amax(dim=1)

    # Get right indices
    right_ids = torch.arange(1, W+1, device=device)
    right_ids = right_ids[None, None, :] * masks
    right_ids = right_ids.flatten(1, 2).amax(dim=1)

    # Get bottom indices
    bot_ids = torch.arange(1, H+1, device=device)
    bot_ids = bot_ids[None, :, None] * masks
    bot_ids = bot_ids.flatten(1, 2).amax(dim=1)

    # Construct bounding boxes
    boxes = torch.stack([left_ids, top_ids, right_ids, bot_ids], dim=1)
    boxes = boxes.float() / torch.tensor([W, H, W, H], dtype=torch.float, device=device)
    boxes = Boxes(boxes, format='xyxy', normalized='img_with_padding', batch_ids=batch_ids)

    return boxes
