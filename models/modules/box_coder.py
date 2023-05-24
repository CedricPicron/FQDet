"""
Collection of box coders.
"""
from abc import ABCMeta, abstractmethod

from mmdet.models.layers.transformer.utils import inverse_sigmoid
import torch
from torch import nn

from models.build import MODELS


class AbstractBoxCoder(nn.Module, metaclass=ABCMeta):
    """
    Class implementing the AbstractBoxCoder module.
    """

    def __init__(self):
        """
        Initializes the AbstractBoxCoder module.
        """

        # Initialization of default nn.Module
        super().__init__()

    @abstractmethod
    def get_box_deltas(self, in_boxes, tgt_boxes):
        """
        Method of the box coder getting the box deltas.

        The method computes the deltas required to transform the input boxes to the target boxes.

        Args:
            in_boxes (Any): Input boxes of size [num_boxes].
            tgt_boxes (Any): Target boxes of size [num_boxes].

        Returns:
            box_deltas (FloatTensor): Tensor containing the box deltas of shape [num_boxes, num_box_dims].
        """

    @abstractmethod
    def apply_box_deltas(self, box_deltas, in_boxes):
        """
        Method of the box coder applying the box deltas.

        The method applies the given box deltas to the given input boxes yielding the new output boxes.

        Args:
            box_deltas (FloatTensor): Tensor containing the box deltas of shape [num_boxes, num_box_dims].
            in_boxes (Any): Input boxes of size [num_boxes].

        Returns:
            out_boxes (Any): Output boxes of size [num_boxes].
        """

    def forward(self, mode, *args, **kwargs):
        """
        Forward method of the AbstractBoxCoder module.

        Args:
            mode (str): String containing the box coder forward mode chosen from ['get', 'apply'].
            args (Tuple): Tuple of positional arguments passed to the chosen underlying forward method.
            kwargs (Dict): Dictionary of keyword arguments passed to the chosen underlying forward method.

        Returns:
            output (Any): Output from the chosen underlying forward method.

         Raises:
            ValueError: Error when an invalid box coder forward mode is provided.
        """

        # Choose underlying forward method
        if mode == 'get':
            output = self.get_box_deltas(*args, **kwargs)

        elif mode == 'apply':
            output = self.apply_box_deltas(*args, **kwargs)

        else:
            error_msg = f"Invalid box coder forward mode (got '{mode}')."
            raise ValueError(error_msg)

        return output


@MODELS.register_module()
class RcnnBoxCoder(AbstractBoxCoder):
    """
    Class implementing the RcnnBoxCoder module.

    Attributes:
        delta_means (FloatTensor): Buffer of shape [1, 4] with 'means' offsetting the box deltas (or None).
        delta_stds (FloatTensor): Buffer of shape [1, 4] with 'standard deviations' scaling the box deltas (or None).
        scale_clamp (float): Maximum relative increase in width and height when applying box deltas.
    """

    def __init__(self, delta_means=None, delta_stds=None, scale_clamp=62.5):
        """
        Initializes the RcnnBoxCoder module.

        Args:
            delta_means (Tuple): Tuple of size [4] with 'means' offsetting the box deltas (default=None).
            delta_stds (Tuple): Tuple of size [4] with 'standard deviations' scaling the box deltas (default=None).
            scale_clamp (float): Maximum relative increase in width and height when applying box deltas (default=62.5).

        Raises:
            ValueError: Error when the provided 'delta_stds' contains non-positive values.
        """

        # Check whether 'delta_stds' values are positive
        if delta_stds is not None:
            for delta_std in delta_stds:
                if delta_std <= 0:
                    error_msg = f"The provided 'delta_stds' should only contain positive values, but got {delta_stds}."
                    raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Register 'delta_means' and 'delta_stds' buffers
        if delta_means is not None:
            delta_means = torch.tensor([delta_means], dtype=torch.float)

        if delta_stds is not None:
            delta_stds = torch.tensor([delta_stds], dtype=torch.float)

        self.register_buffer('delta_means', delta_means, persistent=False)
        self.register_buffer('delta_stds', delta_stds, persistent=False)

        # Set scale_clamp attribute
        self.scale_clamp = scale_clamp

    def get_box_deltas(self, in_boxes, tgt_boxes, **kwargs):
        """
        Method of the R-CNN box coder getting the box deltas.

        Args:
            in_boxes (Boxes): Boxes structure of axis-aligned input boxes of size [num_boxes].
            tgt_boxes (Boxes): Boxes structure of axis-aligned target boxes of size [num_boxes].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            box_deltas (FloatTensor): Tensor containing the box deltas of shape [num_boxes, 4].
        """

        # Check whether box lengths are consistent
        check = len(in_boxes) == len(tgt_boxes)
        assert_msg = f"Inconsistent input and target box lengths (got {len(in_boxes)} and {len(tgt_boxes)})."
        assert check, assert_msg

        # Check whether normalized attributes are consistent
        check = in_boxes.normalized == tgt_boxes.normalized
        assert_msg = f"Inconsistent box normalizations (got {in_boxes.normalized} and {tgt_boxes.normalized}))."
        assert check, assert_msg

        # Check for degenerate boxes (i.e. boxes with non-positive width or height)
        assert in_boxes.well_defined().all().item(), "The 'in_boxes' input contains degenerate boxes."
        assert tgt_boxes.well_defined().all().item(), "The 'tgt_boxes' input contains degenerate boxes."

        # Convert boxes to (center_x, center_y, width, height) format and get box tensors
        in_boxes = in_boxes.clone().to_format('cxcywh').boxes
        tgt_boxes = tgt_boxes.clone().to_format('cxcywh').boxes

        # Get box deltas
        box_deltas = torch.zeros_like(in_boxes)
        box_deltas[:, :2] = (tgt_boxes[:, :2] - in_boxes[:, :2]) / in_boxes[:, 2:]
        box_deltas[:, 2:] = torch.log(tgt_boxes[:, 2:] / in_boxes[:, 2:])

        if self.delta_means is not None:
            box_deltas.sub_(self.delta_means)

        if self.delta_stds is not None:
            box_deltas.div_(self.delta_stds)

        return box_deltas

    def apply_box_deltas(self, box_deltas, in_boxes, **kwargs):
        """
        Method of the R-CNN box coder applying the box deltas.

        Args:
            box_deltas (FloatTensor): Tensor containing the box deltas of shape [num_boxes, 4].
            in_boxes (Boxes): Boxes structure of axis-aligned input boxes of size [num_boxes].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_boxes (Boxes): Boxes structure of axis-aligned output boxes of size [num_boxes].
        """

        # Check whether lengths are consistent
        check = len(box_deltas) == len(in_boxes)
        assert_msg = f"Inconsistent 'box_deltas' and 'in_boxes' lengths (got {len(box_deltas)} and {len(in_boxes)})."
        assert check, assert_msg

        # Check for degenerate boxes (i.e. boxes with non-positive width or height)
        assert in_boxes.well_defined().all().item(), "The 'in_boxes' input contains degenerate boxes."

        # Apply box deltas to get output boxes
        if self.delta_stds is not None:
            box_deltas = self.delta_stds * box_deltas

        if self.delta_means is not None:
            box_deltas = self.delta_means + box_deltas

        in_boxes = in_boxes.clone().to_format('cxcywh')
        out_boxes = in_boxes.clone()

        out_boxes.boxes[:, :2] += box_deltas[:, :2] * in_boxes.boxes[:, 2:]
        out_boxes.boxes[:, 2:] *= torch.exp(box_deltas[:, 2:].clamp(max=self.scale_clamp))

        return out_boxes


@MODELS.register_module()
class SigmoidBoxCoder(AbstractBoxCoder):
    """
    Class implementing the SigmoidBoxCoder module.
    """

    def get_box_deltas(self, in_boxes, tgt_boxes, images=None, **kwargs):
        """
        Method of the sigmoid box coder getting the box deltas.

        Args:
            in_boxes (Boxes): Boxes structure of axis-aligned input boxes of size [num_boxes].
            tgt_boxes (Boxes): Boxes structure of axis-aligned target boxes of size [num_boxes].
            images (Images): Images structure containing the batched images of size [batch_size] (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            box_deltas (FloatTensor): Tensor containing the box deltas of shape [num_boxes, 4].
        """

        # Check whether box lengths are consistent
        check = len(in_boxes) == len(tgt_boxes)
        assert_msg = f"Inconsistent input and target box lengths (got {len(in_boxes)} and {len(tgt_boxes)})."
        assert check, assert_msg

        # Check for degenerate boxes (i.e. boxes with non-positive width or height)
        assert in_boxes.well_defined().all().item(), "The 'in_boxes' input contains degenerate boxes."
        assert tgt_boxes.well_defined().all().item(), "The 'tgt_boxes' input contains degenerate boxes."

        # Get normalized box tensors in desired format
        in_boxes = in_boxes.clone().normalize(images).to_format('cxcywh').boxes
        tgt_boxes = tgt_boxes.clone().normalize(images).to_format('cxcywh').boxes

        # Get box deltas
        in_boxes = inverse_sigmoid(in_boxes, eps=1e-3)
        tgt_boxes = inverse_sigmoid(tgt_boxes, eps=1e-3)
        box_deltas = tgt_boxes - in_boxes

        return box_deltas

    def apply_box_deltas(self, box_deltas, in_boxes, images=None, **kwargs):
        """
        Method of the sigmoid box coder applying the box deltas.

        Args:
            box_deltas (FloatTensor): Tensor containing the box deltas of shape [num_boxes, 4].
            in_boxes (Boxes): Boxes structure of axis-aligned input boxes of size [num_boxes].
            images (Images): Images structure containing the batched images of size [batch_size] (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_boxes (Boxes): Boxes structure of axis-aligned output boxes of size [num_boxes].
        """

        # Check whether lengths are consistent
        check = len(box_deltas) == len(in_boxes)
        assert_msg = f"Inconsistent 'box_deltas' and 'in_boxes' lengths (got {len(box_deltas)} and {len(in_boxes)})."
        assert check, assert_msg

        # Check for degenerate boxes (i.e. boxes with non-positive width or height)
        assert in_boxes.well_defined().all().item(), "The 'in_boxes' input contains degenerate boxes."

        # Apply box deltas to get output boxes
        in_boxes = in_boxes.clone().normalize(images).to_format('cxcywh')
        out_boxes = in_boxes.clone()

        out_boxes.boxes = box_deltas + inverse_sigmoid(in_boxes.boxes, eps=1e-3)
        out_boxes.boxes = out_boxes.boxes.sigmoid()

        return out_boxes
