"""
Collection of interpolation modules.
"""

from torch import nn
import torch.nn.functional as F

from models.build import MODELS


@MODELS.register_module()
class Interpolate(nn.Module):
    """
    Class implementing the Interpolate module.

    Attributes:
        size (int or tuple): Integer or tuple containing the output spatial size.
        scale_factor (float or Tuple): Value or tuple of values scaling the spatial.
        mode (str): String containing the interpolation mode.
        align_corners (bool): Boolean indicating whether values at corners are preserved.
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        """
        Initializes the Interpolate module.

        Args:
            size (int or tuple): Integer or tuple containing the output spatial size (default=None).
            scale_factor (float or Tuple): Value or tuple of values scaling the spatial (default=None).
            mode (str): String containing the interpolation mode (default='nearest').
            align_corners (bool): Boolean indicating whether values at corners are preserved (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the Interpolate module.

        Args:
            in_tensor (FloatTensor): Input tensor to interpolate.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_tensor (FloatTensor): Output tensor obtained by interpolating the input tensor.
        """

        # Get output tensor by interpolating input tensor
        out_tensor = F.interpolate(in_tensor, self.size, self.scale_factor, self.mode, self.align_corners)

        return out_tensor
