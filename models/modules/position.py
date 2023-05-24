"""
Collection of modules related to position encodings.
"""
import math

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class PosEncoder(nn.Module):
    """
    Class implementing the PosEncoder module.

    Attributes
        net (nn.Module): Module implementing the position encoding network.
        normalize (bool): Boolean indicating whether to normalize the (x, y) position coordinates.
    """

    def __init__(self, net_cfg, normalize=False):
        """
        Initializes the PosEncoder module.

        Args:
            net_cfg (Dict): Configuration dictionary specifying the position encoding network.
            normalize (bool): Boolean indicating whether to normalize the (x, y) position coordinates (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build position encoding network
        self.net = build_model(net_cfg)

        # Set normalize attribute
        self.normalize = normalize

    def forward(self, pos_xy):
        """
        Forward method of the PosEncoder module.

        Args:
            pos_xy (FloatTensor): Tensor with position coordinates in (x, y) format of shape [num_pts, 2].

        Returns:
            pos_feats (FloatTensor): Tensor with position features of shape [num_pts, feat_size].
        """

        # Normalize position coordinates if requested
        if self.normalize:
            pos_xy = pos_xy / pos_xy.abs().amax(dim=0)

        # Get position features
        pos_feats = self.net(pos_xy)

        return pos_feats


@MODELS.register_module()
class SineBoxEncoder2d(nn.Module):
    """
    Class implementing the SineBoxEncoder2d module.

    Attributes:
        feat_size (int): Integer containing the feature size of the box features.
        scale_factor (float): Value scaling the boxes after optional normalization.
        max_period (float): Value determining the maximum sine and cosine periods.
    """

    def __init__(self, feat_size, scale_factor=2*math.pi, max_period=1e4):
        """
        Initializes the SineBoxEncoder2d module.

        Args:
            feat_size (int): Integer containing the feature size of the box features.
            scale_factor (float): Value scaling the boxes after optional normalization (default=2*math.pi).
            max_period (float): Value determining the maximum sine and cosine periods (default=1e4).

        Raises:
            ValueError: Error when the provided feature size is not divisible by 8.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check whether the feature size is divisible by 8
        if feat_size % 8 != 0:
            error_msg = f"The feature size (got {feat_size}) of the SineBoxEncoder2d module must be divisible by 8."
            raise ValueError(error_msg)

        # Set remaining attributes
        self.feat_size = feat_size
        self.scale_factor = scale_factor
        self.max_period = max_period

    def forward(self, norm_boxes):
        """
        Forward method of the SineBoxEncoder2d module.

        Args:
            norm_boxes (FloatTensor): Normalized box coordinates in 'cxcywh' format of shape [num_boxes, 4].

        Returns:
            box_feats (FloatTensor): Box features of shape [num_boxes, feat_size].
        """

        # Get scaled box coordinates
        norm_boxes = self.scale_factor * norm_boxes

        # Get periods
        device = norm_boxes.device
        periods = 8 * torch.arange(self.feat_size // 8, dtype=torch.float, device=device) / self.feat_size
        periods = self.max_period ** periods

        # Get box features
        box_feats = norm_boxes[:, :, None] / periods
        box_feats = torch.cat([box_feats.sin(), box_feats.cos()], dim=2)
        box_feats = box_feats.view(-1, self.feat_size)

        return box_feats


@MODELS.register_module()
class SinePosEncoder2d(nn.Module):
    """
    Class implementing the SinePosEncoder2d module.

    Attributes:
        feat_size (int): Integer containing the feature size of the position features.
        normalize (bool): Boolean indicating whether to normalize the input position coordinates.
        scale_factor (float): Value scaling position coordinates after optional normalization.
        max_period (float): Value determining the maximum sine and cosine periods.
    """

    def __init__(self, feat_size, normalize=False, scale_factor=2*math.pi, max_period=1e4):
        """
        Initializes the SinePosEncoder2d module.

        Args:
            feat_size (int): Integer containing the feature size of the position features.
            normalize (bool): Boolean indicating whether to normalize the input position coordinates (default=False).
            scale_factor (float): Value scaling position coordinates after optional normalization (default=2*math.pi).
            max_period (float): Value determining the maximum sine and cosine periods (default=1e4).

        Raises:
            ValueError: Error when the provided feature size is not divisible by 4.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check whether the feature size is divisible by 4
        if feat_size % 4 != 0:
            error_msg = f"The feature size (got {feat_size}) of the SineBoxEncoder2d module must be divisible by 4."
            raise ValueError(error_msg)

        # Set remaining attributes
        self.feat_size = feat_size
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.max_period = max_period

    def forward(self, pos_xy):
        """
        Forward method of the SinePosEncoder2d module.

        Args:
            pos_xy (FloatTensor): Position coordinates of shape [num_pts, 2].

        Returns:
            pos_feats (FloatTensor): Position features of shape [num_pts, feat_size].
        """

        # Get normalized position coordinates if requested
        if self.normalize:
            pos_xy = pos_xy - pos_xy.amin(dim=1)
            pos_xy = pos_xy / (pos_xy.amax(dim=1) + 1e-6)

        # Get scaled position coordinates
        pos_xy = self.scale_factor * pos_xy

        # Get periods
        periods = 4 * torch.arange(self.feat_size // 4, dtype=torch.float, device=pos_xy.device) / self.feat_size
        periods = self.max_period ** periods

        # Get position features
        pos_feats = pos_xy[:, :, None] / periods
        pos_feats = torch.cat([pos_feats.sin(), pos_feats.cos()], dim=2)
        pos_feats = pos_feats.view(-1, self.feat_size)

        return pos_feats
