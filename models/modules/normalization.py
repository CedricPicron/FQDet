"""
Collection of normalization-based modules.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.ops.misc import FrozenBatchNorm2d

from models.build import MODELS


@MODELS.register_module()
class FeatureNorm(nn.Module):
    """
    Class implementing the FeatureNorm normalization module.

    Attributes:
        num_channels (int): Number of expected channels of input map.
        num_groups (int): Number of groups to separate the channels into.
        eps (float): Value added to the normalization denominator for numerical stability.
        affine (bool): Whether or not per-channel learnable weights and biases are used.

        weight (Parameter or None): Tensor of shape [num_channels] with the affine weights (None if affine is False).
        bias (Parameter or None): Tensor of shape [num_channels] with the affine biases (None if affine is False).
    """

    def __init__(self, num_channels, num_groups=1, eps=1e-5, affine=True):
        """
        Initializes the FeatureNorm module.

        Args:
            num_channels (int): Number of expected channels of input map.
            num_groups (int): Number of groups to separate the channels into (default=1).
            eps (float): Value added to the normalization denominator for numerical stability (default=1e-5).
            affine (bool): Whether or not per-channel learnable weights and biases are used (default=True).

        Raises:
            ValueError: Error when 'num_channels' is not divisble by 'num_groups'.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        if num_channels % num_groups != 0:
            raise ValueError(f"Number of channels ({num_channels}) should divide number of groups ({num_groups}).")

        # Set non-learnable attributes
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        # Set affine weight and bias parameters
        if affine:
            self.weight = Parameter(torch.empty(num_channels))
            self.bias = Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Set default initial values of module parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets module parameters to default initial values.
        """

        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, in_feat_map):
        """
        Forward method of the FeatureNorm module.

        Args:
            in_feat_map: Input feature map of shape [batch_size, num_channels, *].

        Returns:
            out_feat_map: Normalized output feature map of shape [batch_size, num_channels, *].
        """

        # Permute input feature map
        permute_vector = [0] + list(range(2, in_feat_map.dim())) + [1]
        feat_map = in_feat_map.permute(*permute_vector)

        # Get normalized feature map
        feat_map = feat_map.view(*feat_map.shape[:-1], self.num_groups, -1)
        feat_map = F.layer_norm(feat_map, (self.num_channels//self.num_groups,), eps=self.eps)
        feat_map = feat_map.view(*feat_map.shape[:-2], -1)
        feat_map = torch.addcmul(self.bias, self.weight, feat_map)

        # Permute normalized feature map back to input feature map shape
        permute_vector = [0, in_feat_map.dim()-1] + list(range(1, in_feat_map.dim()-1))
        out_feat_map = feat_map.permute(*permute_vector)

        return out_feat_map


@MODELS.register_module()
class FrozenBatchNorm2d(FrozenBatchNorm2d):
    """
    Two-dimensional batch normalization layer with frozen statistics.

    Copy from torchvision, but with default eps of 1e-5.

    Attributes:
        num_features (int): Expected number of 2D input feature maps.
        eps (float): Value added to the denominator for numerical stability (defaults to 1e-5).
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__(num_features, eps=eps)
