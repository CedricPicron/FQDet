"""
Implements the deformable modules.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .functions import MSDA3DF


class MSDA3D(nn.Module):
    """
    Class implementing the 3D multi-scale deformable attention (MSDA3D) module.

    Attributes:
        num_heads (int): Integer containing the number of attention heads.
        rad_pts (int): Integer containing the number of radial sample points per head and level.
        lvl_pts (int): Integer containing the number of level sample points per head.

        sample_offsets (nn.Linear): Module computing the sample offsets from the input features.
        attn_weights (nn.Linear): Module computing the unnormalized attention weights from the input features.
        val_proj (nn.Linear): Module computing value features from sample features.
        out_proj (nn.Linear): Module computing output features from weighted value features.
    """

    def __init__(self, in_size, sample_size, out_size=-1, num_heads=8, rad_pts=4, lvl_pts=1, val_size=-1):
        """
        Initializes the MSDA3D module.

        Args:
            in_size (int): Size of input features.
            sample_size (int): Size of sample features.
            out_size (int): Size of output features (default=-1).
            num_heads (int): Integer containing the number of attention heads (default=8).
            rad_pts (int): Integer containing the number of radial sample points per head and level (default=4).
            lvl_pts (int): Integer containing the number of level sample points per head (default=1).
            val_size (int): Size of value features (default=-1).

        Raises:
            ValueError: Error when the input feature size does not divide the number of heads.
            ValueError: Error when the value feature size does not divide the number of heads.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to the number of heads and points
        self.num_heads = num_heads
        self.rad_pts = rad_pts
        self.lvl_pts = lvl_pts

        # Check divisibility input size by number of heads
        if in_size % num_heads != 0:
            error_msg = f"The input feature size ({in_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the sample offsets
        self.sample_offsets = nn.Linear(in_size, num_heads * rad_pts * lvl_pts * 3)
        nn.init.zeros_(self.sample_offsets.weight)

        thetas = torch.arange(num_heads, dtype=torch.float) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(), torch.zeros_like(thetas)], dim=1)
        grid_init = grid_init / grid_init.abs().max(dim=1, keepdim=True)[0]
        grid_init = grid_init.view(num_heads, 1, 1, 3).repeat(1, rad_pts, lvl_pts, 1)

        grid_init = grid_init * torch.arange(1, rad_pts+1, dtype=torch.float).view(1, rad_pts, 1, 1)
        self.sample_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Initialize module computing the unnormalized attention weights
        self.attn_weights = nn.Linear(in_size, num_heads * rad_pts * lvl_pts)
        nn.init.zeros_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)

        # Get and check size of value features
        if val_size == -1:
            val_size = in_size

        elif val_size % num_heads != 0:
            error_msg = f"The value feature size ({val_size}) must divide the number of heads ({num_heads})."
            raise ValueError(error_msg)

        # Initialize module computing the value features
        self.val_proj = nn.Linear(sample_size, val_size)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.zeros_(self.val_proj.bias)

        # Initialize module computing the output features
        out_size = in_size if out_size == -1 else out_size
        self.out_proj = nn.Linear(val_size, out_size)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, in_feats, sample_priors, sample_feats, map_hw, map_offs, map_ids, **kwargs):
        """
        Forward method of the MSDA3D module.

        Args:
            in_feats (FloatTensor): Input features of shape [batch_size, num_in_feats, in_size].
            sample_priors (FloatTensor): Sample priors of shape [batch_size, num_in_feats, {2, 4}].
            sample_feats (FloatTensor): Sample features of shape [batch_size, num_sample_feats, sample_size].
            map_hw (LongTensor): Feature map sizes in (H, W) format of shape [num_maps, 2].
            map_offs (LongTensor): Feature map offsets of shape [num_maps].
            map_ids (LongTensor): Map indices of input features of shape [batch_size, num_in_feats].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [batch_size, num_in_feats, out_size].

        Raises:
            ValueError: Error when the last dimension of 'sample_priors' is different from 2 or 4.
        """

        # Get shapes of input tensors
        batch_size, num_in_feats = in_feats.shape[:2]
        common_shape = (batch_size, num_in_feats, self.num_heads)

        # Get value features
        num_sample_feats = sample_feats.shape[1]
        val_feats = self.val_proj(sample_feats).view(batch_size, num_sample_feats, self.num_heads, -1)

        # Get zero-one normalized sample XYZ
        sample_offs = self.sample_offsets(in_feats).view(*common_shape, self.rad_pts * self.lvl_pts, 3)

        if sample_priors.shape[-1] == 2:
            offset_normalizers = map_hw.fliplr()[map_ids, None, None, :]
            sample_xy = sample_priors[:, :, None, None, :2] + sample_offs[:, :, :, :, :2] / offset_normalizers

        elif sample_priors.shape[-1] == 4:
            offset_factors = 0.5 * sample_priors[:, :, None, None, 2:] / self.rad_pts
            sample_xy = sample_priors[:, :, None, None, :2] + sample_offs[:, :, :, :, :2] * offset_factors

        else:
            error_msg = f"Last dimension of 'sample_priors' must be 2 or 4, but got {sample_priors.shape[-1]}."
            raise ValueError(error_msg)

        sample_z = map_ids[:, :, None, None, None].expand(-1, -1, self.num_heads, self.rad_pts, self.lvl_pts)
        sample_z = sample_z + torch.arange(self.lvl_pts, device=sample_z.device) - 0.5 * (self.lvl_pts - 1)

        sample_z = (sample_z.flatten(3) + sample_offs[:, :, :, :, 2].tanh()) / (len(map_hw) - 1)
        sample_xyz = torch.cat([sample_xy, sample_z[:, :, :, :, None]], dim=4)

        # Get normalized attention weights
        attn_ws = self.attn_weights(in_feats).view(*common_shape, self.rad_pts * self.lvl_pts)
        attn_ws = F.softmax(attn_ws, dim=3)

        # Get attention features consisting of weighted sampled value features
        attn_feats = MSDA3DF.apply(val_feats, map_hw, map_offs, sample_xyz, attn_ws).flatten(2)

        # Get output features
        out_feats = self.out_proj(attn_feats)

        return out_feats
