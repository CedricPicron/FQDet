"""
Collection of anchor-related modules.
"""

import torch
from torch import nn
from torch.nn.modules.utils import _pair as pair

from models.build import MODELS
from structures.boxes import Boxes


@MODELS.register_module()
class AnchorGenerator(nn.Module):
    """
    Class implementing the AnchorGenerator module.

    Attributes:
        strides (Tuple): Tuple of size [num_maps] containing the stride of each feature map.
        cell_anchors (FloatTensor): Buffer containing the cell anchors of shape [num_maps, num_cell_anchors, 4].
        num_cell_anchors (int): Integer containing the number of anchors per cell (i.e. per feature location).
    """

    def __init__(self, map_ids, num_sizes=1, scale_factor=4.0, aspect_ratios=None, cell_grid_size=1):
        """
        Initializes the AnchorGenerator module.

        Args:
            map_ids (Tuple): Tuple of size [num_maps] containing the map ids (i.e. downsampling exponents) of each map.
            num_sizes (int): Integer containing the number of different anchor sizes per aspect ratio (default=1).
            scale_factor (float): Factor scaling the anchors w.r.t. non-overlapping tiling anchors (default=4.0).
            aspect_ratios (Tuple): Tuple [num_aspect_ratios] with different anchor aspect ratios (default=None).
            cell_grid_size (int or Tuple): Integer or tuple in (H, W) format containing the cell grid size (default=1).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get strides
        self.strides = tuple(2**i for i in map_ids)

        # Get cell anchors
        if aspect_ratios is None:
            aspect_ratios = torch.ones(1)
        else:
            aspect_ratios = torch.tensor(aspect_ratios)

        gH, gW = pair(cell_grid_size)
        cell_anchors_list = []

        for stride, i in zip(self.strides, map_ids):
            cell_cx = stride/gW * torch.arange(gW) - stride/2 + stride/gW/2
            cell_cy = stride/gH * torch.arange(gH) - stride/2 + stride/gH/2

            cell_cx = cell_cx[None, :, None].expand(gH, -1, num_sizes * len(aspect_ratios)).flatten()
            cell_cy = cell_cy[:, None, None].expand(-1, gW, num_sizes * len(aspect_ratios)).flatten()

            sizes = torch.tensor([scale_factor * 2**(i+j/num_sizes) for j in range(num_sizes)])
            areas = sizes**2

            cell_w = torch.sqrt(areas[:, None] / aspect_ratios[None, :])
            cell_h = aspect_ratios[None, :] * cell_w

            cell_w = cell_w[None, :, :].expand(gH * gW, -1, -1).flatten()
            cell_h = cell_h[None, :, :].expand(gH * gW, -1, -1).flatten()

            cell_anchors_i = torch.stack([cell_cx, cell_cy, cell_w, cell_h], dim=1)
            cell_anchors_list.append(cell_anchors_i)

        cell_anchors = torch.stack(cell_anchors_list, dim=0)
        self.register_buffer('cell_anchors', cell_anchors, persistent=False)
        self.num_cell_anchors = cell_anchors.size(dim=1)

    def forward(self, feat_maps, **kwargs):
        """
        Forward method of the AnchorGenerator module.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            anchors (Boxes): Structure containing axis-aligned anchor boxes of size [num_feats * num_cell_anchors].
        """

        # Get device
        device = feat_maps[0].device

        # Get anchors
        anchors_list = []

        for i, feat_map in enumerate(feat_maps):
            fH, fW = feat_map.size()[-2:]
            stride = self.strides[i]

            shift_x = stride * torch.arange(fW, device=device) + stride/2
            shift_y = stride * torch.arange(fH, device=device) + stride/2

            shift_x = shift_x[None, :].expand(fH, -1).flatten()
            shift_y = shift_y[:, None].expand(-1, fW).flatten()
            shifts = torch.stack([shift_x, shift_y], dim=1)

            anchors_i = self.cell_anchors[i][None, :, :].expand(fH * fW, -1, -1).clone()
            anchors_i[:, :, :2] += shifts[:, None, :]

            anchors_i = anchors_i.flatten(0, 1)
            anchors_list.append(anchors_i)

        anchors = torch.cat(anchors_list, dim=0)
        anchors = Boxes(anchors, format='cxcywh', normalized='false')

        return anchors
