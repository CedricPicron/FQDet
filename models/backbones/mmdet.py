"""
Module building backbones from MMDetection.
"""

from mmdet.registry import MODELS as MMDET_MODELS
from mmengine.config import Config
from mmengine.registry import build_model_from_cfg
import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class MMDetBackbone(nn.Module):
    """
    Class implementing the MMDetBackbone module.

    Attributes:
        in_norm_mean (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization means.
        in_norm_std (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization standard deviations.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.

        body (nn.Module): Module computing and returning the requested backbone feature maps.
    """

    def __init__(self, cfg_path):
        """
        Initializes the MMDetBackbone module.

        Args:
            cfg_path (str): Path to configuration file specifying the MMDetection backbone.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Register input normalization buffers with values expected by torchvision pretrained backbones
        self.register_buffer('in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Get config specifying the MMDetection backbone
        cfg = Config.fromfile(cfg_path)

        # Set attributes related to output feature maps
        self.out_ids = [i+2 for i in cfg.backbone['out_indices']]
        self.out_sizes = cfg.backbone.pop('out_sizes')

        # Get backbone body
        self.body = build_model_from_cfg(cfg.backbone, registry=MMDET_MODELS)
        self.body.init_weights()

    def forward(self, images):
        """
        Forward method of the MMDetBackbone module.

        Args:
            images (Images): Images structure containing the batched images.

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
        """

        # Normalize input images
        norm_images = images.normalize(self.in_norm_mean, self.in_norm_std)

        # Compute backbone feature maps
        feat_maps = list(self.body(norm_images))

        return feat_maps
