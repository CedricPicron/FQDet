"""
Timm backbone.
"""
import math

import timm
import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class TimmBackbone(nn.Module):
    """
    Class implementing the TimmBackbone module.

    Attributes:
        model (nn.Module): Module computing and returning the requested backbone feature maps.

        in_norm_mean (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization means.
        in_norm_std (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization standard deviations.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, model_name, out_ids, pretrained=True):
        """
        Initializes the TimmBackbone module.

        Args:
            model_name (str): String containing the model name.
            out_ids (Tuple): Tuple containing the output map indices of shape [num_out_maps].
            pretrained (bool): Boolean indicating whether to use pretrained model weights (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Create model
        temp_model = timm.create_model(model_name, features_only=True)
        ids_offset = int(math.log2(temp_model.feature_info.reduction()[0]))

        out_indices = [out_id-ids_offset for out_id in out_ids]
        self.model = timm.create_model(model_name, features_only=True, out_indices=out_indices, pretrained=pretrained)

        # Register input normalization buffers
        in_norm_mean = self.model.pretrained_cfg.get('mean', (0.485, 0.456, 0.406))
        in_norm_std = self.model.pretrained_cfg.get('std', (0.229, 0.224, 0.225))

        self.register_buffer('in_norm_mean', torch.tensor(in_norm_mean))
        self.register_buffer('in_norm_std', torch.tensor(in_norm_std))

        # Set attributes related to output feature maps
        self.out_ids = list(out_ids)
        self.out_sizes = self.model.feature_info.channels()

    def forward(self, images, **kwargs):
        """
        Forward method of the TimmBackbone module.

        Args:
            images (Images): Images structure containing the batched images.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].
        """

        # Normalize input images
        norm_images = images.normalize(self.in_norm_mean, self.in_norm_std)

        # Compute backbone feature maps
        feat_maps = self.model(norm_images)

        return feat_maps
