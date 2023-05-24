"""
ResNet backbone.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from models.build import MODELS
from models.modules.normalization import FrozenBatchNorm2d


@MODELS.register_module()
class ResNet(nn.Module):
    """
    Class implementing the ResNet module.

    Attributes:
        in_norm_mean (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization means.
        in_norm_std (FloatTensor): Tensor (buffer) of shape [3] containing the input normalization standard deviations.

        body (IntermediateLayerGetter): Module computing and returning the requested backbone feature maps.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, name, out_ids, dilation=False):
        """
        Initializes the ResNet module.

        Args:
            name (str): String containing the full name of the desired ResNet model.
            out_ids (List): List of size [num_maps] containing the indices of the feature maps to return.
            dilation (bool): Boolean indicating whether to use dilation last ResNet layer (default=False).

        Raises:
            ValueError: Error when the provided list of ResNet output indices is empty.
            ValueError: Error when a provided ResNet output index does not lie between 2 and 5.
        """

        # Check provided output indices
        if len(out_ids) == 0:
            error_msg = "The provided list of ResNet output indices must be non-empty."
            raise ValueError(error_msg)

        for i in out_ids:
            if i < 2 or i > 5:
                error_msg = f"The ResNet ouput indices must lie between 2 and 5 (got {i})."
                raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Register input normalization buffers with values expected by torchvision pretrained backbones
        self.register_buffer('in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Load ImageNet pretrained ResNet model from torchvision
        resnet_kwargs = {'replace_stride_with_dilation': [False, False, dilation], 'weights': 'IMAGENET1K_V1'}
        resnet_kwargs = {**resnet_kwargs, 'norm_layer': FrozenBatchNorm2d}
        resnet = getattr(torchvision.models, name)(**resnet_kwargs)

        # Determine which backbone parameters should be trained
        max_out_id = max(out_ids)

        for name, parameter in resnet.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

            elif int(name[5]) >= max_out_id:
                parameter.requires_grad_(False)

        # Get body module computing and returning the requested ResNet feature maps
        return_layers = {f'layer{i-1}': str(i) for i in out_ids}
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # Set attributes related to output feature maps
        self.out_ids = out_ids
        out_sizes = [64, 128, 256, 512] if name in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        self.out_sizes = [out_sizes[i-2] for i in out_ids]

    def _load_from_state_dict(self, state_dict, prefix, *args):
        """
        Copies parameters and buffers from given state dictionary into only this module, but not its descendants.

        state_dict (Dict): Dictionary containing model parameters and persistent buffers.
        prefix (str): String containing this module's prefix in the given state dictionary.
        args (Tuple): Tuple containing additional arguments used by the default loading method.
        """

        # Set default input normalization buffers if missing
        state_dict.setdefault(f'{prefix}in_norm_mean', torch.tensor([0.485, 0.456, 0.406]))
        state_dict.setdefault(f'{prefix}in_norm_std', torch.tensor([0.229, 0.224, 0.225]))

        # Continue with default loading
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, images, return_masks=False):
        """
        Forward method of ResNet module.

        Args:
            images (Images): Images structure containing the batched images.
            return_masks (bool): Boolean indicating whether to return feature masks or not (default=False).

        Returns:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            If 'return_masks' is True:
                feat_masks (List): List of size [num_maps] with masks of active features of shape [batch_size, fH, fW].
        """

        # Normalize input images
        norm_images = images.normalize(self.in_norm_mean, self.in_norm_std)

        # Compute backbone feature maps
        feat_maps = list(self.body(norm_images).values())

        # Only return feature maps if desired
        if not return_masks:
            return feat_maps

        # Compute feature masks
        feat_masks = images.masks[None].float()
        feat_masks = [F.interpolate(feat_masks, size=feat_map.shape[-2:]).to(torch.bool)[0] for feat_map in feat_maps]

        return feat_maps, feat_masks
