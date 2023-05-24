"""
Backbone-Core-Heads (BCH) architecture.
"""
from collections import ChainMap
from itertools import chain

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class BCH(nn.Module):
    """
    Class implementing the BCH architecture, consisting of a backbone, a core and heads modules.

    Attributes:
        backbone (nn.Module): Module implementing the backbone.
        core (nn.Module): Module or dictionary of modules of size [num_cores] implementing the core.
        heads (nn.ModuleDict): Dictionary of size [num_heads] with head modules.
    """

    def __init__(self, backbone, core, heads):
        """
        Initializes the BCH module.

        Args:
            backbone (nn.Module): Module implementing the backbone.
            core (nn.Module): Module implementing the core.
            heads (Dict): Dictionary of size [num_heads] with head modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set backbone, core and heads attributes
        self.backbone = backbone
        self.core = core
        self.heads = nn.ModuleDict(heads)

    @staticmethod
    def get_param_families():
        """
        Method returning the BCH parameter families.

        Returns:
            List of strings containing the BCH parameter families.
        """

        return ['backbone', 'core', 'heads']

    def forward(self, images, tgt_dict=None, visualize=False, **kwargs):
        """
        Forward method of the BCH module.

        Args:
            images (Images): Images structure containing the batched images.
            tgt_dict (Dict): Target dictionary with ground-truth information used during trainval (default=None).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).

            kwargs (Dict): Dictionary of keyword arguments, potentially containing following key:
                - extended_analysis (bool): boolean indicating whether to perform extended analyses or not.

        Returns:
            output_dicts (List): List of output dictionaries, potentially containing following items:
                - pred_dicts (List): list of dictionaries with predictions;
                - loss_dict (Dict): dictionary of different loss terms used for backpropagation during training;
                - analysis_dict (Dict): dictionary of different analyses used for logging purposes only;
                - images_dict (Dict): dictionary of different annotated images based on predictions and targets.
        """

        # Apply backbone
        feat_maps = self.backbone(images)

        # Apply core
        feat_maps = self.core(feat_maps, images=images)

        # Apply heads and merge non-prediction dictionaries originating from different heads
        head_kwargs = {'tgt_dict': tgt_dict, 'images': images, 'visualize': visualize, **kwargs}
        head_dicts = [head(feat_maps, **head_kwargs) for head in self.heads.values()]

        if self.training:
            output_dicts = [dict(ChainMap(*dicts)) for dicts in zip(*head_dicts)]
        else:
            zipped_dicts = list(zip(*head_dicts))
            pred_dicts = list(chain.from_iterable(zipped_dicts[0]))
            non_pred_dicts = [dict(ChainMap(*dicts)) for dicts in zipped_dicts[1:]]
            output_dicts = [pred_dicts, *non_pred_dicts]

        return output_dicts
