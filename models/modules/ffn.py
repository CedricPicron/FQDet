"""
Feedforward network (FFN) module.
"""

from torch import nn
import torch.nn.functional as F

from models.build import MODELS


@MODELS.register_module()
class FFN(nn.Module):
    """
    Class implementing the FFN module.

    Attributes:
        layer_norm (nn.LayerNorm): Pre-activaion layer normalization module.
        in_proj (nn.Linear): Linear input projection module projecting normalized features to hidden features.
        out_proj (nn.Linear): Linear output projection module projecting hidden features to delta output features.
    """

    def __init__(self, feat_size, hidden_size):
        """
        Initializes the FFN module.

        Args:
            feat_size (int): Integer containing the input and output feature size.
            hidden_size (int): Integer containing the hidden feature size.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialize layer normalization module
        self.layer_norm = nn.LayerNorm(feat_size)

        # Initialize input and output projection modules
        self.in_proj = nn.Linear(feat_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, feat_size)

    def forward(self, in_feat_list, **kwargs):
        """
        Forward method of the FFN module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Initialize empty output features list
        out_feat_list = []

        # Perform FFN-operation on every set of input features
        for in_feats in in_feat_list:

            # Get FFN delta features
            norm_feats = self.layer_norm(in_feats)
            hidden_feats = self.in_proj(F.relu(norm_feats, inplace=True))
            delta_feats = self.out_proj(F.relu(hidden_feats, inplace=True))

            # Get output features
            out_feats = in_feats + delta_feats
            out_feat_list.append(out_feats)

        return out_feat_list
