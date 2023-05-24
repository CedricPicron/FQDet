"""
Collection of Multi-Layer Perceptron (MLP) modules.
"""
from copy import deepcopy

from torch import nn

from models.build import MODELS


@MODELS.register_module()
class MLP(nn.Module):
    """
    Class implementing the MLP module.

    Attributes:
        mlp (nn.Sequential): Sequence of activation and projection modules.
    """

    def __init__(self, in_size, hidden_size, out_size, layers, **kwargs):
        """
        Initializes the MLP module.

        Args:
            in_size (int): Integer containing the input feature size.
            hidden_size (int): Integer containing the hidden feature size.
            out_size (int): Integer containing the output feature size.
            layers (int): Integer containing the number of hidden layers.

        Raises:
            ValueError: Raised when the requested number of hidden layers is negative.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get MLP head depending on the number of hidden layers
        if layers == 0:
            self.mlp = nn.Sequenatial(nn.Sequential(nn.ReLU(inplace=False), nn.Linear(in_size, out_size)))

        elif layers > 0:
            input_block = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(in_size, hidden_size))
            hidden_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_size, hidden_size))
            output_block = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_size, out_size))

            hidden_blocks = [deepcopy(hidden_block) for _ in range(layers-1)]
            self.mlp = nn.Sequential(input_block, *hidden_blocks, output_block)

        else:
            raise ValueError(f"The number of hidden layers must be non-negative (got {layers}).")

    def forward(self, in_feat_list, **kwargs):
        """
        Forward method of the MLP module.

        Args:
            in_feat_list (List): List [num_feat_sets] containing input features of shape [num_features, feat_size].

        Returns:
            out_feat_list (List): List [num_feat_sets] containing output features of shape [num_features, feat_size].
        """

        # Perform MLP-operation on every set of input features
        out_feat_list = [self.mlp(in_feats) for in_feats in in_feat_list]

        return out_feat_list


@MODELS.register_module()
class OneStepMLP(nn.Module):
    """
    Class implementing the one-step MLP module.

    Attributes:
        norm (nn.Module): Optional normalization module of the OneStepMLP module.
        act_fn (nn.Module): Optional module with the activation function of the OneStepMLP module.
        lin (nn.Linear): Linear projection module of the OneStepMLP module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_size, out_size=-1, norm='', act_fn='', skip=True):
        """
        Initializes the OneStepMLP module.

        Args:
            in_size (int): Size of input features.
            out_size (int): Size of output features (default=-1).
            norm (str): String containing the type of normalization (default='').
            act_fn (str): String containing the type of activation function (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: Error when unsupported type of normalization is provided.
            ValueError: Error when unsupported type of activation function is provided.
            ValueError: Error when input and output feature sizes are different when skip connection is used.
            ValueError: Error when the output feature size is not specified when no skip connection is used.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Initialization of optional normalization module
        if not norm:
            pass
        elif norm == 'layer':
            self.norm = nn.LayerNorm(in_size)
        else:
            error_msg = f"The OneStepMLP module does not support the '{norm}' normalization type."
            raise ValueError(error_msg)

        # Initialization of optional module with activation function
        if not act_fn:
            pass
        elif act_fn == 'gelu':
            self.act_fn = nn.GELU()
        elif act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=False) if not norm and skip else nn.ReLU(inplace=True)
        else:
            error_msg = f"The OneStepMLP module does not support the '{act_fn}' activation function."
            raise ValueError(error_msg)

        # Get and check output feature size
        if skip and out_size == -1:
            out_size = in_size

        elif skip and in_size != out_size:
            error_msg = f"Input ({in_size}) and output ({out_size}) sizes must match when skip connection is used."
            raise ValueError(error_msg)

        elif not skip and out_size == -1:
            error_msg = "The output feature size must be specified when no skip connection is used."
            raise ValueError(error_msg)

        # Initialization of linear module
        self.lin = nn.Linear(in_size, out_size)

        # Set skip attribute
        self.skip = skip

    def forward(self, in_feats, **kwargs):
        """
        Forward method of the OneStepMLP module.

        Args:
            in_feats (FloatTensor): Input features of shape [*, in_size].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [*, out_size].
        """

        # Get delta features
        delta_feats = in_feats
        delta_feats = self.norm(delta_feats) if hasattr(self, 'norm') else delta_feats
        delta_feats = self.act_fn(delta_feats) if hasattr(self, 'act_fn') else delta_feats
        delta_feats = self.lin(delta_feats)

        # Get output features
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats


@MODELS.register_module()
class TwoStepMLP(nn.Module):
    """
    Class implementing the two-step MLP module.

    Attributes:
        mlp1 (OneStepMLP): OneStepMLP module performing the first step.
        mlp2 (OneStepMLP): TwoStepMLP module performing the second step.
        skip (bool): Boolean indicating whether skip connection is used or not (default=True).
    """

    def __init__(self, in_size, hidden_size, out_size=-1, norm1='', norm2='', act_fn1='', act_fn2='', skip=True):
        """
        Initializes the TwoStepMLP module.

        Args:
            in_size (int): Size of input features.
            hidden_size (int): Size of hidden features.
            out_size (int): Size of output features (default=-1).
            norm1 (str): String containing the type of normalization for the first step (default='').
            norm2 (str): String containing the type of normalization for the second step (default='').
            act_fn1 (str): String containing the type of activation function for the first step (default='').
            act_fn2 (str): String containing the type of activation function for the second step (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: Error when input and output feature sizes are different when skip connection is used.
            ValueError: Error when the output feature size is not specified when no skip connection is used.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get and check output feature size
        if skip and out_size == -1:
            out_size = in_size

        elif skip and in_size != out_size:
            error_msg = f"Input ({in_size}) and output ({out_size}) sizes must match when skip connection is used."
            raise ValueError(error_msg)

        elif not skip and out_size == -1:
            error_msg = "The output feature size must be specified when no skip connection is used."
            raise ValueError(error_msg)

        # Get first and second step OneStepMLP modules
        self.mlp1 = OneStepMLP(in_size, hidden_size, norm1, act_fn1, skip=False)
        self.mlp2 = OneStepMLP(hidden_size, out_size, norm2, act_fn2, skip=False)

        # Set skip attribute
        self.skip = skip

    def forward(self, in_feats, **kwargs):
        """
        Forward method of the TwoStepMLP module.

        Args:
            in_feats (FloatTensor): Input features of shape [*, in_size].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_feats (FloatTensor): Output features of shape [*, out_size].
        """

        # Get delta features
        delta_feats = self.mlp1(in_feats)
        delta_feats = self.mlp2(delta_feats)

        # Get output features
        out_feats = in_feats + delta_feats if self.skip else delta_feats

        return out_feats
