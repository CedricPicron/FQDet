"""
Collection of modules implementing mathematical operations.
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.build import MODELS


@MODELS.register_module()
class Exp(nn.Module):
    """
    Class implementing the Exp module computing the exponential of the elements of the input tensor.
    """

    def __init__(self):
        """
        Initializes the Exp module.
        """

        # Initialization of default nn.Module
        super().__init__()

    def forward(self, in_tensor):
        """
        Forward method of the Exp module.

        Args:
            in_tensor (FloatTensor): Input tensor of arbitrary shape.

        Returns:
            out_tensor (FloatTensor): Output tensor of same shape as input tensor.
        """

        # Get output tensor
        out_tensor = torch.exp(in_tensor)

        return out_tensor


@MODELS.register_module()
class Mul(nn.Module):
    """
    Class implementing the Mul module performing element-wise multiplication with optional addition.

    Attributes:
        factor (Parameter): Tensor containing the mulitplication factor of shape [1] or [feat_size].
        bias (Parameter): Tensor containing the optional addition term of shape [1] or [feat_size] (None when missing).
    """

    def __init__(self, feat_dependent=False, feat_size=None, init_factor=1.0, learn_factor=True, bias=False,
                 init_bias=0.0, learn_bias=True):
        """
        Initializes the Mul module.

        Args:
            feat_dependent (bool): Boolean indicating whether parameters should be feature dependent (default=False).
            feat_size (float): Value containing the expected input feature size (default=None).
            init_factor (float): Value containing the initial multiplication factor (default=1.0).
            learn_factor (bool): Boolean indicating whether factor parameter should be learned (default=True).
            bias (bool): Boolean indicating whether bias should be added after multiplication (default=False).
            init_bias (float): Value containing the initial bias value (default=0.0).
            learn_bias (bool): Boolean indicating whether bias parameter should be learned (default=True).

        Raises:
            ValueError: Error when 'feat_dependent' is True and no feature size is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get factor and optional bias parameters
        if not feat_dependent:
            self.factor = Parameter(torch.tensor([init_factor]), requires_grad=learn_factor)
            self.bias = Parameter(torch.tensor([init_bias]), requires_grad=learn_bias) if bias else None

        elif feat_size is not None:
            self.factor = Parameter(torch.full((feat_size,), init_factor), requires_grad=learn_factor)
            self.bias = Parameter(torch.full((feat_size,), init_bias), requires_grad=learn_bias) if bias else None

        else:
            error_msg = "A feature size must be provided when 'feat_dependent' is True."
            raise ValueError(error_msg)

    def forward(self, in_tensor):
        """
        Forward method of the Mul module.

        Args:
            in_tensor (FloatTensor): Input tensor of shape [*, feat_size].

        Returns:
            out_tensor (FloatTensor): Output tensor of shape [*, feat_size].
        """

        # Get output tensor
        if self.bias is None:
            out_tensor = torch.mul(in_tensor, self.factor)
        else:
            out_tensor = torch.addcmul(self.bias, in_tensor, self.factor)

        return out_tensor
