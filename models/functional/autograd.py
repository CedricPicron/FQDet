"""
Collection of custom autograd functions.
"""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class IdConv2d(Function):
    """
    Class implementing the IdConv2d autograd function.

    This custom autograd function avoids keeping intermediate data structures in memory.
    """

    @staticmethod
    def forward(ctx, in_core_feats, aux_feats, conv_ids, weight, bias):
        """
        Forward method of the IdConv2d autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            in_core_feats (FloatTensor): Input core features of shape [num_core_feats, in_channels].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, in_channels].
            conv_ids (LongTensor): Indices selecting convolution features of shape [num_core_feats, kH * kW].
            weight (FloatTensor): Tensor with convolution weights of shape [out_channels, kH * kW * in_channels].
            bias (FloatTensor): Tensor with the convolution biases of shape [out_channels].

        Returns:
            out_core_feats (FloatTensor): Output core features of shape [num_core_feats, out_channels].
        """

        # Compute intermediate data structures
        pad_feat = in_core_feats.new_zeros([1, in_core_feats.size()[1]])
        cat_feats = torch.cat([in_core_feats, aux_feats, pad_feat], dim=0)
        conv_feats = cat_feats[conv_ids].flatten(1)

        # Get output core features
        out_core_feats = torch.mm(conv_feats, weight.t())

        if bias is not None:
            out_core_feats += bias

        # Save input tensors for backward pass
        ctx.save_for_backward(in_core_feats, aux_feats, conv_ids, weight, bias)

        return out_core_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_core_feats):
        """
        Backward method of the IdConv2d autograd function.

        Args:
            ctx (FunctionCtx): Context object storing additional data.
            grad_out_core_feats (FloatTensor): Output core features gradient of shape [num_core_feats, out_channels].

        Returns:
            grad_in_core_feats (FloatTensor): Input core features gradient of shape [num_core_feats, in_channels].
            grad_aux_feats (FloatTensor): Auxiliary features gradient of shape [num_aux_feats, in_channels].
            grad_conv_ids (None): None.
            grad_weight (FloatTensor): Convolution weights gradient of shape [out_channels, kH * kW * in_channels].
            grad_bias (FloatTensor): Convolution biases gradient of shape [out_channels] or None.
        """

        # Recover input tensors from forward method
        in_core_feats, aux_feats, conv_ids, weight, bias = ctx.saved_tensors

        # Recompute intermediate data structures
        pad_feat = aux_feats.new_zeros([1, aux_feats.size()[1]])
        cat_feats = torch.cat([in_core_feats, aux_feats, pad_feat], dim=0)
        conv_feats = cat_feats[conv_ids].flatten(1)

        # Get gradient tensors
        grad_conv_feats = torch.mm(grad_out_core_feats, weight)
        grad_weight = torch.mm(grad_out_core_feats.t(), conv_feats)
        grad_bias = grad_out_core_feats.sum(dim=0) if bias is not None else None

        num_core_feats, in_channels = in_core_feats.size()
        grad_conv_feats = grad_conv_feats.view(-1, in_channels)
        grad_cat_feats = torch.zeros_like(cat_feats).index_add_(0, conv_ids.flatten(), grad_conv_feats)

        grad_in_core_feats = grad_cat_feats[:num_core_feats]
        grad_aux_feats = grad_cat_feats[num_core_feats:-1]
        grad_conv_ids = None

        return grad_in_core_feats, grad_aux_feats, grad_conv_ids, grad_weight, grad_bias
