"""
Collection of functions related to convolutions.
"""

import torch
import torch.nn.functional as F

from models.functional.autograd import IdConv2d


def conv_transpose2d(input, weight, base_map_size=None, stride=1, padding=0, dilation=1, max_counter=100, **kwargs):
    """
    Function performing the 2D transposed convolution operation.

    If a base map size is given, the function uses this size to automatically compute the output padding.

    Args:
        input (FloatTensor): Tensor with input feature map of shape [batch_size, in_channels, iH, iW].
        weight (FloatTensor): Tensor with convolution weights of shape [in_channels, out_channels/groups, kH, kW].
        base_map_size (Tuple): Tuple containing the base map size in (height, width) format (default=None).
        stride (int or Tuple): Stride of the 2D transposed convolution operation (default=1).
        padding (int or Tuple): Padding of the 2D transposed convolution operation (default=0).
        dilation (int or Tuple): Dilation of the 2D transposed convolution operation (default=1).
        max_counter (int): Integer containing the maximum number of map size iterations (default=100).
        kwargs (Dict): Dictionary of keyword arguments to be passed to underlying 2D transposed convolution function.

    Returns:
        output (FloatTensor): Tensor with output feature map of shape [batch_size, out_channels, oH, oW].

    Raises:
        ValueError: Error when the input map size and the base map size do not match.
    """

    # Compute output padding if needed
    if base_map_size is not None:
        in_map_size = torch.tensor(input.size()[-2:])
        map_size = torch.tensor(base_map_size)

        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        kernel_size = torch.tensor(weight.size()[-2:])
        stride = torch.tensor(stride)
        padding = torch.tensor(padding)
        dilation = torch.tensor(dilation)
        counter = 0

        while True:
            counter += 1
            old_map_size = map_size

            map_size = map_size + 2*padding - dilation*(kernel_size-1) - 1
            output_padding = map_size % stride

            map_size = torch.div(map_size, stride, rounding_mode='trunc') + 1
            map_size = map_size.to(torch.int64)

            if torch.equal(map_size, in_map_size):
                break

            elif torch.equal(old_map_size, map_size) or counter >= max_counter:
                in_map_size = tuple(in_map_size.tolist())
                error_msg = f"The input map size {in_map_size} and the base map size {base_map_size} do not match with"
                error_msg += " the given kernel size, stride, padding and dilation."
                raise ValueError(error_msg)

        stride = tuple(stride.tolist())
        padding = tuple(padding.tolist())
        dilation = tuple(dilation.tolist())
        kwargs['output_padding'] = tuple(output_padding.tolist())

    # Perform 2D transposed convolution
    output = F.conv_transpose2d(input, weight, stride=stride, padding=padding, dilation=dilation, **kwargs)

    return output


def id_conv2d(in_core_feats, aux_feats, conv_ids, weight, bias):
    """
    Function implementing the 2D index-based convolution operation.

    This custom implementation does not keep intermediate data structures in memory.

    Args:
        in_core_feats (FloatTensor): Input core features of shape [num_core_feats, in_channels].
        aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, in_channels].
        conv_ids (LongTensor): Indices selecting convolution features of shape [num_core_feats, kH * kW].
        weight (FloatTensor): Tensor with convolution weights of shape [out_channels, kH * kW * in_channels].
        bias (FloatTensor): Tensor with convolution biases of shape [out_channels].

    Returns:
        out_core_feats (FloatTensor): Output core features of shape [num_core_feats, out_channels].
    """

    # Apply custom IdConv2d autograd function
    out_core_feats = IdConv2d.apply(in_core_feats, aux_feats, conv_ids, weight, bias)

    return out_core_feats
