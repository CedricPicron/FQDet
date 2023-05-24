"""
Collection of convolution-based modules.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter

from models.build import MODELS
from models.functional.convolution import conv_transpose2d, id_conv2d


@MODELS.register_module()
class BottleneckConv(nn.Module):
    """
    Class implementing the BottleneckConv module.

    Attributes:
        proj (ProjConv): Module projecting feature map to lower dimensional bottleneck space.
        bottle (ProjConv): Module performing convolution in lower dimensional bottleneck space.
        expansion (ProjConv): Module expanding feature map to desired higher dimensional output space.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_channels, bottle_channels, out_channels=-1, kernel_size=3, norm='', skip=True, **kwargs):
        """
        Initializes the BottleneckConv module.

        Args:
            in_channels (int): Number of input channels.
            bottle_channels (int): Number of bottleneck channels.
            out_channels (int): Number of output channels (default=-1).
            kernel_size (int or Tuple): Size of the convolution kernel (default=3).

            norm (str): String containing the type of normalization chosen from {'', 'group'} (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: When number of input channels does not match output channels when skip connection is used.
            ValueError: When the number of output channels is not specified when no skip connection is used.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get and check number of output channels
        if skip and out_channels == -1:
            out_channels = in_channels

        elif skip and in_channels != out_channels:
            raise ValueError(f"Number of input channels ({in_channels}) must match number of output channels \
                             ({out_channels}) when skip connection is used.")

        elif not skip and out_channels == -1:
            raise ValueError("The number of output channels must be specified when no skip connection is used.")

        # Initialization of three ProjConv modules
        self.proj = ProjConv(in_channels, bottle_channels, norm=norm, skip=False, **kwargs)
        self.bottle = ProjConv(bottle_channels, bottle_channels, kernel_size, norm=norm, skip=False, **kwargs)
        self.expansion = ProjConv(bottle_channels, out_channels, norm=norm, skip=False, **kwargs)

        # Set skip attribute
        self.skip = skip

    def forward(self, map_input, **kwargs):
        """
        Forward method of the BottleneckConv module.

        Args:
            Two types of map-based inputs are supported:
                map_input (List): list [num_maps] with input feature maps of shape [batch_size, feat_size, fH, fW];
                map_input (FloatTensor): input feature map of shape [batch_size, feat_size, fH, fW].

        Returns:
            If 'map_input' was a list:
                map_output (List): list [num_maps] with output feature maps of shape [batch_size, feat_size, fH, fW].

            If map_output was a tensor:
                map_output (FloatTensor): output feature map of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when 'map_input' has invalid type.
        """

        # Place inputs into common structure
        if isinstance(map_input, (list, tuple)):
            in_feat_maps = map_input
        elif torch.is_tensor(map_input):
            in_feat_maps = [map_input]
        else:
            raise ValueError(f"Got invalid type {type(map_input)} for 'map_input'.")

        # Initialize empty list for output maps
        out_feat_maps = []

        # Get output feature map corresponding to each input feature map
        for in_feat_map in in_feat_maps:
            feat_map = self.proj(in_feat_map)
            feat_map = self.bottle(feat_map)
            feat_map = self.expansion(feat_map)

            out_feat_map = in_feat_map + feat_map if self.skip else feat_map
            out_feat_maps.append(out_feat_map)

        # Get output in desired format depending on input
        if isinstance(map_input, (list, tuple)):
            map_output = out_feat_maps
        elif torch.is_tensor(map_input):
            map_output = out_feat_maps[0]

        return map_output


@MODELS.register_module()
class ConvTranspose2d(nn.ConvTranspose2d):
    """
    Class implementing the ConvTranspose2d module.

    It extends the ConvTranspose2d module from torch.nn by automatically computing the output padding if a base map
    size is given.
    """

    def forward(self, input, base_map_size=None):
        """
        Forward method of the ConvTranspose2d module.

        Args:
            input (FloatTensor): Tensor with input feature map of shape [batch_size, in_channels, iH, iW].
            base_map_size (Tuple): Tuple containing the base map size in (height, width) format (default=None).

        Returns:
            output (FloatTensor): Tensor with output feature map of shape [batch_size, out_channels, oH, oW].
        """

        # Perform 2D transposed convolution
        kwargs = {'bias': self.bias, 'stride': self.stride, 'padding': self.padding}
        kwargs = {**kwargs, 'output_padding': self.output_padding, 'groups': self.groups, 'dilation': self.dilation}
        output = conv_transpose2d(input, self.weight, base_map_size=base_map_size, **kwargs)

        return output


@MODELS.register_module()
class IdConv2d(nn.Module):
    """
    Class implementing the IdConv2d module.

    Attributes:
        kernel_size (Tuple): Tuple of size [2] containing the convolution kernel sizes in (H, W) format.
        dilation (Tuple): Tuple of size [2] containing the convolution dilations in (H, W) format.
        weight (Parameter): Parameter with convolution weights of shape [out_channels, kH * kW * in_channels].
        bias (Parameter): Optional parameter with convolution biases of shape [out_channels].
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        """
        Initializes the IdConv2d module.

        Args:
            in_channels (int): Integer containing the number of input channels.
            out_channels (int): Integer containing the number of output channels.
            kernel_size (int or Tuple): Integer ot tuple containing the size of the convolving kernel.
            dilation (int or Tuple): Integer or tuple containing the convolution dilation (default=1).
            bias (bool): Boolean indicating whether learnable bias is added to the output (default=True).

        Raises:
            ValueError: Error when an even kernel size is provided in either dimension.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set convolution attributes
        self.kernel_size = pair(kernel_size)
        self.dilation = pair(dilation)

        # Check kernel size
        if (self.kernel_size[0] % 2 == 0) or (self.kernel_size[1] % 2 == 0):
            error_msg = f"Even kernel sizes are not allowed in either dimension (got {self.kernel_size})."
            raise ValueError(error_msg)

        # Initialize weight parameter
        kH, kW = self.kernel_size
        self.weight = Parameter(torch.empty(out_channels, kH * kW * in_channels))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize optional bias parameter
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            fan_in = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)[0]

            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

        else:
            self.register_parameter('bias', None)

    def forward(self, in_core_feats, aux_feats, id_map, roi_ids, pos_ids, **kwargs):
        """
        Forward method of the IdConv2d module.

        Args:
            in_core_feats (FloatTensor): Input core features of shape [num_core_feats, in_channels].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, in_channels].
            id_map (LongTensor): Index map with feature indices of shape [num_rois, rH, rW].
            roi_ids (LongTensor): RoI indices of core features of shape [num_core_feats].
            pos_ids (LongTensor): RoI-based core position indices in (X, Y) format of shape [num_core_feats, 2].
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_core_feats (FloatTensor): Output core features of shape [num_core_feats, out_channels].
        """

        # Get convolution indices
        kH, kW = self.kernel_size
        dH, dW = self.dilation

        off_y = (kH-1)//2 * dH
        off_x = (kW-1)//2 * dW

        device = aux_feats.device
        offs_y = torch.arange(-off_y, off_y+1, step=dH, device=device)[:, None].expand(-1, kW).flatten()
        offs_x = torch.arange(-off_x, off_x+1, step=dW, device=device)[None, :].expand(kH, -1).flatten()

        pos_y = pos_ids[:, 1, None] + offs_y
        pos_x = pos_ids[:, 0, None] + offs_x

        rH, rW = id_map.size()[1:]
        pad_mask = (pos_y < 0) | (pos_y >= rH) | (pos_x < 0) | (pos_x >= rW)

        roi_ids = roi_ids[:, None].expand(-1, kH*kW)
        pos_y = pos_y.clamp_(min=0, max=rH-1)
        pos_x = pos_x.clamp_(min=0, max=rW-1)

        conv_ids = id_map[roi_ids, pos_y, pos_x]
        conv_ids[pad_mask] = len(in_core_feats) + len(aux_feats)

        # Perform 2D index-based convolution
        out_core_feats = id_conv2d(in_core_feats, aux_feats, conv_ids, self.weight, self.bias)

        return out_core_feats


@MODELS.register_module()
class ProjConv(nn.Module):
    """
    Class implementing the ProjConv module.

    Attributes:
        conv (nn.Conv2d): Module performing the 2D convolution operation.
        norm (nn.Module): Optional attribute containing the normalization module.
        skip (bool): Boolean indicating whether skip connection is used or not.
    """

    def __init__(self, in_channels, out_channels=-1, kernel_size=1, norm='', skip=True, **kwargs):
        """
        Initializes the ProjConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (default=-1).
            kernel_size (int or Tuple): Size of the convolution kernel (default=1).

            norm (str): String containing the type of normalization chosen from {'', 'group'} (default='').
            skip (bool): Boolean indicating whether skip connection is used or not (default=True).

        Raises:
            ValueError: When number of input channels does not match output channels when skip connection is used.
            ValueError: When the number of output channels is not specified when no skip connection is used.
            ValueError: When unsupported normalization type is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get and check number of output channels
        if skip and out_channels == -1:
            out_channels = in_channels

        elif skip and in_channels != out_channels:
            raise ValueError(f"Number of input channels ({in_channels}) must match number of output channels \
                             ({out_channels}) when skip connection is used.")

        elif not skip and out_channels == -1:
            raise ValueError("The number of output channels must be specified when no skip connection is used.")

        # Get convolution keyword arguments
        conv_keys = ('stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode')
        conv_kwargs = {k: v for k, v in kwargs.items() if k in conv_keys}

        # Add default padding to convolution keyword arguments if not yet present
        if 'padding' not in conv_kwargs:
            if isinstance(kernel_size, int):
                conv_kwargs['padding'] = kernel_size//2
            else:
                conv_kwargs['padding'] = (kernel_size[0]//2, kernel_size[1]//2)

        # Initialization of convolution module
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)

        # Initialization of normalization module
        if not norm:
            pass

        elif norm == 'group':
            num_groups = kwargs.pop('num_groups', 8)
            group_keys = ('eps', 'affine')
            group_kwargs = {k: v for k, v in kwargs.items() if k in group_keys}
            self.norm = nn.GroupNorm(num_groups, in_channels, **group_kwargs)

        else:
            raise ValueError(f"The ProjConv module does not support the '{norm}' normalization type.")

        # Set skip attribute
        self.skip = skip

    def forward(self, map_input, **kwargs):
        """
        Forward method of the ProjConv module.

        Args:
            Two types of map-based inputs are supported:
                map_input (List): list [num_maps] with input feature maps of shape [batch_size, feat_size, fH, fW];
                map_input (FloatTensor): input feature map of shape [batch_size, feat_size, fH, fW].

        Returns:
            If 'map_input' was a list:
                map_output (List): list [num_maps] with output feature maps of shape [batch_size, feat_size, fH, fW].

            If map_output was a tensor:
                map_output (FloatTensor): output feature map of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error when 'map_input' has invalid type.
        """

        # Place inputs into common structure
        if isinstance(map_input, (list, tuple)):
            in_feat_maps = map_input
        elif torch.is_tensor(map_input):
            in_feat_maps = [map_input]
        else:
            raise ValueError(f"Got invalid type {type(map_input)} for 'map_input'.")

        # Initialize empty list for output maps
        out_feat_maps = []

        # Get output feature map corresponding to each input feature map
        for in_feat_map in in_feat_maps:
            feat_map = self.norm(in_feat_map) if hasattr(self, 'norm') else in_feat_map
            feat_map = F.relu(feat_map, inplace=hasattr(self, 'norm'))
            feat_map = self.conv(feat_map)

            out_feat_map = in_feat_map + feat_map if self.skip else feat_map
            out_feat_maps.append(out_feat_map)

        # Get output in desired format depending on input
        if isinstance(map_input, (list, tuple)):
            map_output = out_feat_maps
        elif torch.is_tensor(map_input):
            map_output = out_feat_maps[0]

        return map_output
