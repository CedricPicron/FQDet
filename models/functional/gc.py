"""
Collection of feature maps to feature maps operations.
"""
from copy import deepcopy

from mmcv.ops.deform_conv import DeformConv2dPack as DCN
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCNv2
import torch
from torch import nn
import torch.nn.functional as F

from models.modules.attention import Attn2d
from models.modules.normalization import FeatureNorm


def prepare_operation(operation, local_to_global_dict=None):
    """
    Prepares operation dictionary for operation initialization.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        local_to_global_dict (Dict): Optional dictionary specifying the change from local to global ids.

    Returns:
        operation (Dict): Updated dictionary containing the operation ready to be initialized.

    Raises:
        ValueError: Error when sizes of the 'in' and 'out' keys from the operation dictionary do not match.
    """

    # Transform 'in' and 'out' keys to list of lists
    if isinstance(operation['in'], int):
        operation['in'] = [[operation['in']]]
    elif isinstance(operation['in'][0], int):
        operation['in'] = [operation['in']]

    if isinstance(operation['out'], int):
        operation['out'] = [[operation['out']]]
    elif isinstance(operation['out'][0], int):
        operation['out'] = [operation['out']]

    # Check sizes of 'in' and 'out' keys
    if len(operation['in']) != len(operation['out']):
        raise ValueError(f"Sizes of 'in' and 'out' keys from operation dictionary {operation} must match.")

    # Change local map ids to global map ids if requested
    if local_to_global_dict is not None:
        for key in operation:
            if key in ['in', 'out', 'out_to_in']:
                for i, map_ids in enumerate(deepcopy(operation[key])):
                    for j, map_id in enumerate(map_ids):
                        if map_id in local_to_global_dict:
                            operation[key][i][j] = local_to_global_dict[map_id]
                        else:
                            operation[key][i][j] = map_id + local_to_global_dict['offset']

    return operation


def initialize_add(operation, feat_sizes):
    """
    Initializes adding operation for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 2 for map_ids in operation['in']):
        raise ValueError(f"Exactly two maps must be provided per list in {operation['in']}.")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['alpha']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Initialize every add operation
    for map_ids in operation['in']:
        sub_operation = {'function': torch.add, 'in_map_ids': map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_ids[0]]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_attn2d(operation, feat_sizes, modules_offset):
    """
    Initializes 2D attention operation with corresponding modules for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
        ValueError: Error when incorrect output channels input is provided.
    """

    # Check 'in' operation key
    if any(len(map_ids) not in [1, 2] for map_ids in operation['in']):
        raise ValueError(f"One or two maps must be provided per list in {operation['in']}.")

    # Get lists of number of input and output channels
    in_channels_list = [[feat_sizes[map_id] for map_id in map_ids] for map_ids in operation['in']]
    out_channels = operation['out_channels']
    out_channels_list = [out_channels] if isinstance(out_channels, int) else out_channels

    if len(in_channels_list) > 1 and len(out_channels_list) == 1:
        out_channels_list = [out_channels_list[0]] * len(in_channels_list)
    elif len(in_channels_list) != len(out_channels_list):
        raise ValueError("Size of input and output channels list must match when multiple output channels are given.")

    # Get 2D attention keyword arguments
    allowed_keys = ['kernel_size', 'stride', 'padding', 'dilation', 'num_heads', 'bias', 'padding_mode', 'attn_mode']
    allowed_keys.extend(['pos_attn', 'q_stride', 'qk_channels', 'qk_norm'])
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}

    # Initialize list of modules and sub-operations
    modules = []
    sub_operations = []

    # Initialize every 2D attention sub-operation
    for map_ids, in_channels, out_channels in zip(operation['in'], in_channels_list, out_channels_list):
        module = Attn2d(in_channels, out_channels, **sub_operation_kwargs)
        sub_operation = {'in_map_ids': map_ids, 'module_id': modules_offset+len(modules)}

        if 'weight_init' in operation:
            init_kwargs = {k.replace('weight_init_', ''): v for k, v in operation.items() if 'weight_init_' in k}

            for name, parameter in module.named_parameters():
                if 'weight' in parameter:
                    getattr(nn.init, operation['weight_init'])(parameter, **init_kwargs)

        if 'bias_init' in operation:
            init_kwargs = {k.replace('bias_init_', ''): v for k, v in operation.items() if 'bias_init_' in k}

            for name, parameter in module.named_parameters():
                if 'bias' in parameter:
                    getattr(nn.init, operation['bias_init'])(parameter, **init_kwargs)

        modules.append(module)
        sub_operations.append(sub_operation)
        feat_sizes.append(out_channels)

    return modules, sub_operations, feat_sizes


def initialize_conv2d(operation, feat_sizes, modules_offset):
    """
    Initializes 2D convolution operation with corresponding modules for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
        ValueError: Error when incorrect output channels input is provided.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 1 for map_ids in operation['in']):
        raise ValueError(f"Exactly one map must be provided per list in {operation['in']}.")

    # Get lists of number of input and output channels
    in_channels_list = [feat_sizes[map_ids[0]] for map_ids in operation['in']]
    out_channels = operation['out_channels']
    out_channels_list = [out_channels] if isinstance(out_channels, int) else out_channels

    if len(in_channels_list) > 1 and len(out_channels_list) == 1:
        out_channels_list = [out_channels_list[0]] * len(in_channels_list)
    elif len(in_channels_list) != len(out_channels_list):
        raise ValueError("Size of input and output channels list must match when multiple output channels are given.")

    # Get 2D convolution keyword arguments
    allowed_keys = ['kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}

    # Initialize list of modules and sub-operations
    modules = []
    sub_operations = []

    # Initialize every 2D convolution sub-operation
    for map_ids, in_channels, out_channels in zip(operation['in'], in_channels_list, out_channels_list):
        module = nn.Conv2d(in_channels, out_channels, **sub_operation_kwargs)
        sub_operation = {'in_map_ids': map_ids, 'module_id': modules_offset+len(modules)}

        if 'weight_init' in operation:
            init_kwargs = {k.replace('weight_init_', ''): v for k, v in operation.items() if 'weight_init_' in k}
            getattr(nn.init, operation['weight_init'])(module.weight, **init_kwargs)

        if 'bias_init' in operation:
            init_kwargs = {k.replace('bias_init_', ''): v for k, v in operation.items() if 'bias_init_' in k}
            getattr(nn.init, operation['bias_init'])(module.bias, **init_kwargs)

        modules.append(module)
        sub_operations.append(sub_operation)
        feat_sizes.append(out_channels)

    return modules, sub_operations, feat_sizes


def initialize_dcn(operation, feat_sizes, modules_offset):
    """
    Initializes 2D deformable convolution operation with corresponding modules for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
        ValueError: Error when incorrect output channels input is provided.
        ValueError: Error when invalid DCN version number is provided.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 1 for map_ids in operation['in']):
        raise ValueError(f"Exactly one map must be provided per list in {operation['in']}.")

    # Get lists of number of input and output channels
    in_channels_list = [feat_sizes[map_ids[0]] for map_ids in operation['in']]
    out_channels = operation['out_channels']
    out_channels_list = [out_channels] if isinstance(out_channels, int) else out_channels

    if len(in_channels_list) > 1 and len(out_channels_list) == 1:
        out_channels_list = [out_channels_list[0]] * len(in_channels_list)
    elif len(in_channels_list) != len(out_channels_list):
        raise ValueError("Size of input and output channels list must match when multiple output channels are given.")

    # Get 2D deformable convolution module of desired version
    version = operation.setdefault('version', 2)

    if version == 1:
        dcn = DCN
    elif version == 2:
        dcn = DCNv2
    else:
        raise ValueError(f"DCN version number must be 1 or 2, but got {version}.")

    # Get 2D deformable convolution keyword arguments
    allowed_keys = ['kernel_size', 'stride', 'padding', 'dilation', 'groups', 'deform_groups', 'bias']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}

    # Initialize list of modules and sub-operations
    modules = []
    sub_operations = []

    # Initialize every 2D deformable convolution sub-operation
    for map_ids, in_channels, out_channels in zip(operation['in'], in_channels_list, out_channels_list):
        module = dcn(in_channels, out_channels, **sub_operation_kwargs)
        sub_operation = {'in_map_ids': map_ids, 'module_id': modules_offset+len(modules)}

        modules.append(module)
        sub_operations.append(sub_operation)
        feat_sizes.append(out_channels)

    return modules, sub_operations, feat_sizes


def initialize_featurenorm(operation, feat_sizes, modules_offset):
    """
    Initializes featurenorm operation with corresponding modules for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 1 for map_ids in operation['in']):
        raise ValueError(f"Exactly one map must be provided per list in {operation['in']}.")

    # Get featurenorm keyword arguments
    allowed_keys = ['num_groups', 'eps', 'affine']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}

    # Initialize list of modules and sub-operations
    modules = []
    sub_operations = []

    # Initialize every featurenorm sub-operation
    for map_ids in operation['in']:
        num_channels = feat_sizes[map_ids[0]]
        module = FeatureNorm(num_channels=num_channels, **sub_operation_kwargs)
        sub_operation = {'in_map_ids': map_ids, 'module_id': modules_offset+len(modules)}

        modules.append(module)
        sub_operations.append(sub_operation)
        feat_sizes.append(feat_sizes[map_ids[0]])

    return modules, sub_operations, feat_sizes


def initialize_groupnorm(operation, feat_sizes, modules_offset):
    """
    Initializes groupnorm operation with corresponding modules for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 1 for map_ids in operation['in']):
        raise ValueError(f"Exactly one map must be provided per list in {operation['in']}.")

    # Get groupnorm keyword arguments
    allowed_keys = ['num_groups', 'eps', 'affine']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}

    # Initialize list of modules and sub-operations
    modules = []
    sub_operations = []

    # Initialize every groupnorm sub-operation
    for map_ids in operation['in']:
        num_channels = feat_sizes[map_ids[0]]
        module = nn.GroupNorm(num_channels=num_channels, **sub_operation_kwargs)
        sub_operation = {'in_map_ids': map_ids, 'module_id': modules_offset+len(modules)}

        modules.append(module)
        sub_operations.append(sub_operation)
        feat_sizes.append(feat_sizes[map_ids[0]])

    return modules, sub_operations, feat_sizes


def initialize_interpolate(operation, feat_sizes):
    """
    Initializes interpolate operation for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) not in [1, 2] for map_ids in operation['in']):
        raise ValueError(f"One or two maps must be provided per list in {operation['in']}.")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['size', 'scale_factor', 'mode', 'align_corners', 'recompute_scale_factor']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Define shape interpolate if required
    if operation.setdefault('shape', False):
        def shape_interpolate(x, y, **kwargs):
            return F.interpolate(x, size=y.shape[-2:], **kwargs)

    # Initialize every interpolate operation
    for map_ids in operation['in']:
        function = shape_interpolate if operation['shape'] else F.interpolate
        sub_operation = {'function': function, 'in_map_ids': map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_ids[0]]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_layer(layer_operation, feat_sizes, layers, modules_offset):
    """
    Initializes layer operation with corresponding modules for forward computation.

    Args:
        layer_operation (Dict): Dictionary specifying the layer operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        layers (List): List of layers with each of them grouping multiple operations into a single structure.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        layer_modules (List): List of initialized modules used by this layer operation.
        layer_operations (List): List of operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when layer operation with multiple layers does not contain 'out_to_in' key.
        ValueError: Error when shapes of 'out' and 'out_to_in' keys do not match.

        ValueError: Error when unknown layer name is provided.
        ValueError: Error when multiple layers have the same name.
        ValueError: Error when multiple or no sets of inputs are provided by the layer structure.
        ValueError: Error when multiple or no sets of outputs are provided by the layer structure.
        ValueError: Error when input sizes from layer operation do not match layer structure input size.
    """

    # Check layer operation
    if layer_operation.setdefault('num_layers', 1) > 1:
        if 'out_to_in' not in layer_operation:
            raise ValueError(f"Layer operation {layer_operation} with multiple layers must contain 'out_to_in' key.")

        out_shape = [len(map_ids) for map_ids in layer_operation['out']]
        out_to_in_shape = [len(map_ids) for map_ids in layer_operation['out_to_in']]

        if out_shape != out_to_in_shape:
            raise ValueError(f"Layer operation {layer_operation} must have 'out' and 'out_to_in' keys of equal shape.")

    # Get and check layer
    valid_layers = [layer for layer in layers if layer['name'] == layer_operation['name']]

    if len(valid_layers) == 0:
        raise ValueError(f"No layer found with name '{layer_operation['name']}' in {layers}.")
    elif len(valid_layers) == 1:
        layer = valid_layers[0]
    else:
        raise ValueError(f"Multiple layers found with name '{layer_operation['name']}' in {layers}.")

    if len(layer['in']) != 1:
        raise ValueError(f"Exacly one set of inputs must be provided by the layer structure (got {layer['in']}).")
    if len(layer['out']) != 1:
        raise ValueError(f"Exacly one set of outputs must be provided by the layer structure (got {layer['out']}).")
    if any(len(map_ids) != len(layer['in'][0]) for map_ids in layer_operation['in']):
        raise ValueError(f"Input sizes {layer_operation['in']} must match layer structure input size {layer['in']}.")

    # Get layer to layer filter ids when multiple layers are present
    if layer_operation['num_layers'] > 1:
        out_ids = [map_id for map_ids in layer_operation['out'] for map_id in map_ids]
        in_ids = [map_id for map_ids in layer_operation['out_to_in'] for map_id in map_ids]
        layer_filter_ids = list(range(len(feat_sizes)))

        for i in range(len(feat_sizes)):
            if i in in_ids:
                layer_filter_ids[i] = out_ids[in_ids.index(i)]

    # Get local map ids and number of input maps
    local_in_ids = layer['in'][0]
    local_out_ids = layer['out'][0]
    num_in_maps = len(local_in_ids)

    # Initialize layer modules and layer operations
    layer_modules = []
    layer_operations = []

    # Register every operation from layer structure with their corresponding modules
    for layer_id in range(layer_operation['num_layers']):
        if layer_id > 0:
            layer_operations.extend([{'filter_ids': layer_filter_ids}])
            feat_sizes = [feat_sizes[i] for i in layer_filter_ids]

        for global_in_ids in layer_operation['in']:
            num_maps = len(feat_sizes)
            local_to_global_dict = {k: v for k, v in zip(local_in_ids, global_in_ids)}
            local_to_global_dict['offset'] = num_maps - num_in_maps

            for operation in deepcopy(layer['operations']):
                operation = prepare_operation(operation, local_to_global_dict)

                offset = modules_offset + len(layer_modules)
                modules, sub_operations, feat_sizes = initialize_operation(operation, feat_sizes, layers, offset)
                layer_modules.extend(modules)
                layer_operations.extend(sub_operations)

            filter_ids = list(range(num_maps)) + [num_maps+map_id-num_in_maps for map_id in local_out_ids]
            layer_operations.append({'filter_ids': filter_ids})
            feat_sizes = [feat_sizes[i] for i in filter_ids]

    return layer_modules, layer_operations, feat_sizes


def initialize_mul(operation, feat_sizes):
    """
    Initializes multiplication operation for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 1 for map_ids in operation['in']):
        raise ValueError(f"Exactly one map must be provided per list in {operation['in']}.")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['inplace', 'other']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Define multiplication function
    def mul(x, inplace=False, **kwargs):
        if inplace:
            return x.mul_(**kwargs)
        else:
            return x.mul(**kwargs)

    # Initialize every multiplication operation
    for map_ids in operation['in']:
        sub_operation = {'function': mul, 'in_map_ids': map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_ids[0]]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_relu(operation, feat_sizes):
    """
    Initializes ReLU operation for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 1 for map_ids in operation['in']):
        raise ValueError(f"Exactly one map must be provided per list in {operation['in']}.")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['inplace']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Initialize every ReLU operation
    for map_ids in operation['in']:
        sub_operation = {'function': F.relu, 'in_map_ids': map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_ids[0]]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_sub(operation, feat_sizes):
    """
    Initializes subtraction operation for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.

    Returns:
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when incorrect input is provided for the 'in' operation key.
    """

    # Check 'in' operation key
    if any(len(map_ids) != 2 for map_ids in operation['in']):
        raise ValueError(f"Exactly two maps must be provided per list in {operation['in']}.")

    # Get sub-operation keyword arguments and initialize empty list of sub-operations
    allowed_keys = ['alpha']
    sub_operation_kwargs = {k: v for k, v in operation.items() if k in allowed_keys}
    sub_operations = []

    # Initialize every subtraction operation
    for map_ids in operation['in']:
        sub_operation = {'function': torch.sub, 'in_map_ids': map_ids, **sub_operation_kwargs}
        feat_size = feat_sizes[map_ids[0]]

        sub_operations.append(sub_operation)
        feat_sizes.append(feat_size)

    return sub_operations, feat_sizes


def initialize_operation(operation, feat_sizes, layers, modules_offset):
    """
    Initializes operation with corresponding modules for forward computation.

    Args:
        operation (Dict): Dictionary specifying the operation to be performed.
        feat_sizes (List): List of feature sizes of feature maps to be expected during forward computation.
        layers (List): List of layers with each of them grouping multiple operations into a single structure.
        modules_offset (int): Integer containing the offset to be added to the module indices.

    Returns:
        modules (List): List of initialized modules used by this operation.
        sub_operations (List): List of sub-operation dictionaries specifying their behavior during forward computation.
        feat_sizes (List): Updated list of feature sizes of feature maps to be generated during forward computation.

    Raises:
        ValueError: Error when unknown operation type was provided.
    """

    # Prepare operation
    operation = prepare_operation(operation)

    # Get operation type and initialize empty list of modules
    operation_type = operation['type']
    modules = []

    # Initialize operation
    if operation_type == 'add':
        sub_operations, feat_sizes = initialize_add(operation, feat_sizes)
    elif operation_type == 'attn2d':
        modules, sub_operations, feat_sizes = initialize_attn2d(operation, feat_sizes, modules_offset)
    elif operation_type == 'conv2d':
        modules, sub_operations, feat_sizes = initialize_conv2d(operation, feat_sizes, modules_offset)
    elif operation_type == 'dcn':
        modules, sub_operations, feat_sizes = initialize_dcn(operation, feat_sizes, modules_offset)
    elif operation_type == 'featurenorm':
        modules, sub_operations, feat_sizes = initialize_featurenorm(operation, feat_sizes, modules_offset)
    elif operation_type == 'groupnorm':
        modules, sub_operations, feat_sizes = initialize_groupnorm(operation, feat_sizes, modules_offset)
    elif operation_type == 'interpolate':
        sub_operations, feat_sizes = initialize_interpolate(operation, feat_sizes)
    elif operation_type == 'layer':
        modules, sub_operations, feat_sizes = initialize_layer(operation, feat_sizes, layers, modules_offset)
    elif operation_type == 'mul':
        sub_operations, feat_sizes = initialize_mul(operation, feat_sizes)
    elif operation_type == 'relu':
        sub_operations, feat_sizes = initialize_relu(operation, feat_sizes)
    elif operation_type == 'sub':
        sub_operations, feat_sizes = initialize_sub(operation, feat_sizes)
    else:
        raise ValueError(f'Unknown operation type {operation_type} was provided.')

    return modules, sub_operations, feat_sizes
