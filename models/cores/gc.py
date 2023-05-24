"""
GC (Generalized Core) core.
"""
from copy import deepcopy
from pathlib import Path

from torch import nn
from yaml import safe_load

from models.build import MODELS
from models.functional.gc import initialize_operation


@MODELS.register_module()
class GC(nn.Module):
    """
    Class implementing the GC (Generalized Core) module.

    Attributes:
        modules_list (nn.ModuleList): List with modules used by the GC operations.
        operations (List): List with operations transforming input feature maps into output feature maps.

        out_ids (List): List [num_out_maps] containing the indices of the output feature maps.
        out_sizes (List): List [num_out_maps] containing the feature sizes of the output feature maps.
    """

    def __init__(self, in_ids, in_sizes, core_ids, yaml_file):
        """
        Initializes the GC module.

        Args:
            in_ids (List): List [num_in_maps] containing the indices of the input feature maps.
            in_sizes (List): List [num_in_maps] containing the feature sizes of the input feature maps.
            core_ids (List): List [num_maps] containing the indices of the core feature maps.
            yaml_file (str): String containing the path of the yaml-file with the GC specification.

        Raises:
            ValueError: Error when the 'in_ids' length and the 'in_sizes' length do not match.
            ValueError: Error when the 'in_ids' list does not match the first elements of the 'core_ids' list.
        """

        # Check inputs
        if len(in_ids) != len(in_sizes):
            error_msg = f"The 'in_ids' length ({len(in_ids)}) must match the 'in_sizes' length ({len(in_sizes)})."
            raise ValueError(error_msg)

        if in_ids != core_ids[:len(in_ids)]:
            error_msg = f"The 'in_ids' list ({in_ids}) must match the first elements of 'core_ids' list ({core_ids})."
            raise ValueError(error_msg)

        # Initialization of default nn.Module
        super().__init__()

        # Convert yaml-file string to Path object
        yaml_file = Path(yaml_file)

        # Get GC specification dictionary from yaml-file
        with open(yaml_file, 'r') as stream:
            gc_dict = safe_load(stream)

        # Get expected feature sizes from input maps and get layers
        feat_sizes = list(deepcopy(in_sizes))
        layers = gc_dict.setdefault('layers', [])

        # Initialize modules and operations attributes
        self.modules_list = nn.ModuleList([])
        self.operations = []

        # Register every operation with their corresponding modules
        for operation in gc_dict['operations']:
            modules_offset = len(self.modules_list)
            modules, sub_operations, feat_sizes = initialize_operation(operation, feat_sizes, layers, modules_offset)
            self.modules_list.extend(modules)
            self.operations.extend(sub_operations)

        # Append operation getting the final output feature maps
        filter_ids = gc_dict['outputs']
        self.operations.append({'filter_ids': filter_ids})
        feat_sizes = [feat_sizes[i] for i in filter_ids]

        # Set attributes related to output feature maps
        self.out_ids = core_ids
        self.out_sizes = feat_sizes

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the GC module.

        Args:
            in_feat_maps (List): Input feature maps [num_in_maps] of shape [batch_size, feat_size, fH, fW].

        Returns:
            feat_maps (List): Output feature maps [num_out_maps] of shape [batch_size, feat_size, fH, fW].

        Raises:
            ValueError: Error for operation dictionary without key from {'filter_ids', 'module_id', 'function'}.
        """

        # Initialize list of feature maps
        feat_maps = in_feat_maps.copy()

        # Perform every operation
        for operation in self.operations:
            if 'filter_ids' in operation:
                feat_maps = [feat_maps[i] for i in operation['filter_ids']]
                continue

            elif 'module_id' in operation:
                function = self.modules_list[operation['module_id']]

            elif 'function' in operation:
                function = operation['function']

            else:
                required_keys = {'filter_ids', 'module_id', 'function'}
                raise ValueError(f"Operation dictonary should contain a key from {required_keys}.")

            inputs = [feat_maps[i] for i in operation['in_map_ids']]
            kwargs = {k: v for k, v in operation.items() if k not in ['module_id', 'function', 'in_map_ids']}
            outputs = function(*inputs, **kwargs)

            if isinstance(outputs, tuple):
                feat_maps.extend(outputs)
            else:
                feat_maps.append(outputs)

        return feat_maps
