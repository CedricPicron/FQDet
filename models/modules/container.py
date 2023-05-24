"""
Collection of container-type modules.
"""
from copy import deepcopy
from inspect import signature

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class ModuleCat(nn.Module):
    """
    Class implementing the ModuleCat module.

    The ModuleCat module contains a list of sub-modules, where each of the sub-modules are applied on the input and
    where the resulting output tensors are concatenated along the desired dimension to yield the final output tensor
    of the ModuleCat module.

    Attributes:
        sub_modules (nn.ModuleList): List of size [num_sub_modules] containing the sub-modules.
        dim (int): Integer containing the dimension along which to concatenate.
    """

    def __init__(self, sub_module_cfgs, dim=0):
        """
        Initializes the ModuleCat module.

        Args:
            sub_module_cfgs (List): List of configuration dictionaries specifying the sub-modules.
            dim (int): Integer containing the dimension along which to concatenate (default=0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build list with sub-modules
        self.sub_modules = nn.ModuleList([build_model(sub_module_cfg) for sub_module_cfg in sub_module_cfgs])

        # Set dim attribute
        self.dim = dim

    def forward(self, *args, **kwargs):
        """
        Forward method of the ModuleCat module.

        Args:
            args (Tuple): Tuple containing positional arguments passed to the underlying sub-modules.
            kwargs (Dict): Dictionary containing keyword arguments passed to the underlying sub-modules.

        Returns:
            out_tensor (Tensor): Tensor obtained by concatenating the output tensors of the individual sub-modules.
        """

        # Get output tensor
        output = torch.cat([sub_module(*args, **kwargs) for sub_module in self.sub_modules], dim=self.dim)

        return output


@MODELS.register_module()
class ModuleSelector(nn.Module):
    """
    Class implementing the ModuleSelector module.

    The ModuleSelector contains a list of modules, with each module having the same architecture, but different
    weights. During the forward pass, a module is selected either for all of the input features when 'module_id' is
    provided, either for every specific input feature when 'module_ids' is

    Attributes:
        module_list (nn.ModuleList): List of size [num_modules] containing the modules to choose from.
        out_feat_size (int): Integer containing the output feature size (or None).
    """

    def __init__(self, module_cfg, num_modules, out_feat_size=None):
        """
        Initializes the ModuleSelector module.

        Args:
            module_cfg (Dict): Configuration dictionary specifying the architecture of the modules.
            num_modules (int): Integer containing the number of modules to choose from.
            out_feat_size (int): Integer containing the output feature size (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build list with modules to choose from
        self.module_list = nn.ModuleList([build_model(module_cfg) for _ in range(num_modules)])

        # Try inferring the output feature size if needed
        if out_feat_size is None:
            try:
                out_feat_size = module_cfg.get('out_size', None)
            except AttributeError:
                pass

        # Set attribute containing the output feature size
        self.out_feat_size = out_feat_size

    def forward(self, in_feats, module_id=None, module_ids=None, **kwargs):
        """
        Forward method of the ModuleSelector module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, in_feat_size]
            module_ids (LongTensor): Module indices corresponding to each input feature of shape [num_feats].
            kwargs (Dict): Dictionary of keyword arguments passed to the selected module or modules.

        Returns:
            out_feats (FloatTensor): Output features of shape [num_feats, out_feat_size].

        Raises:
            ValueError: Error when neither the 'module_id' nor the 'module_ids' arguments were provided.
            ValueError: Error when both the 'module_id' and the 'module_ids' arguments were provided.
            ValueError: Error when the 'out_feat_size' wasn't set during initialization when using 'module_ids'.
        """

        # Check inputs
        if (module_id is None) and (module_ids is None):
            error_msg = "Either the 'module_id' or the 'module_ids' argument must be provided, but both are missing."
            raise ValueError(error_msg)

        elif (module_id is not None) and (module_ids is not None):
            error_msg = "Either the 'module_id' or the 'module_ids' argument must be provided, but both are given."
            raise ValueError(error_msg)

        # Get output features
        if module_id is not None:
            out_feats = self.module_list[module_id](in_feats, **kwargs)

        elif self.out_feat_size is not None:
            num_feats = in_feats.size(dim=0)
            out_feats = in_feats.new_zeros([num_feats, self.out_feat_size])

            for module_id in range(len(self.module_list)):
                apply_mask = module_ids == module_id
                module = self.module_list[module_id]
                out_feats[apply_mask] = module(in_feats[apply_mask], **kwargs)

        else:
            error_msg = "The output feature size must be set during initialization when using 'module_ids'. "
            error_msg += "The output feature size is set during initialization either when it is inferred through the "
            error_msg += "'out_size' key of the module configuration dictionary, or when it is provided by the "
            error_msg += "'out_feat_size' input argument."
            raise ValueError(error_msg)

        return out_feats


@MODELS.register_module()
class ModuleSum(nn.Module):
    """
    Class implementing the ModuleSum module.

    The ModuleSum module contains a list of sub-modules, where each of the sub-modules are applied on the input and
    where the resulting outputs are added to yield the final output of the ModuleSum module.

    Attributes:
        sub_modules (nn.ModuleList): List of size [num_sub_modules] containing the sub-modules.
    """

    def __init__(self, sub_module_cfgs):
        """
        Initializes the ModuleSum module.

        Args:
            sub_module_cfgs (List): List of configuration dictionaries specifying the sub-modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build list with sub-modules
        self.sub_modules = nn.ModuleList([build_model(sub_module_cfg) for sub_module_cfg in sub_module_cfgs])

    def forward(self, *args, **kwargs):
        """
        Forward method of the ModuleSum module.

        Args:
            args (Tuple): Tuple containing positional arguments passed to the underlying sub-modules.
            kwargs (Dict): Dictionary containing keyword arguments passed to the underlying sub-modules.

        Returns:
            output (Any): Output obtained by summing the outputs of the individual sub-modules.
        """

        # Get output
        output = sum(sub_module(*args, **kwargs) for sub_module in self.sub_modules)

        return output


@MODELS.register_module()
class ModuleWeightedSum(nn.Module):
    """
    Class implementing the ModuleWeightedSum module.

    The ModuleWeightedSum module contains a list of sub-modules, where each of the sub-modules are applied on the input
    and where the resulting outputs are added using a weighted sum with weights computed from the input tensor.

    Attributes:
        sub_modules (nn.ModuleList): List of size [num_sub_modules] containing the sub-modules.
        weight (nn.Module): Module computing the weights from the input tensor.
    """

    def __init__(self, sub_module_cfgs, weight_cfg):
        """
        Initializes the ModuleWeightedSum module.

        Args:
            sub_module_cfgs (List): List of configuration dictionaries specifying the sub-modules.
            weight_cfg (Dict): Configuration dictionary specifying the weight module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build list with sub-modules
        self.sub_modules = nn.ModuleList([build_model(sub_module_cfg) for sub_module_cfg in sub_module_cfgs])

        # Build weight module
        self.weight = build_model(weight_cfg)

    def forward(self, in_tensor, *args, **kwargs):
        """
        Forward method of the ModuleWeightedSum module.

        Args:
            in_tensor (FloatTensor): Input tensor of shape [num_feats, *].
            args (Tuple): Tuple containing additional positional arguments passed to the underlying sub-modules.
            kwargs (Dict): Dictionary containing keyword arguments passed to the underlying sub-modules.

        Returns:
            out_tensor (FloatTensor): Output tensor of shape [num_feats, *].
        """

        # Get output tensor
        terms = torch.stack([sub_module(in_tensor, *args, **kwargs) for sub_module in self.sub_modules], dim=2)
        weights = self.weight(in_tensor)[:, None, :]
        out_tensor = (weights * terms).sum(dim=2)

        return out_tensor


@MODELS.register_module()
class Sequential(nn.Sequential):
    """
    Class implementing the enhanced Sequential module.

    It adds following two features to the forward method:
        1) It allows the use of keyword arguments to be passed to the sub-modules depending on their forward signature.
        2) It adds the possibility to return all intermediate outputs from each of the sub-modules.
    """

    def forward(self, input, return_intermediate=False, **kwargs):
        """
        Forward method of the enhanced Sequential module.

        Args:
            input (Any): Input of the forward method.
            return_intermediate (bool): Whether intermediate outputs should be returned (default=False).
            kwargs (Dict): Dictionary of keyword arguments to be passed to each of the sub-modules.

        Returns:
            * If return_intermediate is False:
                output (Any): Output from the final sub-module.

            * If return_intermediate is True:
                output (List): List of size [num_sub_modules] containing the outputs from each of the sub-modules.
        """

        # Initialize empty list of outputs
        output = []

        # Iterate over all sub-modules
        for module in self:
            module_kwargs = {name: kwargs[name] for name in signature(module.forward).parameters if name in kwargs}
            module_kwargs = kwargs if 'kwargs' in signature(module.forward).parameters else module_kwargs

            input = module(input, **module_kwargs)
            kwargs.update(module_kwargs)
            output.append(input)

        # Select output from final sub-module if requested
        if not return_intermediate:
            output = output[-1]

        return output


@MODELS.register_module()
class Stage(nn.Module):
    """
    Class implementing the Stage module.

    This module implements a sequence of block modules, with the possibility to return intermediate block outputs and
    with support for input and output projection modules.

    Attributes:
        in_proj (nn.Module): Optional module implementing the input projection module (None when missing).
        blocks (Sequential): Sequence of block modules.
        out_proj (nn.Module): Optional module implementing the output projection module (None when missing).
        return_inter_blocks (bool): Boolean indicating whether intermediate block outputs should be returned.
    """

    def __init__(self, block_cfg, num_blocks=1, return_inter_blocks=False, in_proj_cfg=None, out_proj_cfg=None):
        """
        Initializes the Stage module.

        Args:
            block_cfg (Dict): Configuration dictionary specifying a single block module.
            num_blocks (int): Integer containing the number of consecutive block modules (default=1).
            return_inter_blocks (bool): Whether intermediate block outputs should be returned (default=False).
            in_proj_cfg (Dict): Configuration dictionary specifying an input projection module (default=None).
            out_proj_cfg (Dict): Configuration dictionary specifying an output projection module (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build sub-modules
        self.in_proj = build_model(in_proj_cfg) if in_proj_cfg is not None else None
        self.blocks = Sequential(*[build_model(block_cfg) for _ in range(num_blocks)])
        self.out_proj = build_model(out_proj_cfg) if out_proj_cfg is not None else None

        # Set return_inter_blocks attribute
        self.return_inter_blocks = return_inter_blocks

    def forward(self, input):
        """
        Forward method of the Stage module.

        Args:
            input (Any): Input of the forward method.

        Returns:
            output (Any) Output of the forward method.
        """

        # Apply input projection if needed
        if self.in_proj is not None:
            input = self.in_proj(input)

        # Apply sequence of block modules
        output = self.blocks(input, return_intermediate=self.return_inter_blocks)

        # Apply output projection if needed
        if self.out_proj is not None:
            if self.return_inter_blocks:
                output[-1] = self.out_proj(output[-1])
            else:
                output = self.out_proj(output)

        return output


@MODELS.register_module()
class Net(nn.Module):
    """
    Class implementing the Net module.

    This module implements a sequence of stage modules, which itself consists of a sequence of block modules.
    The module allows each stage to have a different amount of blocks, and different feature sizes (i.e. a multiple of
    the base feature sizes). The module provides the possibility to return intermediate stage (and block) outputs.

    Attributes:
        stages (nn.ModuleList): List [num_stages] of stage modules.
        return_inter_stages (bool): Boolean indicating whether intermediate stage outputs should be returned.
        return_inter_blocks (bool): Boolean indicating whether intermediate block outputs should be returned.
    """

    def __init__(self, base_stage_cfg, blocks_per_stage, scale_factors, scale_tags, scale_overwrites=None,
                 return_inter_stages=False):
        """
        Initializes the Net module.

        Args:
            base_stage_cfg (Dict): Configuration dictionary specifying the base stage module.
            blocks_per_stage (List): List [num_stages] containing the number of blocks per stage.
            scale_factors (List): List [num_stages] containing the factors scaling the base stage feature sizes.
            scale_tags (List): List [num_tags] of strings tagging keys from base stage configuration to be scaled.
            scale_overwrites (List): List [num_overwrites] of lists with stage-specific overwrites (default=None).
            return_inter_stages (bool): Whether intermediate stage outputs should be returned (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build stage modules
        num_stages = len(blocks_per_stage)
        return_inter_blocks = base_stage_cfg.pop('return_inter_blocks', False)
        scale_overwrites = [] if scale_overwrites is None else scale_overwrites
        stages = []

        for stage_id in range(num_stages):
            num_blocks = blocks_per_stage[stage_id]
            scale_factor = scale_factors[stage_id]

            def cfg_scaling(cfg, scale, tags):
                if isinstance(cfg, dict):
                    for k, v in cfg.items():
                        if isinstance(v, (dict, list)):
                            cfg_scaling(v, scale, tags)
                            continue

                        for tag in tags:
                            if tag in k:
                                cfg[k] = scale * v

                elif isinstance(cfg, list):
                    for cfg_i in cfg:
                        cfg_scaling(cfg_i, scale, tags)

            stage_cfg = deepcopy(base_stage_cfg)
            cfg_scaling(stage_cfg, scale_factor, scale_tags)

            for overwrite in scale_overwrites:
                if overwrite[0] == stage_id:
                    item = stage_cfg

                    for key in overwrite[1:-2]:
                        item = item[key]

                    key, value = overwrite[-2:]
                    item[key] = value

            block_cfg = stage_cfg.pop('block_cfg')
            in_proj_cfg = stage_cfg.pop('in_proj_cfg', None)
            out_proj_cfg = stage_cfg.pop('out_proj_cfg', None)

            stage = Stage(block_cfg, num_blocks, return_inter_blocks, in_proj_cfg, out_proj_cfg)
            stages.append(stage)

        self.stages = nn.ModuleList(stages)

        # Set additional attributes
        self.return_inter_stages = return_inter_stages
        self.return_inter_blocks = return_inter_blocks

    def forward(self, input):
        """
        Forward method of the Net module.

        Args:
            input (Any): Input of the forward method.

        Returns:
            output (Any) Output of the forward method.
        """

        # Apply sequence of stage modules
        if self.return_inter_stages:
            output = []

        for stage in self.stages:
            stage_output = stage(input)

            if self.return_inter_blocks:
                input = stage_output[-1]
            else:
                input = stage_output

            if self.return_inter_stages:
                if self.return_inter_blocks:
                    output.extend(stage_output)
                else:
                    output.append(stage_output)

        if not self.return_inter_stages:
            output = stage_output

        return output
