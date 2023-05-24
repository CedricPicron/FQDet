"""
Collection of utility modules.
"""

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class ApplyAll(nn.Module):
    """
    Class implementing the ApplyAll module.

    The ApplyAll module applies its underlying module to all inputs from the given input list.

    Attributes:
        module (Sequential): Underlying module applied to all inputs from the input list.
    """

    def __init__(self, module_cfg, **kwargs):
        """
        Initializes the ApplyAll module.

        Args:
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg, sequential=True)

    def forward(self, in_list, **kwargs):
        """
        Forward method of the ApplyAll module.

        Args:
            in_list (List): List with inputs to be processed by the underlying module.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            out_list (List): List with resulting outputs of the underlying module.
        """

        # Apply underlying module to all inputs from input list
        out_list = [self.module(input, **kwargs) for input in in_list]

        return out_list


@MODELS.register_module()
class ApplyToSelected(nn.Module):
    """
    Class implementing the ApplyToSelected module.

    The ApplyToSelected module applies the underlying module to the selected input from the given input list.

    Attributes:
        select_id (int): Integer containing the index of the selected input.
        module (nn.Module): Underlying module applied to the selected input.
    """

    def __init__(self, select_id, module_cfg):
        """
        Initializes the ApplyToSelected module.

        Args:
            select_id (int): Integer containing the index of the selected input.
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set select_id attribute
        self.select_id = select_id

        # Build underlying module
        self.module = build_model(module_cfg)

    def forward(self, in_list, **kwargs):
        """
        Forward method of the ApplyToSelected module.

        Args:
            in_list (List): Input list containing the selected input to be processed by the underlying module.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_list (List): Output list replacing the selected input by its processed output.
        """

        # Get output list
        out_list = in_list.copy()
        out_list[self.select_id] = self.module(in_list[self.select_id])

        return out_list


@MODELS.register_module()
class ApplyOneToOne(nn.Module):
    """
    Class implementing the ApplyOneToOne module.

    The ApplyOneToOne module applies one specific sub-module to each input from the input list.

    Attributes:
        sub_modules (nn.ModuleList): List [num_sub_modules] of sub-modules applied to inputs from input list.
    """

    def __init__(self, sub_module_cfgs):
        """
        Initializes the ApplyOneToOne module.

        Args:
            sub_module_cfgs (List): List [num_sub_modules] of configuration dictionaries specifying the sub-modules.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build sub-modules
        self.sub_modules = nn.ModuleList([build_model(cfg_i, sequential=True) for cfg_i in sub_module_cfgs])

    def forward(self, in_list, **kwargs):
        """
        Forward method of the ApplyOneToOne module.

        Args:
            in_list (List): List with inputs to be processed by the sub-module.
            kwargs (Dict): Dictionary of keyword arguments passed to each sub-module.

        Returns:
            out_list (List): List with resulting outputs of the sub-modules.
        """

        # Apply sub-modules to inputs from input list
        out_list = [module_i(in_i) for in_i, module_i in zip(in_list, self.sub_modules)]

        return out_list


@MODELS.register_module()
class BottomUp(nn.Module):
    """
    Class implementing the BottomUp module.

    Attributes:
        bu (nn.Module): Module computing the residual bottum-up features.
    """

    def __init__(self, bu_cfg):
        """
        Initializes the BottomUp module.

        Args:
            bu_cfg (Dict): Configuration dictionary specifying the residual bottom-up module.
        """

        # Iniialization of default nn.Module
        super().__init__()

        # Build residual bottom-up module
        self.bu = build_model(bu_cfg)

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the BottomUp module.

        Args:
            in_feat_maps (List): List of size [num_maps] with input feature maps.
            kwargs (Dict): Dictionary of keyword arguments passed to the residual bottom-up module.

        Returns:
            out_feat_maps (List): List of size [num_maps] with output feature maps.
        """

        # Get list with output feature maps
        num_maps = len(in_feat_maps)
        out_feat_list = [in_feat_maps[i+1] + self.bu(in_feat_maps[i], **kwargs) for i in range(num_maps-1)]

        return out_feat_list


@MODELS.register_module()
class GetApplyInsert(nn.Module):
    """
    Class implementing the GetApplyInsert module.

    The GetApplyInsert module applies its underlying module to the selected input from the given input list, and
    inserts the resulting output back into the given input list.

    Attributes:
        get_id (int): Index selecting the input for the underlying module from a given input list.
        module (nn.Module): Underlying module applied on the selected input from the given input list.
        insert_id (int): Index inserting the output of the underlying module into the given input list.
    """

    def __init__(self, module_cfg, get_id, insert_id):
        """
        Initializes the GetApplyInsert module.

        Args:
            module_cfg (Dict): Configuration dictionary specifying the underlying module.
            get_id (int): Index selecting the input for the underlying module from a given input list.
            insert_id (int): Index inserting the output of the underlying module into the given input list.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build underlying module
        self.module = build_model(module_cfg)

        # Set index attributes
        self.get_id = get_id
        self.insert_id = insert_id

    def forward(self, in_list, **kwargs):
        """
        Forward method of the GetApplyInsert module.

        Args:
            in_list (List): Input list of size [in_size] containing the input for the underlying module.
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying module.

        Returns:
            in_list (List): Updated input list of size [in_size+1] additionally containing output of underlying module.
        """

        # Get input for underlying module from input list
        input = in_list[self.get_id]

        # Apply underlying module
        output = self.module(input, **kwargs)

        # Insert output from underying module into input list
        in_list.insert(self.insert_id, output)

        return in_list


@MODELS.register_module()
class GetBoxesTensor(nn.Module):
    """
    Class implementing the GetBoxesTensor module.

    Attributes:
        boxes_key (str): String with key to retrieve Boxes structure from storage dictionary (or None).
        clone (bool): Boolean indicating whether to clone Boxes structure.
        detach (bool): Boolean indicating whether to detach Boxes structure.
        format (str): String containing the desired bounding box format (or None).
        normalize (str): String containing the desired bounding box normalization (or None).
    """

    def __init__(self, boxes_key=None, clone=False, detach=False, format=None, normalize=None):
        """
        Initializes the GetBoxesTensor module.

        Args:
            boxes_key (str): String with key to retrieve Boxes structure from storage dictionary (default=None).
            clone (bool): Boolean indicating whether to clone Boxes structure (default=False).
            detach (bool): Boolean indicating whether to detach Boxes structure (default=False).
            format (str): String containing the desired bounding box format (default=None).
            normalize (str): String containing the desired bounding box normalization (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.boxes_key = boxes_key
        self.clone = clone
        self.detach = detach
        self.format = format
        self.normalize = normalize

    def forward(self, *args, storage_dict=None, **kwargs):
        """
        Forward method of the GetBoxesTensor module.

        Args:
            args (Tuple): Tuple with positional arguments (possibly) containing following items:
                - 0: Boxes structure with axis-aligned bounding boxes of size [num_boxes];
                - 1: Images structure containing the batched images of size [batch_size].

            storage_dict (Dict): Optional storage dictionary (possibly) containing following keys:
                - {self.boxes_key} (Boxes): Boxes structure with axis-aligned bounding boxes of size [num_boxes];
                - images (Images): Images structure containing the batched images of size [batch_size].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            boxes_tensor (FloatTensor): Tensor containing the bounding boxes of shape [num_boxes, 4].

        Raises:
            ValueError: Error when an invalid normalization string is provided.
        """

        # Get Boxes structure
        boxes = args[0] if self.boxes_key is None else storage_dict[self.boxes_key]

        # Transform Boxes structure if needed
        if self.clone:
            boxes = boxes.clone()

        if self.detach:
            boxes = boxes.detach()

        if self.format is not None:
            boxes = boxes.to_format(self.format)

        if self.normalize is not None:
            images = args[1] if self.boxes_key is None else storage_dict['images']

            if self.normalize == 'false':
                boxes = boxes.to_img_scale(images)

            elif self.normalize == 'with_padding':
                boxes = boxes.normalize(images, with_padding=True)

            elif self.normalize == 'without_padding':
                boxes = boxes.normalize(images, with_padding=False)

            else:
                error_msg = f"Invalid normalization string in GetBoxesTensor (got {self.normalize})."
                raise ValueError(error_msg)

        # Get tensor with bounding boxes
        boxes_tensor = boxes.boxes

        return boxes_tensor


@MODELS.register_module()
class IdAvg2d(nn.Module):
    """
    Class implementing the IdAvg2d module.
    """

    def __init__(self):
        """
        Initializes the IdAvg2d module.
        """

        # Initialization of default nn.Module
        super().__init__()

    def forward(self, core_feats, aux_feats, id_map, **kwargs):
        """
        Forward method of the IdAvg2d module.

        Args:
            core_feats (FloatTensor): Core features of shape [num_core_feats, feat_size].
            aux_feats (FloatTensor): Auxiliary features of shape [num_aux_feats, feat_size].
            id_map (LongTensor): Index map with feature indices of shape [num_rois, rH, rW].
            kwargs (Dict): Dictionary with unused keyword arguments.

        Returns:
            avg_feat (FloatTensor): Average feature of shape [1, feat_size].
        """

        # Get average feature
        num_core_feats = len(core_feats)
        num_aux_feats = len(aux_feats)
        num_feats = num_core_feats + num_aux_feats

        counts = torch.bincount(id_map.flatten(), minlength=num_feats)
        counts = counts[:, None]

        avg_feat = (counts[:num_core_feats] * core_feats).sum(dim=0, keepdim=True)
        avg_feat += (counts[num_core_feats:] * aux_feats).sum(dim=0, keepdim=True)
        avg_feat *= 1/id_map.numel()

        return avg_feat


@MODELS.register_module()
class SkipConnection(nn.Module):
    """
    Class implementing the SkipConnection module.

    Attributes:
        res (nn.Module): Module computing the residual features from the input features.
    """

    def __init__(self, res_cfg):
        """
        Initializes the SkipConnection module.

        Args:
            res_cfg (Dict): Configuration dictionary specifying the residual module.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build residual module
        self.res = build_model(res_cfg)

    def forward(self, in_feats, **kwargs):
        """
        Forward method of the SkipConnection module.

        Args:
            in_feats (FloatTensor): Input features of shape [num_feats, feat_size].
            kwargs (Dict): Dictionary of keyword arguments passed to the residual module.

        Returns:
            out_feats (FloatTensor): Output features of shape [num_feats, feat_size].
        """

        # Get output features
        out_feats = in_feats + self.res(in_feats, **kwargs)

        return out_feats


@MODELS.register_module()
class TopDown(nn.Module):
    """
    Class implementing the TopDown module.

    Attributes:
        td (nn.Module): Module computing the residual top-down features.
    """

    def __init__(self, td_cfg):
        """
        Initializes the TopDown module.

        Args:
            td_cfg (Dict): Configuration dictionary specifying the residual top-down module.
        """

        # Iniialization of default nn.Module
        super().__init__()

        # Build residual top-down module
        self.td = build_model(td_cfg)

    def forward(self, in_feat_maps, **kwargs):
        """
        Forward method of the TopDown module.

        Args:
            in_feat_maps (List): List of size [num_maps] with input feature maps.
            kwargs (Dict): Dictionary of keyword arguments passed to the residual top-down module.

        Returns:
            out_feat_maps (List): List of size [num_maps] with output feature maps.
        """

        # Get list with output feature maps
        num_maps = len(in_feat_maps)
        out_feat_list = [in_feat_maps[i] + self.td(in_feat_maps[i+1], **kwargs) for i in range(num_maps-1)]

        return out_feat_list


@MODELS.register_module()
class View(nn.Module):
    """
    Class implementing the View module.

    Attributes:
        out_shape (Tuple): Tuple containing the output shape.
    """

    def __init__(self, out_shape):
        """
        Initializes the View module.

        Args:
            out_shape (Tuple): Tuple containing the output shape.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set output shape attribute
        self.out_shape = out_shape

    def forward(self, in_tensor, **kwargs):
        """
        Forward method of the View module.

        Args:
            in_tensor (Tensor): Input tensor of shape [*in_shape].
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_tensor (Tensor): Output tensor of shape [*out_shape].
        """

        # Get output tensor
        out_tensor = in_tensor.view(*self.out_shape)

        return out_tensor
