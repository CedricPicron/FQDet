"""
General Vision Decoder (GVD) head.
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.build import build_model, MODELS


@MODELS.register_module()
class GVD(nn.Module):
    """
    Class implementing the General Vision Decoder (GVD) head.

    Attributes:
        group_init_mode (str): String containing the group initialization mode.

        If group_init_mode is 'learned':
            group_init_feats (Parameter): Parameter with group initialization features [num_groups, group_feat_size].

        If group_init_mode is 'selected':
            group_init_sel (nn.Module): Module obtaining group initialization features by selecting from input maps.

        dec_layers (nn.ModuleList): List of size [num_dec_layers] containing the decoder layers.
        heads (nn.ModuleList): List of size [num_heads] containing the heads.
    """

    def __init__(self, group_init_cfg, dec_layer_cfg, num_dec_layers, head_cfgs, metadata, head_apply_ids=None):
        """
        Initializes the GVD head.

        Args:
            group_init_cfg (Dict): Configuration dictionary specifying the group initialization.
            dec_layer_cfg (Dict): Configuration dictionary specifying a single decoder layer.
            num_dec_layers (int): Integer containing the number decoder layers.
            head_cfgs (List): List of size [num_heads] with the configuration dictionaries specifying the heads.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            head_apply_ids (List): List with integers determining when heads should be applied (default=None).

        Raises:
            ValueError: Error when an invalid group initialization mode is provided.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes related to group initialization
        self.group_init_mode = group_init_cfg['mode']

        if self.group_init_mode == 'learned':
            num_groups = group_init_cfg['num_groups']
            group_feat_size = group_init_cfg['feat_size']
            param_data = torch.randn(num_groups, group_feat_size)
            self.group_init_feats = Parameter(param_data, requires_grad=True)

        elif self.group_init_mode == 'selected':
            sel_cfg = group_init_cfg['sel_cfg']
            self.group_init_sel = build_model(sel_cfg)

        else:
            error_msg = f"Invalid group initialization mode (got '{self.group_init_mode}')."
            raise ValueError(error_msg)

        # Build list with decoder layers
        self.dec_layers = nn.ModuleList([build_model(dec_layer_cfg, sequential=True) for _ in range(num_dec_layers)])

        # Set attributes related to the heads
        self.heads = nn.ModuleList([build_model(head_cfg, metadata=metadata) for head_cfg in head_cfgs])

        for head in self.heads:
            if getattr(head, 'apply_ids', None) is None:
                head.apply_ids = head_apply_ids

    def group_init(self, storage_dict, **kwargs):
        """
        Method performing group initialization, i.e. obtaining the initial group features.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - feat_maps (List): list [num_maps] with maps of shape [batch_size, feat_size, fH, fW].

            kwargs (Dict): Dictionary of additional keyword arguments passed to underlying modules.

        Returns:
            group_init_feats (FloatTensor): Group initialization features of shape [num_groups, group_feat_size].

        Raises:
            ValueError: Error when an invalid group initialization mode is provided.
        """

        # Perform group initialization
        if self.group_init_mode == 'learned':
            feat_maps = storage_dict['feat_maps']

            batch_size = feat_maps[0].size(dim=0)
            group_init_feats = self.group_init_feats[None, :, :].expand(batch_size, -1, -1)
            group_init_feats = group_init_feats.flatten(0, 1)

            device = self.group_init_feats.device
            num_feats_batch = self.group_init_feats.size(dim=0)
            cum_feats_batch = torch.arange(batch_size+1, device=device) * num_feats_batch
            storage_dict['cum_feats_batch'] = cum_feats_batch

        elif self.group_init_mode == 'selected':
            self.group_init_sel(storage_dict=storage_dict, **kwargs)

            group_init_feats = storage_dict.pop('sel_feats')
            storage_dict['prior_boxes'] = storage_dict.pop('sel_boxes', None)
            storage_dict['add_encs'] = storage_dict.pop('sel_box_encs', None)

        else:
            error_msg = f"Invalid group initialization mode (got '{self.group_init_mode}')."
            raise ValueError(error_msg)

        return group_init_feats

    def forward(self, feat_maps, tgt_dict=None, images=None, visualize=False, **kwargs):
        """
        Forward method of the GVD head.

        Args:
            feat_maps (List): List of size [num_maps] with feature maps of shape [batch_size, feat_size, fH, fW].

            tgt_dict (Dict): Optional target dictionary used during trainval possibly containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (BoolTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            images (Images): Images structure of size [batch_size] containing the batched images (default=None).
            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of additional keyword arguments passed to some underlying modules and methods.

        Returns:
            return_list (List): List of size [num_returns] possibly containing following items to return:
                - pred_dicts (List): list of size [num_pred_dicts] with prediction dictionaries (evaluation only);
                - loss_dict (Dict): dictionary with different weighted loss terms used during training (trainval only);
                - analysis_dict (Dict): dictionary with different analyses used for logging purposes only;
                - images_dict (Dict): dictionary with annotated images of predictions/targets (when visualize is True).
        """

        # Initialize storage, loss, analysis and prediction dictionaries
        storage_dict = {'feat_maps': feat_maps, 'images': images}
        loss_dict = {} if tgt_dict is not None else None
        analysis_dict = {}

        # Initialize empty list for prediction dictionaries
        pred_dicts = [] if not self.training else None

        # Initialize empty dictionary for images with visualizations
        images_dict = {} if visualize else None

        # Collect above dictionaries and list into a single dictionary
        local_kwargs = {'storage_dict': storage_dict, 'tgt_dict': tgt_dict, 'loss_dict': loss_dict}
        local_kwargs = {**local_kwargs, 'analysis_dict': analysis_dict, 'pred_dicts': pred_dicts}
        local_kwargs = {**local_kwargs, 'images_dict': images_dict}

        # Perform group initialization
        group_feats = self.group_init(**local_kwargs, **kwargs)
        local_kwargs['qry_feats'] = group_feats

        # Get and add batch indices to storage dictionary
        cum_feats_batch = storage_dict['cum_feats_batch']
        batch_size = len(cum_feats_batch) - 1

        batch_ids = torch.arange(batch_size, device=cum_feats_batch.device)
        batch_ids = batch_ids.repeat_interleave(cum_feats_batch.diff())
        storage_dict['batch_ids'] = batch_ids

        # Apply heads if needed
        for head in self.heads:
            if 0 in head.apply_ids:
                head(mode='pred', id=0, **local_kwargs, **kwargs)

        if tgt_dict is not None:
            for head in self.heads:
                if 0 in head.apply_ids:
                    head(mode='loss', id=0, **local_kwargs, **kwargs)

        # Iterate over decoder layers and apply heads when needed
        for dec_id, dec_layer in enumerate(self.dec_layers, 1):

            # Apply decoder layer
            group_feats = dec_layer(group_feats, **local_kwargs, **kwargs)
            local_kwargs['qry_feats'] = group_feats

            # Apply heads if needed
            for head in self.heads:
                if dec_id in head.apply_ids:
                    head(mode='pred', id=dec_id, **local_kwargs, **kwargs)

            if tgt_dict is not None:
                for head in self.heads:
                    if dec_id in head.apply_ids:
                        head(mode='loss', id=dec_id, **local_kwargs, **kwargs)

        # Get list with items to return
        return_list = [analysis_dict]
        return_list.insert(0, loss_dict) if tgt_dict is not None else None
        return_list.insert(0, pred_dicts) if not self.training else None
        return_list.append(images_dict) if visualize else None

        return return_list
