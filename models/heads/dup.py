"""
Collection of duplicate heads.
"""

import torch
from torch import nn

from models.build import build_model, MODELS


@MODELS.register_module()
class BaseDuplicateHead(nn.Module):
    """
    Class implementing the BaseDuplicateHead module.

    Attributes:
        pre_fusion (nn.Module): Optional module updating the query features before fusion.
        fusion_type (str): String containing the fusion type.
        post_fusion (nn.Module): Optional module updating the fused query features.
        matcher (nn.Module): Optional module determining the target duplicate labels.
        loss (nn.Module): Module computing the duplicate loss.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, loss_cfg, pre_fusion_cfg=None, fusion_type='mul', post_fusion_cfg=None, matcher_cfg=None,
                 apply_ids=None, **kwargs):
        """
        Initializes the BaseDuplicateHead module.

        Args:
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            pre_fusion_cfg (Dict): Configuration dictionary specifying the pre-fusion module (default=None).
            fusion_type (str): String containing the fusion type (default='mul').
            post_fusion_cfg (Dict): Configuration dictionary specifying the post-fusion module (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build pre-fusion and post-fusion modules if needed
        self.pre_fusion = build_model(pre_fusion_cfg) if pre_fusion_cfg is not None else None
        self.post_fusion = build_model(post_fusion_cfg) if post_fusion_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set attribute containing the fusion type
        self.fusion_type = fusion_type

        # Set apply_ids attribute
        self.apply_ids = apply_ids

    def forward_pred(self, qry_feats, storage_dict, **kwargs):
        """
        Forward prediction method of the BaseDuplicateHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_feats, qry_feat_size].
            storage_dict (Dict): Dictionary storing all kinds of key-value pairs of interest.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - dup_logits (FloatTensor): duplicate logits of shape [num_feats, num_feats].

        Raises:
            ValueError: Error when an invalid fusion type is provided.
        """

        # Apply pre-fusion module if needed
        if self.pre_fusion is not None:
            qry_feats = self.pre_fusion(qry_feats)

        # Apply fusion operation
        if self.fusion_type == 'mul':
            dup_feats = qry_feats[:, None, :] * qry_feats[None, :, :]

        else:
            error_msg = f"Invalid fusion type (got '{self.fusion_type}')."
            raise ValueError(error_msg)

        # Apply post-fusion module if needed
        if self.post_fusion is not None:
            num_feats = len(qry_feats)
            dup_feats = self.post_fusion(dup_feats.flatten(0, 1))
            dup_feats = dup_feats.view(num_feats, num_feats, -1)

        # Get duplicate logits
        dup_logits = dup_feats.squeeze(dim=2)
        storage_dict['dup_logits'] = dup_logits

        return storage_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseDuplicateHead module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys (after matching):
                - dup_logits (FloatTensor): duplicate logits of shape [num_feats, num_feats];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_queries];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_queries].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - labels (LongTensor): target class indices of shape [num_targets].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - dup_loss (FloatTensor): duplicate loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - dup_acc (FloatTensor): duplicate accuracy of shape [].
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve desired items from storage dictionary
        dup_logits = storage_dict['dup_logits']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device and number of positive matches
        device = dup_logits.device
        num_matches = len(matched_qry_ids)

        # Handle case where there are no positive matches
        if num_matches == 0:

            # Get duplicate loss
            dup_loss = 0.0 * dup_logits.sum()
            key_name = f'dup_loss_{id}' if id is not None else 'dup_loss'
            loss_dict[key_name] = dup_loss

            # Get duplicate accuracy if needed
            if analysis_dict is not None:
                dup_acc = 1.0 if len(tgt_dict['labels']) == 0 else 0.0
                dup_acc = torch.tensor(dup_acc, dtype=dup_loss.dtype, device=device)

                key_name = f'dup_acc_{id}' if id is not None else 'dup_acc'
                analysis_dict[key_name] = 100 * dup_acc

            return loss_dict, analysis_dict

        # Get duplicate loss
        dup_logits = dup_logits[matched_qry_ids][:, matched_qry_ids]
        dup_targets = (matched_tgt_ids[:, None] == matched_tgt_ids[None, :]).float()
        dup_loss = num_matches * self.loss(dup_logits, dup_targets)

        key_name = f'dup_loss_{id}' if id is not None else 'dup_loss'
        loss_dict[key_name] = dup_loss

        # Get duplicate accuracy if needed
        if analysis_dict is not None:
            dup_preds = dup_logits > 0.0
            dup_acc = (dup_preds == dup_targets.bool()).sum() / dup_preds.numel()

            key_name = f'dup_acc_{id}' if id is not None else 'dup_acc'
            analysis_dict[key_name] = 100 * dup_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseDuplicateHead module.

        Args:
            mode (str): String containing the forward mode chosen from ['pred', 'loss'].
            kwargs (Dict): Dictionary of keyword arguments passed to the underlying forward method.

        Raises:
            ValueError: Error when an invalid forward mode is provided.
        """

        # Choose underlying forward method
        if mode == 'pred':
            self.forward_pred(**kwargs)

        elif mode == 'loss':
            self.forward_loss(**kwargs)

        else:
            error_msg = f"Invalid forward mode (got '{mode}')."
            raise ValueError(error_msg)
