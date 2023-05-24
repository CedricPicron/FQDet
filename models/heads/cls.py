"""
Collection of classification heads.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS
from structures.boxes import box_iou


@MODELS.register_module()
class BaseClsHead(nn.Module):
    """
    Class implementing the BaseClsHead module.

    Attributes:
        logits (nn.Module): Module computing the classification logits.
        matcher (nn.Module): Optional module determining the target classification labels.
        soft_label_type (str): Optional string containing the soft label type.
        loss (nn.Module): Module computing the classification loss.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, logits_cfg, loss_cfg, matcher_cfg=None, soft_label_type=None, apply_ids=None, **kwargs):
        """
        Initializes the BaseClsHead module.

        Args:
            logits_cfg (Dict): Configuration dictionary specifying the logits module.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            soft_label_type (str): String containing the soft label type (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build logits module
        self.logits = build_model(logits_cfg)

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Set remaining attributes
        self.soft_label_type = soft_label_type
        self.apply_ids = apply_ids

    def forward_pred(self, qry_feats, storage_dict, **kwargs):
        """
        Forward prediction method of the BaseClsHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].
            storage_dict (Dict): Dictionary storing all kinds of key-value pairs of interest.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional key:
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels].
        """

        # Get classification logits
        cls_logits = self.logits(qry_feats)
        storage_dict['cls_logits'] = cls_logits

        return storage_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseClsHead module.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys (after matching):
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - match_labels (LongTensor): match labels corresponding to each query of shape [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys].

            tgt_dict (Dict): Target dictionary (possibly) containing following keys:
                - labels (LongTensor): target class indices of shape [num_targets];
                - boxes (Boxes): target 2D bounding boxes of size [num_targets].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary containing following additional key:
                - cls_loss (FloatTensor): classification loss of shape [].

            analysis_dict (Dict): Analysis dictionary containing following additional key (if not None):
                - cls_acc (FloatTensor): classification accuracy of shape [].

        Raises:
            ValueError: Error when an invalid soft label type is provided.
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve classification logits and matching results
        cls_logits = storage_dict['cls_logits']
        match_labels = storage_dict['match_labels']
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get classification logits with corresponding targets
        device = cls_logits.device
        num_labels = cls_logits.size(dim=1)

        pos_cls_logits = cls_logits[matched_qry_ids]
        pos_cls_targets = tgt_dict['labels'][matched_tgt_ids]

        if self.soft_label_type is None:
            pos_cls_targets_oh = F.one_hot(pos_cls_targets, num_classes=num_labels)

        elif self.soft_label_type == 'box_iou':
            num_pos_matches = len(matched_qry_ids)
            pos_cls_targets_oh = torch.zeros(num_pos_matches, num_labels, device=device)

            if num_pos_matches > 0:
                pred_boxes = storage_dict['pred_boxes'][matched_qry_ids].detach()
                tgt_boxes = tgt_dict['boxes'][matched_tgt_ids]
                ious = box_iou(pred_boxes, tgt_boxes).diag()

                match_ids = torch.arange(num_pos_matches, device=device)
                pos_cls_targets_oh[match_ids, pos_cls_targets] = ious

        else:
            error_msg = f"Invalid soft label type (got '{self.soft_label_type}') in BaseClsHead module."
            raise ValueError(error_msg)

        neg_mask = match_labels == 0
        num_negs = neg_mask.sum().item()

        neg_cls_logits = cls_logits[neg_mask]
        neg_cls_targets = torch.full([num_negs], num_labels-1, device=cls_logits.device)
        neg_cls_targets_oh = F.one_hot(neg_cls_targets, num_classes=num_labels)

        cls_logits = torch.cat([pos_cls_logits, neg_cls_logits], dim=0)
        cls_targets = torch.cat([pos_cls_targets, neg_cls_targets], dim=0)
        cls_targets_oh = torch.cat([pos_cls_targets_oh, neg_cls_targets_oh], dim=0)

        # Get classification loss
        cls_loss = self.loss(cls_logits, cls_targets_oh)

        key_name = f'cls_loss_{id}' if id is not None else 'cls_loss'
        loss_dict[key_name] = cls_loss

        # Get classification accuracy if needed
        if analysis_dict is not None:
            cls_preds = cls_logits.argmax(dim=1)
            cls_acc = (cls_preds == cls_targets).sum() / len(cls_preds)

            key_name = f'cls_acc_{id}' if id is not None else 'cls_acc'
            analysis_dict[key_name] = 100 * cls_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseClsHead module.

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
