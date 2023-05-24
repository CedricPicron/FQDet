"""
Module building cores from MMDetection.
"""

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures.mask import BitmapMasks
from mmengine.config import Config
from mmengine.registry import build_model_from_cfg
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.build import MODELS
from structures.boxes import Boxes


@MODELS.register_module()
class MMDetArch(nn.Module):
    """
    Class implementing the MMDetArch module.

    Attributes:
        requires_masks (bool): Boolean indicating whether target masks are required during training.
        arch (nn.Module): Module containing the MMDetection architecture.
    """

    def __init__(self, cfg_path, backbone=None, core=None):
        """
        Initializes the MMDetArch module.

        Args:
            cfg_path (str): Path to configuration file specifying the MMDetection architecture.
            backbone (nn.Module): Module overwriting the backbone if requested (default=None).
            core (nn.Module): Module overwriting the core if requested (default=None).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Get config specifying the MMDetection architecture
        cfg = Config.fromfile(cfg_path)

        # Pop information from configuration dictionary
        overwrite_backbone = cfg.model.pop('overwrite_backbone')
        overwrite_neck = cfg.model.pop('overwrite_neck')
        self.requires_masks = cfg.model.pop('requires_masks')

        # Get architecture
        self.arch = build_model_from_cfg(cfg.model, registry=MMDET_MODELS)
        self.arch.init_weights()

        # Replace backbone and core/neck with given input modules if requested
        if overwrite_backbone:
            self.arch.backbone = backbone

        if hasattr(self.arch, 'neck') and overwrite_neck:
            self.arch.neck = core

    @staticmethod
    def get_param_families():
        """
        Method returning the recurring MMDetArch parameter families.

        Returns:
            List of strings containing the recurring MMDetArch parameter families.
        """

        return ['backbone', 'neck']

    def forward(self, images, tgt_dict=None, visualize=False, **kwargs):
        """
        Forward method of the MMDetArch module.

        Args:
            images (Images): Images structure containing the batched images.

            tgt_dict (Dict): Optional target dictionary used during trainval (possibly) containing following keys:
                - labels (LongTensor): tensor of shape [num_targets_total] containing the class indices;
                - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_targets_total];
                - sizes (LongTensor): tensor of shape [batch_size+1] with the cumulative target sizes of batch entries;
                - masks (BoolTensor): padded segmentation masks of shape [num_targets_total, max_iH, max_iW].

            visualize (bool): Boolean indicating whether to compute dictionary with visualizations (default=False).
            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            out_dicts (List): List of output dictionaries, potentially containing following items:
                - pred_dicts (List): list of dictionaries with predictions;
                - loss_dict (Dict): dictionary of different loss terms used for backpropagation during training;
                - analysis_dict (Dict): dictionary of different analyses used for logging purposes only.

        Raises:
            ValueError: Error when visualizations are requested.
        """

        # Check whether 'visualize' is False
        if visualize:
            raise ValueError("The MMDetArch architecture does not provide visualizations.")

        # Get MMDetection image metas
        batch_size = len(images)
        img_metas = [{} for _ in range(batch_size)]

        img_shapes = [(iH, iW, 3) for iW, iH in images.size(mode='without_padding')]
        scale_factors = images.resize_ratios()
        hflipped = images.hflipped()
        padded_width, padded_height = images.size(mode='with_padding')
        orig_img_shapes = [(iH, iW, 3) for iW, iH in images.size(mode='original')]

        for i in range(batch_size):
            img_metas[i]['img_shape'] = img_shapes[i]
            img_metas[i]['scale_factor'] = np.array([*scale_factors[i], *scale_factors[i]])
            img_metas[i]['flip'] = hflipped[i]
            img_metas[i]['pad_shape'] = (padded_height, padded_width, 3)
            img_metas[i]['ori_shape'] = orig_img_shapes[i]

        # Get loss and analysis dictionaries if desired
        if tgt_dict is not None:
            tgt_boxes = tgt_dict['boxes']
            tgt_boxes = tgt_boxes.to_format('xyxy').to_img_scale(images).boxes

            tgt_sizes = tgt_dict['sizes']
            tgt_boxes = [tgt_boxes[i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
            tgt_labels = [tgt_dict['labels'][i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]

            arch_kwargs = {}
            arch_kwargs['gt_bboxes'] = tgt_boxes
            arch_kwargs['gt_labels'] = tgt_labels

            if self.requires_masks:
                tgt_masks = tgt_dict['masks'].cpu().numpy()
                height, width = tgt_masks.shape[-2:]

                tgt_masks = [tgt_masks[i0:i1] for i0, i1 in zip(tgt_sizes[:-1], tgt_sizes[1:])]
                tgt_masks = [BitmapMasks(tgt_masks_i, height, width) for tgt_masks_i in tgt_masks]
                arch_kwargs['gt_masks'] = tgt_masks

            out_dict = self.arch(images, img_metas, return_loss=True, **arch_kwargs)
            out_dict.update({k: sum(v) for k, v in out_dict.items() if isinstance(v, list)})

            loss_dict = {k: v for k, v in out_dict.items() if 'loss' in k}
            analysis_dict = {k: v for k, v in out_dict.items() if 'loss' not in k}

        # Get prediction dictionaries if desired
        if not self.training:
            preds = self.arch([images], [img_metas], return_loss=False)

            pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
            pred_dict = {pred_key: [] for pred_key in pred_keys}

            if self.requires_masks:
                num_classes = len(preds[0][0])
                pred_dict['masks'] = []
            else:
                num_classes = len(preds[0])

            for i, preds_i in enumerate(preds):
                box_preds = preds_i[0] if self.requires_masks else preds_i

                cls_freqs = torch.tensor([len(cls_box_preds) for cls_box_preds in box_preds])
                labels = torch.arange(num_classes).repeat_interleave(cls_freqs, dim=0)
                batch_ids = torch.full_like(labels, i)

                box_preds = np.concatenate(box_preds, axis=0)
                box_preds = torch.from_numpy(box_preds)

                boxes = Boxes(box_preds[:, :4], format='xyxy', batch_ids=batch_ids)
                scores = box_preds[:, 4]

                pred_dict['labels'].append(labels)
                pred_dict['boxes'].append(boxes)
                pred_dict['scores'].append(scores)
                pred_dict['batch_ids'].append(batch_ids)

                if self.requires_masks:
                    padded_width, padded_height = images.size(mode='with_padding')

                    if len(labels) > 0:
                        seg_preds = [np.stack(cls_seg_preds, axis=0) for cls_seg_preds in preds_i[1] if cls_seg_preds]
                        seg_preds = np.concatenate(seg_preds, axis=0)
                        seg_preds = torch.from_numpy(seg_preds)

                        height, width = seg_preds.size()[-2:]
                        padding = (0, padded_width - width, 0, padded_height - height)
                        seg_preds = F.pad(seg_preds, padding)

                    else:
                        seg_preds = torch.empty(0, padded_height, padded_width, dtype=torch.bool)

                    pred_dict['masks'].append(seg_preds)

            pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k != 'boxes'})
            pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])
            pred_dicts = [pred_dict]

        # Get architecture output dictionaries
        if self.training:
            out_dicts = [loss_dict, analysis_dict]
        elif tgt_dict is not None:
            out_dicts = [pred_dicts, loss_dict, analysis_dict]
        else:
            out_dicts = [pred_dicts]

        return out_dicts
