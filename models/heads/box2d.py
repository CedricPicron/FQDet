"""
Collection of 2D bounding box heads.
"""

from detectron2.layers import batched_nms
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
from mmcv.ops import soft_nms
from mmdet.models.layers.transformer.utils import inverse_sigmoid
import torch
from torch import nn
import torchvision.transforms.functional as T

from models.build import build_model, MODELS
from structures.boxes import Boxes, box_iou


@MODELS.register_module()
class BaseBox2dHead(nn.Module):
    """
    Class implementing the BaseBox2dHead module.

    Attributes:
        detach_qry_feats (bool): Boolean indicating whether to use detached query features.
        logits (nn.Module): Module computing the 2D bounding box logits.
        box_coder (nn.Module): Module containing the 2D box coder.
        update_prior_boxes (bool): Boolean indicating whether to update prior boxes.
        box_encoder (nn.Module): Optional module updating the box encodings based on the updated prior boxes.
        box_score (nn.Module): Optional module computing the unnormalized 2D bounding box scores.
        get_dets (bool): Boolean indicating whether to get 2D object detection predictions.

        score_attrs (Dict): Dictionary specifying the scoring mechanism possibly containing following keys:
            - cls_power (float): value containing the classification score power;
            - box_power (float): value containing the box score power.

        dup_attrs (Dict): Dictionary specifying the duplicate removal or rescoring mechanism possibly containing:
            - type (str): string containing the type of duplicate removal or rescoring mechanism;
            - dup_thr (float): value thresholding the predicted duplicate scores;
            - nms_candidates (int): integer containing the maximum number of candidate detections retained before NMS;
            - nms_thr (float): IoU threshold used during NMS or Soft-NMS to remove or rescore duplicate detections;
            - soft_dup_thr (float): value thresholding the predicted duplicate score during soft duplicate removal;
            - soft dup_power (float): value containing the duplicate score power during soft duplicate removal;
            - soft_nms_sigma (float): value containing the standard deviation of the Soft-NMS gaussian;
            - soft_nms_min_score (float): value containing the minimum detection score to be considered by Soft-NMS;
            - soft_nms_method (str): string containing the type of Soft-NMS method;
            - ub_tgt_thr (float): minimum IoU value required to match target during upper bound removal;
            - ub_pred_thr (float): minumum IoU value required to match other prediction during upper bound removal.

        max_dets (int): Optional integer with the maximum number of returned 2D object detection predictions.
        metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
        matcher (nn.Module): Optional module determining the 2D target boxes.
        report_match_stats (bool): Boolean indicating whether to report matching statistics.
        loss (nn.Module): Module computing the 2D bounding box loss.
        box_score_loss (nn.Module): Optional module computing the 2D bounding box score loss.
        apply_ids (List): List with integers determining when the head should be applied.
    """

    def __init__(self, logits_cfg, box_coder_cfg, metadata, loss_cfg, detach_qry_feats=False, update_prior_boxes=False,
                 box_encoder_cfg=None, box_score_cfg=None, get_dets=True, score_attrs=None, dup_attrs=None,
                 max_dets=None, matcher_cfg=None, report_match_stats=True, box_score_loss_cfg=None, apply_ids=None,
                 **kwargs):
        """
        Initializes the BaseBox2dHead module.

        Args:
            logits_cfg (Dict): Configuration dictionary specifying the logits module.
            box_coder_cfg (Dict): Configuration dictionary specifying the box coder module.
            metadata (detectron2.data.Metadata): Metadata instance containing additional dataset information.
            loss_cfg (Dict): Configuration dictionary specifying the loss module.
            detach_qry_feats (bool): Boolean indicating whether to use detached query features (default=False).
            update_prior_boxes (bool): Boolean indicating whether to update prior boxes (default=False).
            box_encoder_cfg (Dict): Configuration dictionary specifying the box encoder module (default=None).
            box_score_cfg (Dict): Configuration dictionary specifying the box score module (default=None).
            get_dets (bool): Boolean indicating whether to get 2D object detection predictions (default=True).
            score_attrs (Dict): Attribute dictionary specifying the scoring mechanism (default=None).
            dup_attrs (Dict): Dictionary specifying the duplicate removal or rescoring mechanism (default=None).
            max_dets (int): Integer with the maximum number of returned 2D object detection predictions (default=None).
            matcher_cfg (Dict): Configuration dictionary specifying the matcher module (default=None).
            report_match_stats (bool): Boolean indicating whether to report matching statistics (default=True).
            box_score_loss_cfg (Dict): Configuration dictionary specifying the box score loss module (default=None).
            apply_ids (List): List with integers determining when the head should be applied (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build logits module
        self.logits = build_model(logits_cfg)

        # Build box coder module
        self.box_coder = build_model(box_coder_cfg)

        # Build box encoder module if needed
        self.box_encoder = build_model(box_encoder_cfg) if box_encoder_cfg is not None else None

        # Build box score module if needed
        self.box_score = build_model(box_score_cfg) if box_score_cfg is not None else None

        # Build matcher module if needed
        self.matcher = build_model(matcher_cfg) if matcher_cfg is not None else None

        # Build loss module
        self.loss = build_model(loss_cfg)

        # Build box score loss module if needed
        self.box_score_loss = build_model(box_score_loss_cfg) if box_score_loss_cfg is not None else None

        # Set remaining attributes
        self.detach_qry_feats = detach_qry_feats
        self.update_prior_boxes = update_prior_boxes
        self.get_dets = get_dets
        self.score_attrs = score_attrs if score_attrs is not None else dict()
        self.dup_attrs = dup_attrs if dup_attrs is not None else dict()
        self.max_dets = max_dets
        self.metadata = metadata
        self.report_match_stats = report_match_stats
        self.apply_ids = apply_ids

    @torch.no_grad()
    def compute_dets(self, storage_dict, pred_dicts, tgt_dict=None, **kwargs):
        """
        Method computing the 2D object detection predictions.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys:
                - cls_logits (FloatTensor): classification logits of shape [num_qrys, num_labels];
                - dup_logits (FloatTensor): duplicate logits of shape [num_qrys, num_qrys];
                - cum_feats_batch (LongTensor): cumulative number of queries per batch entry [batch_size+1];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - box_scores (FloatTensor): unnormalized 2D bounding box scores of shape [num_qrys].

            pred_dicts (List): List of size [num_pred_dicts] collecting various prediction dictionaries.

            tgt_dict (Dictionary): Optional target dictionary containing at least following key:
                - labels (LongTensor): target class indices of shape [num_targets];
                - boxes (Boxes): target 2D bounding boxes of size [num_targets].

            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            pred_dicts (List): List with prediction dictionaries containing following additional entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

        Raises:
            ValueError: Error when an invalid type of duplicate removal or rescoring mechanism is provided.
        """

        # Retrieve desired items from storage dictionary
        cls_logits = storage_dict['cls_logits']
        cum_feats_batch = storage_dict['cum_feats_batch']
        pred_boxes = storage_dict['pred_boxes']
        box_scores = storage_dict.get('box_scores', None)

        # Get number of queries, number of labels, device and batch size
        num_qrys, num_labels = cls_logits.size()
        device = cls_logits.device
        batch_size = len(cum_feats_batch) - 1

        # Only keep entries with well-defined boxes
        well_defined = pred_boxes.well_defined()
        cls_logits = cls_logits[well_defined]
        pred_boxes = pred_boxes[well_defined]

        # Get 2D detections for each combination of box and class label
        num_qrys = len(cls_logits)
        num_classes = num_labels - 1

        pred_labels = torch.arange(num_classes, device=device)[None, :].expand(num_qrys, -1).flatten()
        pred_boxes = pred_boxes.expand(num_classes)
        pred_scores = cls_logits[:, :-1].sigmoid().flatten()
        batch_ids = pred_boxes.batch_ids

        # Update prediction scores if needed
        if box_scores is not None:
            box_scores = box_scores.sigmoid()
            box_scores = box_scores[:, None].expand(-1, num_classes).flatten()

            cls_power = self.score_attrs.get('cls_power', 1.0)
            box_power = self.score_attrs.get('box_power', 1.0)
            pred_scores = (pred_scores**cls_power) * (box_scores**box_power)

        # Get duplicate-relared mask or scores if needed
        dup_type = self.dup_attrs.get('type', None)

        if dup_type == 'learned':
            dup_logits = storage_dict['dup_logits']
            dup_logits = dup_logits[well_defined, :][:, well_defined]

            dup_thr = self.dup_attrs.get('dup_thr', 0.5)
            dup_mask = dup_logits.sigmoid() > dup_thr

        elif dup_type == 'soft-learned':
            dup_logits = storage_dict['dup_logits']
            dup_logits = dup_logits[well_defined, :][:, well_defined]

            dup_thr = self.dup_attrs.get('soft_dup_thr', 0.0)
            dup_power = self.dup_attrs.get('soft_dup_power', 2.0)

            dup_scores = (dup_logits.sigmoid() - dup_thr) / (1 - dup_thr)
            dup_scores = dup_scores.clamp(min=0)**dup_power
            non_dup_scores = 1 - dup_scores

        # Initialize prediction dictionary
        pred_keys = ('labels', 'boxes', 'scores', 'batch_ids')
        pred_dict = {pred_key: [] for pred_key in pred_keys}

        # Iterate over every batch entry
        for i in range(batch_size):

            # Get predictions corresponding to batch entry
            batch_mask = batch_ids == i

            pred_labels_i = pred_labels[batch_mask]
            pred_boxes_i = pred_boxes[batch_mask]
            pred_scores_i = pred_scores[batch_mask]

            # Remove duplicate predictions if needed
            if self.dup_attrs:

                if dup_type == 'learned':
                    dup_batch_mask = batch_mask.view(-1, num_classes)[:, 0]
                    dup_mask_i = dup_mask[dup_batch_mask, :][:, dup_batch_mask]
                    dup_mask_i = dup_mask_i[:, :, None].expand(-1, -1, num_classes)

                    scores = pred_scores_i.view(-1, num_classes)
                    lower_score_mask = scores[:, None, :] < scores[None, :, :]
                    base_mask = dup_mask_i & lower_score_mask

                    old_mask = base_mask
                    new_mask = base_mask & (~old_mask.any(dim=1)[None, :, :])

                    while not torch.equal(old_mask, new_mask):
                        old_mask = new_mask
                        new_mask = base_mask & (~old_mask.any(dim=1)[None, :, :])

                    dup_mask_i = new_mask.any(dim=1).flatten()
                    non_dup_mask = ~dup_mask_i

                    pred_labels_i = pred_labels_i[non_dup_mask]
                    pred_boxes_i = pred_boxes_i[non_dup_mask]
                    pred_scores_i = pred_scores_i[non_dup_mask]

                elif dup_type == 'nms':
                    num_candidates = self.dup_attrs.get('nms_candidates', 1000)
                    candidate_ids = pred_scores_i.topk(num_candidates)[1]

                    pred_labels_i = pred_labels_i[candidate_ids]
                    pred_boxes_i = pred_boxes_i[candidate_ids].to_format('xyxy')
                    pred_scores_i = pred_scores_i[candidate_ids]

                    iou_thr = self.dup_attrs.get('nms_thr', 0.65)
                    non_dup_ids = batched_nms(pred_boxes_i.boxes, pred_scores_i, pred_labels_i, iou_thr)

                    pred_labels_i = pred_labels_i[non_dup_ids]
                    pred_boxes_i = pred_boxes_i[non_dup_ids]
                    pred_scores_i = pred_scores_i[non_dup_ids]

                elif dup_type == 'soft-learned':
                    dup_batch_mask = batch_mask[::num_classes]
                    non_dup_scores_i = non_dup_scores[dup_batch_mask, :][:, dup_batch_mask]

                    scores = pred_scores_i.view(-1, num_classes).t()
                    num_preds = scores.size(dim=1)

                    orig_ids = torch.arange(num_preds, device=device)[None, :].expand(num_classes, -1)
                    cls_ids = torch.arange(num_classes, device=device)

                    orig_ids_list = []
                    scores_list = []

                    for j in range(num_preds):
                        scores_j, local_ids = scores.max(dim=1)
                        orig_ids_j = orig_ids[cls_ids, local_ids]

                        orig_ids_list.append(orig_ids_j)
                        scores_list.append(scores_j)

                        if j < num_preds-1:
                            mask = torch.ones_like(scores, dtype=torch.bool)
                            mask[cls_ids, local_ids] = False

                            orig_ids = orig_ids[mask].view(num_classes, num_preds-j-1)
                            scores = scores[mask].view(num_classes, num_preds-j-1)

                            orig_ids_j = orig_ids_j[:, None].expand(-1, num_preds-j-1)
                            non_dup_scores_j = non_dup_scores_i[orig_ids_j, orig_ids]
                            scores = non_dup_scores_j * scores

                    orig_ids = torch.stack(orig_ids_list, dim=0)
                    scores = torch.stack(scores_list, dim=0)

                    pred_scores_i = torch.empty(num_preds, num_classes, device=device)
                    cls_ids = cls_ids[None, :].expand(num_preds, -1)

                    pred_scores_i[orig_ids, cls_ids] = scores
                    pred_scores_i = pred_scores_i.flatten()

                elif dup_type == 'soft-nms':
                    soft_nms_kwargs = {}
                    soft_nms_kwargs['iou_threshold'] = self.dup_attrs.get('nms_thr', 0.3)
                    soft_nms_kwargs['sigma'] = self.dup_attrs.get('soft_nms_sigma', 0.5)
                    soft_nms_kwargs['min_score'] = self.dup_attrs.get('soft_nms_min_score', 1e-3)
                    soft_nms_kwargs['method'] = self.dup_attrs.get('soft_nms_method', 'gaussian')

                    nms_boxes = pred_boxes_i.to_format('xyxy').boxes
                    offset = nms_boxes.max().item() + 1

                    nms_boxes = nms_boxes + offset * pred_labels_i[:, None]
                    dets, ids = soft_nms(nms_boxes, pred_scores_i, **soft_nms_kwargs)

                    pred_labels_i = pred_labels_i[ids]
                    pred_boxes_i = pred_boxes_i[ids]
                    pred_scores_i = dets[:, 4]

                elif dup_type == 'upper_bound':
                    if tgt_dict is not None:
                        tgt_batch_mask = tgt_dict['boxes'].batch_ids == i
                        tgt_labels_i = tgt_dict['labels'][tgt_batch_mask]
                        tgt_boxes_i = tgt_dict['boxes'][tgt_batch_mask]

                        if len(tgt_boxes_i) > 0:
                            tgt_thr = self.dup_attrs.get('ub_tgt_thr', 0.5)
                            pred_thr = self.dup_attrs.get('ub_pred_thr', 0.5)

                            pred_boxes_ic = pred_boxes_i[::num_classes]
                            max_ious, tgt_ids = box_iou(pred_boxes_ic, tgt_boxes_i).max(dim=1)

                            tgt_ids[max_ious < tgt_thr] = -1
                            cls_ids = tgt_labels_i[tgt_ids]

                            num_preds = len(tgt_ids)
                            pred_ids = torch.arange(num_preds, device=device)

                            tgt_ids_m = torch.full([num_preds, num_classes], -1, dtype=torch.int64, device=device)
                            tgt_ids_m[pred_ids, cls_ids] = tgt_ids

                            tgt_unmatched = torch.ones(len(tgt_boxes_i)+1, dtype=torch.bool, device=device)
                            tgt_unmatched[-1] = False

                            scores = pred_scores_i.view(-1, num_classes)
                            scores, sort_ids = scores.sort(dim=0, descending=True)

                            pred_ids = pred_ids[:, None].expand(-1, num_classes)
                            pred_ids = torch.gather(pred_ids, dim=0, index=sort_ids)

                            cls_ids = torch.arange(num_classes, device=device)
                            pred_iou_mask = box_iou(pred_boxes_ic, pred_boxes_ic) > pred_thr
                            non_dup_mask_i = torch.empty(num_preds, num_classes, dtype=torch.bool, device=device)

                            for j in range(num_preds):
                                pred_ids_j = pred_ids[j]
                                tgt_ids_j = tgt_ids_m[pred_ids_j, cls_ids]

                                matches_unmatched_tgt = tgt_unmatched[tgt_ids_j]
                                tgt_unmatched[tgt_ids_j] = False

                                pred_iou_mask_i = pred_iou_mask[pred_ids[:j], pred_ids_j[None, :].expand(j, -1)]
                                overlaps_matched_pred = (non_dup_mask_i[:j] & pred_iou_mask_i).any(dim=0)
                                non_dup_mask_i[j] = matches_unmatched_tgt | (~overlaps_matched_pred)

                            orig_ids = torch.arange(num_preds*num_classes, device=device).view(num_preds, num_classes)
                            orig_ids = torch.gather(orig_ids, dim=0, index=sort_ids)
                            non_dup_ids = orig_ids[non_dup_mask_i]

                            pred_labels_i = pred_labels_i[non_dup_ids]
                            pred_boxes_i = pred_boxes_i[non_dup_ids]
                            pred_scores_i = pred_scores_i[non_dup_ids]

                else:
                    error_msg = f"Invalid type of duplicate removal or rescoring mechanism (got '{dup_type}')."
                    raise ValueError(error_msg)

            # Only keep top predictions if needed
            if self.max_dets is not None:
                if len(pred_scores_i) > self.max_dets:
                    top_pred_ids = pred_scores_i.topk(self.max_dets)[1]

                    pred_labels_i = pred_labels_i[top_pred_ids]
                    pred_boxes_i = pred_boxes_i[top_pred_ids]
                    pred_scores_i = pred_scores_i[top_pred_ids]

            # Add predictions to prediction dictionary
            pred_dict['labels'].append(pred_labels_i)
            pred_dict['boxes'].append(pred_boxes_i)
            pred_dict['scores'].append(pred_scores_i)
            pred_dict['batch_ids'].append(torch.full_like(pred_labels_i, i))

        # Concatenate predictions of different batch entries
        pred_dict.update({k: torch.cat(v, dim=0) for k, v in pred_dict.items() if k not in ['boxes']})
        pred_dict['boxes'] = Boxes.cat(pred_dict['boxes'])

        # Add prediction dictionary to list of prediction dictionaries
        pred_dicts.append(pred_dict)

        return pred_dicts

    @torch.no_grad()
    def draw_dets(self, storage_dict, images_dict, pred_dicts, tgt_dict=None, vis_score_thr=0.4, id=None, **kwargs):
        """
        Draws predicted and target 2D object detections on the corresponding images.

        Boxes must have a score of at least the score threshold to be drawn. Target boxes get a default 100% score.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - images (Images): images structure containing the batched images of size [batch_size].

            images_dict (Dict): Dictionary with annotated images of predictions/targets.

            pred_dicts (List): List with prediction dictionaries containing as last entry:
                pred_dict (Dict): Prediction dictionary containing following keys:
                    - labels (LongTensor): predicted class indices of shape [num_preds];
                    - boxes (Boxes): structure containing axis-aligned bounding boxes of size [num_preds];
                    - scores (FloatTensor): normalized prediction scores of shape [num_preds];
                    - batch_ids (LongTensor): batch indices of predictions of shape [num_preds].

            tgt_dict (Dict): Optional target dictionary containing at least following keys when given:
                - labels (LongTensor): target class indices of shape [num_targets];
                - boxes (Boxes): target 2D bounding boxes of size [num_targets];
                - sizes (LongTensor): cumulative number of targets per batch entry of size [batch_size+1].

            vis_score_thr (float): Threshold indicating the minimum score for a box to be drawn (default=0.4).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            images_dict (Dict): Dictionary containing additional images annotated with 2D object detections.
        """

        # Retrieve images from storage dictionary
        images = storage_dict['images']

        # Initialize list of draw dictionaries and list of dictionary names
        draw_dicts = []
        dict_names = []

        # Get prediction draw dictionary and dictionary name
        pred_dict = pred_dicts[-1]

        pred_boxes = pred_dict['boxes'].to_img_scale(images).to_format('xyxy')
        well_defined = pred_boxes.well_defined()

        pred_scores = pred_dict['scores'][well_defined]
        sufficient_score = pred_scores >= vis_score_thr

        pred_labels = pred_dict['labels'][well_defined][sufficient_score]
        pred_boxes = pred_boxes.boxes[well_defined][sufficient_score]
        pred_scores = pred_scores[sufficient_score]
        pred_batch_ids = pred_dict['batch_ids'][well_defined][sufficient_score]

        pred_sizes = [0] + [(pred_batch_ids == i).sum() for i in range(len(images))]
        pred_sizes = torch.tensor(pred_sizes, device=pred_scores.device).cumsum(dim=0)

        draw_dict_keys = ['labels', 'boxes', 'scores', 'sizes']
        draw_dict_values = [pred_labels, pred_boxes, pred_scores, pred_sizes]

        draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}
        draw_dicts.append(draw_dict)

        dict_name = f"box2d_pred_{id}" if id is not None else "box2d_pred"
        dict_names.append(dict_name)

        # Get target draw dictionary and dictionary name if needed
        if tgt_dict is not None and not any('box2d_tgt' in key for key in images_dict.keys()):
            tgt_labels = tgt_dict['labels']
            tgt_boxes = tgt_dict['boxes'].to_img_scale(images).to_format('xyxy').boxes
            tgt_scores = torch.ones_like(tgt_labels, dtype=torch.float)
            tgt_sizes = tgt_dict['sizes']

            draw_dict_values = [tgt_labels, tgt_boxes, tgt_scores, tgt_sizes]
            draw_dict = {k: v for k, v in zip(draw_dict_keys, draw_dict_values)}

            draw_dicts.append(draw_dict)
            dict_names.append('box2d_tgt')

        # Get number of images and image sizes without padding in (width, height) format
        num_images = len(images)
        img_sizes = images.size(mode='without_padding')

        # Draw 2D object detections on images and add them to images dictionary
        for dict_name, draw_dict in zip(dict_names, draw_dicts):
            sizes = draw_dict['sizes']

            for image_id, i0, i1 in zip(range(num_images), sizes[:-1], sizes[1:]):
                img_size = img_sizes[image_id]
                img_size = (img_size[1], img_size[0])

                image = images.images[image_id].clone()
                image = T.crop(image, 0, 0, *img_size)
                image = image.permute(1, 2, 0) * 255
                image = image.to(torch.uint8).cpu().numpy()

                metadata = self.metadata
                visualizer = Visualizer(image, metadata=metadata)

                if i1 > i0:
                    img_labels = draw_dict['labels'][i0:i1].cpu().numpy()
                    img_boxes = draw_dict['boxes'][i0:i1].cpu().numpy()
                    img_scores = draw_dict['scores'][i0:i1].cpu().numpy()

                    instances = Instances(img_size, pred_classes=img_labels, pred_boxes=img_boxes, scores=img_scores)
                    visualizer.draw_instance_predictions(instances)

                annotated_image = visualizer.output.get_image()
                images_dict[f'{dict_name}_{image_id}'] = annotated_image

        return images_dict

    def forward_pred(self, qry_feats, storage_dict, images_dict=None, **kwargs):
        """
        Forward prediction method of the BaseBox2dHead module.

        Args:
            qry_feats (FloatTensor): Query features of shape [num_qrys, qry_feat_size].

            storage_dict (Dict): Storage dictionary possibly containing following key:
                - images (Images): images structure containing the batched images of size [batch_size];
                - prior_boxes (Boxes): prior 2D bounding boxes of size [num_qrys].

            images_dict (Dict): Dictionary with annotated images of predictions/targets (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying methods.

        Returns:
            storage_dict (Dict): Storage dictionary containing following additional or updated keys:
                - box_logits (FloatTensor): 2D bounding box logits of shape [num_qrys, 4];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - prior_boxes (Boxes): possibly updated prior 2D bounding boxes of size [num_qrys];
                - add_encs (FloatTensor): possibly updated box encodings of shape [num_qrys, feat_size];
                - box_scores (FloatTensor): unnormalized 2D bounding box scores of shape [num_qrys].

            images_dict (Dict): Dictionary containing additional images annotated with 2D object detections (if given).
        """

        # Detach query features if needed
        if self.detach_qry_feats:
            qry_feats = qry_feats.detach()

        # Get 2D bounding box logits
        box_logits = self.logits(qry_feats)
        storage_dict['box_logits'] = box_logits

        # Get predicted 2D bounding boxes
        prior_boxes = storage_dict['prior_boxes']
        images = storage_dict['images']

        pred_boxes = self.box_coder('apply', box_logits, prior_boxes, images=images)
        storage_dict['pred_boxes'] = pred_boxes

        # Update prior boxes if needed
        if self.update_prior_boxes:
            storage_dict['prior_boxes'] = pred_boxes

        # Update box encodings if needed
        if self.update_prior_boxes and self.box_encoder is not None:
            norm_boxes = pred_boxes.clone().detach().normalize(images).to_format('cxcywh')
            storage_dict['add_encs'] = self.box_encoder(norm_boxes.boxes)

        # Get box scores if needed
        if self.box_score is not None:
            box_scores = self.box_score(qry_feats)
            storage_dict['box_scores'] = box_scores

        # Get 2D object detection predictions if needed
        if self.get_dets and not self.training:
            self.compute_dets(storage_dict=storage_dict, **kwargs)

        # Draw predicted and target 2D object detections if needed
        if self.get_dets and images_dict is not None:
            self.draw_dets(storage_dict=storage_dict, images_dict=images_dict, **kwargs)

        return storage_dict, images_dict

    def forward_loss(self, storage_dict, tgt_dict, loss_dict, analysis_dict=None, id=None, **kwargs):
        """
        Forward loss method of the BaseBox2dHead module.

        Args:
            storage_dict (Dict): Storage dictionary (possibly) containing following keys (after matching):
                - images (Images): images structure containing the batched images of size [batch_size];
                - box_logits (FloatTensor): 2D bounding box logits of shape [num_qrys, 4];
                - pred_boxes (Boxes): predicted 2D bounding boxes of size [num_qrys];
                - prior_boxes (Boxes): prior 2D bounding boxes of size [num_qrys];
                - box_scores (FloatTensor): unnormalized 2D bounding box scores of shape [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].

            tgt_dict (Dict): Target dictionary containing at least following key:
                - boxes (Boxes): target 2D bounding boxes of size [num_targets].

            loss_dict (Dict): Dictionary containing different weighted loss terms.
            analysis_dict (Dict): Dictionary containing different analyses (default=None).
            id (int): Integer containing the head id (default=None).
            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            loss_dict (Dict): Loss dictionary (possibly) containing following additional keys:
                - box_loss (FloatTensor): 2D bounding box loss of shape [];
                - box_score_loss (FloatTensor): 2D bounding box score loss of shape [].

            analysis_dict (Dict): Analysis dictionary (possibly) containing following additional keys (if not None):
                - box_multi_tgt_qry (FloatTensor): percentage of queries matched to multiple targets of shape [];
                - box_matched_qry (FloatTensor): percentage of matched queries of shape [];
                - box_matched_tgt (FloatTensor): percentage of matched targets of shape [];
                - box_acc (FloatTensor): 2D bounding box accuracy of shape [];
                - box_score_acc (FloatTensor): 2D bounding box score accuracy of shape [].
        """

        # Perform matching if matcher is available
        if self.matcher is not None:
            self.matcher(storage_dict=storage_dict, tgt_dict=tgt_dict, analysis_dict=analysis_dict, **kwargs)

        # Retrieve desired items from storage dictionary
        box_logits = storage_dict['box_logits']
        box_scores = storage_dict.get('box_scores', None)
        matched_qry_ids = storage_dict['matched_qry_ids']
        matched_tgt_ids = storage_dict['matched_tgt_ids']

        # Get device
        device = box_logits.device

        # Report matching statistics if needed
        if self.report_match_stats and analysis_dict is not None:

            # Get percentage of queries matched to multiple targets
            num_qrys = len(box_logits)
            qry_counts = matched_qry_ids.unique(return_counts=True)[1]

            box_multi_tgt_qry = (qry_counts > 1).sum().item() / num_qrys if num_qrys > 0 else 0.0
            box_multi_tgt_qry = torch.tensor(box_multi_tgt_qry, device=device)

            key_name = f'box_multi_tgt_qry_{id}' if id is not None else 'box_multi_tgt_qry'
            analysis_dict[key_name] = 100 * box_multi_tgt_qry

            # Get percentage of matched queries
            box_matched_qry = len(qry_counts) / num_qrys if num_qrys > 0 else 1.0
            box_matched_qry = torch.tensor(box_matched_qry, device=device)

            key_name = f'box_matched_qry_{id}' if id is not None else 'box_matched_qry'
            analysis_dict[key_name] = 100 * box_matched_qry

            # Get percentage of matched targets
            num_tgts = len(tgt_dict['boxes'])

            box_matched_tgt = len(matched_tgt_ids.unique()) / num_tgts if num_tgts > 0 else 1.0
            box_matched_tgt = torch.tensor(box_matched_tgt, device=device)

            key_name = f'box_matched_tgt_{id}' if id is not None else 'box_matched_tgt'
            analysis_dict[key_name] = 100 * box_matched_tgt

        # Handle case where there are no positive matches
        if len(matched_qry_ids) == 0:

            # Get 2D bounding box loss
            box_loss = 0.0 * box_logits.sum()
            key_name = f'box_loss_{id}' if id is not None else 'box_loss'
            loss_dict[key_name] = box_loss

            # Get 2D bounding box score loss if needed
            if box_scores is not None:
                box_score_loss = 0.0 * box_scores.sum()
                key_name = f'box_score_loss_{id}' if id is not None else 'box_score_loss'
                loss_dict[key_name] = box_score_loss

            # Get accuracies if needed
            if analysis_dict is not None:

                # Get 2D bounding box accuracy
                box_acc = 1.0 if len(tgt_dict['boxes']) == 0 else 0.0
                box_acc = torch.tensor(box_acc, dtype=box_loss.dtype, device=device)

                key_name = f'box_acc_{id}' if id is not None else 'box_acc'
                analysis_dict[key_name] = 100 * box_acc

                # Get 2D bounding box score accuracy if needed
                if box_scores is not None:
                    box_score_acc = 1.0 if len(tgt_dict['boxes']) == 0 else 0.0
                    box_score_acc = torch.tensor(box_score_acc, dtype=box_loss.dtype, device=device)

                    key_name = f'box_score_acc_{id}' if id is not None else 'box_score_acc'
                    analysis_dict[key_name] = 100 * box_score_acc

            return loss_dict, analysis_dict

        # Get 2D bounding box logits with corresponding predicted and target boxes
        box_logits = box_logits[matched_qry_ids]
        pred_boxes = storage_dict['pred_boxes'][matched_qry_ids]
        tgt_boxes = tgt_dict['boxes'][matched_tgt_ids]

        # Get 2D bounding box targets
        prior_boxes = storage_dict['prior_boxes'][matched_qry_ids]
        images = storage_dict['images']
        box_targets = self.box_coder('get', prior_boxes, tgt_boxes, images=images)

        # Get 2D bounding box loss
        box_loss = self.loss(box_logits, box_targets, pred_boxes, tgt_boxes)
        key_name = f'box_loss_{id}' if id is not None else 'box_loss'
        loss_dict[key_name] = box_loss

        # Get 2D bounding box score loss if needed
        if box_scores is not None:
            pred_boxes = storage_dict['pred_boxes'].detach()
            pred_boxes = pred_boxes[matched_qry_ids]
            box_ious = box_iou(pred_boxes, tgt_boxes, images=images).diag()

            pred_scores = box_scores[matched_qry_ids]
            tgt_scores = inverse_sigmoid(box_ious, eps=1e-3)
            box_score_loss = self.box_score_loss(pred_scores, tgt_scores)

            key_name = f'box_score_loss_{id}' if id is not None else 'box_score_loss'
            loss_dict[key_name] = box_score_loss

        # Get accuracies if needed
        if analysis_dict is not None:

            # Get 2D bounding box accuracy
            if box_scores is None:
                pred_boxes = storage_dict['pred_boxes'].detach()
                pred_boxes = pred_boxes[matched_qry_ids]
                box_acc = box_iou(pred_boxes, tgt_boxes, images=images).diag().mean()

            else:
                box_acc = box_ious.mean()

            key_name = f'box_acc_{id}' if id is not None else 'box_acc'
            analysis_dict[key_name] = 100 * box_acc

            # Get 2D bounding box score accuracy if needed
            if box_scores is not None:
                box_score_err = pred_scores.detach().sigmoid() - box_ious
                box_score_acc = 1 - box_score_err.abs().mean()

                key_name = f'box_score_acc_{id}' if id is not None else 'box_score_acc'
                analysis_dict[key_name] = 100 * box_score_acc

        return loss_dict, analysis_dict

    def forward(self, mode, **kwargs):
        """
        Forward method of the BaseBox2dHead module.

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
