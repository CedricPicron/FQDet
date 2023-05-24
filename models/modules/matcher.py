"""
Collection of matcher modules.
"""

import torch
from torch import nn
from torch_scatter import scatter_min

from models.build import build_model, MODELS
from structures.boxes import box_giou, box_iou


@MODELS.register_module()
class BoxMatcher(nn.Module):
    """
    Class implementing the BoxMatcher module.

    The module matches query boxes with target boxes, assigning one of following match labels to each query:
        - positive label with value 1 (query matches with at least one target);
        - negative label with value 0 (query does not match with any of the targets);
        - ignore label with value -1 (query has no matching verdict).

    Attributes:
        qry_key (str): String with key to retrieve the queries boxes from the storage dictionary.
        share_qry_boxes (bool): Boolean indicating whether to share query boxes between images.
        box_metric (str): String containing the metric computing similarities between query and target boxes.
        center_in_mask (bool): Boolean indicating whether match requires the query box center inside the target mask.
        sim_matcher (nn.Module): Module matching queries with targets based on the given similarity matrix.
    """

    def __init__(self, qry_key, sim_matcher_cfg, share_qry_boxes=False, box_metric='iou', center_in_mask=False):
        """
        Initializes the BoxMatcher module.

        Args:
            qry_key (str): String with key to retrieve the queries boxes from the storage dictionary.
            sim_matcher_cfg (Dict): Configuration dictionary specifying the similarity matcher.
            share_qry_boxes (bool): Boolean indicating whether to share query boxes between images (default=False).
            box_metric (str): String with metric computing similarities between query and target boxes (default='iou').
            center_in_mask (bool): Whether match requires the query box center inside the target mask (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build similarity matcher
        self.sim_matcher = build_model(sim_matcher_cfg)

        # Set remaining attributes
        self.qry_key = qry_key
        self.share_qry_boxes = share_qry_boxes
        self.box_metric = box_metric
        self.center_in_mask = center_in_mask

    def forward(self, storage_dict, tgt_dict, **kwargs):
        """
        Forward method of the BoxMatcher module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following keys:
                - images (Images): images structure containing the batched images of size [batch_size];
                - {self.qry_key} (Boxes): structure containing axis-aligned query boxes of size [num_qrys].

            tgt_dict (Dict): Target dictionary (possibly) containing following keys:
                - boxes (Boxes): structure containing axis-aligned target boxes of size [num_targets].
                - masks (BoolTensor): target segmentation masks of shape [num_targets, iH, iW].

            kwargs (Dict): Dictionary of keyword arguments passed to some underlying modules.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional keys:
                - match_labels (LongTensor): match labels corresponding to each query of shape [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys];
                - top_qry_ids (LongTensor): optional top query indices per target of shape [top_limit, num_targets].

        Raises:
            ValueError: Error when an invalid bounding box metric is provided.
        """

        # Retrieve query and target boxes
        qry_boxes = storage_dict[self.qry_key]
        tgt_boxes = tgt_dict['boxes']

        # Retrieve target masks and get mask sizes if needed
        if self.center_in_mask:
            tgt_masks = tgt_dict['masks']
            iH, iW = tgt_masks.size()[1:]

        # Get device
        device = qry_boxes.boxes.device

        # Get list of query and target boxes per image
        images = storage_dict['images']
        batch_size = len(images)

        if self.share_qry_boxes:
            assert (qry_boxes.batch_ids == 0).all().item()
            qry_boxes_list = [qry_boxes] * batch_size

        else:
            qry_boxes_list = [qry_boxes[qry_boxes.batch_ids == i] for i in range(batch_size)]

        tgt_boxes_list = [tgt_boxes[tgt_boxes.batch_ids == i] for i in range(batch_size)]

        # Initialize query and target offsets
        qry_offset = 0
        tgt_offset = 0

        # Intialize lists for per image outputs
        match_labels_list = []
        matched_qry_ids_list = []
        matched_tgt_ids_list = []

        if self.sim_matcher.get_top_qry_ids:
            top_qry_ids_list = []

        # Perform matching per image
        for qry_boxes_i, tgt_boxes_i in zip(qry_boxes_list, tgt_boxes_list):

            # Get number of query and target boxes
            num_qrys = len(qry_boxes_i)
            num_targets = len(tgt_boxes_i)

            # Case where there are both queries and targets
            if (num_qrys > 0) and (num_targets > 0):

                # Get similarity matrix between query and target boxes
                if self.box_metric == 'giou':
                    sim_matrix = box_giou(qry_boxes_i, tgt_boxes_i)

                elif self.box_metric == 'iou':
                    sim_matrix = box_iou(qry_boxes_i, tgt_boxes_i)

                else:
                    error_msg = f"Invalid bounding box metric (got '{self.box_metric}')."
                    raise ValueError(error_msg)

                # Add center constraints if requested
                if self.center_in_mask:
                    i0 = tgt_offset
                    i1 = tgt_offset + num_targets
                    tgt_masks_i = tgt_masks[i0:i1]

                    qry_xy = qry_boxes_i.to_format('cxcywh').to_img_scale(images).boxes[:, :2].long()
                    qry_xy[:, 0].clamp_(max=iW-1)
                    qry_xy[:, 1].clamp_(max=iH-1)

                    center_in_mask = tgt_masks_i[:, qry_xy[:, 1], qry_xy[:, 0]].t()
                    sim_matrix[~center_in_mask] = 0.0

                # Perform matching based on similarity matrix
                match_results = self.sim_matcher(sim_matrix)
                match_labels_i, matched_qry_ids_i, matched_tgt_ids_i = match_results[:3]

            # Case where there are no queries or targets
            else:
                match_labels_i = torch.zeros(num_qrys, dtype=torch.int64, device=device)
                matched_qry_ids_i = torch.zeros(0, dtype=torch.int64, device=device)
                matched_tgt_ids_i = torch.zeros(0, dtype=torch.int64, device=device)

            # Add query and target offsets to indices
            matched_qry_ids_i = matched_qry_ids_i + qry_offset
            matched_tgt_ids_i = matched_tgt_ids_i + tgt_offset

            # Add image-specific matching results to lists
            match_labels_list.append(match_labels_i)
            matched_qry_ids_list.append(matched_qry_ids_i)
            matched_tgt_ids_list.append(matched_tgt_ids_i)

            # Get image-specific top query indices if requested
            if self.sim_matcher.get_top_qry_ids:

                if (num_qrys > 0) and (num_targets > 0):
                    top_qry_ids_i = match_results[3]

                else:
                    top_limit = self.sim_matcher.top_limit
                    top_qry_ids_i = torch.zeros(top_limit, num_targets, dtype=torch.int64, device=device)

                top_qry_ids_i = top_qry_ids_i + qry_offset
                top_qry_ids_list.append(top_qry_ids_i)

            # Update query and target offsets
            qry_offset += num_qrys
            tgt_offset += num_targets

        # Concatenate matching results and top query indices if requested
        storage_dict['match_labels'] = torch.cat(match_labels_list, dim=0)
        storage_dict['matched_qry_ids'] = torch.cat(matched_qry_ids_list, dim=0)
        storage_dict['matched_tgt_ids'] = torch.cat(matched_tgt_ids_list, dim=0)

        if self.sim_matcher.get_top_qry_ids:
            storage_dict['top_qry_ids'] = torch.cat(top_qry_ids_list, dim=1)

        return storage_dict


@MODELS.register_module()
class SimMatcher(nn.Module):
    """
    Class implementing the SimMatcher module.

    The module matches queries with targets based on the given query-target similarity matrix.

    The module assigns one of following match labels to each query:
        - positive label with value 1 (query matches with at least one target);
        - negative label with value 0 (query does not match with any of the targets);
        - ignore label with value -1 (query has no matching verdict).

    Attributes:
        mode (str): String containing the matching mode.
        static_mode (str): String containing the static matching mode.
        abs_pos (float): Absolute threshold determining positive query labels during static matching.
        abs_neg (float): Absolute threshold determining negative query labels during static matching.
        rel_pos (int): Relative threshold determining positive query labels during static matching.
        rel_neg (int): Relative threshold determining negative query labels during static matching.
        get_top_qry_ids (bool): Boolean indicating whether to get top query indices per target.
        top_limit (int): Integer limiting the number of top query indices returned per target.
        allow_multi_tgt (bool): Boolean indicating whether to allow multiple targets per query.
    """

    def __init__(self, mode='static', static_mode='rel', abs_pos=0.5, abs_neg=0.3, rel_pos=5, rel_neg=10,
                 get_top_qry_ids=False, top_limit=15, allow_multi_tgt=True):
        """
        Initializes the SimMatcher module.

        Args:
            mode (str): String containing the matching mode (default='static').
            static_mode (str): String containing the static matching mode (default='rel').
            abs_pos (float): Absolute threshold determining positive labels during static matching (default=0.5).
            abs_neg (float): Absolute threshold determining negative labels during static matching (default=0.3).
            rel_pos (int): Relative threshold determining positive labels during static matching (default=5).
            rel_neg (int): Relative threshold determining negative labels during static matching (default=10).
            get_top_qry_ids (bool): Boolean indicating whether to get top query indices per target (default=False).
            top_limit (int): Integer limiting the number of top query indices returned per target (default=15).
            allow_multi_tgt (bool): Boolean indicating whether to allow multiple targets per query (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.mode = mode
        self.static_mode = static_mode
        self.abs_pos = abs_pos
        self.abs_neg = abs_neg
        self.rel_pos = rel_pos
        self.rel_neg = rel_neg
        self.get_top_qry_ids = get_top_qry_ids
        self.top_limit = top_limit
        self.allow_multi_tgt = allow_multi_tgt

    def forward(self, sim_matrix):
        """
        Forward method of the SimMatcher module.

        Args:
            sim_matrix (FloatTensor): Query-target similarity matrix of shape [num_qrys, num_targets].

        Returns:
            return_list (List): List of size [num_returns] possibly containing following items to return:
                - match_labels (LongTensor): match labels corresponding to each query of shape [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys];
                - top_qry_ids (LongTensor): optional top query indices per target of shape [top_limit, num_targets].

        Raises:
            ValueError: Error when an invalid static matching mode is provided.
            ValueError: Error when an invalid matching mode is provided.
        """

        # Get number of queries and targets
        num_qrys, num_targets = sim_matrix.size()

        # Get device
        device = sim_matrix.device

        # Handle case where there are no queries or targets and return
        if (num_qrys == 0) or (num_targets == 0):
            match_labels = torch.zeros(num_qrys, dtype=torch.int64, device=device)
            matched_qry_ids = torch.zeros(0, dtype=torch.int64, device=device)
            matched_tgt_ids = torch.zeros(0, dtype=torch.int64, device=device)

            if self.get_top_qry_ids:
                top_qry_ids = torch.zeros(self.top_limit, num_targets, dtype=torch.int64, device=device)

            return_list = [match_labels, matched_qry_ids, matched_tgt_ids]
            return_list.append(top_qry_ids) if self.get_top_qry_ids else None

            return return_list

        # Get positive and negative label masks
        if self.mode == 'static':

            if 'abs' in self.static_mode:
                abs_pos_mask = sim_matrix > self.abs_pos
                abs_neg_mask = sim_matrix <= self.abs_neg

            if 'rel' in self.static_mode:
                top_limit = max(self.rel_neg, int(self.get_top_qry_ids) * self.top_limit)
                top_qry_ids = torch.topk(sim_matrix, top_limit, dim=0, sorted=True).indices

                pos_ids = top_qry_ids[:self.rel_pos, :]
                rel_pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
                rel_pos_mask[pos_ids, torch.arange(num_targets)] = True

                non_neg_ids = top_qry_ids[:self.rel_neg, :]
                rel_neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
                rel_neg_mask[non_neg_ids, torch.arange(num_targets)] = False

            if self.static_mode == 'abs':
                pos_mask = abs_pos_mask
                neg_mask = abs_neg_mask

            elif self.static_mode == 'abs_and_rel':
                pos_mask = abs_pos_mask & rel_pos_mask
                neg_mask = abs_neg_mask | rel_neg_mask

            elif self.static_mode == 'abs_or_rel':
                pos_mask = abs_pos_mask | rel_pos_mask
                neg_mask = abs_neg_mask & rel_neg_mask

            elif self.static_mode == 'rel':
                pos_mask = rel_pos_mask
                neg_mask = rel_neg_mask

            else:
                error_msg = f"Invalid static matching mode (got {self.static_mode})."
                raise ValueError(error_msg)

        else:
            error_msg = f"Invalid matching mode (got {self.mode})."
            raise ValueError(error_msg)

        # Get top query indices if requested
        if self.get_top_qry_ids:

            if self.mode == 'static' and 'rel' in self.static_mode:
                top_qry_ids = top_qry_ids[:self.top_limit]
            else:
                top_qry_ids = torch.topk(sim_matrix, self.top_limit, dim=0, sorted=True).indices

        # Remove matches if multiple targets per query are not allowed
        if not self.allow_multi_tgt:
            best_qry_ids = torch.arange(num_qrys, device=device)
            best_tgt_ids = torch.argmax(pos_mask.to(torch.float) * sim_matrix, dim=1)

            best_tgt_mask = torch.zeros_like(pos_mask)
            best_tgt_mask[best_qry_ids, best_tgt_ids] = True
            pos_mask = pos_mask & best_tgt_mask

        # Get match labels
        match_labels = torch.full(size=[num_qrys], fill_value=-1, device=device)
        match_labels[pos_mask.sum(dim=1) > 0] = 1
        match_labels[neg_mask.sum(dim=1) == num_targets] = 0

        # Get query and target indices of matches
        matched_qry_ids, matched_tgt_ids = pos_mask.nonzero(as_tuple=True)

        # Return matching results and top query indices if requested
        return_list = [match_labels, matched_qry_ids, matched_tgt_ids]
        return_list.append(top_qry_ids) if self.get_top_qry_ids else None

        return return_list


@MODELS.register_module()
class TopMatcher(nn.Module):
    """
    Class implementing the TopMatcher module.

    The module matches queries with targets, assigning one of following match labels to each query:
        - positive label with value 1 (query matches with at least one target);
        - negative label with value 0 (query does not match with any of the targets);
        - ignore label with value -1 (query has no matching verdict).

    Attributes:
        ids_key (str): String with key to retrieve the top query indices per target.
        qry_key (str): String with key to retrieve object from which the number of queries can be inferred.
        top_pos (int): Integer with maximum query rank to be considered positive w.r.t. target.
        top_neg (int): Integer with maximum query rank to be considered non-negative w.r.t. target.
        allow_multi_tgt (bool): Boolean indicating whether to allow multiple targets per query.
    """

    def __init__(self, ids_key, qry_key, top_pos=15, top_neg=15, allow_multi_tgt=True):
        """
        Initializes the TopMatcher module.

        Args:
            ids_key (str): String with key to retrieve the top query indices per target.
            qry_key (str): String with key to retrieve object from which the number of queries can be inferred.
            top_pos (int): Integer with maximum query rank to be considered positive w.r.t. target (default=15).
            top_neg (int): Integer with maximum query rank to be considered non-negative w.r.t. target (default=15).
            allow_multi_tgt (bool): Boolean indicating whether to allow multiple targets per query (default=True).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.ids_key = ids_key
        self.qry_key = qry_key
        self.top_pos = top_pos
        self.top_neg = top_neg
        self.allow_multi_tgt = allow_multi_tgt

    def forward(self, storage_dict, **kwargs):
        """
        Forward method of the TopMatcher module.

        Args:
            storage_dict (Dict): Storage dictionary containing at least following key:
                - {self.ids_key} (LongTensor): top query indices per target of shape [top_limit, num_targets];
                - (self.qry_key) (object): object from which the number of queries can be inferred.

            kwargs (Dict): Dictionary of keyword arguments not used by this module.

        Returns:
            storage_dict (Dict): Storage dictionary possibly containing following additional keys:
                - match_labels (LongTensor): match labels corresponding to each query of shape [num_qrys];
                - matched_qry_ids (LongTensor): indices of matched queries of shape [num_pos_qrys];
                - matched_tgt_ids (LongTensor): indices of corresponding matched targets of shape [num_pos_qrys].
        """

        # Retrieve top query indices per target
        top_qry_ids = storage_dict[self.ids_key]

        # Get number of queries
        num_qrys = len(storage_dict[self.qry_key])

        # Get device, top limit and number of targets
        device = top_qry_ids.device
        top_limit, num_targets = top_qry_ids.size()

        # Handle case where there are no queries or targets and return
        if (num_qrys == 0) or (num_targets == 0):
            storage_dict['match_labels'] = torch.zeros(num_qrys, dtype=torch.int64, device=device)
            storage_dict['matched_qry_ids'] = torch.zeros(0, dtype=torch.int64, device=device)
            storage_dict['matched_tgt_ids'] = torch.zeros(0, dtype=torch.int64, device=device)

            return storage_dict

        # Get query, target and rank indices
        qry_ids = top_qry_ids.flatten()

        tgt_ids = torch.arange(num_targets, device=device)
        tgt_ids = tgt_ids[None, :].expand(top_limit, -1).flatten()

        rank_ids = torch.arange(top_limit, device=device)
        rank_ids = rank_ids[:, None].expand(-1, num_targets).flatten()

        # Filter query, target and rank indices
        filter_mask = qry_ids >= 0

        qry_ids = qry_ids[filter_mask]
        tgt_ids = tgt_ids[filter_mask]
        rank_ids = rank_ids[filter_mask]

        # Get positive and non-negative label masks
        pos_mask = rank_ids < self.top_pos
        non_neg_mask = rank_ids < self.top_neg

        # Get query and target indices of matches
        matched_qry_ids = qry_ids[pos_mask]
        matched_tgt_ids = tgt_ids[pos_mask]

        # Remove matches if multiple targets per query are not allowed
        if not self.allow_multi_tgt:
            matched_qry_ids, inv_ids = matched_qry_ids.unique(return_inverse=True)
            argmin_ids = scatter_min(rank_ids[pos_mask], inv_ids, dim=0)[1]
            matched_tgt_ids = matched_tgt_ids[argmin_ids]

        # Get non-negative query indices
        non_neg_qry_ids = qry_ids[non_neg_mask]

        # Get match labels
        match_labels = torch.zeros(num_qrys, dtype=torch.int64, device=device)
        match_labels[non_neg_qry_ids] = -1
        match_labels[matched_qry_ids] = 1

        # Add matching results to storage dictionary
        storage_dict['match_labels'] = match_labels
        storage_dict['matched_qry_ids'] = matched_qry_ids
        storage_dict['matched_tgt_ids'] = matched_tgt_ids

        return storage_dict
