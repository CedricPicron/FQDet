"""
Collection of PyTorch-based insert functions.
"""

import torch


def pytorch_maps_insert_2d(map_feats, map_wh, map_offs, insert_feats, insert_xy, insert_map_ids):
    """
    Function inserting features at 2D locations of collection of feature maps using PyTorch functions.

    Args:
        map_feats (FloatTensor): Concatenated map features of shape [batch_size, num_map_feats, feat_size].
        map_wh (LongTensor): Map sizes in (W, H) format of shape [num_maps, 2].
        map_offs (LongTensor): Map offsets of shape [num_maps].
        insert_feats (FloatTensor): Features to be inserted of shape [batch_size, num_inserts, feat_size].
        insert_xy (FloatTensor): Zero-one normalized (X, Y) insert locations of shape [batch_size, num_inserts, 2].
        insert_map_ids (LongTensor): Feature map indices of insert locations of shape [batch_size, num_inserts].

    Returns:
        map_feats (FloatTensor): Map features with added insertions of shape [batch_size, num_map_feats, feat_size].
    """

    # Get feature size
    batch_size, _, feat_size = map_feats.size()
    _, num_inserts, _ = insert_xy.size()

    # Clamp insert locations between 0 and 1
    insert_xy = insert_xy.clamp_(min=0.0, max=1.0)

    # Get insert widths, heights and offsets
    insert_wh = map_wh[insert_map_ids]
    insert_w = insert_wh[:, :, 0]
    insert_offs = map_offs[insert_map_ids]

    # Get unnormalized insert locations
    insert_xy = insert_xy * (insert_wh - 1)

    # Get insert indices
    insert_ij = insert_xy.floor().to(dtype=torch.int64)
    insert_ij = torch.min(torch.stack([insert_ij, insert_wh-2], dim=3), dim=3)[0]

    top_left_ids = insert_offs + insert_w * insert_ij[:, :, 1] + insert_ij[:, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + insert_w
    bot_right_ids = bot_left_ids + 1

    insert_ids = torch.stack([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=2)
    insert_ids = insert_ids.view(batch_size, num_inserts*4, 1).expand(-1, -1, feat_size)

    # Get weighted insert features
    right_weights, bot_weights = (insert_xy - insert_ij).split(1, dim=2)
    left_weights, top_weights = (1-right_weights, 1-bot_weights)

    top_left_ws = left_weights * top_weights
    top_right_ws = right_weights * top_weights
    bot_left_ws = left_weights * bot_weights
    bot_right_ws = right_weights * bot_weights

    insert_weights = torch.stack([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=2)
    insert_feats = insert_weights * insert_feats[:, :, None, :]
    insert_feats = insert_feats.view(batch_size, num_inserts*4, feat_size)

    # Add weighted insert features to map features
    map_feats.scatter_add_(dim=1, index=insert_ids, src=insert_feats)

    return map_feats


def pytorch_maps_insert_3d(map_feats, map_wh, map_offs, insert_feats, insert_xyz):
    """
    Function inserting features at 3D locations of collection of feature maps using PyTorch functions.

    Args:
        map_feats (FloatTensor): Concatenated map features of shape [batch_size, num_map_feats, feat_size].
        map_wh (LongTensor): Map sizes in (W, H) format of shape [num_maps, 2].
        map_offs (LongTensor): Map offsets of shape [num_maps].
        insert_feats (FloatTensor): Features to be inserted of shape [batch_size, num_inserts, feat_size].
        insert_xyz (FloatTensor): Zero-one normalized (X, Y, Z) insert locations of shape [batch_size, num_inserts, 3].

    Returns:
        map_feats (FloatTensor): Map features with added insertions of shape [batch_size, num_map_feats, feat_size].
    """

    # Get some sizes
    batch_size, _, feat_size = map_feats.size()
    num_maps, _ = map_wh.size()
    _, num_inserts, _ = insert_xyz.size()

    # Clamp insert locations between 0 and 1
    insert_xyz = insert_xyz.clamp_(min=0.0, max=1.0)

    # Get insert map indices
    insert_z = insert_xyz[:, :, 2] * (num_maps - 1)
    insert_k = insert_z.floor().to(dtype=torch.int64).clamp_(max=num_maps-2)
    insert_map_ids = torch.stack([insert_k, insert_k+1], dim=2)

    # Get insert widths, heights and offsets
    insert_wh = map_wh[insert_map_ids]
    insert_w = insert_wh[:, :, :, 0]
    insert_offs = map_offs[insert_map_ids]

    # Get unnormalized insert locations
    insert_xy = insert_xyz[:, :, None, :2] * (insert_wh - 1)

    # Get insert indices
    insert_ij = insert_xy.floor().to(dtype=torch.int64)
    insert_ij = torch.min(torch.stack([insert_ij, insert_wh-2], dim=4), dim=4)[0]

    top_left_ids = insert_offs + insert_w * insert_ij[:, :, :, 1] + insert_ij[:, :, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + insert_w
    bot_right_ids = bot_left_ids + 1

    insert_ids = torch.stack([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=3)
    insert_ids = insert_ids.view(batch_size, num_inserts*8, 1).expand(-1, -1, feat_size)

    # Get weighted insert features
    right_weights, bot_weights = (insert_xy - insert_ij).split(1, dim=3)
    left_weights, top_weights = (1-right_weights, 1-bot_weights)

    back_weights = insert_z - insert_k
    front_weights = 1-back_weights
    front_back_weights = torch.stack([front_weights, back_weights], dim=2).unsqueeze(dim=3)

    top_left_ws = left_weights * top_weights * front_back_weights
    top_right_ws = right_weights * top_weights * front_back_weights
    bot_left_ws = left_weights * bot_weights * front_back_weights
    bot_right_ws = right_weights * bot_weights * front_back_weights

    insert_weights = torch.stack([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=3)
    insert_weights = insert_weights.view(batch_size, num_inserts, 8, 1)

    insert_feats = insert_weights * insert_feats[:, :, None, :]
    insert_feats = insert_feats.view(batch_size, num_inserts*8, feat_size)

    # Add weighted insert features to map features
    map_feats.scatter_add_(dim=1, index=insert_ids, src=insert_feats)

    return map_feats
