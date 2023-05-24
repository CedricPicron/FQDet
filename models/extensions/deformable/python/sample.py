"""
Collection of PyTorch-based sample functions.
"""

import torch


def pytorch_maps_sample_2d(feats, feat_map_wh, feat_map_offs, sample_xy, sample_map_ids, return_derivatives=False):
    """
    Function sampling 2D locations from collection of 2D feature maps using PyTorch functions.

    Args:
        feats (FloatTensor): Features to sample from of shape [batch_size, num_feats, feat_size].
        feat_map_wh (LongTensor): Feature map sizes in (W, H) format of shape [num_maps, 2].
        feat_map_offs (LongTensor): Feature map offsets of shape [num_maps].
        sample_xy (FloatTensor): Zero-one normalized (X, Y) sample locations of shape [batch_size, num_samples, 2].
        sample_map_ids (LongTensor): Feature map indices to sample from of shape [batch_size, num_samples].
        return_derivatives (bool): Boolean indicating whether derivatives should be returned or not (default=False).

    Returns:
        sampled_feats (FloatTensor): Sampled features of shape [batch_size, num_samples, feat_size].

        If return_derivatives is True:
            dx (FloatTensor): X derivative of shape [batch_size, num_samples, feat_size].
            dy (FloatTensor): Y derivative of shape [batch_size, num_samples, feat_size].
    """

    # Get feature size
    batch_size, _, feat_size = feats.size()
    _, num_samples, _ = sample_xy.size()

    # Clamp sample locations between 0 and 1
    sample_xy = sample_xy.clamp_(min=0.0, max=1.0)

    # Get sample widths, heights and offsets
    sample_wh = feat_map_wh[sample_map_ids]
    sample_w = sample_wh[:, :, 0]
    sample_offs = feat_map_offs[sample_map_ids]

    # Get unnormalized sample locations
    sample_xy = sample_xy * (sample_wh - 1)

    # Get corner features
    sample_ij = sample_xy.floor().to(dtype=torch.int64)
    sample_ij = torch.min(torch.stack([sample_ij, sample_wh-2], dim=3), dim=3)[0]

    top_left_ids = sample_offs + sample_w * sample_ij[:, :, 1] + sample_ij[:, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + sample_w
    bot_right_ids = bot_left_ids + 1

    sample_ids = torch.cat([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=1)
    corner_feats = torch.gather(feats, dim=1, index=sample_ids[:, :, None].expand(-1, -1, feat_size))

    # Get corner weights
    right_weights, bot_weights = (sample_xy - sample_ij).split(1, dim=2)
    left_weights, top_weights = (1-right_weights, 1-bot_weights)

    top_left_ws = left_weights * top_weights
    top_right_ws = right_weights * top_weights
    bot_left_ws = left_weights * bot_weights
    bot_right_ws = right_weights * bot_weights
    corner_weights = torch.cat([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=1)

    # Get 2D sampled features
    sampled_feats = corner_weights * corner_feats
    sampled_feats = sampled_feats.view(batch_size, 4, num_samples, feat_size).sum(dim=1)

    # Return if no derivatives are requested
    if not return_derivatives:
        return sampled_feats

    # Get X and Y derivatives
    top_left_feats, top_right_feats, bot_left_feats, bot_right_feats = torch.chunk(corner_feats, chunks=4, dim=1)
    dx = top_weights * (top_right_feats-top_left_feats) + bot_weights * (bot_right_feats-bot_left_feats)
    dy = left_weights * (bot_left_feats-top_left_feats) + right_weights * (bot_right_feats-top_right_feats)

    return sampled_feats, dx, dy


def pytorch_maps_sample_3d(feats, feat_map_wh, feat_map_offs, sample_xyz, return_derivatives=False):
    """
    Function sampling 3D locations from collection of 2D feature maps using PyTorch functions.

    Args:
        feats (FloatTensor): Features to sample from of shape [batch_size, num_feats, feat_size].
        feat_map_wh (LongTensor): Feature map sizes in (W, H) format of shape [num_maps, 2].
        feat_map_offs (LongTensor): Feature map offsets of shape [num_maps].
        sample_xyz (FloatTensor): Zero-one normalized (X, Y, Z) sample locations of shape [batch_size, num_samples, 3].
        return_derivatives (bool): Boolean indicating whether derivatives should be returned or not (default=False).

    Returns:
        sampled_feats (FloatTensor): Sampled features of shape [batch_size, num_samples, feat_size].

        If return_derivatives is True:
            dx (FloatTensor): X derivative of shape [batch_size, num_samples, feat_size].
            dy (FloatTensor): Y derivative of shape [batch_size, num_samples, feat_size].
            dz (FloatTensor): Z derivative of shape [batch_size, num_samples, feat_size].
    """

    # Get some sizes
    batch_size, _, feat_size = feats.size()
    num_maps, _ = feat_map_wh.size()
    _, num_samples, _ = sample_xyz.size()

    # Clamp sample locations between 0 and 1
    sample_xyz = sample_xyz.clamp_(min=0.0, max=1.0)

    # Get sample map indices
    sample_z = sample_xyz[:, :, 2] * (num_maps - 1)
    sample_k = sample_z.floor().to(dtype=torch.int64).clamp_(max=num_maps-2)
    sample_map_ids = torch.stack([sample_k, sample_k+1], dim=2)

    # Get sample widths, heights and offsets
    sample_wh = feat_map_wh[sample_map_ids]
    sample_w = sample_wh[:, :, :, 0]
    sample_offs = feat_map_offs[sample_map_ids]

    # Get unnormalized sample locations
    sample_xy = sample_xyz[:, :, None, :2] * (sample_wh - 1)

    # Get unweighted corner features
    sample_ij = sample_xy.floor().to(dtype=torch.int64)
    sample_ij = torch.min(torch.stack([sample_ij, sample_wh-2], dim=4), dim=4)[0]

    top_left_ids = sample_offs + sample_w * sample_ij[:, :, :, 1] + sample_ij[:, :, :, 0]
    top_right_ids = top_left_ids + 1
    bot_left_ids = top_left_ids + sample_w
    bot_right_ids = bot_left_ids + 1

    sample_ids = torch.cat([top_left_ids, top_right_ids, bot_left_ids, bot_right_ids], dim=1).flatten(1)
    corner_feats = torch.gather(feats, dim=1, index=sample_ids[:, :, None].expand(-1, -1, feat_size))
    corner_feats = corner_feats.view(batch_size, 4*num_samples, 2, feat_size)

    # Get 2D corner weights
    right_weights, bot_weights = (sample_xy - sample_ij).split(1, dim=3)
    left_weights, top_weights = (1-right_weights, 1-bot_weights)

    top_left_ws = left_weights * top_weights
    top_right_ws = right_weights * top_weights
    bot_left_ws = left_weights * bot_weights
    bot_right_ws = right_weights * bot_weights
    corner_weights = torch.cat([top_left_ws, top_right_ws, bot_left_ws, bot_right_ws], dim=1)

    # Get 2D sampled features
    sampled_feats_2d = corner_weights * corner_feats
    sampled_feats_2d = sampled_feats_2d.view(batch_size, 4, num_samples, 2, feat_size).sum(dim=1)

    # Get 3D sampled features
    back_weights = sample_z - sample_k
    front_weights = 1-back_weights

    front_back_weights = torch.stack([front_weights, back_weights], dim=2).unsqueeze(dim=3)
    sampled_feats = (front_back_weights*sampled_feats_2d).sum(dim=2)

    # Return if no derivatives are requested
    if not return_derivatives:
        return sampled_feats

    # Get X, Y and Z derivatives
    top_left_feats, top_right_feats, bot_left_feats, bot_right_feats = torch.chunk(corner_feats, chunks=4, dim=1)
    dx = top_weights * (top_right_feats-top_left_feats) + bot_weights * (bot_right_feats-bot_left_feats)
    dy = left_weights * (bot_left_feats-top_left_feats) + right_weights * (bot_right_feats-top_right_feats)

    dx = (front_back_weights*dx).sum(dim=2)
    dy = (front_back_weights*dy).sum(dim=2)
    dz = sampled_feats_2d[:, :, 1, :] - sampled_feats_2d[:, :, 0, :]

    return sampled_feats, dx, dy, dz
