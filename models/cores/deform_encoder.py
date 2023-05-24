"""
Deformable encoder core.
"""

from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmdet.models.detectors import DeformableDETR
import torch
from torch import nn
import torch.nn.functional as F

from models.build import build_model, MODELS


@MODELS.register_module()
class DeformEncoder(nn.Module):
    """
    Class implementing the DeformEncoder module.

    Attributes:
        pos_embed (nn.Module): Optional module computing position embeddings.
        lvl_embed (nn.Parameter): Optional parameter containing the level embeddings.
        encoder (nn.Module): Encoder module updating the feature maps.
    """

    def __init__(self, num_levels, feat_size, encoder_cfg, pos_embed_cfg=None, with_lvl_embed=False):
        """
        Initializes the DeformEncoder module.

        Args:
            num_levels (int): Integer containing the number of levels (i.e. the number of feature maps).
            feat_size (int): Integer containing the feature size.
            encoder_cfg (Dict): Configuration dictionary specifying the encoder module.
            pos_embed_cfg (Dict): Configuration dictionary specifying the position embedding module (default=None).
            with_lvl_embed (bool): Boolean indicating whether to use learnable level embeddings (default=False).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Build position embedding module if needed
        self.pos_embed = build_model(pos_embed_cfg) if pos_embed_cfg is not None else None

        # Initialize level embeddings if needed
        if with_lvl_embed:
            self.lvl_embed = nn.Parameter(torch.empty(num_levels, feat_size))
            nn.init.normal_(self.lvl_embed)

        else:
            self.lvl_embed = None

        # Build encoder module
        self.encoder = build_model(encoder_cfg)

        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        for module in self.encoder.modules():
            if isinstance(module, MultiScaleDeformableAttention):
                module.init_weights()

    def forward(self, in_feat_maps, images, **kwargs):
        """
        Forward method the DeformEncoder module.

        Args:
            in_feat_maps (List): Input feature maps [num_maps] of shape [batch_size, feat_size, fH, fW].
            images (Images): Images structure of size [batch_size] containing the batched images.
            kwargs (Dict): Dictionary of unused keyword arguments.

        Returns:
            out_feat_maps (List): Output feature maps [num_maps] of shape [batch_size, feat_size, fH, fW].
        """

        # Get batch size and device
        batch_size = len(in_feat_maps[0])
        device = in_feat_maps[0].device

        # Get encoder inputs
        iW, iH = images.size()
        base_pad_mask = torch.ones(batch_size, iH, iW, dtype=torch.float, device=device)

        for i, (iW, iH) in enumerate(images.size(mode='without_padding')):
            base_pad_mask[i, :iH, :iW] = 0.0

        in_feats_list = []
        embeds_list = []
        pad_masks = []
        spatial_shapes = []
        valid_ratios = []

        for i, in_feat_map in enumerate(in_feat_maps):
            in_feats = in_feat_map.flatten(2).transpose(1, 2)
            embeds = torch.zeros_like(in_feats)

            spatial_shape = tuple(in_feat_map.size()[2:])
            pad_mask = F.interpolate(base_pad_mask[None], size=spatial_shape).bool()[0]
            valid_ratio = DeformableDETR.get_valid_ratio(pad_mask)

            if self.pos_embed is not None:
                embeds += self.pos_embed(pad_mask).flatten(2).transpose(1, 2)

            if self.lvl_embed is not None:
                embeds += self.lvl_embed[i]

            in_feats_list.append(in_feats)
            embeds_list.append(embeds)
            pad_masks.append(pad_mask.flatten(1))
            spatial_shapes.append(spatial_shape)
            valid_ratios.append(valid_ratio)

        in_feats = torch.cat(in_feats_list, dim=1)
        embeds = torch.cat(embeds_list, dim=1)
        pad_mask = torch.cat(pad_masks, dim=1)
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device)
        lvl_start_ids = torch.cat([spatial_shapes.new_zeros([1]), spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1]])
        valid_ratios = torch.stack(valid_ratios, dim=1)

        # Apply encoder
        input_dict = {'query': in_feats, 'query_pos': embeds, 'key_padding_mask': pad_mask}
        input_dict = {**input_dict, 'spatial_shapes': spatial_shapes, 'level_start_index': lvl_start_ids}
        input_dict = {**input_dict, 'valid_ratios': valid_ratios}
        out_feats = self.encoder(**input_dict)

        # Get output feature maps
        feats_per_map = spatial_shapes.prod(dim=1).tolist()
        out_feat_maps = out_feats.split(feats_per_map, dim=1)
        out_feat_maps = [out_feat_maps[i].transpose(1, 2).view_as(in_feat_maps[i]) for i in range(len(in_feat_maps))]

        return out_feat_maps
