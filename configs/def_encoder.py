deform_encoder = dict(
    type='DeformEncoder',
    num_levels=5,
    feat_size=256,
    pos_embed_cfg=dict(
        type='mmdet.SinePositionalEncoding',
        num_feats=128,
        temperature=20,
        normalize=True,
        offset=0.0,
    ),
    with_lvl_embed=True,
    encoder_cfg=dict(
        type='mmdet.DeformableDetrTransformerEncoder',
        pop_num_layers=False,
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_levels=5,
                dropout=0.0,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
        ),
    ),
    out_ids=(3, 4, 5, 6, 7),
    out_sizes=(256, 256, 256, 256, 256),
)
