core = dict(
    type='FPN',
    in_channels=[512, 1024, 2048],
    out_channels=256,
    num_outs=5,
    out_ids=(3, 4, 5, 6, 7),
    out_sizes=(256, 256, 256, 256, 256),
    add_extra_convs='on_lateral',
    relu_before_extra_convs=True,
)
