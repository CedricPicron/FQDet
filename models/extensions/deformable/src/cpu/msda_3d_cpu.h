#pragma once
#include <torch/extension.h>

at::Tensor msda_3d_cpu_forward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws);

std::tuple<at::Tensor, at::Tensor, at::Tensor> msda_3d_cpu_backward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws,
    const at::Tensor &grad_out_feats);
