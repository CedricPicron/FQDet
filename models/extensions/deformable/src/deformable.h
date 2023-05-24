#pragma once

#include "cpu/msda_3d_cpu.h"

#ifdef WITH_CUDA
#include "cuda/msda_3d_cuda.h"
#endif

/*
=================== 
 MSDA 3D (forward)
===================

Args:
    in_feats (FloatTensor): Features to sample from of shape [batch_size, num_in_feats, num_heads, channels].
    map_hw (LongTensor): Feature map sizes in (H, W) format of shape [num_maps, 2].
    map_offs (LongTensor): Feature map offsets of shape [num_maps].
    sample_xyz (FloatTensor): Zero-one sample XYZ of shape [batch_size, num_out_feats, num_heads, num_pts, 3].
    attn_ws (FloatTensor): Attention weights of shape [batch_size, num_out_feats, num_heads, num_pts].

Returns:
    Output features (FloatTensor) of weighted samples of shape [batch_size, num_out_feats, num_heads, channels].

Remarks:
    We assume the 'sample_xyz' input argument is normalized between 0 and 1. This is guaranteed to be the case when
    using the MSDA3DF autograd function. However, if this function is used directly, please make sure the provided
    input argument 'sample_xyz' is indeed normalized between 0 and 1.
*/

at::Tensor msda_3d_forward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws)
{
    if (in_feats.device().is_cuda())
    {
        #ifdef WITH_CUDA
        return msda_3d_cuda_forward(in_feats, map_hw, map_offs, sample_xyz, attn_ws);

        #else
        AT_ERROR("Not compiled with GPU support.");
        #endif
    }
    return msda_3d_cpu_forward(in_feats, map_hw, map_offs, sample_xyz, attn_ws);
}

/*
=================== 
 MSDA 3D (backward)
===================

Args:
    in_feats (FloatTensor): Features to sample from of shape [batch_size, num_in_feats, num_heads, channels].
    map_hw (LongTensor): Feature map sizes in (H, W) format of shape [num_maps, 2].
    map_offs (LongTensor): Feature map offsets of shape [num_maps].
    sample_xyz (FloatTensor): Zero-one sample XYZ of shape [batch_size, num_out_feats, num_heads, num_pts, 3].
    attn_ws (FloatTensor): Attention weights of shape [batch_size, num_out_feats, num_heads, num_pts].
    grad_out_feats (FloatTensor): Grad output feats of shape [batch_size, num_out_feats, num_heads, channels].

Returns:
    Tuple containing following items:
        - Gradient (FloatTensor) of input features of shape [batch_size, num_in_feats, num_heads, channels].
        - Gradient (FloatTensor) of zero-one sample XYZ of shape [batch_size, num_out_feats, num_heads, num_pts, 3].
        - Gradient (FloatTensor) of attention weights of shape [batch_size, num_out_feats, num_heads, num_pts].
Remarks:
    We assume the 'sample_xyz' input argument is normalized between 0 and 1. This is guaranteed to be the case when
    using the MSDA3DF autograd function. However, if this function is used directly, please make sure the provided
    input argument 'sample_xyz' is indeed normalized between 0 and 1.
*/

std::tuple<at::Tensor, at::Tensor, at::Tensor> msda_3d_backward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws,
    const at::Tensor &grad_out_feats)
{
    if (in_feats.device().is_cuda())
    {
        #ifdef WITH_CUDA
        return msda_3d_cuda_backward(in_feats, map_hw, map_offs, sample_xyz, attn_ws, grad_out_feats);

        #else
        AT_ERROR("Not compiled with GPU support.");
        #endif
    }
    return msda_3d_cpu_backward(in_feats, map_hw, map_offs, sample_xyz, attn_ws, grad_out_feats);
}
