#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/macros/Macros.h>

constexpr int NUM_THREADS = 256;

using namespace at::cuda::detail;

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(NUM_THREADS)
__global__ void msda_3d_cuda_forward_kernel(
  const int num_kernels,
  TensorInfo<scalar_t, int> in_feats,
  TensorInfo<int64_t, int> map_hw,
  TensorInfo<int64_t, int> map_offs,
  TensorInfo<scalar_t, int> sample_xyz,
  TensorInfo<scalar_t, int> attn_ws,
  TensorInfo<scalar_t, int> out_feats)
{
  // Get sizes
  const int H = in_feats.sizes[2];
  const int C = in_feats.sizes[3];
  const int M = map_hw.sizes[0];
  const int F = sample_xyz.sizes[1];
  const int P = sample_xyz.sizes[3];

  // Get strides
  const int in_sN = in_feats.strides[0];
  const int in_sF = in_feats.strides[1];
  const int in_sH = in_feats.strides[2];
  const int in_sC = in_feats.strides[3];

  const int map_sM = map_hw.strides[0];
  const int map_sC = map_hw.strides[1];

  const int sample_sN = sample_xyz.strides[0];
  const int sample_sF = sample_xyz.strides[1];
  const int sample_sH = sample_xyz.strides[2];
  const int sample_sP = sample_xyz.strides[3];
  const int sample_sC = sample_xyz.strides[4];

  const int attn_sN = attn_ws.strides[0];
  const int attn_sF = attn_ws.strides[1];
  const int attn_sH = attn_ws.strides[2];
  const int attn_sP = attn_ws.strides[3];

  const int out_sN = out_feats.strides[0];
  const int out_sF = out_feats.strides[1];
  const int out_sH = out_feats.strides[2];
  const int out_sC = out_feats.strides[3];

  CUDA_KERNEL_LOOP(index, num_kernels)
  {
    // Get kernel-specific indices
    const int c = index % C;
    const int h = (index / C) % H;
    const int f = (index / (C * H)) % F;
    const int n = index / (C * H * F);

    // Get initial sample and attention offsets
    int sample_offset = n * sample_sN + f * sample_sF + h * sample_sH;
    int attn_offset = n * attn_sN + f * attn_sF + h * attn_sH;

    // Get base input pointer
    auto in_ptr = in_feats.data + n * in_sN + h * in_sH + c * in_sC;

    // Get output pointer
    auto out_ptr = out_feats.data + n * out_sN + f * out_sF + h * out_sH + c * out_sC;

    // Loop over points
    for (int p = 0; p < P; ++p, sample_offset += sample_sP, attn_offset += attn_sP)
    {
      // Get normalized sample locations
      scalar_t ix = sample_xyz.data[sample_offset];
      scalar_t iy = sample_xyz.data[sample_offset + sample_sC];
      scalar_t iz = sample_xyz.data[sample_offset + 2 * sample_sC];

      // Get unnormalized Z sample locations
      float iz_f = static_cast<float>(iz) * (M-1);
      int iz_0 = static_cast<int>(::min(::floor(iz_f), static_cast<float>(M-2)));
      int iz_1 = iz_0 + 1;

      // Get map widths and heights
      int64_t h_0 = map_hw.data[iz_0 * map_sM];
      int64_t w_0 = map_hw.data[iz_0 * map_sM + map_sC];
      int64_t h_1 = map_hw.data[iz_1 * map_sM];
      int64_t w_1 = map_hw.data[iz_1 * map_sM + map_sC];

      // Get unnormalized X and Y sample locations
      float ix_f = static_cast<float>(ix);
      float iy_f = static_cast<float>(iy);

      float ix_f0 = ix_f * static_cast<float>(w_0 - 1);
      float ix_f1 = ix_f * static_cast<float>(w_1 - 1);      
      float iy_f0 = iy_f * static_cast<float>(h_0 - 1);
      float iy_f1 = iy_f * static_cast<float>(h_1 - 1);

      scalar_t ix_0 = static_cast<scalar_t>(ix_f0);
      scalar_t ix_1 = static_cast<scalar_t>(ix_f1);
      scalar_t iy_0 = static_cast<scalar_t>(iy_f0);
      scalar_t iy_1 = static_cast<scalar_t>(iy_f1);

      int ix_00 = static_cast<int>(::min(::floor(ix_f0), static_cast<float>(w_0-2)));
      int ix_10 = static_cast<int>(::min(::floor(ix_f1), static_cast<float>(w_1-2)));
      int iy_00 = static_cast<int>(::min(::floor(iy_f0), static_cast<float>(h_0-2)));
      int iy_10 = static_cast<int>(::min(::floor(iy_f1), static_cast<float>(h_1-2)));

      int ix_01 = ix_00 + 1;
      int ix_11 = ix_10 + 1;
      int iy_01 = iy_00 + 1;
      int iy_11 = iy_10 + 1;

      // Get X and Y sampling weights
      scalar_t wx_00 = ix_01 - ix_0;
      scalar_t wy_00 = iy_01 - iy_0;
      scalar_t wx_10 = ix_11 - ix_1;
      scalar_t wy_10 = iy_11 - iy_1;
      scalar_t wx_01 = ix_0 - ix_00;
      scalar_t wy_01 = iy_0 - iy_00;
      scalar_t wx_11 = ix_1 - ix_10;
      scalar_t wy_11 = iy_1 - iy_10;

      // Get Z sampling weights
      iz = static_cast<scalar_t>(iz_f);
      scalar_t wz_0 = iz_1 - iz;
      scalar_t wz_1 = iz - iz_0;

      // Get map offsets
      int64_t off_0 = map_offs.data[iz_0];
      int64_t off_1 = map_offs.data[iz_1];

      // Get input pointers
      auto in_ptr_000 = in_ptr + (off_0 + w_0 * iy_00 + ix_00) * in_sF;
      auto in_ptr_001 = in_ptr + (off_1 + w_1 * iy_10 + ix_10) * in_sF;
      auto in_ptr_010 = in_ptr + (off_0 + w_0 * iy_01 + ix_00) * in_sF;
      auto in_ptr_011 = in_ptr + (off_1 + w_1 * iy_11 + ix_10) * in_sF;
      auto in_ptr_100 = in_ptr + (off_0 + w_0 * iy_00 + ix_01) * in_sF;
      auto in_ptr_101 = in_ptr + (off_1 + w_1 * iy_10 + ix_11) * in_sF;
      auto in_ptr_110 = in_ptr + (off_0 + w_0 * iy_01 + ix_01) * in_sF;
      auto in_ptr_111 = in_ptr + (off_1 + w_1 * iy_11 + ix_11) * in_sF;

      // Add weighted sampled inputs to output
      scalar_t val_0 = 0;
      val_0 += (*in_ptr_000 * wx_00 + *in_ptr_100 * wx_01) * wy_00;
      val_0 += (*in_ptr_010 * wx_00 + *in_ptr_110 * wx_01) * wy_01;
      val_0 *= wz_0;

      scalar_t val_1 = 0;
      val_1 += (*in_ptr_001 * wx_10 + *in_ptr_101 * wx_11) * wy_10;
      val_1 += (*in_ptr_011 * wx_10 + *in_ptr_111 * wx_11) * wy_11;
      val_1 *= wz_1;

      scalar_t w = attn_ws.data[attn_offset];
      *out_ptr += (val_0 + val_1) * w;
    }
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(NUM_THREADS)
__global__ void msda_3d_cuda_backward_kernel(
  const int num_kernels,
  TensorInfo<scalar_t, int> in_feats,
  TensorInfo<int64_t, int> map_hw,
  TensorInfo<int64_t, int> map_offs,
  TensorInfo<scalar_t, int> sample_xyz,
  TensorInfo<scalar_t, int> attn_ws,
  TensorInfo<scalar_t, int> grad_out_feats,
  TensorInfo<scalar_t, int> grad_in_feats,
  TensorInfo<scalar_t, int> grad_sample_xyz,
  TensorInfo<scalar_t, int> grad_attn_ws)
{
  // Get sizes
  const int H = in_feats.sizes[2];
  const int C = in_feats.sizes[3];
  const int M = map_hw.sizes[0];
  const int F = sample_xyz.sizes[1];
  const int P = sample_xyz.sizes[3];

  // Get strides
  const int in_sN = in_feats.strides[0];
  const int in_sF = in_feats.strides[1];
  const int in_sH = in_feats.strides[2];
  const int in_sC = in_feats.strides[3];

  const int map_sM = map_hw.strides[0];
  const int map_sC = map_hw.strides[1];

  const int sample_sN = sample_xyz.strides[0];
  const int sample_sF = sample_xyz.strides[1];
  const int sample_sH = sample_xyz.strides[2];
  const int sample_sP = sample_xyz.strides[3];
  const int sample_sC = sample_xyz.strides[4];

  const int attn_sN = attn_ws.strides[0];
  const int attn_sF = attn_ws.strides[1];
  const int attn_sH = attn_ws.strides[2];
  const int attn_sP = attn_ws.strides[3];

  const int gOut_sN = grad_out_feats.strides[0];
  const int gOut_sF = grad_out_feats.strides[1];
  const int gOut_sH = grad_out_feats.strides[2];
  const int gOut_sC = grad_out_feats.strides[3];

  const int gIn_sN = grad_in_feats.strides[0];
  const int gIn_sF = grad_in_feats.strides[1];
  const int gIn_sH = grad_in_feats.strides[2];
  const int gIn_sC = grad_in_feats.strides[3];

  const int gSample_sN = grad_sample_xyz.strides[0];
  const int gSample_sF = grad_sample_xyz.strides[1];
  const int gSample_sH = grad_sample_xyz.strides[2];
  const int gSample_sP = grad_sample_xyz.strides[3];
  const int gSample_sC = grad_sample_xyz.strides[4];

  const int gAttn_sN = grad_attn_ws.strides[0];
  const int gAttn_sF = grad_attn_ws.strides[1];
  const int gAttn_sH = grad_attn_ws.strides[2];
  const int gAttn_sP = grad_attn_ws.strides[3];

  CUDA_KERNEL_LOOP(index, num_kernels)
  {
    // Get kernel-specific indices
    const int c = index % C;
    const int h = (index / C) % H;
    const int f = (index / (C * H)) % F;
    const int n = index / (C * H * F);

    // Get initial sample and attention offsets
    int sample_offset = n * sample_sN + f * sample_sF + h * sample_sH;
    int attn_offset = n * attn_sN + f * attn_sF + h * attn_sH;

    // Get base (gradient) input pointers
    auto in_ptr = in_feats.data + n * in_sN + h * in_sH + c * in_sC;
    auto gIn_ptr = grad_in_feats.data + n * gIn_sN + h * gIn_sH + c * gIn_sC;

    // Get output gradient
    scalar_t gOut = grad_out_feats.data[n * gOut_sN + f * gOut_sF + h * gOut_sH + c * gOut_sC];

    // Get initial gradient sample and gradient attention pointers
    auto gSample_ptr = grad_sample_xyz.data + n * gSample_sN + f * gSample_sF + h * gSample_sH;
    auto gAttn_ptr = grad_attn_ws.data + n * gAttn_sN + f * gAttn_sF + h * gAttn_sH;

    // Loop over points
    for (int p = 0; p < P; ++p, sample_offset += sample_sP, attn_offset += attn_sP, gSample_ptr += gSample_sP,
         gAttn_ptr += gAttn_sP)
    {
      // Get normalized sample locations
      scalar_t ix = sample_xyz.data[sample_offset];
      scalar_t iy = sample_xyz.data[sample_offset + sample_sC];
      scalar_t iz = sample_xyz.data[sample_offset + 2 * sample_sC];

      // Get unnormalized Z sample locations
      float iz_f = static_cast<float>(iz) * (M-1);
      int iz_0 = static_cast<int>(::min(::floor(iz_f), static_cast<float>(M-2)));
      int iz_1 = iz_0 + 1;

      // Get map widths and heights
      int64_t h_0 = map_hw.data[iz_0 * map_sM];
      int64_t w_0 = map_hw.data[iz_0 * map_sM + map_sC];
      int64_t h_1 = map_hw.data[iz_1 * map_sM];
      int64_t w_1 = map_hw.data[iz_1 * map_sM + map_sC];

      // Get unnormalized X and Y sample locations
      float ix_f = static_cast<float>(ix);
      float iy_f = static_cast<float>(iy);

      float ix_f0 = ix_f * static_cast<float>(w_0 - 1);
      float ix_f1 = ix_f * static_cast<float>(w_1 - 1);      
      float iy_f0 = iy_f * static_cast<float>(h_0 - 1);
      float iy_f1 = iy_f * static_cast<float>(h_1 - 1);

      scalar_t ix_0 = static_cast<scalar_t>(ix_f0);
      scalar_t ix_1 = static_cast<scalar_t>(ix_f1);
      scalar_t iy_0 = static_cast<scalar_t>(iy_f0);
      scalar_t iy_1 = static_cast<scalar_t>(iy_f1);

      int ix_00 = static_cast<int>(::min(::floor(ix_f0), static_cast<float>(w_0-2)));
      int ix_10 = static_cast<int>(::min(::floor(ix_f1), static_cast<float>(w_1-2)));
      int iy_00 = static_cast<int>(::min(::floor(iy_f0), static_cast<float>(h_0-2)));
      int iy_10 = static_cast<int>(::min(::floor(iy_f1), static_cast<float>(h_1-2)));

      int ix_01 = ix_00 + 1;
      int ix_11 = ix_10 + 1;
      int iy_01 = iy_00 + 1;
      int iy_11 = iy_10 + 1;

      // Get X and Y sampling weights
      scalar_t wx_00 = ix_01 - ix_0;
      scalar_t wy_00 = iy_01 - iy_0;
      scalar_t wx_10 = ix_11 - ix_1;
      scalar_t wy_10 = iy_11 - iy_1;
      scalar_t wx_01 = ix_0 - ix_00;
      scalar_t wy_01 = iy_0 - iy_00;
      scalar_t wx_11 = ix_1 - ix_10;
      scalar_t wy_11 = iy_1 - iy_10;

      // Get Z sampling weights
      iz = static_cast<scalar_t>(iz_f);
      scalar_t wz_0 = iz_1 - iz;
      scalar_t wz_1 = iz - iz_0;

      // Get map offsets
      int64_t off_0 = map_offs.data[iz_0];
      int64_t off_1 = map_offs.data[iz_1];

      // Get input pointers
      auto in_ptr_000 = in_ptr + (off_0 + w_0 * iy_00 + ix_00) * in_sF;
      auto in_ptr_001 = in_ptr + (off_1 + w_1 * iy_10 + ix_10) * in_sF;
      auto in_ptr_010 = in_ptr + (off_0 + w_0 * iy_01 + ix_00) * in_sF;
      auto in_ptr_011 = in_ptr + (off_1 + w_1 * iy_11 + ix_10) * in_sF;
      auto in_ptr_100 = in_ptr + (off_0 + w_0 * iy_00 + ix_01) * in_sF;
      auto in_ptr_101 = in_ptr + (off_1 + w_1 * iy_10 + ix_11) * in_sF;
      auto in_ptr_110 = in_ptr + (off_0 + w_0 * iy_01 + ix_01) * in_sF;
      auto in_ptr_111 = in_ptr + (off_1 + w_1 * iy_11 + ix_11) * in_sF;

      // Get gradient input pointers
      auto gIn_ptr_000 = gIn_ptr + (off_0 + w_0 * iy_00 + ix_00) * gIn_sF;
      auto gIn_ptr_001 = gIn_ptr + (off_1 + w_1 * iy_10 + ix_10) * gIn_sF;
      auto gIn_ptr_010 = gIn_ptr + (off_0 + w_0 * iy_01 + ix_00) * gIn_sF;
      auto gIn_ptr_011 = gIn_ptr + (off_1 + w_1 * iy_11 + ix_10) * gIn_sF;
      auto gIn_ptr_100 = gIn_ptr + (off_0 + w_0 * iy_00 + ix_01) * gIn_sF;
      auto gIn_ptr_101 = gIn_ptr + (off_1 + w_1 * iy_10 + ix_11) * gIn_sF;
      auto gIn_ptr_110 = gIn_ptr + (off_0 + w_0 * iy_01 + ix_01) * gIn_sF;
      auto gIn_ptr_111 = gIn_ptr + (off_1 + w_1 * iy_11 + ix_11) * gIn_sF;

      // Get attention weight
      scalar_t w = attn_ws.data[attn_offset];

      // Get input gradients
      scalar_t gVal = gOut * w;

      scalar_t gVal_0 = gVal * wz_0;
      scalar_t gVal_1 = gVal * wz_1;

      scalar_t gVal_00 = gVal_0 * wy_00;
      scalar_t gVal_01 = gVal_0 * wy_01;
      scalar_t gVal_10 = gVal_1 * wy_10;
      scalar_t gVal_11 = gVal_1 * wy_11;

      atomicAdd(gIn_ptr_000, gVal_00 * wx_00);
      atomicAdd(gIn_ptr_100, gVal_00 * wx_01);
      atomicAdd(gIn_ptr_010, gVal_01 * wx_00);
      atomicAdd(gIn_ptr_110, gVal_01 * wx_01);
      atomicAdd(gIn_ptr_001, gVal_10 * wx_10);
      atomicAdd(gIn_ptr_101, gVal_10 * wx_11);
      atomicAdd(gIn_ptr_011, gVal_11 * wx_10);
      atomicAdd(gIn_ptr_111, gVal_11 * wx_11);

      // Get sample gradients
      scalar_t grad_sample_x0 = gVal_00 * (*in_ptr_100 - *in_ptr_000);
      grad_sample_x0 += gVal_01 * (*in_ptr_110 - *in_ptr_010);
      grad_sample_x0 *= (w_0 - 1);

      scalar_t grad_sample_x1 = gVal_10 * (*in_ptr_101 - *in_ptr_001);
      grad_sample_x1 += gVal_11 * (*in_ptr_111 - *in_ptr_011);
      grad_sample_x1 *= (w_1 - 1);

      scalar_t grad_sample_x = grad_sample_x0 + grad_sample_x1;
      atomicAdd(gSample_ptr, grad_sample_x);

      scalar_t val_00 = *in_ptr_000 * wx_00 + *in_ptr_100 * wx_01;
      scalar_t val_01 = *in_ptr_010 * wx_00 + *in_ptr_110 * wx_01;
      scalar_t val_10 = *in_ptr_001 * wx_10 + *in_ptr_101 * wx_11;
      scalar_t val_11 = *in_ptr_011 * wx_10 + *in_ptr_111 * wx_11;

      scalar_t grad_sample_y = gVal_0 * (val_01 - val_00) * (h_0 - 1);
      grad_sample_y += gVal_1 * (val_11 - val_10) * (h_1 - 1);
      atomicAdd(gSample_ptr + gSample_sC, grad_sample_y);

      scalar_t val_0 = val_00 * wy_00 + val_01 * wy_01;
      scalar_t val_1 = val_10 * wy_10 + val_11 * wy_11;

      scalar_t grad_sample_z = gVal *(val_1 - val_0) * (M-1);
      atomicAdd(gSample_ptr + 2 * gSample_sC,  grad_sample_z);

      // Get attention gradients
      atomicAdd(gAttn_ptr, gOut * (val_0 * wz_0 + val_1 * wz_1));
    }
  }
}
