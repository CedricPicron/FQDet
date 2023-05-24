#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/msda_3d_cuda.cuh"

inline int BLOCKS(const int64_t N, const int64_t max_threads_per_block=NUM_THREADS) {
  TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  auto block_num = (N - 1) / max_threads_per_block + 1;
  TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

  return static_cast<int>(block_num);
}

using namespace at::cuda::detail;

at::Tensor msda_3d_cuda_forward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws)
{
    AT_ASSERTM(in_feats.device().is_cuda(), "in_feats must be a CUDA tensor");
    AT_ASSERTM(map_hw.device().is_cuda(), "map_hw must be a CUDA tensor");
    AT_ASSERTM(map_offs.device().is_cuda(), "map_offs must be a CUDA tensor");
    AT_ASSERTM(sample_xyz.device().is_cuda(), "sample_xyz must be a CUDA tensor");
    AT_ASSERTM(attn_ws.device().is_cuda(), "attn_ws must be a CUDA tensor");

    auto batch_size = in_feats.size(0);
    auto num_out_feats = sample_xyz.size(1);
    auto num_heads = in_feats.size(2);
    auto channels = in_feats.size(3);

    auto out_feats = at::zeros({batch_size, num_out_feats, num_heads, channels}, in_feats.options());
    int64_t num_kernels = batch_size * num_out_feats * num_heads * channels;

    if (num_kernels > 0)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feats.scalar_type(), "msda_3d_cuda_forward", [&] {
            msda_3d_cuda_forward_kernel<scalar_t>
            <<<BLOCKS(num_kernels), NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(num_kernels),
            getTensorInfo<scalar_t, int>(in_feats),
            getTensorInfo<int64_t, int>(map_hw),
            getTensorInfo<int64_t, int>(map_offs),
            getTensorInfo<scalar_t, int>(sample_xyz),
            getTensorInfo<scalar_t, int>(attn_ws),
            getTensorInfo<scalar_t, int>(out_feats));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }

    return out_feats;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> msda_3d_cuda_backward(
    const at::Tensor &in_feats, 
    const at::Tensor &map_hw,
    const at::Tensor &map_offs,
    const at::Tensor &sample_xyz,
    const at::Tensor &attn_ws,
    const at::Tensor &grad_out_feats)
{
    AT_ASSERTM(in_feats.device().is_cuda(), "in_feats must be a CUDA tensor");
    AT_ASSERTM(map_hw.device().is_cuda(), "map_hw must be a CUDA tensor");
    AT_ASSERTM(map_offs.device().is_cuda(), "map_offs must be a CUDA tensor");
    AT_ASSERTM(sample_xyz.device().is_cuda(), "sample_xyz must be a CUDA tensor");
    AT_ASSERTM(attn_ws.device().is_cuda(), "attn_ws must be a CUDA tensor");
    AT_ASSERTM(grad_out_feats.device().is_cuda(), "grad_out_feats must be a CUDA tensor");

    auto batch_size = in_feats.size(0);
    auto num_out_feats = sample_xyz.size(1);
    auto num_heads = in_feats.size(2);
    auto channels = in_feats.size(3);

    auto grad_in_feats = at::zeros_like(in_feats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto grad_sample_xyz = at::zeros_like(sample_xyz, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto grad_attn_ws = at::zeros_like(attn_ws, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    int64_t num_kernels = batch_size * num_out_feats * num_heads * channels;

    if (num_kernels > 0)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(in_feats.scalar_type(), "msda_3d_cuda_backward", [&] {
            msda_3d_cuda_backward_kernel<scalar_t>
            <<<BLOCKS(num_kernels), NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(num_kernels),
            getTensorInfo<scalar_t, int>(in_feats),
            getTensorInfo<int64_t, int>(map_hw),
            getTensorInfo<int64_t, int>(map_offs),
            getTensorInfo<scalar_t, int>(sample_xyz),
            getTensorInfo<scalar_t, int>(attn_ws),
            getTensorInfo<scalar_t, int>(grad_out_feats),
            getTensorInfo<scalar_t, int>(grad_in_feats),
            getTensorInfo<scalar_t, int>(grad_sample_xyz),
            getTensorInfo<scalar_t, int>(grad_attn_ws));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }

    return std::make_tuple(grad_in_feats, grad_sample_xyz, grad_attn_ws);
}