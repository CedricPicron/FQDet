"""
Script testing deformable functions.
"""

import torch

from deformable import msda_3d_backward, msda_3d_forward
from python.sample import pytorch_maps_sample_2d, pytorch_maps_sample_3d


# Test MSDA 3D forward
print('\n-----------------------')
print('|  MSDA 3D (forward)  |')
print('-----------------------\n')

batch_size = 1
num_heads = 1
channels = 1
num_maps = 2
num_out_feats = 1
num_pts = 1

map_height = 2
map_width = 2
num_in_feats = num_maps * map_height * map_width

map_hw = torch.tensor([[map_height, map_width] for _ in range(num_maps)], device='cuda')
map_offs = torch.prod(map_hw, dim=1).cumsum(dim=0)
map_offs = torch.cat([map_offs.new_zeros((1,)), map_offs[:-1]], dim=0)

sample_xyz = torch.rand(batch_size, num_out_feats, num_heads, num_pts, 3, device='cuda')
attn_ws = torch.rand(batch_size, num_out_feats, num_heads, num_pts, device='cuda')

sqz_sample_xyz = sample_xyz.squeeze()
inv_sample_xyz = 1 - sqz_sample_xyz
sqz_attn_ws = attn_ws.squeeze()

in_feats_000 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_001 = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_010 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_011 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_100 = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_101 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_110 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)
in_feats_111 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device='cuda').view(1, num_in_feats, 1, 1)

pred_000 = msda_3d_forward(in_feats_000, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_001 = msda_3d_forward(in_feats_001, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_010 = msda_3d_forward(in_feats_010, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_011 = msda_3d_forward(in_feats_011, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_100 = msda_3d_forward(in_feats_100, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_101 = msda_3d_forward(in_feats_101, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_110 = msda_3d_forward(in_feats_110, map_hw, map_offs, sample_xyz, attn_ws).item()
pred_111 = msda_3d_forward(in_feats_111, map_hw, map_offs, sample_xyz, attn_ws).item()

tgt_000 = (sqz_attn_ws * inv_sample_xyz[0] * inv_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_001 = (sqz_attn_ws * sqz_sample_xyz[0] * inv_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_010 = (sqz_attn_ws * inv_sample_xyz[0] * sqz_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_011 = (sqz_attn_ws * sqz_sample_xyz[0] * sqz_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_100 = (sqz_attn_ws * inv_sample_xyz[0] * inv_sample_xyz[1] * sqz_sample_xyz[2]).item()
tgt_101 = (sqz_attn_ws * sqz_sample_xyz[0] * inv_sample_xyz[1] * sqz_sample_xyz[2]).item()
tgt_110 = (sqz_attn_ws * inv_sample_xyz[0] * sqz_sample_xyz[1] * sqz_sample_xyz[2]).item()
tgt_111 = (sqz_attn_ws * sqz_sample_xyz[0] * sqz_sample_xyz[1] * sqz_sample_xyz[2]).item()

print(f'Error front top left: {abs(pred_000 - tgt_000)}')
print(f'Error front top right: {abs(pred_001 - tgt_001)}')
print(f'Error front bottom left: {abs(pred_010 - tgt_010)}')
print(f'Error front bottom right: {abs(pred_011 - tgt_011)}')
print(f'Error back top left: {abs(pred_100 - tgt_100)}')
print(f'Error back top right: {abs(pred_101 - tgt_101)}')
print(f'Error back bottom left: {abs(pred_110 - tgt_110)}')
print(f'Error back bottom right: {abs(pred_111 - tgt_111)}')

# Test PyTorch 2D sampler
print('\n------------------------')
print('|  PyTorch 2D sampler  |')
print('------------------------\n')

batch_size = 1
num_heads = 1
channels = 1
num_maps = 1
num_out_feats = 1
num_pts = 1

map_height = 2
map_width = 2
num_in_feats = num_maps * map_height * map_width

map_wh = torch.tensor([[map_width, map_height] for _ in range(num_maps)], device='cuda')
map_offs = torch.prod(map_wh, dim=1).cumsum(dim=0)
map_offs = torch.cat([map_offs.new_zeros((1,)), map_offs[:-1]], dim=0)

sample_xy = torch.rand(batch_size * num_heads, num_out_feats * num_pts, 2, device='cuda')
sample_map_ids = torch.randint(high=num_maps, size=(batch_size, num_out_feats), device='cuda')

sqz_sample_xy = sample_xy.squeeze()
inv_sample_xy = 1 - sqz_sample_xy

in_feats_00 = torch.tensor([1, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_01 = torch.tensor([0, 1, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_10 = torch.tensor([0, 0, 1, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_11 = torch.tensor([0, 0, 0, 1], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)

pred_00 = pytorch_maps_sample_2d(in_feats_00, map_wh, map_offs, sample_xy, sample_map_ids).item()
pred_01 = pytorch_maps_sample_2d(in_feats_01, map_wh, map_offs, sample_xy, sample_map_ids).item()
pred_10 = pytorch_maps_sample_2d(in_feats_10, map_wh, map_offs, sample_xy, sample_map_ids).item()
pred_11 = pytorch_maps_sample_2d(in_feats_11, map_wh, map_offs, sample_xy, sample_map_ids).item()

tgt_00 = (inv_sample_xy[0] * inv_sample_xy[1]).item()
tgt_01 = (sqz_sample_xy[0] * inv_sample_xy[1]).item()
tgt_10 = (inv_sample_xy[0] * sqz_sample_xy[1]).item()
tgt_11 = (sqz_sample_xy[0] * sqz_sample_xy[1]).item()

print(f'Error top left: {abs(pred_00 - tgt_00)}')
print(f'Error top right: {abs(pred_01 - tgt_01)}')
print(f'Error bottom left: {abs(pred_10 - tgt_10)}')
print(f'Error bottom right: {abs(pred_11 - tgt_11)}')

# Test PyTorch 3D sampler
print('\n------------------------')
print('|  PyTorch 3D sampler  |')
print('------------------------\n')

batch_size = 1
num_heads = 1
channels = 1
num_maps = 2
num_out_feats = 1
num_pts = 1

map_height = 2
map_width = 2
num_in_feats = num_maps * map_height * map_width

map_wh = torch.tensor([[map_width, map_height] for _ in range(num_maps)], device='cuda')
map_offs = torch.prod(map_wh, dim=1).cumsum(dim=0)
map_offs = torch.cat([map_offs.new_zeros((1,)), map_offs[:-1]], dim=0)

sample_xyz = torch.rand(batch_size * num_heads, num_out_feats * num_pts, 3, device='cuda')
sqz_sample_xyz = sample_xyz.squeeze()
inv_sample_xyz = 1 - sqz_sample_xyz

in_feats_000 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_001 = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_010 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_011 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_100 = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_101 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_110 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)
in_feats_111 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device='cuda').view(1, num_in_feats, 1)

pred_000 = pytorch_maps_sample_3d(in_feats_000, map_wh, map_offs, sample_xyz).item()
pred_001 = pytorch_maps_sample_3d(in_feats_001, map_wh, map_offs, sample_xyz).item()
pred_010 = pytorch_maps_sample_3d(in_feats_010, map_wh, map_offs, sample_xyz).item()
pred_011 = pytorch_maps_sample_3d(in_feats_011, map_wh, map_offs, sample_xyz).item()
pred_100 = pytorch_maps_sample_3d(in_feats_100, map_wh, map_offs, sample_xyz).item()
pred_101 = pytorch_maps_sample_3d(in_feats_101, map_wh, map_offs, sample_xyz).item()
pred_110 = pytorch_maps_sample_3d(in_feats_110, map_wh, map_offs, sample_xyz).item()
pred_111 = pytorch_maps_sample_3d(in_feats_111, map_wh, map_offs, sample_xyz).item()

tgt_000 = (inv_sample_xyz[0] * inv_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_001 = (sqz_sample_xyz[0] * inv_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_010 = (inv_sample_xyz[0] * sqz_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_011 = (sqz_sample_xyz[0] * sqz_sample_xyz[1] * inv_sample_xyz[2]).item()
tgt_100 = (inv_sample_xyz[0] * inv_sample_xyz[1] * sqz_sample_xyz[2]).item()
tgt_101 = (sqz_sample_xyz[0] * inv_sample_xyz[1] * sqz_sample_xyz[2]).item()
tgt_110 = (inv_sample_xyz[0] * sqz_sample_xyz[1] * sqz_sample_xyz[2]).item()
tgt_111 = (sqz_sample_xyz[0] * sqz_sample_xyz[1] * sqz_sample_xyz[2]).item()

print(f'Error front top left: {abs(pred_000 - tgt_000)}')
print(f'Error front top right: {abs(pred_001 - tgt_001)}')
print(f'Error front bottom left: {abs(pred_010 - tgt_010)}')
print(f'Error front bottom right: {abs(pred_011 - tgt_011)}')
print(f'Error back top left: {abs(pred_100 - tgt_100)}')
print(f'Error back top right: {abs(pred_101 - tgt_101)}')
print(f'Error back bottom left: {abs(pred_110 - tgt_110)}')
print(f'Error back bottom right: {abs(pred_111 - tgt_111)}')

# Compare MSDA 3D with variant using PyTorch 3D sampler
print('\n--------------------------------------')
print('|  MSDA 3D  vs.  PyTorch 3D sampler  |')
print('--------------------------------------\n')

batch_size = 2
channels = 256
num_out_feats = 300
num_heads = 8
num_pts = 4

h3, w3 = (128, 128)
h4, w4 = (64, 64)
h5, w5 = (32, 32)
h6, w6 = (16, 16)
h7, w7 = (8, 8)

feat_map3 = torch.randn(batch_size, channels, h3, w3, device='cuda')
feat_map4 = torch.randn(batch_size, channels, h4, w4, device='cuda')
feat_map5 = torch.randn(batch_size, channels, h5, w5, device='cuda')
feat_map6 = torch.randn(batch_size, channels, h6, w6, device='cuda')
feat_map7 = torch.randn(batch_size, channels, h7, w7, device='cuda')
feat_maps = [feat_map3, feat_map4, feat_map5, feat_map6, feat_map7]

in_feats = torch.cat([feat_map.flatten(2).transpose(1, 2) for feat_map in feat_maps], dim=1)
in_feats = in_feats.view(batch_size, -1, num_heads, channels // num_heads)

map_hw = torch.tensor([[h3, w3], [h4, w4], [h5, w5], [h6, w6], [h7, w7]], device='cuda')
map_offs = torch.cat([torch.zeros(1, dtype=torch.int64, device='cuda'), map_hw.prod(dim=1).cumsum(dim=0)[:-1]], dim=0)

sample_xyz = torch.randn(batch_size, num_out_feats, num_heads, num_pts, 3, device='cuda').clamp_(min=0.0, max=1.0)
attn_ws = torch.randn(batch_size, num_out_feats, num_heads, num_pts, device='cuda').softmax(dim=3)
out_feats_msda_3d = msda_3d_forward(in_feats, map_hw, map_offs, sample_xyz, attn_ws)

grad_out = torch.ones_like(out_feats_msda_3d)
grads_msda_3d = msda_3d_backward(in_feats, map_hw, map_offs, sample_xyz, attn_ws, grad_out)
grad_in_msda_3d, grad_sample_msda_3d, grad_attn_msda_3d = grads_msda_3d

in_feats.requires_grad_()
sample_xyz.requires_grad_()
attn_ws.requires_grad_()

map_wh = map_hw.fliplr()
in_feats_reshaped = in_feats.transpose(1, 2).reshape(batch_size * num_heads, -1, channels // num_heads)
sample_xyz_reshaped = sample_xyz.transpose(1, 2).reshape(batch_size * num_heads, num_out_feats * num_pts, 3)

sampled_feats = pytorch_maps_sample_3d(in_feats_reshaped, map_wh, map_offs, sample_xyz_reshaped)
sampled_feats = sampled_feats.view(batch_size, num_heads, -1, num_pts, channels // num_heads).transpose(1, 2)
out_feats_pytorch_3d = (attn_ws[:, :, :, :, None] * sampled_feats).sum(dim=3)
out_feats_pytorch_3d.sum().backward()

max_diff_out = (out_feats_msda_3d - out_feats_pytorch_3d).max().item()
max_diff_grad_in = (grad_in_msda_3d - in_feats.grad).max().item()
max_diff_grad_sample = (grad_sample_msda_3d - sample_xyz.grad).max().item()
max_diff_grad_attn = (grad_attn_msda_3d - attn_ws.grad).max().item()

print(f'Max diff. output: {max_diff_out}')
print(f'Max diff. grad. input: {max_diff_grad_in}')
print(f'Max diff. grad. sample: {max_diff_grad_sample}')
print(f'Max diff. grad. attention: {max_diff_grad_attn}\n')
