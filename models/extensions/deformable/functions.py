"""
Implements deformable autograd functions.
"""

from torch.autograd import Function
from torch.autograd.function import once_differentiable

from deformable import msda_3d_backward, msda_3d_forward


class MSDA3DF(Function):
    """
    Class implementing the MSDA3DF autograd function.
    """

    @staticmethod
    def forward(ctx, in_feats, map_hw, map_offs, sample_xyz, attn_ws):
        """
        Forward method of the MSDA3DF autograd function.

        Args:
            ctx (Object): Context object storing additional data.
            in_feats (FloatTensor): Features to sample from of shape [batch_size, num_in_feats, num_heads, channels].
            map_hw (LongTensor): Feature map sizes in (H, W) format of shape [num_maps, 2].
            map_offs (LongTensor): Feature map offsets of shape [num_maps].
            sample_xyz (FloatTensor): Zero-one sample XYZ of shape [batch_size, num_out_feats, num_heads, num_pts, 3].
            attn_ws (FloatTensor): Attention weights of shape [batch_size, num_out_feats, num_heads, num_pts].

        Returns:
            out_feats (FloatTensor): Sampled features of shape [batch_size, num_out_feats, num_heads, channels].
        """

        sample_xyz = sample_xyz.clamp_(min=0.0, max=1.0)
        ctx.save_for_backward(in_feats, map_hw, map_offs, sample_xyz, attn_ws)
        out_feats = msda_3d_forward(in_feats, map_hw, map_offs, sample_xyz, attn_ws)

        return out_feats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out_feats):
        """
        Backward method of the MSDA3DF autograd function.

        Args:
            ctx (Object): Context object storing additional data.
            grad_out_feats (FloatTensor): Grad output feats of shape [batch_size, num_out_feats, num_heads, channels].

        Returns:
            grad_in_feats (FloatTensor): Grad input feats of shape [batch_size, num_in_feats, num_heads, channels].
            grad_map_hw (None): None.
            grad_map_offs (None): None.
            grad_sample_xyz (FloatTensor): Grad sample XYZ of shape [batch_size, num_out_feats, num_heads, num_pts, 3].
            grad_attn_ws (FloatTensor): Grad attn weights of shape [batch_size, num_out_feats, num_heads, num_pts].
        """

        in_feats, map_hw, map_offs, sample_xyz, attn_ws = ctx.saved_tensors
        backward_output = msda_3d_backward(in_feats, map_hw, map_offs, sample_xyz, attn_ws, grad_out_feats)

        grad_in_feats, grad_sample_xyz, grad_attn_ws = backward_output
        grad_map_hw = grad_map_offs = None

        return grad_in_feats, grad_map_hw, grad_map_offs, grad_sample_xyz, grad_attn_ws
