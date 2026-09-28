"""Pure-torch pad/unpad helpers, identical in behavior to flash-attn's bert_padding.

These are the only parts of flash-attn that verl's remove-padding path uses, and the
upstream originals are already pure torch/einops (no CUDA extension), so they work
unmodified on this node.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """flash-attn >= 2.5.8 signature: returns a 5-tuple including used seqlens."""
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = torch.arange(
        seqlen, device=length.device, dtype=length.dtype
    ).expand(len(length), seqlen) < length.unsqueeze(1)
    real_indices_idx = torch.nonzero(attention_mask_in_length.flatten(), as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
