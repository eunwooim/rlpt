"""Pure-torch (SDPA-backed) stand-ins for flash-attn kernels.

Real flash-attn won't load on this RHEL8/glibc-2.28 node, but verl hard-requires the
`flash_attn` package (bert_padding) and vLLM's Qwen2.5-VL vision encoder imports
`flash_attn_varlen_func`. These implementations reproduce the flash-attn call signatures
on top of torch.nn.functional.scaled_dot_product_attention. They are numerically
equivalent (not kernel-identical) and slower than real flash-attn; in this setup only
the small vision encoder routes through them.

Unsupported flash-attn extras raise instead of silently degrading: non-default
window_size (sliding window), softcap, and alibi_slopes.
"""

import torch
import torch.nn.functional as F


def _check_unsupported(window_size, softcap, alibi_slopes, return_attn_probs):
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("flash_attn shim: sliding window is not supported")
    if softcap not in (0.0, None):
        raise NotImplementedError("flash_attn shim: softcap is not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("flash_attn shim: alibi_slopes is not supported")
    if return_attn_probs:
        raise NotImplementedError("flash_attn shim: return_attn_probs is not supported")


def _expand_kv(q, k, v):
    """Repeat KV heads for grouped-query attention. q,k,v: (..., heads, seq, dim)."""
    if k.shape[-3] != q.shape[-3]:
        rep = q.shape[-3] // k.shape[-3]
        k = k.repeat_interleave(rep, dim=-3)
        v = v.repeat_interleave(rep, dim=-3)
    return k, v


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    **kwargs,
):
    """Fixed-length attention. q: (B, Sq, H, D); k, v: (B, Sk, Hk, D). Returns (B, Sq, H, D)."""
    _check_unsupported(window_size, softcap, alibi_slopes, return_attn_probs)
    q_t = q.transpose(1, 2)  # (B, H, Sq, D)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    k_t, v_t = _expand_kv(q_t, k_t, v_t)
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
    )
    return out.transpose(1, 2).contiguous()


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    **kwargs,
):
    """Variable-length attention via per-segment SDPA. q:(Tq,H,D) k,v:(Tk,Hk,D)."""
    _check_unsupported(window_size, softcap, alibi_slopes, return_attn_probs)
    if block_table is not None:
        raise NotImplementedError("flash_attn shim: paged KV (block_table) is not supported")
    out = torch.empty_like(q)
    n = cu_seqlens_q.numel() - 1
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    for i in range(n):
        qs, qe = cu_q[i], cu_q[i + 1]
        ks, ke = cu_k[i], cu_k[i + 1]
        if qe == qs:
            continue
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)  # (1, H, sq, D)
        k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)
        k_i, v_i = _expand_kv(q_i, k_i, v_i)
        o_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        out[qs:qe] = o_i.squeeze(0).transpose(0, 1)
    return out


def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
    return flash_attn_func(q, kv[:, :, 0], kv[:, :, 1], *args, **kwargs)


def flash_attn_varlen_kvpacked_func(q, kv, *args, **kwargs):
    return flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], *args, **kwargs)
