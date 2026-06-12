"""Pure-torch (SDPA-backed) stand-ins for flash-attn kernels.

Real flash-attn won't load on this RHEL8/glibc-2.28 node, but verl hard-requires the
`flash_attn` package (bert_padding) and vLLM's Qwen2.5-VL vision encoder imports
`flash_attn_varlen_func`. The policy/training attention is forced to SDPA, and the LLM
rollout uses vLLM's own vendored kernels; only the (small) vision encoder routes through
here, so an SDPA fallback is correct and fast enough.

Implements the call shapes verl/vLLM use:
  flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)
    q:(total_q, nheads, hd)  k,v:(total_k, nheads_k, hd) -> (total_q, nheads, hd)
  flash_attn_func(q, k, v, ...)   q,k,v:(b, s, nheads, hd) -> (b, s, nheads, hd)
"""

import torch
import torch.nn.functional as F


def _repeat_kv(x_heads, n_rep):
    # x_heads: (..., nheads_k, L, hd) -> (..., nheads_k*n_rep, L, hd)
    return x_heads.repeat_interleave(n_rep, dim=-3) if n_rep > 1 else x_heads


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                           max_seqlen_k=None, dropout_p=0.0, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=0.0,
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, block_table=None, **kwargs):
    """Variable-length attention via per-segment SDPA. q:(Tq,H,D) k,v:(Tk,Hk,D)."""
    H, D = q.shape[1], q.shape[2]
    Hk = k.shape[1]
    n_rep = H // Hk
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cu_q) - 1):
        qs, qe = cu_q[i], cu_q[i + 1]
        ks, ke = cu_k[i], cu_k[i + 1]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)   # (1, H, lq, D)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)   # (1, Hk, lk, D)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        ki = _repeat_kv(ki, n_rep)
        vi = _repeat_kv(vi, n_rep)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=scale,
                                            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0)
        outs.append(oi.squeeze(0).transpose(0, 1))   # (lq, H, D)
    out = torch.cat(outs, dim=0)
    return (out, None, None) if return_attn_probs else out


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, **kwargs):
    """Batched attention via SDPA. q,k,v: (b, s, nheads, hd)."""
    D = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    qt = q.transpose(1, 2)   # (b, H, s, D)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    n_rep = qt.shape[1] // kt.shape[1]
    kt = _repeat_kv(kt, n_rep)
    vt = _repeat_kv(vt, n_rep)
    o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, scale=scale,
                                       dropout_p=dropout_p if torch.is_grad_enabled() else 0.0)
    out = o.transpose(1, 2)
    return (out, None, None) if return_attn_probs else out


# some callers import these names too
flash_attn_varlen_kvpacked_func = None
flash_attn_kvpacked_func = None
