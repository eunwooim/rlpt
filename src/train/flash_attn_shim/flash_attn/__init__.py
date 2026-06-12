"""Shim `flash_attn` package for Sol (RHEL8 / glibc 2.28).

Real flash-attn torch-2.7 wheels need glibc 2.32 (won't load here) and building from
source is heavy. verl HARD-requires `flash_attn` (bert_padding, used unconditionally in
its log-prob path) and vLLM's Qwen2.5-VL vision imports `flash_attn_varlen_func`, so the
package can't simply be absent. This shim provides:
  - bert_padding  : the real pure-torch unpad/pad helpers (what verl uses)
  - flash_attn_interface : SDPA-backed flash_attn_varlen_func/flash_attn_func (vision)
The policy/training attention is forced to SDPA and the LLM rollout uses vLLM's own
vendored kernels, so only the small vision encoder routes through the SDPA fallback.
Put this dir on PYTHONPATH (a sibling flash_attn-*.dist-info supplies the version).
"""

__version__ = "2.6.0"  # >=2.1 so is_flash_attn_2_available() is True and callers use these fns

from .flash_attn_interface import (  # noqa: E402,F401
    flash_attn_func,
    flash_attn_varlen_func,
)
