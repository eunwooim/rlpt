"""Tier-subset variant of the production marker_split — identical semantics
(same regexes, same slicing, first tier with >= 2 chunks wins), but the tier
hierarchy is configurable. Used by the Exp 4 list-tier ablation and the
production relevance check. The production function itself is unchanged."""
import sys

sys.path.insert(0, "/scratch/sghos104/rlpt/chunk_canonical")
from marker_split import STEP_RE, LIST_RE, PARA_RE, _slice_chunks  # noqa: E402

_TIERS = {"step": STEP_RE, "list": LIST_RE, "para": PARA_RE}


def marker_split_tiers(text, tiers):
    """tiers: ordered tuple drawn from {'step','list','para'}."""
    for name in tiers:
        rx = _TIERS[name]
        if name == "para":
            starts = [m.end() for m in rx.finditer(text)]
        else:
            starts = [m.start() for m in rx.finditer(text)]
        if not starts:
            continue
        chunks = _slice_chunks(text, starts)
        if len(chunks) >= 2:
            return name, chunks
    return None, None


if __name__ == "__main__":
    from marker_split import marker_split
    t = "Intro.\n1. first item\n2. second item\n\nClosing paragraph."
    assert marker_split_tiers(t, ("step", "list", "para")) == marker_split(t)
    tier, ch = marker_split_tiers(t, ("step", "para"))
    assert tier == "para" and len(ch) == 2, (tier, ch)
    tier, ch = marker_split_tiers(t, ("step",))
    assert tier is None and ch is None
    t2 = "Step 1: do a thing.\nStep 2: do another."
    assert marker_split_tiers(t2, ("step",))[0] == "step"
    print("marker_variants self-test OK")
