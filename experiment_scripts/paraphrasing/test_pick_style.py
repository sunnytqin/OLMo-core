#!/usr/bin/env python3
"""
Sanity check for `pick_style_for_doc` in paraphrase_shard.py.

Run:
    /n/holylabs/dam_lab/Lab/sqin/envs/openrlhf/bin/python test_pick_style.py

Verifies:
  1. Deterministic: same (doc_idx, seed) → same style across calls.
  2. Seed-dependent: same doc_idx + different seed → different styles (high prob.)
  3. Uniform across 4 styles: over 4000 docs, each style gets 900-1100 assignments
     (±10% of the uniform 1000, well within 3σ of a binomial(4000, 0.25)).
"""

import sys
from collections import Counter
from pathlib import Path

# Import without running main()
sys.path.insert(0, str(Path(__file__).parent))
from paraphrase_shard import pick_style_for_doc, MIXED_STYLES


def test_deterministic():
    for idx in range(0, 100000, 997):
        a = pick_style_for_doc(idx, seed=2)
        b = pick_style_for_doc(idx, seed=2)
        assert a == b, f"non-deterministic at idx={idx}: {a} vs {b}"
    print("✓ deterministic across calls")


def test_seed_sensitivity():
    # At least 60% of docs should differ between seed 2 and seed 3 (expected ~75%)
    diff = sum(1 for i in range(1000)
               if pick_style_for_doc(i, seed=2) != pick_style_for_doc(i, seed=3))
    assert diff > 600, f"too few doc-style differences between seeds 2 and 3: {diff}/1000"
    print(f"✓ seed-sensitive ({diff}/1000 docs differ between seed 2 and 3)")


def test_uniform_distribution():
    c = Counter(pick_style_for_doc(i, seed=2) for i in range(4000))
    assert len(c) == len(MIXED_STYLES), f"missing styles in 4k samples: {c}"
    for style, n in c.items():
        assert 900 <= n <= 1100, f"style {style!r} count {n} outside 900-1100 range"
    print(f"✓ uniform over 4000 docs: {dict(c)}")


def test_canonical_styles_match():
    assert set(MIXED_STYLES) == {"faq", "math", "table", "tutorial"}
    print(f"✓ MIXED_STYLES = {MIXED_STYLES}")


if __name__ == "__main__":
    test_canonical_styles_match()
    test_deterministic()
    test_seed_sensitivity()
    test_uniform_distribution()
    print("\nAll tests passed.")
