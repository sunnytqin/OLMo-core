"""
95% confidence intervals for the one-go joint triple fit parameters,
via parametric bootstrap.

Methodology follows extended_lse_synergy_model.py:
  1. Compute the canonical k=15 fit on the full data (Option C anchor).
  2. Subset to the kept rows from that anchor fit.
  3. For B bootstrap samples:
       a. Resample the kept rows with replacement.
       b. Re-fit the 11-param model warm-started from the anchor.
       c. Store the bootstrap parameter vector.
  4. Compute median, 2.5%, 97.5% percentiles per parameter
     (= proper 95% CI; the reference uses 5/95 which is 90%).

Usage:
    python bootstrap_onego.py --anchor-json _onego_json/onego_all_para_dense.json \\
        --B 200 --out-json _onego_json/bootstrap_all.json --seed 42
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(__file__))

# Palatino styling (for plots if added later)
import matplotlib.pyplot as plt
for _f in glob.glob("/usr/share/fonts/urw-base35/P052-*.otf"):
    font_manager.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["P052", "Palatino", "TeX Gyre Pagella", "serif"],
    "mathtext.fontset": "cm",
    "font.size":        12,
})

from fit_joint_triple import (DELTA, SOURCE_NONE, SOURCE_PARA,  # noqa: E402
                                SOURCE_REPEAT, collect_pooled_triple,
                                make_triple_forward, topk_drop_sweep)
from fit_lse import fit_lse  # noqa: E402

PARAM_NAMES = [
    "e", "a", "b", "alpha", "beta",
    "log_K_rep", "rho_rep", "sigma_rep",
    "log_K_para", "rho_para", "sigma_para",
]


def get_data(variant, scale_min_para=0.5):
    full = collect_pooled_triple(scale_min_para=scale_min_para,
                                   exclude_sizes_1ep=(),
                                   exclude_sizes_rep=(),
                                   exclude_sizes_para=())
    if variant == "all":
        return full
    if variant == "drop14m":
        keep = ~((full["tags"] == "14m") & (full["source"] != SOURCE_PARA))
        return {k: v[keep] for k, v in full.items()}
    raise ValueError(f"unknown variant: {variant}")


def fit_one(data, init_params, delta=DELTA, max_iter=200):
    """Single warm-start LBFGS fit at init_params on `data`."""
    fwd = make_triple_forward(data["N"], data["D"], data["Dp"], data["source"])
    log_L = torch.tensor(np.log(data["L"]), dtype=torch.float64)
    grid = [{k: float(v) for k, v in init_params.items()}]
    return fit_lse(fwd, log_L, grid, delta=delta, verbose=False)


def get_canonical_kept_set(data, anchor_params, k_canonical=15):
    """Reproduce the canonical k=15 fit's kept-set mask by re-running
    the topk_drop_sweep with the anchor as warm-start.

    Returns (kept_data, kept_mask).
    """
    sweep = topk_drop_sweep(data, anchor_params, k_values=[0, 5, 10, k_canonical])
    rc = sweep[k_canonical]
    kept_mask = rc["keep"]
    kept_data = {k: v[kept_mask] for k, v in data.items()}
    return kept_data, kept_mask, rc["params"]


def bootstrap_fit(kept_data, anchor_params, B=200, seed=42):
    """Run B bootstrap fits, warm-started from `anchor_params`.

    Returns a list of param dicts (one per bootstrap sample).
    """
    rng = np.random.default_rng(seed)
    n = len(kept_data["L"])
    results = []
    losses = []
    t0 = time.time()
    for b in range(B):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        boot_data = {k: v[idx] for k, v in kept_data.items()}
        try:
            res = fit_one(boot_data, anchor_params)
            params = {k: float(v) for k, v in res["params"].items()}
            loss = float(res.get("rmse_logL", float("nan")))
        except Exception as e:
            print(f"  bootstrap {b}: failed ({e})", flush=True)
            continue
        results.append(params)
        losses.append(loss)
        if (b + 1) % 20 == 0 or b == B - 1:
            dt = time.time() - t0
            rate = (b + 1) / dt
            eta = (B - b - 1) / rate
            print(f"  bootstrap {b+1}/{B}  ({dt:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining)", flush=True)
    return results, losses


def summarize(results, anchor_params):
    """Compute median, 2.5%, 97.5% percentiles per parameter."""
    summary = {}
    for name in PARAM_NAMES:
        vals = np.array([r[name] for r in results if name in r])
        if len(vals) == 0:
            continue
        anchor_val = float(anchor_params[name])
        summary[name] = dict(
            anchor=anchor_val,
            mean=float(np.mean(vals)),
            median=float(np.median(vals)),
            std=float(np.std(vals)),
            ci_2_5=float(np.percentile(vals, 2.5)),
            ci_97_5=float(np.percentile(vals, 97.5)),
            ci_low_90=float(np.percentile(vals, 5)),
            ci_high_90=float(np.percentile(vals, 95)),
        )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-json", required=True,
                     help="Path to onego fit JSON to use as anchor")
    ap.add_argument("--variant", required=True, choices=["all", "drop14m"])
    ap.add_argument("--B", type=int, default=200,
                     help="Number of bootstrap samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale-min-para", type=float, default=0.5)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    print(f"# Bootstrap CI run")
    print(f"#   anchor JSON  = {args.anchor_json}")
    print(f"#   variant      = {args.variant}")
    print(f"#   B            = {args.B}")
    print(f"#   seed         = {args.seed}")
    print(f"#   scale_min_p  = {args.scale_min_para}")

    # Load anchor params
    with open(args.anchor_json) as f:
        anchor_json = json.load(f)
    anchor_params = anchor_json["canonical"]["params"]
    print(f"\n# Anchor params (canonical k=15):")
    for k in PARAM_NAMES:
        print(f"  {k:<14} = {anchor_params[k]:+.4f}")

    # Get data
    data = get_data(args.variant, args.scale_min_para)
    print(f"\n# Full pool n={len(data['L'])}")

    # Get the kept set from the canonical fit (reproduces the drop sweep)
    print(f"\n[Drop-sweep] reproducing canonical k=15 kept set...")
    kept_data, kept_mask, anchor_after_drop = get_canonical_kept_set(
        data, anchor_params, k_canonical=15)
    print(f"  n_kept = {kept_mask.sum()} (out of {len(kept_mask)})")
    print(f"  anchor params after drop:")
    for k in PARAM_NAMES:
        print(f"    {k:<14} = {anchor_after_drop[k]:+.4f}")

    # Bootstrap
    print(f"\n[Bootstrap] B={args.B} samples...")
    results, losses = bootstrap_fit(
        kept_data, anchor_after_drop, B=args.B, seed=args.seed)
    print(f"\n  bootstrap fits succeeded: {len(results)}/{args.B}")

    summary = summarize(results, anchor_after_drop)

    # Print headline CI table
    print("\n" + "=" * 96)
    print(f"95% CIs for one-go triple fit  (variant={args.variant}, "
          f"B={len(results)})")
    print("=" * 96)
    print(f"  {'param':<14}  {'anchor':>10}  {'median':>10}  "
          f"{'std':>9}  {'95% CI (2.5-97.5)':>25}  "
          f"{'90% CI (5-95)':>22}")
    for k in PARAM_NAMES:
        s = summary[k]
        ci95 = f"[{s['ci_2_5']:+.3f}, {s['ci_97_5']:+.3f}]"
        ci90 = f"[{s['ci_low_90']:+.3f}, {s['ci_high_90']:+.3f}]"
        print(f"  {k:<14}  {s['anchor']:>+10.4f}  {s['median']:>+10.4f}  "
              f"{s['std']:>9.4f}  {ci95:>25s}  {ci90:>22s}")

    # Sign analysis for σ_para and ρ_para
    print("\n" + "=" * 96)
    print(f"Sign analysis (does the 95% CI include zero?)")
    print("=" * 96)
    for k in ["sigma_para", "rho_para", "sigma_rep", "rho_rep"]:
        s = summary[k]
        zero_in = s["ci_2_5"] <= 0 <= s["ci_97_5"]
        zero_in_90 = s["ci_low_90"] <= 0 <= s["ci_high_90"]
        sign = "NEG" if s["median"] < 0 else "POS"
        signif = "STAT SIG" if not zero_in else "not 95% sig"
        signif_90 = "STAT SIG (90%)" if not zero_in_90 else "not 90% sig"
        print(f"  {k:<14}  median {s['median']:+.3f}  ({sign})  "
              f"95% CI {'crosses 0' if zero_in else 'excludes 0'}  ({signif})  "
              f"|  90% CI: {signif_90}")

    # Save JSON
    out_dict = dict(
        anchor_json=args.anchor_json,
        variant=args.variant,
        B=args.B,
        seed=args.seed,
        B_succeeded=len(results),
        n_full=int(len(data["L"])),
        n_kept=int(kept_mask.sum()),
        anchor_params=anchor_params,
        anchor_after_drop=serialize(anchor_after_drop),
        summary=summary,
        bootstrap_samples=results,
        bootstrap_losses=losses,
    )
    with open(args.out_json, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nWrote {args.out_json}")


def serialize(p):
    return {k: float(v) for k, v in p.items()}


if __name__ == "__main__":
    main()
