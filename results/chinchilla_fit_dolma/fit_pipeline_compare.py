"""
Reviewer response: staged (frozen-backbone) vs one-go (joint) pipelines.

Answers, on ONE consistent corpus (current n≈370, k=0 / no residual drop so
that *only the pipeline differs*):

  1. The joint (non-staged) fit — one-go 11-param.
  2. How far the stage-1 backbone drifts when paraphrase is added:
       - raw (E, A, alpha) drift  (large, but a degeneracy)
       - E_eff(N)=E+A/N^alpha drift  (tiny — the identified backbone
         prediction barely moves, so the raw drift "does not matter").
  3. η_para from BOTH pipelines with parametric-bootstrap 95% CIs, so the
     reviewer can see the difference *with uncertainty*.

Staged pipeline:
    stage-1: fit 9-param backbone (E,A,B,alpha,beta + eta_rep) on 1ep+rep.
    freeze it; fit eta_para (3 params) on paraphrase rows.
    bootstrap: resample paraphrase rows, refit eta_para w/ backbone frozen.

One-go pipeline:
    fit all 11 params jointly on 1ep+rep+para.
    bootstrap: resample all rows, warm-start refit 11 params (matches §6.6).

Usage:
    python fit_pipeline_compare.py --B 200 --out-json _onego_json/pipeline_compare.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fit_joint_triple import (GRID_REP_ONLY, PARA_GRID, SOURCE_PARA,
                              SOURCE_REPEAT, DELTA, make_forward_rep_only,
                              make_triple_forward)
from fit_joint_triple_onego import GRID_CATALOG, get_data, topk_drop_sweep
from fit_lse import expand_grid, fit_lse
from data import SIZES

NS = [SIZES[s][0] for s in ['14m', '30m', '60m', '100m', '190m', '370m', '600m']]
NLAB = ['14M', '30M', '60M', '100M', '190M', '370M', '600M']
ETA_PARA = ['log_K_para', 'rho_para', 'sigma_para']

# Compact but reliable one-go init grid (brackets the sigma_para<0 basin).
ONEGO_GRID = expand_grid({
    "e": [0.0, 1.0], "a": [3.0, 12.0], "b": [9.0], "alpha": [0.2, 0.35],
    "beta": [0.35, 0.45], "log_K_rep": [12.0], "rho_rep": [-0.4],
    "sigma_rep": [-0.4], "log_K_para": [10.0, 14.0],
    "rho_para": [-2.0, -1.0], "sigma_para": [-1.5, 0.0],
})


# ── forwards ──────────────────────────────────────────────────────────

def _wrap_fixed(forward_fn, fixed):
    fixed_t = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in fixed.items()}
    return lambda p: forward_fn({**p, **fixed_t})


def fit_backbone(data):
    """Stage-1: 9-param backbone on 1ep+rep (no paraphrase)."""
    keep = data['source'] != SOURCE_PARA
    fwd = make_forward_rep_only(data['N'][keep], data['D'][keep], data['Dp'][keep],
                                is_multi_arr=(data['source'][keep] == SOURCE_REPEAT))
    log_L = torch.tensor(np.log(data['L'][keep]), dtype=torch.float64)
    return fit_lse(fwd, log_L, GRID_REP_ONLY, delta=DELTA, verbose=False)['params']


def fit_eta_para_frozen(para_data, backbone, init_grid):
    """Fit 3 eta_para params with the backbone frozen."""
    base = make_triple_forward(para_data['N'], para_data['D'],
                               para_data['Dp'], para_data['source'])
    fwd = _wrap_fixed(base, backbone)
    log_L = torch.tensor(np.log(para_data['L']), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=DELTA, verbose=False)['params']


def fit_onego(data, init_grid):
    """Joint 11-param fit."""
    fwd = make_triple_forward(data['N'], data['D'], data['Dp'], data['source'])
    log_L = torch.tensor(np.log(data['L']), dtype=torch.float64)
    return fit_lse(fwd, log_L, init_grid, delta=DELTA, verbose=False)['params']


def e_eff(p, N):
    return np.exp(p['e']) + np.exp(p['a']) / N ** p['alpha']


# ── bootstrap ─────────────────────────────────────────────────────────

def boot_staged(para_data, backbone, anchor, B, seed):
    rng = np.random.default_rng(seed)
    n = len(para_data['L']); out = []
    grid = [{k: float(anchor[k]) for k in ETA_PARA}]
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        bd = {k: v[idx] for k, v in para_data.items()}
        try:
            out.append(fit_eta_para_frozen(bd, backbone, grid))
        except Exception as e:
            print(f"  staged boot {b} failed: {e}", flush=True)
    return out


def boot_onego(data, anchor, B, seed):
    rng = np.random.default_rng(seed)
    n = len(data['L']); out = []
    grid = [{k: float(v) for k, v in anchor.items()}]
    t0 = time.time()
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        bd = {k: v[idx] for k, v in data.items()}
        try:
            out.append(fit_onego(bd, grid))
        except Exception as e:
            print(f"  onego boot {b} failed: {e}", flush=True)
        if (b + 1) % 25 == 0:
            print(f"  onego boot {b+1}/{B} ({time.time()-t0:.0f}s)", flush=True)
    return out


def ci(samples, name):
    v = np.array([s[name] for s in samples if name in s])
    return dict(median=float(np.median(v)), std=float(np.std(v)),
                lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all')
    ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=0,
                    help="shared residual-drop count; 0=no drop (equal footing "
                         "at low beta), 15=writeup headline (shared kept-set).")
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    full = get_data(args.variant)

    # ---- shared kept-set: one-go pooled drop defines the outliers, then
    #      BOTH pipelines fit on the same survivors (equal footing at the
    #      headline beta). k=0 -> no drop, use everything. ----
    if args.k > 0:
        p0 = fit_onego(full, ONEGO_GRID)
        sweep = topk_drop_sweep(full, p0, [args.k])
        keep = sweep[args.k]['keep']
        data = {kk: vv[keep] for kk, vv in full.items()}
        onego = sweep[args.k]['params']          # joint k-drop anchor
        n_drop = int((~keep).sum())
    else:
        data = full
        onego = fit_onego(data, ONEGO_GRID)
        n_drop = 0

    para = {k: v[data['source'] == SOURCE_PARA] for k, v in data.items()}
    n_per = {s: int((data['source'] == s).sum()) for s in [0, 1, 2]}
    print(f"corpus n={len(full['L'])}  kept n={len(data['L'])} "
          f"(dropped {n_drop}; 1ep={n_per[0]}, rep={n_per[1]}, para={n_per[2]}), "
          f"k={args.k}, B={args.B}\n")

    # ---- point estimates on the (shared) kept set ----
    backbone = fit_backbone(data)                      # stage-1 backbone
    staged = fit_eta_para_frozen(para, backbone, PARA_GRID)   # frozen eta_para

    # ---- (2) backbone drift: raw vs E_eff(N) ----
    print("=" * 68)
    print("BACKBONE DRIFT: stage-1 (no para)  vs  one-go (with para)")
    print("=" * 68)
    print(f"{'param':<8}{'stage-1':>12}{'one-go':>12}{'rel drift':>12}")
    for nm, fn in [('E', lambda p: np.exp(p['e'])), ('A', lambda p: np.exp(p['a'])),
                   ('B', lambda p: np.exp(p['b'])), ('alpha', lambda p: p['alpha']),
                   ('beta', lambda p: p['beta'])]:
        vb, vo = fn(backbone), fn(onego)
        print(f"{nm:<8}{vb:>12.4g}{vo:>12.4g}{(vo-vb)/abs(vb)*100:>11.1f}%")
    print(f"\n{'E_eff(N) = E + A/N^alpha  (the identified backbone prediction)'}")
    print(f"{'N':<8}{'stage-1':>12}{'one-go':>12}{'rel drift':>12}")
    md = 0.0
    e_eff_rows = []
    for N, lab in zip(NS, NLAB):
        eb, eo = e_eff(backbone, N), e_eff(onego, N)
        rel = (eo - eb) / eb * 100; md = max(md, abs(rel))
        e_eff_rows.append(dict(N=lab, stage1=eb, onego=eo, drift_pct=rel))
        print(f"{lab:<8}{eb:>12.4f}{eo:>12.4f}{rel:>11.2f}%")
    print(f"  --> max |E_eff drift| = {md:.2f}%   "
          f"(raw E/A/alpha drift is a degeneracy; the prediction is stable)")

    # ---- (3) eta_para with bootstrap 95% CI, both pipelines ----
    print("\n[bootstrap] staged (frozen backbone, resample para rows)...", flush=True)
    bs = boot_staged(para, backbone, staged, args.B, args.seed)
    print(f"[bootstrap] one-go (joint, resample all rows)...", flush=True)
    bo = boot_onego(data, onego, args.B, args.seed)

    print("\n" + "=" * 84)
    print("eta_para: staged (frozen backbone)  vs  one-go (joint)   [median, 95% CI]")
    print("=" * 84)
    print(f"{'param':<12}{'staged pt':>10}{'staged 95% CI':>22}"
          f"{'one-go pt':>11}{'one-go 95% CI':>22}")
    table = {}
    for nm in ETA_PARA:
        cs, co = ci(bs, nm), ci(bo, nm)
        table[nm] = dict(staged_pt=float(staged[nm]), staged_ci=cs,
                         onego_pt=float(onego[nm]), onego_ci=co)
        s_ci = f"[{cs['lo']:+.3f}, {cs['hi']:+.3f}]"
        o_ci = f"[{co['lo']:+.3f}, {co['hi']:+.3f}]"
        print(f"{nm:<12}{staged[nm]:>+10.3f}{s_ci:>22}"
              f"{onego[nm]:>+11.3f}{o_ci:>22}")

    # sign summary for sigma_para (the headline flip)
    print("\nsigma_para sign:")
    for lab, s, c in [('staged', staged, ci(bs, 'sigma_para')),
                      ('one-go', onego, ci(bo, 'sigma_para'))]:
        sign = 'NEG' if c['hi'] < 0 else ('POS' if c['lo'] > 0 else 'STRADDLES 0')
        print(f"  {lab:<8} pt={s['sigma_para']:+.3f}  95%CI=[{c['lo']:+.3f},{c['hi']:+.3f}]  -> {sign}")

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(dict(n=len(data['L']), k=args.k, n_dropped=n_drop,
                           n_per_source=n_per, B=args.B,
                           backbone=backbone, staged_eta_para=staged,
                           onego_params=onego, e_eff=e_eff_rows,
                           eta_para_ci=table), f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == '__main__':
    main()
