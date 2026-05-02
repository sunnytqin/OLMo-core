"""
Bootstrap CIs and train/test holdout for the two pipelines:

  Two-shot:  fit joint Chinchilla on 1-ep, then η on multi-ep.
  One-shot:  fit (E, A, B, α, β, η-params) jointly on 1-ep + multi-ep.

For each pipeline:
  - n_boot bootstrap resamples (with-replacement on rows)
  - Each resample randomly holds out test_frac points as test
  - Fit on train, evaluate on test
  - Collect parameter samples + train/test RMSE
  - Report median, 5/95 percentiles for parameters and RMSE

Goal: figure out which pipeline gives tighter CIs and lower test RMSE,
and whether either pipeline brings down the per-point η > 1 fraction.

Uses exp-sat R*(N) as the canonical η form.
"""

import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data import OVERFIT_EXCLUDE, SIZES, TTP_RATIO, extract_1epoch  # noqa: E402
from data import extract_multi_epoch  # noqa: E402
from fit_chinchilla_joint import (collect_1epoch_all_sizes, fit_joint,  # noqa: E402
                                   predict)
from fit_eta import (FORMS, collect_multi_epoch_all_sizes, fit_form,  # noqa: E402
                     per_point_eta_diff)
from fit_joint_all import (collect_pooled, fit_joint_all_sweep,  # noqa: E402
                            ETA_FORMS as JOINT_ETA_FORMS)

DELTA = 0.1


# ──────────────────────────────────────────────────────────────────────
# Pipelines (no resampling)
# ──────────────────────────────────────────────────────────────────────

def fit_two_shot(N_1ep, D_1ep, L_1ep,
                 N_me, D_me, Dp_me, L_me,
                 eta_form="Muennighoff R*(N)", delta=DELTA):
    """Stage 1: joint Chinchilla on 1-ep.  Stage 2: η on multi-ep."""
    chinch = fit_joint(N_1ep, D_1ep, L_1ep, delta=delta)
    E, A, B = chinch["E"], chinch["A"], chinch["B"]
    a, b = chinch["alpha"], chinch["beta"]
    E_eff_arr = E + A / N_me ** a
    res = fit_form(FORMS[eta_form], D_me, Dp_me, N_me, L_me,
                   E_eff_arr, B, b, delta=delta)
    return dict(chinch=chinch, eta=res, eta_form=eta_form)


def fit_one_shot(data_pool, eta_form="Muennighoff", delta=DELTA, k=20):
    """Joint fit on pooled data (1-ep + multi-ep) with iterative residual drop."""
    sweep = fit_joint_all_sweep(form_name=eta_form, delta=delta,
                                 k_values=(k,))
    return sweep[k]


# ──────────────────────────────────────────────────────────────────────
# Eval
# ──────────────────────────────────────────────────────────────────────

def predict_two_shot(N, D, Dp, fit, eta_form):
    """L_pred from a two-shot fit."""
    chinch = fit["chinch"]
    E_eff = chinch["E"] + chinch["A"] / N ** chinch["alpha"]
    res = fit["eta"]
    p_t = {k: torch.tensor(v, dtype=torch.float64)
           for k, v in res["params"].items()}
    D_t = torch.tensor(D, dtype=torch.float64)
    Dp_t = torch.tensor(Dp, dtype=torch.float64)
    N_t = torch.tensor(N, dtype=torch.float64)
    with torch.no_grad():
        eta = FORMS[eta_form]["fn"](p_t, D_t, Dp_t, N_t).numpy()
    # For 1-ep points D' = 0 ⇒ η has no effect (we use 0 there).
    eta_safe = np.where(Dp > 0, eta, 0.0)
    D_eff = D + eta_safe * Dp
    return E_eff + chinch["B"] / D_eff ** chinch["beta"]


def predict_one_shot(N, D, Dp, fit):
    p = fit["raw_params"]
    log_K = p["log_K"] if "log_K" in p else None
    chinch = fit["chinchilla"]
    eta_p = fit["eta_params"]
    # use the exp-sat / Muennighoff fn from JOINT_ETA_FORMS
    name = fit["form_name"]
    eta_fn = JOINT_ETA_FORMS[name]["fn"]
    p_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in p.items()}
    D_t = torch.tensor(D, dtype=torch.float64)
    Dp_t = torch.tensor(Dp, dtype=torch.float64)
    N_t = torch.tensor(N, dtype=torch.float64)
    with torch.no_grad():
        eta = eta_fn(p_t, D_t, Dp_t, N_t).numpy()
    eta_safe = np.where(Dp > 0, eta, 0.0)
    D_eff = D + eta_safe * Dp
    return chinch["E"] + chinch["A"] / N ** chinch["alpha"] + chinch["B"] / D_eff ** chinch["beta"]


def rmse_logL(L_pred, L_obs):
    return float(np.sqrt(np.mean((np.log(L_pred) - np.log(L_obs)) ** 2)))


# ──────────────────────────────────────────────────────────────────────
# Bootstrap with random train/test holdout
# ──────────────────────────────────────────────────────────────────────

def bootstrap_compare(n_boot=50, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)

    # Load data
    tags_1ep, N_1ep, _, D_1ep, L_1ep = collect_1epoch_all_sizes(scale_min=0.0)
    tags_me, N_me, _, D_me, _, Dp_me, L_me, _ = collect_multi_epoch_all_sizes(0.5)
    pool = collect_pooled()

    n_1ep = len(L_1ep)
    n_me = len(L_me)
    n_pool = len(pool["L"])

    print(f"Bootstrap: n_1ep={n_1ep}, n_me={n_me}, n_pool={n_pool}, "
          f"n_boot={n_boot}, test_frac={test_frac}")

    two_shot_records: List[dict] = []
    one_shot_records: List[dict] = []

    for b in range(n_boot):
        if b % 10 == 0:
            print(f"  bootstrap step {b}/{n_boot}")

        # Two-shot: separate test split for 1-ep and multi-ep
        idx_1ep = rng.permutation(n_1ep)
        n_test_1ep = int(test_frac * n_1ep)
        test_1ep = idx_1ep[:n_test_1ep]
        train_1ep = idx_1ep[n_test_1ep:]
        # bootstrap-resample the train side (with replacement)
        train_1ep_bs = rng.choice(train_1ep, size=len(train_1ep), replace=True)

        idx_me = rng.permutation(n_me)
        n_test_me = int(test_frac * n_me)
        test_me = idx_me[:n_test_me]
        train_me = idx_me[n_test_me:]
        train_me_bs = rng.choice(train_me, size=len(train_me), replace=True)

        try:
            ts = fit_two_shot(
                N_1ep[train_1ep_bs], D_1ep[train_1ep_bs], L_1ep[train_1ep_bs],
                N_me[train_me_bs], D_me[train_me_bs], Dp_me[train_me_bs],
                L_me[train_me_bs], eta_form="Muennighoff R*(N)")
            # Eval on test (multi-ep test for total RMSE)
            L_pred_test = predict_two_shot(
                N_me[test_me], D_me[test_me], Dp_me[test_me], ts,
                "Muennighoff R*(N)")
            test_rmse = rmse_logL(L_pred_test, L_me[test_me])
            two_shot_records.append(dict(
                E=ts["chinch"]["E"], A=ts["chinch"]["A"],
                B=ts["chinch"]["B"], alpha=ts["chinch"]["alpha"],
                beta=ts["chinch"]["beta"],
                log_K=ts["eta"]["params"]["log_K"],
                rho=ts["eta"]["params"]["rho"],
                sigma=ts["eta"]["params"]["sigma"],
                test_rmse=test_rmse,
            ))
        except Exception as e:
            pass

        # One-shot: hold out from pooled
        idx_pool = rng.permutation(n_pool)
        n_test_pool = int(test_frac * n_pool)
        test_pool = idx_pool[:n_test_pool]
        train_pool = idx_pool[n_test_pool:]
        train_pool_bs = rng.choice(train_pool, size=len(train_pool),
                                    replace=True)

        try:
            # We need to override collect_pooled for the bootstrap.
            # Simplest path: monkey-patch the data inside fit_joint_all
            # by reaching into the resampled arrays.  Since fit_joint_all
            # uses collect_pooled internally, refactor would take time —
            # for now, pass the data as a cached module-level dict.
            from fit_joint_all import collect_pooled as orig_collect
            import fit_joint_all as fja
            cached = dict(
                tags=pool["tags"][train_pool_bs],
                N=pool["N"][train_pool_bs],
                D=pool["D"][train_pool_bs],
                Dp=pool["Dp"][train_pool_bs],
                L=pool["L"][train_pool_bs],
                is_multi=pool["is_multi"][train_pool_bs],
            )
            fja.collect_pooled = lambda: cached
            try:
                os_fit = fit_one_shot(cached, eta_form="Muennighoff", k=15)
            finally:
                fja.collect_pooled = orig_collect
            # Eval on test
            L_pred_test = predict_one_shot(
                pool["N"][test_pool], pool["D"][test_pool],
                pool["Dp"][test_pool], os_fit)
            test_rmse = rmse_logL(L_pred_test, pool["L"][test_pool])
            ch = os_fit["chinchilla"]
            ep = os_fit["eta_params"]
            one_shot_records.append(dict(
                E=ch["E"], A=ch["A"], B=ch["B"],
                alpha=ch["alpha"], beta=ch["beta"],
                R0=ep.get("R0", float('nan')),
                rho=ep.get("rho", float('nan')),
                test_rmse=test_rmse,
            ))
        except Exception:
            pass

    return two_shot_records, one_shot_records


def summarize(records, name):
    if not records:
        print(f"  {name}: no successful fits")
        return
    print(f"\n  {name}  ({len(records)} successful fits)")
    keys = [k for k in records[0] if k != "test_rmse"] + ["test_rmse"]
    print(f"  {'param':<10s}  {'median':>10s}  {'5%':>10s}  {'95%':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k in keys:
        vals = np.array([r[k] for r in records])
        print(f"  {k:<10s}  {np.median(vals):>10.4f}  "
              f"{np.percentile(vals, 5):>10.4f}  "
              f"{np.percentile(vals, 95):>10.4f}")


def eta_gt1_fraction(N, D, Dp, L, L_1ep, B, beta):
    """Fraction of multi-ep points with per-point η > 1 using ΔL solver."""
    eta = per_point_eta_diff(D, Dp, L, L_1ep, B, beta)
    valid = ~np.isnan(eta)
    n_v = int(valid.sum())
    n_gt = int(np.sum(eta[valid] > 1))
    return n_gt, n_v


def report_eta_gt1_for_anchors():
    """Compare η>1 violations under different 1-ep anchors."""
    from fit_chinchilla_joint import topk_residual_drop_sweep
    tags_me, N_me, _, D_me, _, Dp_me, L_me, L_1ep_me = \
        collect_multi_epoch_all_sizes(0.5)
    print(f"\nη > 1 fraction across 1-ep anchor choices  "
          f"(per-point η via ΔL solver, scale ≥ 0.5×, n={len(L_me)}):")

    tags_1ep, N_1ep, _, D_1ep, L_1ep = collect_1epoch_all_sizes(scale_min=0.0)

    # Anchor 1: two-shot joint Chinchilla (k=20 residual drop, canonical)
    drop_sweep, _, _ = topk_residual_drop_sweep(
        tags_1ep, N_1ep, D_1ep, L_1ep,
        k_values=[20], delta=DELTA, iterative=True)
    p2 = drop_sweep[20]["p"]
    n_gt, n_v = eta_gt1_fraction(N_me, D_me, Dp_me, L_me, L_1ep_me,
                                  p2["B"], p2["beta"])
    print(f"  two-shot k=20  (B={p2['B']:>8.0f}, β={p2['beta']:.3f}):  "
          f"η>1 = {n_gt:>2d}/{n_v} = {n_gt/n_v:.1%}")

    # Anchor 2: one-shot at canonical k=20
    sweep = fit_joint_all_sweep(form_name="Muennighoff", delta=DELTA,
                                 k_values=(15, 20, 25))
    for k in (15, 20, 25):
        if k not in sweep:
            continue
        p1 = sweep[k]["chinchilla"]
        n_gt, n_v = eta_gt1_fraction(N_me, D_me, Dp_me, L_me, L_1ep_me,
                                      p1["B"], p1["beta"])
        print(f"  one-shot k={k}  (B={p1['B']:>8.0f}, β={p1['beta']:.3f}):  "
              f"η>1 = {n_gt:>2d}/{n_v} = {n_gt/n_v:.1%}")

    # Anchor 3: β-sweep around the canonical β with B fixed at two-shot value
    print("\n  β-sweep (B fixed from two-shot k=20, find β minimising η>1):")
    print(f"  {'β':>5s}  {'η>1':>10s}    n_valid")
    for bs in np.linspace(0.40, 0.60, 11):
        n_gt, n_v = eta_gt1_fraction(N_me, D_me, Dp_me, L_me, L_1ep_me,
                                      p2["B"], bs)
        if n_v == 0:
            print(f"  {bs:>5.3f}    (all NaN)")
            continue
        pct = n_gt / n_v
        print(f"  {bs:>5.3f}  {n_gt:>3d}/{n_v:<3d}  =  {pct:>5.1%}")


# ──────────────────────────────────────────────────────────────────────

def report_eta_gt1_delta_sweep():
    """Sweep δ on the 1-ep fit (with residual drop) and check η > 1."""
    from fit_chinchilla_joint import topk_residual_drop_sweep
    tags_me, N_me, _, D_me, _, Dp_me, L_me, L_1ep_me = \
        collect_multi_epoch_all_sizes(0.5)
    tags_1ep, N_1ep, _, D_1ep, L_1ep = collect_1epoch_all_sizes(scale_min=0.0)

    print(f"\nδ sweep on 1-ep joint Chinchilla (residual drop k=20):")
    print(f"  {'δ':>8s}  {'B':>8s}  {'β':>6s}  {'η>1':>10s}")
    for delta in [10.0, 1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-5]:
        try:
            drop_sweep, _, _ = topk_residual_drop_sweep(
                tags_1ep, N_1ep, D_1ep, L_1ep,
                k_values=[20], delta=delta, iterative=True)
            p = drop_sweep[20]["p"]
            n_gt, n_v = eta_gt1_fraction(N_me, D_me, Dp_me, L_me, L_1ep_me,
                                          p["B"], p["beta"])
            pct = n_gt / max(n_v, 1)
            print(f"  {delta:>8.0e}  {p['B']:>8.0f}  {p['beta']:>6.3f}  "
                  f"{n_gt:>3d}/{n_v:<3d} = {pct:>5.1%}")
        except Exception as e:
            print(f"  {delta:>8.0e}  failed: {type(e).__name__}")


if __name__ == "__main__":
    print("=" * 80)
    print("PART 0: δ sweep on 1-ep fit, check η > 1 sensitivity")
    print("=" * 80)
    report_eta_gt1_delta_sweep()
    print("\n" + "=" * 80)
    print("PART 1: η > 1 fraction across 1-ep anchor choices")
    print("=" * 80)
    report_eta_gt1_for_anchors()

    print("\nDONE (skipping bootstrap PART 2 — already in writeup)")
