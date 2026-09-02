"""
Export the exact dataset behind the §6 one-go triple joint fit
(1-epoch + repetition + paraphrase) as tidy CSV + JSON.

Row set is byte-for-byte the pool that `fit_joint_triple_onego.py
--variant all --scale-min-para 0.5` optimizes on (verified by an
assertion against `collect_pooled_triple`).  Each row carries the raw
observation (N, D, D', L) plus the derived quantities and the canonical
k=15 fit's prediction, so the file is self-contained.

Usage:
    python export_fit_data.py                      # current data + current anchor
    python export_fit_data.py --anchor-json _onego_json/onego_all_default.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (OVERFIT_EXCLUDE, SIZES, TTP_RATIO, extract_1epoch,  # noqa: E402
                  extract_multi_epoch, extract_paraphrase, load_with_para)
from fit_joint_triple import (SOURCE_NONE, SOURCE_PARA, SOURCE_REPEAT,  # noqa: E402
                               make_triple_forward)
from fit_joint_triple_onego import get_data, topk_drop_sweep  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_NAME = {SOURCE_NONE: "1epoch", SOURCE_REPEAT: "repeat",
            SOURCE_PARA: "paraphrase"}


def collect_rows(scale_min_para=0.5):
    """Same pool as collect_pooled_triple, but keeping epochs / K / scale."""
    rows = []
    for size in SIZES:
        N, datasets, parap = load_with_para(size)
        s1, D1, L1 = extract_1epoch(datasets, N, scale_min=0.0)
        for s, d, l in zip(s1, D1, L1):
            rows.append(dict(source=SOURCE_NONE, size=size, N=N, scale=s,
                             D=d, Dp=0.0, L=l, epochs=1, K=None))
        sm, Dm, em, Dpm, Lm, _ = extract_multi_epoch(
            datasets, N, scale_min=0.0,
            exclude_overfit=OVERFIT_EXCLUDE.get(size, set()))
        for s, d, e, dp, l in zip(sm, Dm, em, Dpm, Lm):
            rows.append(dict(source=SOURCE_REPEAT, size=size, N=N, scale=s,
                             D=d, Dp=dp, L=l, epochs=int(e), K=None))
        if parap:
            sp, Dp_, Kp, Dpp, Lp, _ = extract_paraphrase(
                datasets, parap, N, scale_min=scale_min_para)
            for s, d, k, dp, l in zip(sp, Dp_, Kp, Dpp, Lp):
                rows.append(dict(source=SOURCE_PARA, size=size, N=N, scale=s,
                                 D=d, Dp=dp, L=l, epochs=1, K=int(k)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-json", default="_onego_json/triple_onego_k64_update.json",
                    help="one-go fit JSON supplying the k=0 warm start + canonical params")
    ap.add_argument("--canonical-k", type=int, default=15)
    ap.add_argument("--scale-min-para", type=float, default=0.5)
    ap.add_argument("--outdir", default="data_export")
    args = ap.parse_args()

    # ---- data ----------------------------------------------------------
    rows = collect_rows(args.scale_min_para)
    data = get_data("all", args.scale_min_para)
    # the two collectors must agree row-for-row
    assert len(rows) == len(data["L"]), (len(rows), len(data["L"]))
    for i, r in enumerate(rows):
        assert r["source"] == data["source"][i] and r["size"] == data["tags"][i]
        assert np.isclose(r["D"], data["D"][i]) and np.isclose(r["Dp"], data["Dp"][i])
        assert np.isclose(r["L"], data["L"][i])
    print(f"[export] pooled n = {len(rows)}  "
          f"(1ep={sum(r['source']==SOURCE_NONE for r in rows)}, "
          f"rep={sum(r['source']==SOURCE_REPEAT for r in rows)}, "
          f"para={sum(r['source']==SOURCE_PARA for r in rows)})")

    # ---- canonical fit: reproduce the k-drop from the stored k=0 anchor --
    anchor_path = os.path.join(SCRIPT_DIR, args.anchor_json)
    anchor = json.load(open(anchor_path))
    k0 = next(s for s in anchor["sweep"] if s["k"] == 0)["params"]
    print(f"[export] reproducing k={args.canonical_k} residual drop "
          f"warm-started from {os.path.basename(anchor_path)} k=0 ...")
    # replay the anchor's own k grid up to canonical_k so the warm-start
    # chain (and hence the optimum) matches the stored fit exactly
    k_grid = sorted({s["k"] for s in anchor["sweep"] if s["k"] <= args.canonical_k}
                    | {args.canonical_k})
    swept = topk_drop_sweep(data, k0, k_grid)[args.canonical_k]
    keep = swept["keep"]
    # Use the anchor's published canonical parameters for the prediction
    # columns (the replayed drop lands in the same flat basin but LBFGS
    # noise moves the loose params slightly); the replay supplies `keep`.
    params = {k: float(v) for k, v in anchor["canonical"]["params"].items()}
    print("[export] replayed-fit vs published params max |rel diff| = "
          f"{max(abs(swept['params'][k] - params[k]) / max(abs(params[k]), 1e-9) for k in params):.3f}")

    # ---- per-row model quantities --------------------------------------
    fwd = make_triple_forward(data["N"], data["D"], data["Dp"], data["source"])
    with torch.no_grad():
        pred = fwd({k: torch.tensor(v, dtype=torch.float64)
                    for k, v in params.items()}).numpy()
    resid = np.log(data["L"]) - pred

    D, Dp, N = data["D"], data["Dp"], data["N"]
    x = np.where(Dp > 0, Dp / D, 1.0)
    Rstar = np.full(len(rows), np.nan)
    for src, sfx in [(SOURCE_REPEAT, "rep"), (SOURCE_PARA, "para")]:
        m = data["source"] == src
        Rstar[m] = np.exp(params[f"log_K_{sfx}"]
                          + params[f"rho_{sfx}"] * np.log(D[m] / N[m])
                          + params[f"sigma_{sfx}"] * np.log(N[m]))
    eta = np.where(Dp > 0, Rstar * (1.0 - np.exp(-x / Rstar)) / x, np.nan)
    D_eff = D + np.where(Dp > 0, eta, 0.0) * Dp

    for i, r in enumerate(rows):
        r.update(Rstar=Rstar[i], eta=eta[i], D_eff=D_eff[i],
                 L_pred=float(np.exp(pred[i])), resid_log=float(resid[i]),
                 kept=bool(keep[i]))

    # ---- ordering: source, then N, then scale, then D'/D ----------------
    order = {SOURCE_NONE: 0, SOURCE_REPEAT: 1, SOURCE_PARA: 2}
    rows.sort(key=lambda r: (order[r["source"]], r["N"], r["scale"],
                             r["Dp"] / r["D"]))

    # ---- write ----------------------------------------------------------
    outdir = os.path.join(SCRIPT_DIR, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    FIELDS = ["source", "size", "N_params", "chinchilla_scale", "D_tokens",
              "epochs", "K_paraphrase", "D_prime_tokens", "tokens_trained",
              "Dprime_over_D", "D_over_N", "val_loss", "eta_fit",
              "R_star_fit", "D_eff_fit", "val_loss_pred", "resid_log",
              "kept_in_canonical_fit"]

    def record(r):
        return {
            "source": SRC_NAME[r["source"]],
            "size": r["size"],
            "N_params": int(r["N"]),
            "chinchilla_scale": r["scale"],
            "D_tokens": int(round(r["D"])),
            "epochs": r["epochs"],
            "K_paraphrase": r["K"],
            "D_prime_tokens": int(round(r["Dp"])),
            "tokens_trained": int(round(r["D"] + r["Dp"])),
            "Dprime_over_D": round(r["Dp"] / r["D"], 6),
            "D_over_N": round(r["D"] / r["N"], 4),
            "val_loss": r["L"],
            "eta_fit": None if np.isnan(r["eta"]) else round(float(r["eta"]), 6),
            "R_star_fit": None if np.isnan(r["Rstar"]) else round(float(r["Rstar"]), 6),
            "D_eff_fit": int(round(r["D_eff"])),
            "val_loss_pred": round(r["L_pred"], 6),
            "resid_log": round(r["resid_log"], 6),
            "kept_in_canonical_fit": r["kept"],
        }

    recs = [record(r) for r in rows]

    csv_path = os.path.join(outdir, "chinchilla_triple_fit_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in recs:
            w.writerow({k: ("" if v is None else v) for k, v in rec.items()})

    # per-source split files (same columns, same ordering)
    for name in ("1epoch", "repeat", "paraphrase"):
        sub = [r for r in recs if r["source"] == name]
        sp = os.path.join(outdir, f"chinchilla_triple_fit_data_{name}.csv")
        with open(sp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for rec in sub:
                w.writerow({k: ("" if v is None else v) for k, v in rec.items()})
        print(f"[export] wrote {sp}  (n={len(sub)}, "
              f"kept={sum(r['kept_in_canonical_fit'] for r in sub)})")

    E, A, B = np.exp(params["e"]), np.exp(params["a"]), np.exp(params["b"])
    meta = {
        "description": "Dolma multi-epoch / paraphrase scaling-law dataset — the exact "
                       "pool behind the one-go triple joint fit (writeup.md §6).",
        "generated_by": "results/chinchilla_fit_dolma/export_fit_data.py",
        "n_total": len(recs),
        "n_by_source": {name: sum(r["source"] == name for r in recs)
                        for name in ("1epoch", "repeat", "paraphrase")},
        "conventions": {
            "D_tokens": "fresh unique Dolma tokens = chinchilla_scale * 20 * N",
            "D_prime_tokens": "second-stream tokens: (epochs-1)*D for repeat, "
                              "tokens_trained - D for paraphrase, 0 for 1epoch",
            "val_loss": "Dolma held-out validation cross-entropy",
            "filters": {
                "paraphrase_scale_min": args.scale_min_para,
                "repeat_overfit_excluded": {k: sorted(map(list, v))
                                            for k, v in OVERFIT_EXCLUDE.items() if v},
                "note": "u-shape overfit (scale, epoch) pairs are excluded from the "
                        "repetition stream; 1-epoch and paraphrase use no scale floor "
                        "beyond paraphrase_scale_min.",
            },
            "chinchilla_tokens_per_param": TTP_RATIO,
        },
        "model": {
            "loss": "L = E + A/N^alpha + B/(D + eta_src * D')^beta",
            "eta": "eta_src = R*_src (1 - exp(-x/R*_src)) / x,  x = D'/D",
            "R_star": "log R*_src = log K_src + rho_src * log(D/N) + sigma_src * log N",
            "canonical_k": args.canonical_k,
            "anchor_json": os.path.basename(anchor_path),
            "params_raw": {k: float(v) for k, v in params.items()},
            "params_natural": {"E": float(E), "A": float(A), "B": float(B),
                               "alpha": float(params["alpha"]),
                               "beta": float(params["beta"])},
            "rmse_log_kept": float(np.sqrt(np.mean(resid[keep] ** 2))),
            "rmse_log_all": float(np.sqrt(np.mean(resid ** 2))),
        },
        "columns": FIELDS,
    }
    json_path = os.path.join(outdir, "chinchilla_triple_fit_data.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": meta, "data": recs}, f, indent=1)

    print(f"[export] wrote {csv_path}")
    print(f"[export] wrote {json_path}")
    print(f"[export] E={E:.3f} A={A:.1f} B={B:.0f} "
          f"alpha={params['alpha']:.3f} beta={params['beta']:.3f}")
    print(f"[export] rmse(log L) kept={meta['model']['rmse_log_kept']:.4f} "
          f"all={meta['model']['rmse_log_all']:.4f}  "
          f"n_dropped={int((~keep).sum())}")


if __name__ == "__main__":
    main()
