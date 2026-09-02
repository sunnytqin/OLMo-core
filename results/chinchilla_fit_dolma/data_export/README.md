# Dolma scaling-law dataset — one-go triple joint fit (writeup.md §6)

Every observation behind the headline fit
`L = E + A/N^α + B/(D + η_src·D')^β`, with separate exp-sat η surfaces
for repetition and paraphrase.

Regenerate with:

```
python ../export_fit_data.py            # rewrites everything in this dir
```

## Files

| file | rows |
|---|---|
| `chinchilla_triple_fit_data.csv` | all 370 |
| `chinchilla_triple_fit_data_1epoch.csv` | 56 |
| `chinchilla_triple_fit_data_repeat.csv` | 197 |
| `chinchilla_triple_fit_data_paraphrase.csv` | 117 |
| `chinchilla_triple_fit_data.json` | all 370 + fit metadata block |
| [`hparam_sweeps/`](hparam_sweeps/) | all 2556 runs across the lr × wd grid |

Rows are ordered by source → model size `N` → chinchilla scale → `D'/D`.
The JSON carries the same records plus the fitted parameters, the
functional form, and the exact filters applied.

## Columns

| column | meaning |
|---|---|
| `source` | `1epoch` / `repeat` / `paraphrase` |
| `size`, `N_params` | model size label and non-embedding parameter count |
| `chinchilla_scale` | multiplier on the Chinchilla-optimal token budget |
| `D_tokens` | fresh unique Dolma tokens = `chinchilla_scale · 20 · N` |
| `epochs` | passes over `D` (repetition stream; 1 elsewhere) |
| `K_paraphrase` | number of paraphrase seeds per document (paraphrase stream) |
| `D_prime_tokens` | second-stream tokens: `(epochs−1)·D` for repeat, `tokens_trained − D` for paraphrase, `0` for 1-epoch |
| `tokens_trained` | `D + D'` |
| `Dprime_over_D`, `D_over_N` | the two ratios the η surface is a function of |
| `val_loss` | **the measurement** — Dolma held-out validation cross-entropy |
| `eta_fit`, `R_star_fit` | η and R\* predicted by the canonical fit at this row (blank for 1-epoch) |
| `D_eff_fit` | `D + η·D'` |
| `val_loss_pred` | loss predicted by the canonical fit |
| `resid_log` | `log(val_loss) − log(val_loss_pred)` |
| `kept_in_canonical_fit` | `False` for the 15 points removed by the residual-greedy drop (`k=15`) |

Only `val_loss` and the columns left of it are data; everything from
`eta_fit` rightwards is derived from the fit and can be recomputed.

## Full hyperparameter sweeps

The files above hold one row per `(size, scale, epochs/K)` — the run
whose `(lr, wd)` won its sweep. Every run we ever evaluated, including
the losers, is in [`hparam_sweeps/`](hparam_sweeps/) (2,556 rows), with
a `selected_for_scaling_fit` flag marking the 505 that feed
`dolma_*.py`. See [hparam_sweeps/README.md](hparam_sweeps/README.md).

## Filters already applied

- Paraphrase rows are restricted to `chinchilla_scale ≥ 0.5` (matching §5/§6).
- Repetition rows drop the u-shape overfit `(scale, epoch)` pairs listed
  under `metadata.conventions.filters` in the JSON.
- 1-epoch rows have no scale floor; small-scale points are handled by the
  residual drop instead (`kept_in_canonical_fit`).

## Canonical fit (k=15, n=370)

```
E = 1.396   A = 225.3   B = 17738   α = 0.289   β = 0.438
log R*_rep  = 10.74 − 0.416·log(D/N) − 0.403·log N
log R*_para = 26.87 − 1.303·log(D/N) − 1.150·log N
```

RMSE(log L) = 0.0326 on the 355 kept points (0.0618 including the 15 dropped).

Note: writeup.md §6 tables were written at `n = 356` (paraphrase `K ≤ 32`).
This export is the current corpus, `n = 370`, which adds the `K = 64`
paraphrase runs; parameters shift slightly but every sign and conclusion
in §6 is unchanged.
