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
| `run_name` | the training run behind this point; joins to [`hparam_sweeps/`](hparam_sweeps/) |
| `learning_rate`, `weight_decay` | the (lr, wd) that won this cell's sweep |
| `D_files` | the fresh-data `.npy` file(s) the run read, `;`-separated |
| `D_dir` | directory of `D_files`, relative to the data root |
| `D_prime_files` | the K paraphrase seed files (paraphrase rows only) |
| `D_prime_dir` | directory of `D_prime_files`, relative to the data root |
| `D_prime_corpus` | which D′ pool (`sized_smollm2_mixed`) |
| `paraphrase_seeds` | seed range consumed, e.g. `1-8` |
| `data_mix` | the `DataMix` name backing `D_files` (repetition rows) |
| `tokens_evaluated` | size of the validation prefix this loss was measured over |
| `val_loss` | **the measurement** — Dolma held-out validation cross-entropy |
| `eta_fit`, `R_star_fit` | η and R\* predicted by the canonical fit at this row (blank for 1-epoch) |
| `D_eff_fit` | `D + η·D'` |
| `val_loss_pred` | loss predicted by the canonical fit |
| `resid_log` | `log(val_loss) − log(val_loss_pred)` |
| `kept_in_canonical_fit` | `False` for the 15 points removed by the residual-greedy drop (`k=15`) |

Only `val_loss` and the columns left of it are data; everything from
`eta_fit` rightwards is derived from the fit and can be recomputed.

## Which data and settings each point came from

`run_name` identifies the training run, and `D_files` / `D_prime_files`
name the exact `.npy` files it read. Full paths are
`<data_root>/<D_dir>/<tokenizer_id>/<file>` with
`tokenizer_id = allenai/dolma2-tokenizer`. Repetition rows read `D_files`
`epochs` times and have no second stream; paraphrase rows read `D_files`
once plus the K seed files, which paraphrase *the same documents* as `D`.

These are reconstructed from source — `(size, chinchilla_scale)` → the
training script's `_DATASET_LOOKUP` → `DataMix` → the file list in
`src/olmo_core/data/mixes/syn_data_scaling/dolma/*.txt` — rather than
scraped off the cluster, because netscratch's 90-day purge has removed
most runs' own `data_paths.txt`. Of the 2,556 runs in `hparam_sweeps/`,
1,151 still have theirs, and all 1,151 reproduce exactly:

```
python ../run_data_provenance.py --verify
```

Two things to keep in view:

- **`D_tokens` is nominal, `D_files` is actual.** At 190M with scale ≥ 2
  the runs reused the 370M shard family, so scale = 2 is nominally 7.6B
  but read `train_7.4B.npy`. Where they disagree, `D_files` is what
  trained.
- **`tokens_evaluated` is not constant.** Every run evaluates on the same
  `validation.npy`, but over a different-length prefix of it: 3.48M
  tokens is the default, while an earlier round of the hyperparameter grid
  (30M and 370M repetition runs, April 2026) used 13.28M. Losses are
  directly comparable within a group; check the column before comparing
  across.

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
