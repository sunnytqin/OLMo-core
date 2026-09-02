# Hyperparameter sweeps — every Dolma val-loss record

The parent directory holds the **winners**: one row per
`(size, scale, epochs/K)`, the run whose `(lr, wd)` was selected. This
folder holds **everything we ever evaluated** — all 2,556 runs across
the learning-rate × weight-decay grid.

Regenerate with:

```
python ../../export_hparam_sweeps.py
```

## Files

| file | rows | of which selected |
|---|---|---|
| `hparam_sweep_all.csv` | 2556 | 505 |
| `hparam_sweep_repeat.csv` | 2074 | 265 |
| `hparam_sweep_paraphrase.csv` | 423 | 230 |
| `hparam_sweep_selfdistill.csv` | 59 | 10 |
| `hparam_sweep_all.json` | 2556 + metadata | — |

Ordered by stream → `N` → scale → epochs/K → lr → wd, so each sweep is a
contiguous block.

## Sweep coverage

Regenerate this section with `python ../../export_hparam_sweeps.py --coverage`.

**The three grids are not the same size.** A "full" paraphrase sweep
explores 4 settings; a full repetition sweep explores 20. They are not
comparable in tuning depth — worth remembering before reading anything
into a para-vs-repeat loss gap.

| stream | full grid | lr values | wd values |
|---|---|---|---|
| repeat | **20** (4 lr × 5 wd) | 1e-4, 3e-4, 1e-3, 3e-3 | 0.1, 0.2, 0.4, 0.8, 1.6 |
| paraphrase | **4** (2 lr × 2 wd) | 1e-3, 3e-3 | 0.1, 0.2 |
| selfdistill | **6** (2 lr × 3 wd) | 1e-3, 3e-3 | 0.1, 0.2, 0.4 |

In the maps below, columns are chinchilla scale and each character is one
epoch (repeat) or K (paraphrase / self-distill) level, ascending:
`F` = full grid, `p` = partial (2+ points), `.` = single point, no sweep.

### repeat — 2074 runs, 266 cells (71 F / 77 p / 118 ·)

```
size        0.05      0.1     0.25      0.5        1        2        4        8       16
14m     ppFFpppF pppppFFF  ppFFFFp   pFFFFp   FFFp.p   FFF..p    .....    .....        -
30m     FFFFFFFp FFFFFFpp  FFFFFFF  FFFFFFF   FpFppF   ppppFp    Fpppp    .....    .....
60m      ..p.F.p  ......p  ......p  ......p   ......   ......    .....    .....        -
100m           p        p        p        p        p        p        p        -        -
190m     ppF.F.F   ppp...   ppp.p.   ppppp.   ppp...     ....      ...       ..       ..
370m     FFFFpFF  FFFFFpp   FFFFpp   ....pp    ....p     ....        .        .        -
600m        p...     p...     p...     p...      p..       p.        p        -        -
```

epochs 1→128. `D` spans 14M–60.8B tokens, but **full grids only span 14M–2.4B**.

- **30M is the workhorse** — full grid across essentially all epoch levels
  at 0.05–0.5×, thinning at 1–4×.
- **370M is the surprise** — genuinely well-swept at 0.05–0.25×, then falls
  off a cliff at 0.5× and above.
- **14M** is solid at 0.25–2×, patchier at the smallest scales.
- **60M, 100M and 600M have essentially no sweep.** 60M has exactly one
  full cell (0.05×, 16ep); everything else is a single inherited setting.
  100M is 2 points per cell throughout; 600M is single points.
- **Everything at 4× and above is a single point**, at every size.
- A handful of 14M/30M cells extend wd to 3.2 / 6.4 (22–28 points).

### paraphrase — 423 runs, 230 cells (42 F / 66 p / 122 ·)

```
size        0.05      0.1     0.25      0.5        1        2        4        8       16
14m         pppp     pppp     pppp     pppp     pppp     pppp     pppp     pppp     pppp
30m      pppp...  pppp...  pppp...     FFFF  FFFF...  FFFF...  FFFF...  FFFF...        -
60m      pppp...  pppp...  pppp...  FFFF...  FFFF...  FFFF...  FFFF...        -        -
190m     .......  .......  .......  FF.....   FF....   FF....        -        -        -
370m     pp.....  p......        -   ......   ppp...        -        -        -        -
600m     .......  .......   ......   ......        -        -        -        -        -
```

K 1→64. `D` spans 14M–7.6B; full grids span 300M–7.6B.

- The 4-point grid exists **only at 30M and 60M, scale ≥ 0.5×, and only for
  K = 1, 2, 4, 8**. Every K = 16 / 32 / 64 run is a single point.
- Small scales (0.05–0.25×) get 2 points — wd ∈ {0.1, 0.2} at lr = 3e-3 only.
- **190M gets 4 points at K = 1, 2 only**; 370M and 600M are effectively unswept.
- 14M is 2 points everywhere and stops at K = 8 — the gap writeup §6.4 flags.

### selfdistill — 59 runs, 10 cells (9 F / 1 p)

```
size        0.25      0.5
30m        FFFFp    FFFFF
```

K 1→16, `D` = 150M–300M. Proportionally the best-swept of the three — but
it is one model size at two token budgets, which is why writeup §7 notes
σ_sd is unidentifiable.

### What this means

The high-epoch and high-K tail — exactly where the saturation parameter
R\* is identified — is almost entirely single-point. The η saturation
results therefore lean on runs whose hyperparameters were inherited from a
neighbouring cell rather than tuned in place. Since LR is known not to
extrapolate across scales here, that is the sharpest gap in the coverage:
if an inherited LR is off at K = 32/64 or at 64/128 epochs, those points
read as *worse* than they should, which biases R\* downward — the same
direction as the "paraphrase saturates faster at large N" conclusion in
writeup §5.5 / §6.2. This has not been tested; a 4-point sweep at 30M,
K = 32 would settle it cheaply.

## Columns

| column | meaning |
|---|---|
| `stream` | `repeat` (multi-epoch, includes the 1-epoch baselines) / `paraphrase` / `selfdistill` |
| `run_name` | the training run's identifier, as on disk |
| `size`, `N_params` | model size label and parameter count |
| `chinchilla_scale`, `D_tokens` | token budget; `D = chinchilla_scale · 20 · N` |
| `epochs` | passes over `D` (repeat stream) |
| `K` | paraphrase / self-distill seeds per document |
| `learning_rate`, `weight_decay`, `seed` | the swept hyperparameters |
| `val_loss`, `perplexity` | the measurement |
| `tokens_evaluated` | size of the validation set used |
| `run_complete` | `False` for runs that never reached their final step |
| `selected_for_scaling_fit` | `True` for the run feeding `dolma_<size>.py` and the writeup fits |
| `source_file` | path (relative to `results/`) the record was read from |

## Provenance

Union of the per-run JSONs and the aggregated ones — neither alone is
complete:

- `results/dolma_val_loss/`, `results/dolma_para_val_loss/`,
  `results/dolma_sd_val_loss/` — 2,554 per-run files.
- `results/hparam/merged/{multi_epoch,para,selfdistill}/` — 2,463
  aggregated records; **missing all 100M and 600M runs**, but the only
  home of 2 incomplete 60M runs.

Where both exist the per-run file wins (identical payloads).

## The `selected_for_scaling_fit` flag

`dolma_<size>.py` stores `learning_rate` and `weight_decay` alongside
each `validation_loss`, so selection is an exact key match on
`(stream, size, scale, epochs|K, lr, wd)` — not a loss-value guess.

Verified: all 505 `dolma_*.py` entries match exactly one run here, and
every selected row's `val_loss` agrees with `dolma_*.py` to 4 decimals.
All 370 rows of `../chinchilla_triple_fit_data.csv` are drawn from these
505 (the rest are filtered out by the paraphrase scale floor and the
u-shape overfit exclusions).

Selection took the sweep minimum. Example — 30M, 0.5×, 16 epochs, 20
runs, winner `lr=3e-3, wd=0.2` at 3.9728:

```
lr=1e-4  wd 0.1→1.6   5.596 … 5.700
lr=3e-4  wd 0.1→1.6   4.684 … 4.860
lr=1e-3  wd 0.1→1.6   4.155 … 4.438
lr=3e-3  wd 0.1→1.6   3.992  3.973 ←  3.995  4.095  4.273
```

## Caveats

- **Two validation-set sizes** are present: 3,478,624 and 13,279,917
  tokens (`tokens_evaluated`). Losses are comparable within a set;
  keep the column in view before comparing across sets.
- **Grid coverage is uneven** — see [Sweep coverage](#sweep-coverage)
  above. The grid was pruned once the optimum was located, so
  high-epoch / large-size cells often hold a single point. Never assume
  a full grid; check the cell.
- `run_complete=False` on 2 runs (60M, 0.5× 64ep and 8× 32ep); their
  `val_loss` is the best eval seen, not a final-step number.
