<div align="center">
  <h1>Bridging Compute- and Data-Optimal Pretraining</h1>
  <p>
    <a href="https://arxiv.org/abs/2607.25271">Tian&nbsp;Qin</a><sup>1</sup> ·
    Kimia&nbsp;Hamidieh<sup>2</sup> ·
    David&nbsp;Alvarez-Melis<sup>1</sup>
  </p>
  <p><sup>1</sup>Harvard University &nbsp;&nbsp; <sup>2</sup>MIT CSAIL</p>
  <p>
    <a href="https://arxiv.org/abs/2607.25271"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2607.25271-b31b1b.svg"></a>
  </p>
</div>

Code and data release for the paper. We introduce **Compute-Data (CD) scaling
laws**, which unify the compute-optimal and data-optimal regimes through a token
effectiveness function **η** measuring how derived tokens compare to fresh data.
The measurements behind the paper are ~2,500 training runs at 14M–600M
parameters over two ways of deriving tokens — multi-epoch repetition and
paraphrasing.

This repo is a fork of [allenai/OLMo-core](https://github.com/allenai/OLMo-core);
see the [upstream README](https://github.com/allenai/OLMo-core#readme) for
installation and the library itself. Everything below covers only what this
paper adds.

---

## Contents

| what | where |
|---|---|
| Every loss we measured, joined to its data and settings | [`results/chinchilla_fit_dolma/data_export/`](results/chinchilla_fit_dolma/data_export/) |
| Training entry points | [`src/scripts/official/OLMo-scale-train-*.py`](src/scripts/official/) |
| SLURM submission wrappers | [`experiment_scripts/train_scale*.sh`](experiment_scripts/) |
| Data download + sharding | [`experiment_scripts/download_data.sh`](experiment_scripts/download_data.sh), [`create_dolma3_splits.py`](experiment_scripts/create_dolma3_splits.py) |
| Paraphrase generation | [`experiment_scripts/paraphrasing/`](experiment_scripts/paraphrasing/) |
| Evaluation | [`experiment_scripts/run_eval_batch.py`](experiment_scripts/run_eval_batch.py) |
| Per-size result tables | [`results/dolma_<size>.py`](results/) |

Throughout, `D` is fresh unique Dolma data and `D'` is **derived** data. `D'`
comes from one of two mechanisms, and they are stored differently:

- **Repetition** — `D'` is simply training on repeated `D`. There are no extra
  files; the number of passes is the `epochs` column.
- **Paraphrasing** — `D'` is a separate corpus of model-rewritten versions of
  `D`'s documents, `K` per document. These are real files, listed per run.

---

## 1. Trained models: validation losses and the runs behind the fits

Start at [`data_export/`](results/chinchilla_fit_dolma/data_export/). Two tables,
same schema for the data columns:

- **[`chinchilla_triple_fit_data.csv`](results/chinchilla_fit_dolma/data_export/chinchilla_triple_fit_data.csv)** — the 370 points behind the headline fit, one per `(size, scale, epochs|K)`.
- **[`hparam_sweeps/hparam_sweep_all.csv`](results/chinchilla_fit_dolma/data_export/hparam_sweeps/hparam_sweep_all.csv)** — all 2,556 runs including the losers of every lr × wd sweep, with `selected_for_scaling_fit` marking the 505 that feed the per-size tables.

Every row names both its settings and its data:

| column | meaning |
|---|---|
| `run_name` | the run's identifier; joins the two tables |
| `learning_rate`, `weight_decay` | the swept hyperparameters |
| `seed` | training seed (42 throughout) |
| `epochs` / `K_paraphrase` | how `D'` was derived — passes over `D`, or paraphrases per document |
| `D_files`, `D_dir` | the exact fresh-data `.npy` file(s) trained on |
| `D_prime_files`, `D_prime_dir` | the K paraphrase seed files (paraphrase rows; empty for repetition, where `epochs` carries it) |
| `val_loss` | the measurement |

Full paths are `<data_root>/<D_dir>/allenai/dolma2-tokenizer/<file>`.

Everything else follows deterministically from `(row, training script)`:
architecture from `size`, global batch fixed at `512 × 4096` = 2.1M tokens,
sequence length 4096, and total steps from the on-disk token count of the files
named in the row. There are no free parameters left.

Two things to keep in view when reading the tables:

- **`D_tokens` vs `D_files`.** `D_tokens` is the fresh-data budget, computed
  exactly as `chinchilla_scale × 20 × N`. `D_files` names the shard we actually
  trained on. The two can differ slightly, since a shard is cut to whole
  documents and a few rungs reuse a neighbouring size's shard — the largest gap
  is 190M at scale ≥ 2, which uses the 370M shards (`train_7.4B.npy` against a
  nominal 7.6B, ~3%). The differences are minor, but `D_files` is the ground
  truth for what trained.
- **`D'` tokens for paraphrase runs are measured, not derived.** A paraphrase is
  not guaranteed to match its source document's length — in practice it is
  shorter — so the paraphrase token count cannot be inferred from `K`. We
  compute it empirically from the files on disk, which is why it is *not*
  `D_tokens × (1 + K)`. See [section 3](#3-the-paraphrase-corpus).

---

## 2. Training and validation data

All experiments train on a 150B-token sample of the OLMo 0625 mix, tokenized
with **`allenai/dolma2-tokenizer`** (vocab 100,278; padded 100,352;
EOS = 100,257). The file list is
[`OLMo-mix-0625-150Bsample.txt`](src/olmo_core/data/mixes/OLMo-mix-0625-150Bsample.txt)
— 6,915 files, ~600 GB. (On-disk paths read `dolma2-0625`; this is the same
corpus the paper refers to as Dolma-3.)

From that corpus we cut one validation set and a ladder of nested training
shards. **We recommend rebuilding these rather than asking us to send them** —
it is a download plus one deterministic pass, and moves ~600 GB instead of the
multi-TB alternative. (If that is awkward on your side, get in touch and we will
work something out.)

```bash
# 1. fetch the source corpus (~600 GB, ~2h at 8-way parallel)
bash experiment_scripts/download_data.sh

# 2. cut validation + the nested train shards (deterministic given --seed)
python experiment_scripts/create_dolma3_splits.py \
    --data-dir  <corpus>/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer \
    --output-dir <corpus>/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer \
    --seed 42
```

**How the splits work.** All documents across all 6,915 files are indexed, shuffled
once globally under `--seed 42`, then assigned greedily: validation first
(~500M tokens, disjoint from train), then each training shard as a **nested
prefix** of the same order. So `train_0.03B.npy` is a byte-exact prefix of
`train_0.06B.npy`, and so on up. Which shard each `(model_size, chinchilla)`
rung uses is declared in `_PER_MODEL_SHARDS` in the same script. Shards ≥ 20B
are written as `.partNN` files, which are contiguous slices of that same order.

Output files are flat `uint32` token streams with `EOS = 100257` between
documents. Despite the `.npy` extension they carry **no numpy header**, so
`np.load` will not read them — use:

```python
np.memmap(path, dtype=np.uint32, mode="r")
```

### Verifying your rebuild matches ours

- `doc_index_meta.json` (written by step 2) carries
  `file_list_hash: 357446410583ab31` over the 6,915 source files. If yours
  matches, your corpus is ours.
- Per-shard `(num_docs, num_tokens)` land in `shard_manifest.json`
  ([`build_shard_manifest.py`](experiment_scripts/build_shard_manifest.py)).
- **Check content, not just size.** `write_split` uses `np.memmap`, which
  preallocates the full file zero-filled and fills it progressively. A write
  interrupted partway leaves a file with the *correct* byte size and *correct*
  metadata whose tail is all zeros — it passes a size-and-manifest check and
  would silently train on garbage. Sample windows across each shard and assert
  non-zero fraction > 95%, max token ≤ vocab size, EOS present, and last token
  == EOS.

---

## 3. The paraphrase corpus

This section covers the paraphrase mechanism only. For repetition, `D'` needs no
data of its own — it is simply training on repeated `D`, recorded as `epochs`.

**How we scale up paraphrase tokens.** Every source document is paraphrased
repeatedly, once per *generation seed*, and a run at `K` trains on seeds
`1..K`. So `K` is the number of distinct paraphrases held per document, and
raising it is how the paraphrase corpus grows. These generation seeds are a
property of the **data**, and are unrelated to the training seed in the results
tables (which is 42 for every run).

Paraphrases were generated with **SmolLM2-1.7B-Instruct** via vLLM at
temperature 1.0, over the 4,866,732 documents of `train_7.4B.npy`, for
**64 independent seeds**. Each document is rewritten under one of four prompt
styles — `faq`, `math`, `table`, `tutorial`, with a math→wikipedia fallback for
documents with too few numeric tokens — chosen deterministically by
`pick_style_for_doc(doc_idx, seed)`, so the style mix differs across seeds too.

**Paraphrases are typically shorter than their sources** (~0.3× the source token
count per seed), and the ratio is not fixed. This is why paraphrase token counts
are measured from the files rather than computed from `K`, and why the training
script budgets from actual on-disk sizes (see [section 4](#4-training-settings)).

[`experiment_scripts/paraphrasing/PARAPHRASE_DATA_README.md`](experiment_scripts/paraphrasing/PARAPHRASE_DATA_README.md)
documents the layout and the doc-ordering invariant in full. The short version:

> The k-th document of `seed{N}/shard_{i}.npy` is a paraphrase of the same
> source document for every `N`. So `paraphrase_of(train_X.B)` is the first
> `num_docs(train_X.B)` documents of any seed's concatenated output — which is
> what makes `D'` line up with `D` at every rung.

**Critically, raising `K` adds no new source documents** — every document in `D'`
is a rewrite of one already in `D`. That is what isolates η.

Two forms live on disk:

| | what | size |
|---|---|---|
| `paraphrased/train_7.4B_smollm2_mixed_seed{1..64}/` | generator output: 32 shards per seed, each a `.npy` plus a `.jsonl` carrying the **paraphrase text** and its `prompt_style` | ~1.4 TB |
| `paraphrased/sized_smollm2_mixed/` | `D'` pre-cut to each rung (`train_<X>B_seed<N>.npy`) — what the paraphrase runs train on | ~3.5 TB |

The full `paraphrased/` tree is ~4.9 TB, but most of that is redundant: the
sized files are larger than the seeds they come from because every rung re-cuts
the same paraphrases at a different size. They are a deterministic function of
the raw seeds, rebuilt with
[`build_sized_paraphrase.py`](experiment_scripts/paraphrasing/build_sized_paraphrase.py).

So the raw seeds are the thing worth having: 1.4 TB rather than 4.9 TB, they
regenerate the rest, and the `.jsonl` side is human-readable, which the
tokenized sized files are not.

Generation is the expensive step (tens of thousands of GPU-hours), so unlike the
original data, regenerating this corpus is not the sensible path — **contact the
authors and we will arrange a transfer** (see [section 6](#6-obtaining-the-data)).

Relevant code: [`paraphrase_shard.py`](experiment_scripts/paraphrasing/paraphrase_shard.py)
(generator), [`check_paraphrase_integrity.py`](experiment_scripts/paraphrasing/check_paraphrase_integrity.py),
[`inspect_paraphrased_data.py`](experiment_scripts/paraphrasing/inspect_paraphrased_data.py).

---

## 4. Training settings

| stream | entry point | submission wrapper |
|---|---|---|
| repetition (incl. 1-epoch) | [`OLMo-scale-train-multiepoch-dolma.py`](src/scripts/official/OLMo-scale-train-multiepoch-dolma.py) | [`train_scale.sh`](experiment_scripts/train_scale.sh) |
| paraphrase (`D + D'×K`) | [`OLMo-scale-train-paraphrase-dolma.py`](src/scripts/official/OLMo-scale-train-paraphrase-dolma.py) | [`train_scale_paraphrase.sh`](experiment_scripts/train_scale_paraphrase.sh) |
| self-distillation | [`OLMo-scale-train-selfdistill-dolma.py`](src/scripts/official/OLMo-scale-train-selfdistill-dolma.py) | [`train_scale_selfdistill.sh`](experiment_scripts/train_scale_selfdistill.sh) |

A run is fully specified by `MODEL_SIZE`, `CHIN` (chinchilla multiplier), `LR`,
`WD`, and either `EPOCHS` (repetition) or `NUM_SEEDS` = K (paraphrase):

```bash
sbatch --export=ALL,MODEL_SIZE=30M,CHIN=1,NUM_SEEDS=8,LR=3e-3,WD=0.1,MICROBATCH_MULT=16 \
       experiment_scripts/train_scale_paraphrase.sh
```

Fixed across every run in the paper:

| | |
|---|---|
| architecture | `TransformerConfig.olmo3_<size>` for 14M/30M/60M/100M/190M/370M/600M |
| global batch | `512 × 4096` = 2,097,152 tokens |
| sequence length | 4096 |
| init seed | 42 |
| z-loss | 1e-5 |
| max grad norm | 1.0 |
| total steps | `total_training_tokens // global_batch_size` |

Swept: learning rate and weight decay (grids differ by stream — see
[`hparam_sweeps/README.md`](results/chinchilla_fit_dolma/data_export/hparam_sweeps/README.md),
which also maps how thoroughly each cell was actually explored).

`MICROBATCH_MULT` is gradient accumulation only (A100 = 8, H100 = 16,
H200 = 32); it changes throughput and numerics, not the global batch or the
optimizer math.

**One-epoch guarantee.** Because paraphrases are shorter than their sources, the
naive `params × 20 × chin × (1+K)` budget would over-shoot and start a partial
second pass. The training script instead sums actual on-disk sizes
(`st_size // 4`) and sets `max_duration` from that, giving exactly one pass over
`D + D'`.

Every run also wrote a complete `config.json` (model, optimizer, scheduler,
dataset) and a `data_paths.txt` into its checkpoint directory; 2,884 of these
survive and can be shared on request as a ~20 MB bundle.

---

## 5. Evaluation

[`run_eval_batch.py`](experiment_scripts/run_eval_batch.py) scores a checkpoint's
cross-entropy on the held-out `validation.npy` (`DataMix.OLMo_dolma_val`),
batching at `4 × 4096` tokens and caching eval batches across checkpoints.
Per-run outputs land in [`results/dolma_val_loss/`](results/dolma_val_loss/),
[`dolma_para_val_loss/`](results/dolma_para_val_loss/), and
[`dolma_sd_val_loss/`](results/dolma_sd_val_loss/), all committed here.

Every run is evaluated on the same held-out `validation.npy`; the default is a
3,478,624-token prefix of it, recorded per run in `tokens_evaluated`.

---

## 6. Obtaining the data

| | size | recommendation |
|---|---|---|
| source corpus (150B sample) | ~600 GB | download via [`download_data.sh`](experiment_scripts/download_data.sh) |
| `D` shards + `validation.npy` | ~115 GB | rebuild with `create_dolma3_splits.py --seed 42` |
| `doc_index.npz` (global doc order) | 551 MB | rebuilt by the same script, or **ask us** for a copy to skip the corpus scan |
| paraphrase raw seeds (64 × 32 shards, `.npy` + `.jsonl`) | ~1.4 TB | **contact the authors** |
| paraphrase sized `D'` | ~3.5 TB | rebuild from raw seeds with `build_sized_paraphrase.py` |

**Please get in touch for the paraphrase corpus** — it is the one piece that
cannot reasonably be regenerated, and we are happy to share it. Globus is the
practical route at that volume, and we can stage a subset (specific sizes or
seed ranges) if the full 1.4 TB is more than you need. 

---

## Citation

```bibtex
@article{qin2026bridging,
  title  = {Bridging Compute- and Data-Optimal Pretraining},
  author = {Qin, Tian and Hamidieh, Kimia and Alvarez-Melis, David},
  journal = {arXiv preprint arXiv:2607.25271},
  year   = {2026}
}
```
