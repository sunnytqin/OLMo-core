# Paraphrased dolma data — layout and how to build (D + D') training mixes

## Goal

We train models on `D + D'` where:
- **`D`** = a slice of original dolma training data (e.g. `train_2.4B.npy`)
- **`D'`** = paraphrased versions of *exactly the same documents that appear in D*
- We sweep two axes: the size of `D` (chinchilla-like ladder) and the ratio `D'/D ∈ {0, 1, 2, 4, 8, 16}` controlled by adding more paraphrase seeds.

**Critical property:** when we increase `D'/D` by adding another seed, we are NOT introducing new source documents — every doc in `D'` is a paraphrase of a doc that already exists in `D`. Different seeds give different paraphrases of the *same* documents.

---

## Where the data lives

Root: `/n/netscratch/barak_lab/Everyone/sqin/olmo/preprocessed/dolma2-0625/resharded/allenai/dolma2-tokenizer/`

```
train_7.4B.npy                          # ORIGINAL — 4,866,732 docs
train_3.8B.npy, train_1.9B.npy, ...     # ORIGINAL nested-prefix shards
shard_manifest.json                     # {shard_name: {num_docs, num_tokens}}
paraphrased/
    train_7.4B_seed0/                   # ← V1: OLMo-3-7B + Wikipedia prompt (legacy)
    train_7.4B_smollm2_mixed_seed{1..8}/   # ← V2: SmolLM2-1.7B + 4 mixed prompts
        shard_0000.npy   # ~150K paraphrased docs, dolma2-tokenized, EOS-separated
        shard_0000.jsonl # checkpoint with {idx, text, prompt_style} per doc
        ...
        shard_0031.npy
        shard_0031.jsonl
```

- 32 shards per seed; each shard covers the same 152,085-doc slice of `train_7.4B` across all seeds.
- `.npy` is a flat `np.uint32` memmap of dolma2 token IDs with `EOS_TOKEN_ID = 100257` between docs.

---

## The doc-ordering invariant (this makes everything work)

**The k-th doc of every paraphrased shard `seed{N}/shard_{i}.npy` is a paraphrase of the same source doc.** Specifically: it is a paraphrase of the k-th doc in the i-th 152,085-doc slice of `train_7.4B.npy`, regardless of `N`.

This means if you take the first `M` docs of seed 1's concatenated paraphrase output, those are paraphrases of the first `M` docs of `train_7.4B`. And the first `M` docs of `train_7.4B` are exactly `train_X.B` for the appropriate `X` — that's how the original splits were built (see [create_dolma3_splits.py](../create_dolma3_splits.py): nested prefixes of one globally-shuffled doc order).

So:
> **paraphrase_of(train_X.B) = first num_docs(train_X.B) docs of any seed's concatenated paraphrase output.**

And combining multiple seeds for a given target size:
> **D' at ratio K for size train_X.B = concat of (paraphrase_of(train_X.B) from K different seeds).**

---

## Document counts you'll need

From `shard_manifest.json`:
- `train_0.03B.npy`: 19,658 docs
- `train_0.19B.npy`: 123,879 docs
- `train_0.6B.npy`: 395,851 docs
- `train_1.9B.npy`: 1,254,047 docs
- `train_2.4B.npy`: 1,582,708 docs
- `train_3.8B.npy`: 2,507,923 docs
- `train_7.4B.npy`: 4,866,732 docs
- (full list in `shard_manifest.json`)

Per paraphrased shard: 152,085 docs (shards 0–30), 152,097 docs (shard 31). All 32 shards together = 4,866,732 docs (= 32 × 152,085 + 12 leftover; verify per-shard via `wc -l shard_{i}.jsonl` if exact counts matter).

---

## Recipe for building D' at size `train_X.B` from seeds `{s_1, ..., s_K}`

1. Look up `N = num_docs(train_X.B)` in `shard_manifest.json`.
2. Compute `shard_idx_full = N // 152085` (number of paraphrase shards consumed in full).
3. Compute `docs_in_partial_shard = N % 152085` (docs to take from the final partial shard).
4. For each seed `s` in `{s_1, ..., s_K}`:
   - Concatenate `seed{s}/shard_0000.npy ... shard_{shard_idx_full - 1}.npy` (full shards).
   - From `seed{s}/shard_{shard_idx_full}.npy`, take the first `docs_in_partial_shard` docs — split by counting `EOS_TOKEN_ID` (100257) positions.
5. Concatenate the per-seed outputs in seed order.

This gives you a single `D'` token array. Wrap it with EOS sentinels as needed by the data loader.

### Worked example: `D = train_2.4B`, `D'/D = 2` using seeds 1 and 2

```
N = 1,582,708 docs
shard_idx_full = 1,582,708 // 152,085 = 10        # consume seed{N}/shard_0000..0009 fully
docs_in_partial_shard = 1,582,708 % 152,085 = 61,858  # then 61,858 docs from shard_0010

D' = (
    seed1/shard_0000.npy + ... + seed1/shard_0009.npy + first 61,858 docs of seed1/shard_0010.npy
    +
    seed2/shard_0000.npy + ... + seed2/shard_0009.npy + first 61,858 docs of seed2/shard_0010.npy
)
```

Then training data is `train_2.4B.npy + D'` (concat, with EOS between docs as usual).

### Higher ratios

`D'/D = 4` → use seeds {1,2,3,4} the same way.
`D'/D = 8` → use seeds {1..8}.
We have 8 seeds available, so the maximum supported ratio is 8.

---

## What's the same and what differs across seeds 1–8

**Same across all 8 seeds:**
- Source documents (the 4.87M docs of `train_7.4B.npy` in identical order).
- Per-document prompt assignment (deterministic from `(seed_inside_seed=2, doc_idx)`... wait, see below).
- Math→Wikipedia fallback rule (when a doc has < 2 numerical tokens).

**Differs across seeds:**
- vLLM sampling seed (so generated text differs even when the prompt is identical).
- Per-document prompt assignment IS reseeded per seed: `pick_style_for_doc(doc_idx, seed)` uses both as inputs. So in practice each seed produces a *different* mixture of (faq/math/table/tutorial/wikipedia-fallback) outputs for any given doc. This is intentional — it maximizes per-doc diversity across seeds.

The `prompt_style` field in each JSONL line records which prompt was used for that (seed, doc_idx). Useful for filtering/diagnostics.

---

## V1 vs V2

- `paraphrased/train_7.4B_seed0/` — generated with **OLMo-3-7B-Instruct + a single Wikipedia-style prompt**. Older "V1" data; can be used as-is.
- `paraphrased/train_7.4B_smollm2_mixed_seed{1..8}/` — generated with **SmolLM2-1.7B-Instruct + 4 mixed prompts** (faq, math, table, tutorial) with math→wikipedia fallback. The intended training data.

The two versions are NOT interchangeable per-doc (different paraphrases). Treat them as separate D' pools.

---

## Helpful code locations

- [paraphrase_shard.py](paraphrase_shard.py) — generator. `pick_style_for_doc(doc_idx, seed)` is the deterministic prompt picker. `MIXED_STYLES` is the canonical 4-tuple of styles sampled in mixed mode.
- [build_shard_manifest.py](build_shard_manifest.py) — produces `shard_manifest.json`.
- [inspect_pilot.py](inspect_pilot.py) — runs per-style length/refusal stats on a paraphrase JSONL; example of how to walk the JSONL + match against the source `.npy`.
- `pick_style_for_doc` and `MIXED_STYLES` can be imported from `paraphrase_shard` if the trainer needs to reconstruct which prompt produced which doc.

---

## TL;DR for the training-script agent

1. Load `shard_manifest.json` to get `num_docs` for whatever original shard `train_X.B.npy` you're matching.
2. For each seed in your seed-set, take the first `num_docs(train_X.B)` paraphrased docs by concatenating the right number of full shards plus a doc-counted partial slice of the next shard.
3. Concat seeds in order to get D' at ratio `K = len(seed_set)`.
4. Final training data = `train_X.B.npy` + D' (via OLMo's existing DataMix concatenation).
