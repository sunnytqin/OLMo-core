# 60M Scaling Sweep — Findings

Sweep: chinchilla ∈ {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8}, epochs ∈ {1, 2, 4, 8, 16, 32, 64}.
Phase 0 full WD×LR grid at chin=0.05, ep∈{16, 64}. Production runs for all other (chin, ep).

## Key findings

### 1. LR does NOT extrapolate from 30M/370M
- 30M: **LR=3e-3** optimal in 100% of settings
- 370M: **LR=3e-3** optimal in 83% of settings
- 60M: **LR=1e-3** optimal in 100% of settings — the one exception at chin=0.05 ep=64 where LR=3e-3 edges ahead is within noise.

**Takeaway:** Do not assume LR is constant across scales. The Phase 0 calibration sweep at chin=0.05 is essential.

### 2. Optimal WD for 60M

| Epochs | Best WD | Notes |
|--------|---------|-------|
| 1 | 0.1 | Only 0.1 tested broadly |
| 2 | 0.1 | |
| 4 | 0.1 | |
| 8 | 0.1 | |
| 16 | 0.2 | Phase 0 confirmed (WD=0.2 vs 0.1 within ~0.01) |
| 32 | 0.4 | |
| 64 | 0.4 to 0.8 | WD=0.8 wins for chin≤0.1 (prevents end-of-training rise) |

WD scales more gently for 60M than for 370M:
- 60M at ep=64: best WD ≈ 0.4-0.8
- 370M at ep=64: best WD ≈ 0.8-1.6

### 3. Short-epoch overfitting at ep=64
For small fresh-data sizes (chin≤0.25), WD=0.4 bottoms out early then the final val loss rises above the minimum. WD=0.8 fixes this:

| setting | WD=0.4 final | WD=0.8 final | min (WD=0.4) |
|---------|-------------|-------------|--------------|
| chin=0.05 ep=64 | 4.934 | **4.604** | 4.455 |
| chin=0.1 ep=64 | 4.413 | **4.136** | 4.016 |
| chin=0.25 ep=64 | 3.753 | **3.600** | 3.696 |

**Takeaway:** For ep=64 at low chin, default to WD=0.8. For high chin (≥0.5) where data is larger, WD=0.4 is fine.

### 4. `eval_interval=200` is too coarse for short runs
Short runs (chin=0.05 ep=4 → ~115 steps) never trigger an eval. Fix: `eval_interval=min(200, total_steps // 2)` — already applied to training script.

## Best val loss table (final val loss, all LR=1e-3 unless noted)

| chin | ep=1 | ep=2 | ep=4 | ep=8 | ep=16 | ep=32 | ep=64 |
|------|------|------|------|------|-------|-------|-------|
| 0.05 | 8.08 | 7.46 | — | 5.99 | 5.27 | 4.69 | 4.53¹ |
| 0.1 | 7.47 | 6.74 | 5.96 | 5.28 | 4.38 | 4.03 | 4.14² |
| 0.25 | 6.44 | 5.72 | 5.08 | — | 3.75 | 3.66 | 3.60² |
| 0.5 | 5.72 | 5.13 | 4.12 | 3.69 | 3.52 | 3.41 | 3.38 |
| 1 | 5.11 | — | 3.64 | 3.44 | 3.33 | 3.29 | 3.26 |
| 2 | 4.10 | — | 3.42 | 3.30 | 3.23 | 3.23 | 3.22³ |
| 4 | 3.63 | — | 3.30 | 3.21 | 3.18 | 3.20³ | — |
| 8 | 3.45 | — | 3.20 | 3.16 | 3.16³ | 3.25³ | — |

¹ WD=0.8 LR=3e-3 (from Phase 0 sweep). ² WD=0.8. ³ min (run incomplete).

## Operational lessons

### Cluster issues
- **4 bad A100 nodes** (broken CUDA/NVML): `holygpu8a19301, 19104, 19305, 19405`. Always submit with `--exclude` for these.
- **1 bad H200 node** found: `holygpu8a12501`.
- Bad-node failures are distinctive: exit code 1 within ~15-30 seconds, `CUDA driver initialization failed` or `NVML: Failed to get usage(999)` in log.

### Time limits
- kempner (A100): up to **7 days**. Use `--time=7-00:00` for long jobs.
- gpu_h200 (H200): up to **3 days**. Use `--time=3-00:00`.
- kempner_h100 (H100): 20h (use for short jobs only).
- Previous default of 20h caused many timeouts on chin≥1 ep≥32 — use longer limits from start.

### Throughput (60M)
- A100: ~12 sec/step (BPS ≈ 0.084)
- H200: ~4.6 sec/step (BPS ≈ 0.215)
- Global batch = 512 × 4096 = 2.1M tokens

### Partition strategy for 60M-scale runs
- chin ≤ 0.5: A100 (kempner) fine for all epochs
- chin = 1: A100 OK through ep=16, H200 preferred for ep=32-64
- chin ≥ 2: H200 strongly preferred for ep≥16
- chin=8 ep=32: ~120h compute — needs multiple 3-day H200 resubmissions (resume via checkpoint)

### Disk quota
Long runs on `chin≥4 ep≥32` accumulate huge `global_indices` files in the dataset cache. One run failed with `OSError: [Errno 122] Disk quota exceeded`. Periodically clean old global_indices files, or request larger quota.

### Resuming runs
- Training script auto-resumes from latest checkpoint if save folder is the same.
- **MICROBATCH_MULT must match the checkpoint** — switching an A100 (MBM=8) checkpoint to H200 (MBM=32) may fail. Resume on the same partition/GPU type.
- eval_interval is tied to `total_steps` which is computed per run, so the fixed-eval_interval trick means different runs can have different eval cadence.
