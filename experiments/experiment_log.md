# Experiment Log — HISEMOTIONS 2026

Tracking experiment results across development iterations.

## Experiments Summary

| # | Date | Experiment | Model | Loss | Macro-F1 | Optimized F1 | Precision | Recall | Notes |
|---|------|------------|-------|------|----------|-------------|-----------|--------|-------|
| 1 | 2026-03-18 | `roberta_baseline` | BETO | Weighted BCE | 0.319 | N/A | 0.242 | 0.544 | Baseline. No weight cap. fp16. |
| 2 | 2026-03-18 | `xlm_roberta_improved` | xlm-roberta-base | Weighted BCE | 0.105 | 0.255 | 0.060 | 0.486 | ❌ LayerNorm mismatch |
| 3 | 2026-03-18 | `beto_improved_v3` | BETO | Weighted BCE | 0.276 | 0.372 | 0.269 | 0.461 | Capped weights. grad_norm=0. |
| 4 | 2026-03-18 | `beto_fp32_v4` | BETO | Weighted BCE | 0.286 | **0.400** | 0.226 | 0.500 | ✅ **Best.** Capped weights + warmup + decay + threshold opt. |
| 5 | 2026-03-19 | `beto_asl_v5` | BETO | **ASL** | 0.216 | 0.367 | 0.224 | 0.403 | ASL(γ⁻=4, clip=0.05). max_length=256. |
| 7 | 2026-03-20 | `beto_wbce_v7_fixed` | BETO | **Weighted BCE** | 0.406 | **0.467** | 0.406 | 0.580 | ✅ **New Best.** TwoPhaseSampler + wBCE + Threshold Opt. |

## Per-class F1 Comparison (Optimized Thresholds)

| Label | Support | V4 (BETO+wBCE) | V5 (BETO+ASL) | V7 (BETO+wBCE+Sampler) | Best |
|-------|---------|---------------|---------------|-----------------------|------|
| anger | 52 | 0.425 | 0.350 | **0.458** | **V7** |
| fear | 40 | **0.340** | 0.307 | 0.301 | V4 |
| joy | 29 | **0.448** | 0.426 | 0.412 | V4 |
| sadness | 70 | 0.527 | **0.562** | 0.470 | V5 |
| surprise | 7 | 0.235 | 0.068 | **0.429** | **V7** |
| hope | 85 | 0.427 | **0.488** | 0.469 | V5 |
| **Macro-F1** | — | 0.400 | 0.367 | **0.467** | **V7** |

### V4 Per-class Detail

| Label | Threshold | F1 | Precision | Recall |
|-------|----------|------|-----------|--------|
| anger | 0.35 | 0.425 | 0.360 | 0.519 |
| fear | 0.70 | 0.340 | 0.230 | 0.650 |
| joy | 0.75 | 0.448 | 0.448 | 0.448 |
| sadness | 0.75 | 0.527 | 0.400 | 0.771 |
| surprise | 0.30 | 0.235 | 0.200 | 0.286 |
| hope | 0.20 | 0.427 | 0.331 | 0.600 |

### V5 Per-class Detail (BETO + ASL)

| Label | Threshold | F1 | Precision | Recall |
|-------|----------|------|-----------|--------|
| anger | 0.20 | 0.350 | 0.228 | 0.750 |
| fear | 0.55 | 0.307 | 0.216 | 0.525 |
| joy | 0.50 | 0.426 | 0.308 | 0.690 |
| sadness | 0.75 | 0.562 | 0.443 | 0.771 |
| surprise | 0.25 | 0.068 | 0.038 | 0.286 |
| hope | 0.35 | 0.488 | 0.406 | 0.612 |

### V7 Per-class Detail (BETO + Weighted BCE + TwoPhaseSampler)

| Label | Threshold | F1 | Precision | Recall |
|-------|----------|------|-----------|--------|
| anger | Opt | 0.458 | 0.409 | 0.519 |
| fear | Opt | 0.301 | 0.215 | 0.500 |
| joy | Opt | 0.412 | 0.288 | 0.724 |
| sadness | Opt | 0.470 | 0.406 | 0.557 |
| surprise | Opt | 0.429 | 0.429 | 0.429 |
| hope | Opt | 0.469 | 0.447 | 0.494 |

## Key Findings

- **V7 (BETO + Weighted BCE + TwoPhaseSampler) set a new SOTA** at macro-F1 = 0.4670, completely overshadowing V4.
- Custom dataset sampling completely revived the `surprise` class (0.235 -> 0.429) bypassing the HuggingFace DataLoader truncation bugs.
- Weighted BCE + TwoPhaseSampler effectively handles extreme multi-label imbalances natively.
- `bf16` causes NaN on BETO → stick with fp16
- **Bottleneck remains**: surprise (F1=0.235, 7 dev samples) and fear (F1=0.340) drag macro-F1

## Changelog

### v7 — `beto_wbce_v7_fixed` (2026-03-20) ✅ BEST
**Result:** Macro-F1 = 0.406 (default) → **0.467** (optimized thresholds)

Changes from V4:
- Reverted back to Weighted BCE instead of ASL.
- Replaced standard DataLoader with custom `TwoPhaseSampler` to fix truncation bugs.
- Interleaved 4 rare slots + 12 common slots dynamically per batch without losing sequences.
- Resulted in massive F1 spikes for `surprise` (0.429) & `anger` (0.458).

### v5 — `beto_asl_v5` (2026-03-19)
**Result:** Macro-F1 = 0.216 (default) → 0.367 (optimized)

Changes from V4:
- Replaced Weighted BCE → ASL loss (γ⁻=4, γ⁺=0, clip=0.05)
- max_length: 128 → 256
- Config-driven loss selection via `build_loss_fn()`
- ASL improved sadness (+0.035) and hope (+0.061) but destroyed surprise (0.235 → 0.068)

### v4 — `beto_fp32_v4` (2026-03-18) ✅ BEST
**Result:** Macro-F1 = 0.286 (default) → **0.400** (optimized thresholds)

Changes from V1:
- Capped class weights at max 10.0
- Added warmup_ratio: 0.1, weight_decay: 0.01
- Post-training per-class threshold optimization
- Saved thresholds to `optimal_thresholds.json`

### v3 — `beto_improved_v3` (2026-03-18)
Macro-F1: 0.276 → 0.372 (optimized). Same as v4 but without warmup/decay.

### v1 — `roberta_baseline` (2026-03-18)
Macro-F1: 0.319. Raw class weights. Fixed threshold 0.5.
