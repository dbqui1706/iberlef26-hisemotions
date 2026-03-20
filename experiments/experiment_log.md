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
| 6 | 2026-03-19 | `xlmr_asl_v6` | xlm-roberta-base | **ASL** | 0.125 | 0.335 | 0.081 | 0.292 | ASL + XLM-R. LayerNorm fix applied. lr=1e-5. |

## Per-class F1 Comparison (Optimized Thresholds)

| Label | Support | V4 (BETO+wBCE) | V5 (BETO+ASL) | V6 (XLM-R+ASL) | Best |
|-------|---------|---------------|---------------|-----------------|------|
| anger | 52 | **0.425** | 0.350 | 0.313 | V4 |
| fear | 40 | **0.340** | 0.307 | **0.390** | V6 |
| joy | 29 | **0.448** | 0.426 | 0.267 | V4 |
| sadness | 70 | 0.527 | **0.562** | 0.488 | **V5** |
| surprise | 7 | **0.235** | 0.068 | 0.154 | V4 |
| hope | 85 | 0.427 | **0.488** | 0.392 | **V5** |
| **Macro-F1** | — | **0.400** | 0.367 | 0.334 | **V4** |

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

### V6 Per-class Detail (XLM-R + ASL)

| Label | Threshold | F1 | Precision | Recall |
|-------|----------|------|-----------|--------|
| anger | 0.20 | 0.313 | 0.192 | 0.846 |
| fear | 0.50 | 0.390 | 0.263 | 0.750 |
| joy | 0.40 | 0.267 | 0.258 | 0.276 |
| sadness | 0.70 | 0.488 | 0.367 | 0.729 |
| surprise | 0.30 | 0.154 | 0.167 | 0.143 |
| hope | 0.30 | 0.392 | 0.258 | 0.812 |

## Key Findings

- **V4 (BETO + Weighted BCE) remains best** at macro-F1 = 0.400
- **Threshold optimization is the single biggest boost**: V4 jumped from 0.286 → 0.400 (+40%)
- **ASL (γ⁻=4) hurt performance** on this small dataset — too aggressive suppression of negatives reduced recall without sufficient precision gain
- **XLM-R now trains successfully** (V6) thanks to `ignore_mismatched_sizes=True` — old V2 crash was a code bug, not a model issue
- **XLM-R underperforms BETO** on this Spanish-specific task (0.335 vs 0.400)
- `bf16` causes NaN on BETO → stick with fp16
- **Bottleneck remains**: surprise (F1=0.235, 7 dev samples) and fear (F1=0.340) drag macro-F1

### ASL Analysis
- ASL improved sadness (0.527→0.562) and hope (0.427→0.488) but destroyed surprise (0.235→0.068)
- `gamma_neg=4` too aggressive for rare classes with <15 train samples
- XLM-R's best class was fear (0.390 > V4's 0.340) — potential ensemble candidate
- Next steps: try `gamma_neg=2, clip=0.02` or per-class gamma tuning

## Changelog

### v6 — `xlmr_asl_v6` (2026-03-19)
**Result:** Macro-F1 = 0.125 (default) → 0.335 (optimized)

Changes:
- Switched model to `xlm-roberta-base`
- Fixed LayerNorm mismatch via `ignore_mismatched_sizes=True`
- ASL loss (γ⁻=4, γ⁺=0, clip=0.05)
- max_length: 256, lr: 1e-5

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

### v2 — `xlm_roberta_improved` (2026-03-18) ❌
Macro-F1: 0.105. LayerNorm key mismatch — fixed in V6.

### v1 — `roberta_baseline` (2026-03-18)
Macro-F1: 0.319. Raw class weights. Fixed threshold 0.5.
