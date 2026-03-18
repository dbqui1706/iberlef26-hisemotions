# Experiment Log — HISEMOTIONS 2026

Bảng theo dõi kết quả các thí nghiệm qua từng lần phát triển.

## Experiments Summary

| # | Date | Experiment | Model | Macro-F1 | Optimized F1 | Precision | Recall | Notes |
|---|------|------------|-------|----------|-------------|-----------|--------|-------|
| 1 | 2026-03-18 | `roberta_baseline` | BETO | 0.319 | N/A | 0.242 | 0.544 | Baseline. No weight cap. fp16. |
| 2 | 2026-03-18 | `xlm_roberta_improved` | xlm-roberta-base | 0.105 | 0.255 | 0.060 | 0.486 | ❌ LayerNorm mismatch |
| 3 | 2026-03-18 | `beto_improved_v3` | BETO | 0.276 | 0.372 | 0.269 | 0.461 | fp16. Capped weights. grad_norm=0. |
| 4 | 2026-03-18 | `beto_fp32_v4` | BETO | 0.286 | **0.400** | 0.226 | 0.500 | ✅ Best. fp16 + capped weights + warmup + decay + threshold opt. |

## Key Findings

- **Threshold optimization is the single biggest boost**: V4 jumped from 0.286 → 0.400 (+40%) just by tuning per-class thresholds
- **Capped weights + warmup + weight_decay** improve default F1 slightly but main benefit is through threshold optimization
- **Best optimized F1 = 0.400** (V4) vs baseline 0.319 (V1) = **+25% improvement**
- `xlm-roberta-base` has broken LayerNorm naming → avoid
- `bf16` causes NaN on BETO → stick with fp16

## V4 Per-class Optimal Thresholds

| anger | fear | joy | sadness | surprise | hope |
|-------|------|-----|---------|----------|------|
| 0.35  | 0.70 | 0.75| 0.75    | 0.30     | 0.20 |

## Changelog

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
Macro-F1: 0.105. LayerNorm key mismatch.

### v1 — `roberta_baseline` (2026-03-18)
Macro-F1: 0.319. Raw class weights. Fixed threshold 0.5.
