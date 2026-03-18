# Experiment Log — HISEMOTIONS 2026

Bảng theo dõi kết quả các thí nghiệm qua từng lần phát triển.

## Experiments Summary

| # | Date | Experiment | Model | Epochs | Batch | Macro-F1 | Precision | Recall | Notes |
|---|------|------------|-------|--------|-------|----------|-----------|--------|-------|
| 1 | 2026-03-18 | `roberta_baseline` | `dccuchile/bert-base-spanish-wwm-uncased` | 5 | 8 | 0.319 | 0.242 | 0.544 | Baseline. Extreme class weights (anger:220, surprise:294). Over-predicts. |
| 2 | 2026-03-18 | `xlm_roberta_improved` | `xlm-roberta-base` | 5 | 8 | *pending* | *pending* | *pending* | Capped weights (max 10), warmup 0.1, weight_decay 0.01, threshold optimization |

## Changelog

### v2 — `xlm_roberta_improved` (2026-03-18)
- Capped class weights at max 10.0 (anger: 220→10, surprise: 294→10)
- Switched backbone: `bert-base-spanish-wwm-uncased` → `xlm-roberta-base`
- Added `warmup_ratio: 0.1`, `weight_decay: 0.01`
- Added post-training per-class threshold optimization
- Thresholds + Macro-F1 saved to `optimal_thresholds.json`

### v1 — `roberta_baseline` (2026-03-18)
- Initial baseline with BETO model
- `BCEWithLogitsLoss` with raw class weights (no cap)
- Fixed threshold 0.5 for all classes
- Macro-F1: **0.319**
