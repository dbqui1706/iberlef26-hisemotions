# Experiments Directory

This directory tracks model development progress and results.

## Structure

```
experiments/
├── experiment_log.md        # Full results table for all experiments (V1-V6)
├── data_issues.md           # Data analysis: 5 issues with severity ratings
├── charts/                  # 7 visualization charts
│   ├── 01_train_vs_dev_distribution.png
│   ├── 02_distribution_mismatch.png
│   ├── 03_class_imbalance_ratio.png
│   ├── 04_labels_per_sample.png
│   ├── 05_label_cooccurrence.png
│   ├── 06_text_length_distribution.png
│   └── 07_text_length_by_emotion.png
├── configs/                 # Config snapshot for each experiment run
│   ├── v1_roberta_baseline.yaml
│   ├── v2_xlm_roberta_improved.yaml
│   ├── v3_beto_improved.yaml
│   ├── v4_beto_fp32.yaml
│   ├── v5_beto_asl.yaml
│   └── v6_xlmr_asl.yaml
└── README.md                # This file
```

## How to Use

1. **Before running a new experiment:** Copy config YAML to `experiments/configs/` with prefix `vN_`
2. **After running:** Update tables in `experiment_log.md` with results and per-class F1
3. **Important notes:** Write changelog entry describing changes from previous version

## Current Best

**V4 (BETO + Weighted BCE)** → macro-F1 = 0.400 (optimized thresholds)

## Quick Reference

| Experiment | Model | Loss | Optimized F1 |
|-----------|-------|------|-------------|
| V1 | BETO | wBCE | 0.319 |
| V2 | XLM-R | wBCE | 0.255 ❌ |
| V3 | BETO | wBCE | 0.372 |
| **V4** | **BETO** | **wBCE** | **0.400** ✅ |
| V5 | BETO | ASL | 0.367 |
| V6 | XLM-R | ASL | 0.335 |
