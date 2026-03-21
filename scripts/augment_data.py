"""
Generate augmented training CSV for rare emotion classes.

Usage:
    python scripts/augment_data.py \
        --input data/processed/train.csv \
        --output data/processed/train_augmented.csv \
        --target_min 100 \
        --methods random contextual
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.data_deps.augmentation import augment_rare_classes

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.csv")
    parser.add_argument("--output", default="data/processed/train_augmented.csv")
    parser.add_argument("--target_min", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=["random"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Original: {len(df)} samples")
    for l in LABEL_COLS:
        print(f"  {l:12s}: {int(df[l].sum())}")

    result = augment_rare_classes(df, LABEL_COLS, args.target_min, args.methods, args.seed)

    print(f"\nAugmented: {len(result)} samples")
    for l in LABEL_COLS:
        print(f"  {l:12s}: {int(result[l].sum())}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
