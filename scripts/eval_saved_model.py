import argparse
import yaml
import os
import sys
import json
import pandas as pd
import torch

try:
    from torchvision.io import VideoReader  # noqa: F401
except ImportError:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _DummyVideoReader:
            pass
        _tvio.VideoReader = _DummyVideoReader

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_deps.preprocessing import prepare_dataset
from src.evaluate.metrics import compute_multilabel_metrics, find_optimal_thresholds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Note: Double quotes inside f-string interpolation fixed
    exp_name = config["experiment_name"]
    save_dir = os.path.join(config["training"]["output_dir"], f"final_model_{exp_name}")
    print(f"Loading model from: {save_dir}")
    
    if not os.path.exists(save_dir):
        print("Error: Model path does not exist!")
        sys.exit(1)

    # 1. Load data
    dev_df = pd.read_csv(config["data"]["dev_path"])
    label_cols = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]

    # 2. Prepare dataset
    dev_dataset = prepare_dataset(
        dev_df, config["model"]["name"], label_cols,
        max_length=config["model"]["max_length"]
    )

    # 3. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        save_dir,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 4. Trainer for eval
    training_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dev_dataset,
        compute_metrics=compute_multilabel_metrics,
    )

    print("\nEvaluating (threshold = 0.5)...")
    eval_results = trainer.evaluate()
    print(f"Results: {eval_results}")

    print("\nOptimizing per-class thresholds...")
    dev_preds = trainer.predict(dev_dataset)
    optimal_thresholds, optimized_f1, class_report, conf_matrix = find_optimal_thresholds(
        dev_preds.predictions, dev_preds.label_ids, label_cols
    )

    print(f"Optimized Macro-F1: {optimized_f1:.4f}")
    print(f"Per-class report:\n{class_report}")
    print(f"Confusion Matrix:")
    for name, matrix in zip(label_cols, conf_matrix):
        print(f"--- {name} ---")
        print(matrix)

if __name__ == "__main__":
    main()
