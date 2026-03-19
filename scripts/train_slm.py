import argparse
import yaml
import os
import sys

# Fix nightly torchvision compatibility
try:
    from torchvision.io import VideoReader  # noqa: F401
except ImportError:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _DummyVideoReader:
            pass
        _tvio.VideoReader = _DummyVideoReader

# Set wandb project name
os.environ["WANDB_PROJECT"] = "hisemotions_2026"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.data_deps.preprocessing import prepare_dataset
from src.evaluate.metrics import compute_multilabel_metrics, find_optimal_thresholds
from src.losses.asl import AsymmetricLoss

# ---------------------------------------------------------------------------
# Trainer with ASL loss (drop-in replacement for WeightedMultiLabelTrainer)
# ---------------------------------------------------------------------------

class ASLTrainer(Trainer):
    """
    Trainer using Asymmetric Loss instead of BCEWithLogitsLoss.

    Supported YAML config (all optional, has defaults):
        loss:
          type: "asl"        # "asl" | "weighted_bce"
          gamma_neg: 4       # ASL: penalize easy negatives
          gamma_pos: 0       # ASL: do not discount hard positives
          clip: 0.05         # ASL: clip negative probabilities
          max_weight: 10.0   # weighted_bce: cap pos_weight
    """

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        # If no loss_fn provided, fallback to plain BCE
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Ensure loss_fn runs on the correct device
        loss = self.loss_fn(logits.to(torch.float32), labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_loss_fn(config: dict, train_df: pd.DataFrame, label_cols: list, device: torch.device):
    """
    Read config['loss'] and return the appropriate loss function.
    Default: ASL with gamma_neg=4, gamma_pos=0, clip=0.05
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "weighted_bce").lower()

    if loss_type == "asl":
        gamma_neg = float(loss_cfg.get("gamma_neg", 4.0))
        gamma_pos = float(loss_cfg.get("gamma_pos", 0.0))
        clip      = float(loss_cfg.get("clip", 0.05))
        print(f"  Loss: AsymmetricLoss(gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip})")
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    elif loss_type == "weighted_bce":
        max_weight = float(loss_cfg.get("max_weight", 10.0))
        weights = _calculate_pos_weights(train_df, label_cols, max_weight).to(device)
        print(f"  Loss: WeightedBCE  pos_weights={weights.tolist()}")
        return nn.BCEWithLogitsLoss(pos_weight=weights)

    else:
        print(f"  Loss: BCEWithLogitsLoss (fallback — unknown type '{loss_type}')")
        return nn.BCEWithLogitsLoss()


def _calculate_pos_weights(df: pd.DataFrame, labels: list, max_weight: float = 10.0) -> torch.Tensor:
    weights = []
    for label in labels:
        pos = df[label].sum()
        neg = len(df) - pos
        weights.append(min(neg / (pos + 1e-6), max_weight))
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SLM for HISEMOTIONS")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Experiment : {config['experiment_name']}")
    print(f"Model      : {config['model']['name']}")
    print(f"{'='*60}\n")

    # Initialize wandb if available and configured
    if config["training"].get("report_to") == "wandb":
        try:
            import wandb
            wandb.init(
                project="hisemotions_2026",
                name=config["experiment_name"],
                config=config,
                reinit=True,
            )
        except ImportError:
            print("Warning: wandb not installed, disabling wandb reporting")
            config["training"]["report_to"] = "none"

    # 1. Load data
    train_df = pd.read_csv(config["data"]["train_path"])
    dev_df   = pd.read_csv(config["data"]["dev_path"])
    label_cols = ["anger", "fear", "joy", "sadness", "surprise", "hope"]

    # Clean data via preprocessing pipeline
    from src.data_deps.preprocessing import clean_dataframe, round_robin_balance
    print("\nCleaning Train Data:")
    train_df = clean_dataframe(train_df, label_cols)
    print("Cleaning Dev Data:")
    dev_df   = clean_dataframe(dev_df, label_cols)
    
    # Optional Class Balancing
    if config["data"].get("balance_classes") == "round_robin":
        print("\nBalancing Train Data (Round-Robin):")
        train_df = round_robin_balance(train_df, label_cols)

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 3. Loss function
    loss_fn = build_loss_fn(config, train_df, label_cols, device)

    # 4. Prepare datasets
    train_dataset = prepare_dataset(
        train_df, config["model"]["name"],
        max_length=config["model"]["max_length"]
    )
    dev_dataset = prepare_dataset(
        dev_df, config["model"]["name"],
        max_length=config["model"]["max_length"]
    )

    # 5. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,   # avoid classifier head size mismatch errors
    )

    model.to(device)

    # warmup steps 
    total_steps = (len(train_dataset) // config["training"]["per_device_train_batch_size"]) \
                  * config["training"]["num_train_epochs"]
    warmup_ratio = config["training"].get("warmup_ratio", 0.1)
    warmup_steps = int(total_steps * warmup_ratio)

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        save_total_limit=config["training"].get("save_total_limit", 2),
        metric_for_best_model=config["training"]["metric_for_best_model"],
        report_to=config["training"]["report_to"],
        run_name=config["experiment_name"],
        logging_steps=10,
        fp16=(config["training"].get("precision", "fp16") == "fp16" and torch.cuda.is_available()),
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        warmup_steps=warmup_steps,
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # 7. Trainer
    trainer = ASLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_multilabel_metrics,
        loss_fn=loss_fn,
    )

    # 8. Train
    print("Training...\n")
    trainer.train()

    # 9. Eval default threshold
    print("\nEvaluating (threshold = 0.5)...")
    eval_results = trainer.evaluate()
    print(f"Results: {eval_results}")

    # 10. Optimize per-class thresholds
    print("\nOptimizing per-class thresholds...")
    dev_preds = trainer.predict(dev_dataset)
    optimal_thresholds, optimized_f1 = find_optimal_thresholds(
        dev_preds.predictions, dev_preds.label_ids, label_cols
    )
    print(f"Optimized Macro-F1: {optimized_f1:.4f}")

    # 11. Save
    save_dir = os.path.join(config["training"]["output_dir"], f"final_model_{config['experiment_name']}")
    trainer.save_model(save_dir)
    with open(os.path.join(save_dir, "optimal_thresholds.json"), "w") as f:
        json.dump(
            {
                "thresholds": optimal_thresholds,
                "labels": label_cols,
                "optimized_macro_f1": optimized_f1,
                "loss_type": config.get("loss", {}).get("type", "asl"),
                "experiment": config["experiment_name"],
            },
            f, indent=2,
        )
    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()
