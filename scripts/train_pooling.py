"""
Unified training script for pooling technique experiments.

Supports 3 pooling strategies:
  mean     — MeanPoolingClassifier
  attention — AttentionPoolingClassifier
  sentpair — NLI-style sentence-pair (AutoModelForSequenceClassification)

Usage:
    python scripts/train_pooling.py --config configs/exp04_pooling_mean_wbce.yaml
    python scripts/train_pooling.py --config configs/exp05_pooling_attention.yaml
    python scripts/train_pooling.py --config configs/exp06_pooling_sentpair.yaml
"""
import argparse
import yaml
import os
import sys
import json
import shutil

from dotenv import load_dotenv
load_dotenv()

try:
    from torchvision.io import VideoReader
except ImportError:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _DummyVR: pass
        _tvio.VideoReader = _DummyVR

os.environ["WANDB_PROJECT"] = "hisemotions_2026"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from src.data_deps.preprocessing import prepare_dataset
from src.evaluate.metrics import compute_multilabel_metrics, find_optimal_thresholds
from src.losses.asl import AsymmetricLoss
from src.losses.focal import WeightedFocalLoss
from src.trainer import HisemotionTrainer
from src.models.pooling_models import MeanPoolingClassifier, AttentionPoolingClassifier
from src.experiment_logger import log_experiment

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]


def _calculate_pos_weights(df, labels, max_weight=10.0):
    weights = []
    for label in labels:
        pos = df[label].sum()
        neg = len(df) - pos
        weights.append(min(neg / (pos + 1e-6), max_weight))
    return torch.tensor(weights, dtype=torch.float)


def build_loss_fn(config, train_df, label_cols, device):
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "").lower()
    if loss_type == "weighted_bce":
        max_weight = float(loss_cfg.get("max_weight", 5.0))
        weights = _calculate_pos_weights(train_df, label_cols, max_weight).to(device)
        print(f"  Loss: WeightedBCE  pos_weights {weights}")
        return nn.BCEWithLogitsLoss(pos_weight=weights)
    elif loss_type == "focal":
        gamma = float(loss_cfg.get("gamma", 2.0))
        max_weight = float(loss_cfg.get("max_weight", 5.0))
        weights = _calculate_pos_weights(train_df, label_cols, max_weight).to(device)
        print(f"  Loss: Focal (gamma={gamma})  pos_weights {weights}")
        return WeightedFocalLoss(gamma=gamma, pos_weight=weights)
    elif loss_type == "asl":
        print(f"  Loss: ASL")
        return AsymmetricLoss(
            gamma_neg=float(loss_cfg.get("gamma_neg", 4)),
            gamma_pos=float(loss_cfg.get("gamma_pos", 0)),
            clip=float(loss_cfg.get("clip", 0.05)),
        )
    else:
        return nn.BCEWithLogitsLoss()


# ===========================================================================
# Train Mean / Attention pooling (multi-label, 7 classes)
# ===========================================================================
# BestModelCallback: manually save/restore best weights for custom PyTorch models.
# HF Trainer's load_best_model_at_end only works for PreTrainedModel, NOT for
# custom nn.Module subclasses like MeanPoolingClassifier / AttentionPoolingClassifier.
# ---------------------------------------------------------------------------
class BestModelCallback(TrainerCallback):
    """Saves model.state_dict() whenever eval macro_f1 improves."""
    def __init__(self, model: torch.nn.Module, save_path: str):
        self.model = model
        self.save_path = save_path
        self.best_f1 = -1.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        f1 = metrics.get("eval_macro_f1", 0.0) if metrics else 0.0
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(self.model.state_dict(), self.save_path)
            print(f"  [BestModelCallback] ✓ macro_f1={f1:.4f} → saved best weights")


def train_custom_pooling(config, pooling_type, exp_model_dir):
    print("=" * 60)
    print(f"  Training: {pooling_type.upper()} POOLING")
    print("=" * 60)

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    train_df = pd.read_csv(config["data"]["train_path"])
    dev_df = pd.read_csv(config["data"]["dev_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = prepare_dataset(train_df, model_name, LABEL_COLS, max_length)
    dev_ds = prepare_dataset(dev_df, model_name, LABEL_COLS, max_length)

    # Model
    num_labels = len(LABEL_COLS)
    if pooling_type == "mean":
        model = MeanPoolingClassifier(model_name, num_labels, dropout=config.get("dropout", 0.1))
    elif pooling_type == "attention":
        model = AttentionPoolingClassifier(model_name, num_labels, dropout=config.get("dropout", 0.1))
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
    model.to(device)

    # Loss
    loss_fn = build_loss_fn(config, train_df, LABEL_COLS, device)

    # Sampler
    sampler_cfg = config.get("data", {}).get("sampler", None)
    labels_array = None
    if sampler_cfg:
        labels_array = np.array(train_ds["labels"])
        print(f"  Sampler: rare_threshold={sampler_cfg.get('rare_threshold', 300)}")

    t_cfg = config["training"]
    checkpoint_dir = os.path.join(exp_model_dir, "checkpoints")
    final_dir = os.path.join(exp_model_dir, "final")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    report_to = config.get("report_to", "none")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=float(t_cfg.get("learning_rate", 1e-5)),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 8),
        num_train_epochs=t_cfg.get("num_train_epochs", 20),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="macro_f1",
        report_to=report_to,
        run_name=config.get("experiment_name", f"pooling_{pooling_type}"),
        logging_steps=10,
        fp16=False,
        warmup_ratio=t_cfg.get("warmup_ratio", 0.1),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        max_grad_norm=t_cfg.get("max_grad_norm", 1.0),
    )

    # FGM adversarial training config
    fgm_cfg = config.get("fgm", {})
    use_fgm = fgm_cfg.get("enabled", False)
    fgm_epsilon = float(fgm_cfg.get("epsilon", 1.0))
    if use_fgm:
        print(f"  FGM: enabled (epsilon={fgm_epsilon})")

    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    best_model_cb = BestModelCallback(model, best_ckpt_path)

    trainer = HisemotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_multilabel_metrics,
        loss_fn=loss_fn,
        labels_array=labels_array,
        sampler_cfg=sampler_cfg,
        use_fgm=use_fgm,
        fgm_epsilon=fgm_epsilon,
        callbacks=[best_model_cb],
    )

    trainer.train()

    # Restore best weights
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"\n  Restored best model (macro_f1={best_model_cb.best_f1:.4f}) for eval/save")
    else:
        print("\n  Warning: best checkpoint not found, using final weights")

    # Eval default threshold
    print("\nEvaluating (threshold = 0.5)...")
    eval_results = trainer.evaluate()
    macro_f1_default = eval_results.get("eval_macro_f1", 0.0)
    macro_f1_6class_default = eval_results.get("eval_macro_f1_6class", 0.0)
    print(f"Default Macro-F1 (7-class): {macro_f1_default:.4f}")
    print(f"Default Macro-F1 (6-class, competition): {macro_f1_6class_default:.4f}")

    # Optimize thresholds (6-class competition metric)
    print("\nOptimizing per-class thresholds (6-class competition metric)...")
    EVAL_INDICES = list(range(6))
    preds = trainer.predict(dev_ds)
    opt_thresh, opt_f1, report, conf = find_optimal_thresholds(
        preds.predictions, preds.label_ids, LABEL_COLS,
        eval_class_indices=EVAL_INDICES,
    )
    print(f"  Optimized Macro-F1 (6-class, competition): {opt_f1:.4f}")
    print(f"  Thresholds: {dict(zip(LABEL_COLS, opt_thresh))}")

    # Save model
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    with open(os.path.join(final_dir, "results.json"), "w") as f:
        json.dump({
            "pooling": pooling_type,
            "optimized_macro_f1": opt_f1,
            "thresholds": opt_thresh,
            "labels": LABEL_COLS,
            "per_class_report": report,
        }, f, indent=2)
    print(f"  Saved to: {final_dir}")

    return macro_f1_default, macro_f1_6class_default, opt_f1, opt_thresh, report


# ===========================================================================
# Train Sentence-Pair (NLI-style, binary per emotion)
# ===========================================================================

def train_sentpair(config, exp_model_dir):
    from src.data_deps.sentpair_preprocessing import (
        prepare_sentpair_dataset, collate_sentpair_predictions,
    )
    from sklearn.metrics import f1_score

    print("=" * 60)
    print("  Training: SENTENCE-PAIR (NLI-style)")
    print("=" * 60)

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    train_df = pd.read_csv(config["data"]["train_path"])
    dev_df = pd.read_csv(config["data"]["dev_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_labels = len(LABEL_COLS)
    print(f"  Expanding data: train {len(train_df)} -> {len(train_df)*n_labels}, "
          f"dev {len(dev_df)} -> {len(dev_df)*n_labels}")

    train_ds = prepare_sentpair_dataset(train_df, model_name, LABEL_COLS, max_length)
    dev_ds = prepare_sentpair_dataset(dev_df, model_name, LABEL_COLS, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    ).to(device)

    def compute_binary_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
        acc = (preds == labels).mean()
        return {"f1": f1, "accuracy": acc}

    t_cfg = config["training"]
    checkpoint_dir = os.path.join(exp_model_dir, "checkpoints")
    final_dir = os.path.join(exp_model_dir, "final")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    report_to = config.get("report_to", "none")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=float(t_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 32),
        num_train_epochs=t_cfg.get("num_train_epochs", 5),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        report_to=report_to,
        run_name=config.get("experiment_name", "pooling_sentpair"),
        logging_steps=20,
        fp16=False,
        warmup_ratio=t_cfg.get("warmup_ratio", 0.1),
        weight_decay=t_cfg.get("weight_decay", 0.01),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_binary_metrics,
    )
    trainer.train()

    # Evaluate: reshape predictions to multi-label format
    print("\nEvaluating in multi-label format...")
    preds = trainer.predict(dev_ds)
    logits = preds.predictions
    probs_positive = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    prob_matrix = collate_sentpair_predictions(probs_positive, len(dev_df), n_labels)

    gt = dev_df[LABEL_COLS].values

    # Optimize per-class thresholds
    thresholds_dict = {}
    for i, col in enumerate(LABEL_COLS):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            pred = (prob_matrix[:, i] >= t).astype(int)
            f = f1_score(gt[:, i], pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, round(float(t), 2)
        thresholds_dict[col] = best_t

    final_preds = np.zeros_like(gt)
    for i, col in enumerate(LABEL_COLS):
        final_preds[:, i] = (prob_matrix[:, i] >= thresholds_dict[col]).astype(int)

    macro_f1 = f1_score(gt, final_preds, average='macro', zero_division=0)
    per_f1 = f1_score(gt, final_preds, average=None, zero_division=0)
    macro_f1_default = 0.0  # sentpair doesn't have a simple 0.5 threshold equivalent

    print(f"  Optimized Macro-F1: {macro_f1:.4f}")
    for col, f in zip(LABEL_COLS, per_f1):
        print(f"    {col:12s}: {f:.4f}")

    report = "\n".join([f"{col}: F1={f:.4f}" for col, f in zip(LABEL_COLS, per_f1)])

    # Save
    trainer.save_model(final_dir)
    with open(os.path.join(final_dir, "results.json"), "w") as f:
        json.dump({
            "pooling": "sentpair",
            "optimized_macro_f1": float(macro_f1),
            "thresholds": thresholds_dict,
            "labels": LABEL_COLS,
            "per_class_f1": dict(zip(LABEL_COLS, per_f1.tolist())),
        }, f, indent=2)
    print(f"  Saved to: {final_dir}")

    return macro_f1_default, 0.0, macro_f1, list(thresholds_dict.values()), report


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config.get("experiment_name", "pooling_experiment")
    pooling_type = config.get("pooling_type", "mean")

    # ── Directory structure: models/{exp_name}/ ──────────────────────────
    exp_model_dir = os.path.join("models", exp_name)
    os.makedirs(exp_model_dir, exist_ok=True)

    # Copy config for reproducibility
    shutil.copy2(args.config, os.path.join(exp_model_dir, "config.yaml"))

    print(f"\nExperiment: {exp_name}")
    print(f"Pooling:    {pooling_type}")
    print(f"Save to:    {exp_model_dir}\n")

    if pooling_type in ("mean", "attention"):
        f1_default, f1_6class_default, f1_opt, thresholds, report = train_custom_pooling(config, pooling_type, exp_model_dir)
    elif pooling_type == "sentpair":
        f1_default, f1_6class_default, f1_opt, thresholds, report = train_sentpair(config, exp_model_dir)
    else:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")

    # Log experiment
    log_experiment(
        experiment_name=exp_name,
        config=config,
        macro_f1_default=f1_default,
        macro_f1_optimized=f1_opt,
        thresholds=thresholds,
        per_class_report=report,
        label_cols=LABEL_COLS,
        save_dir=os.path.join(exp_model_dir, "final"),
        macro_f1_6class_default=f1_6class_default,
        macro_f1_6class_optimized=f1_opt,
    )

    print(f"\n{'='*60}")
    print(f"  DONE: {exp_name} ({pooling_type} pooling)")
    print(f"  Optimized Macro-F1: {f1_opt:.4f}")
    print(f"  Model: {exp_model_dir}/final/")
    print(f"  Log:   experiments/{exp_name}.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
