"""
Experiment logger — saves per-experiment JSON logs to experiments/{experiment_name}.json
"""
import json
import os
from datetime import datetime


def log_experiment(
    experiment_name: str,
    config: dict,
    macro_f1_default: float,
    macro_f1_optimized: float,
    thresholds: list,
    per_class_report: str,
    label_cols: list,
    save_dir: str,
    extra: dict = None,
):
    """Save experiment results to experiments/{experiment_name}.json"""
    log_dir = "experiments"
    os.makedirs(log_dir, exist_ok=True)

    log = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model": config.get("model", {}).get("name", ""),
        "pooling_type": config.get("pooling_type", "none"),
        "loss": config.get("loss", {}),
        "training": {
            "epochs": config.get("training", {}).get("num_train_epochs"),
            "learning_rate": config.get("training", {}).get("learning_rate"),
            "batch_size": config.get("training", {}).get("per_device_train_batch_size"),
            "warmup_ratio": config.get("training", {}).get("warmup_ratio"),
            "weight_decay": config.get("training", {}).get("weight_decay"),
        },
        "results": {
            "macro_f1_default": round(macro_f1_default, 4),
            "macro_f1_optimized": round(macro_f1_optimized, 4),
            "thresholds": dict(zip(label_cols, thresholds)),
        },
        "per_class_report": per_class_report,
        "model_save_dir": save_dir,
    }

    if extra:
        log.update(extra)

    log_path = os.path.join(log_dir, f"{experiment_name}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Experiment log saved: {log_path}")
    return log_path
