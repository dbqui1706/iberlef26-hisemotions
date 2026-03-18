import argparse
import yaml
import os
import sys

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    AutoTokenizer
)
from src.data_deps.preprocessing import prepare_dataset
from src.evaluate.metrics import compute_multilabel_metrics

class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Move weights to correct device
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

def calculate_pos_weights(df, labels):
    """Calculate weights for imbalanced classes: pos_weight = (neg_count / pos_count)"""
    weights = []
    for label in labels:
        pos_count = df[label].sum()
        neg_count = len(df) - pos_count
        weight = neg_count / (pos_count + 1e-6) # prevent div by zero
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float)

def main():
    parser = argparse.ArgumentParser(description="Train SLM for HISEMOTIONS")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Starting experiment: {config['experiment_name']}")
    
    # 1. Load raw data
    train_df = pd.read_csv(config['data']['train_path'])
    dev_df = pd.read_csv(config['data']['dev_path'])
    
    label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'hope']
    pos_weights = calculate_pos_weights(train_df, label_cols)
    print(f"Calculated class weights: {pos_weights}")
    
    # 2. Prepare datasets
    train_dataset = prepare_dataset(
        train_df, 
        config['model']['name'], 
        max_length=config['model']['max_length']
    )
    dev_dataset = prepare_dataset(
        dev_df, 
        config['model']['name'], 
        max_length=config['model']['max_length']
    )
    
    # 3. Load Model and send to Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # Send class weights to device so the loss function can use them
    pos_weights = pos_weights.to(device)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        num_train_epochs=config['training']['num_train_epochs'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        report_to=config['training']['report_to'],
        run_name=config['experiment_name'],
        fp16=torch.cuda.is_available(),
        logging_steps=10,
    )
    
    # 5. Initialize Trainer
    trainer = WeightedMultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_multilabel_metrics,
        class_weights=pos_weights
    )
    
    # 6. Train & Evaluate
    print("Starting training...")
    trainer.train()
    
    print("Evaluating on dev set...")
    eval_results = trainer.evaluate()
    print(f"Final Evaluation results: {eval_results}")
    
    # 7. Save final model
    trainer.save_model(os.path.join(config['training']['output_dir'], "final_model"))
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
