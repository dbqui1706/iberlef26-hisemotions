"""
Sentence-pair preprocessing for NLI-style emotion classification.

Transforms each sample into N binary classification pairs:
  [CLS] text [SEP] emotion_name [SEP] -> 0/1

This allows BERT to leverage its sentence-pair pre-training (NSP).
"""
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Spanish emotion labels for sentence B
EMOTION_NAMES_ES = {
    'anger': 'ira',
    'fear': 'miedo',
    'joy': 'alegria',
    'sadness': 'tristeza',
    'surprise': 'sorpresa',
    'hope': 'esperanza',
    'neutral': 'neutral',
}


def prepare_sentpair_dataset(df, model_name, label_cols, max_length=128):
    """Expand each sample into len(label_cols) sentence pairs.
    
    Input:  N samples with C label columns
    Output: N*C samples, each with binary label (0/1)
    
    Example:
        text="Estoy triste", labels=[0,1,0,1,0,0,0]
        ->  [CLS] Estoy triste [SEP] ira      [SEP] -> 0
            [CLS] Estoy triste [SEP] miedo    [SEP] -> 1
            [CLS] Estoy triste [SEP] alegria  [SEP] -> 0
            ...
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    rows = []
    for _, row in df.iterrows():
        text = str(row.get('text', ''))
        for col in label_cols:
            rows.append({
                'text_a': text,
                'text_b': EMOTION_NAMES_ES.get(col, col),
                'labels': int(row[col]),
            })

    expanded_df = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(expanded_df[['text_a', 'text_b', 'labels']])

    def tokenize_fn(examples):
        return tokenizer(
            examples['text_a'], examples['text_b'],
            padding='max_length', truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    cols = ['input_ids', 'attention_mask', 'labels']
    # Include token_type_ids if tokenizer produces them (BERT does, RoBERTa does not)
    if 'token_type_ids' in dataset.column_names:
        cols.append('token_type_ids')
    dataset.set_format(type='torch', columns=cols)
    return dataset


def collate_sentpair_predictions(logits_or_probs, n_samples, n_labels):
    """Reshape flat sentence-pair predictions back to (n_samples, n_labels).
    
    Input:  (n_samples * n_labels,) probabilities
    Output: (n_samples, n_labels) probability matrix
    """
    import torch
    if isinstance(logits_or_probs, torch.Tensor):
        return logits_or_probs.reshape(n_samples, n_labels)
    import numpy as np
    return np.array(logits_or_probs).reshape(n_samples, n_labels)
