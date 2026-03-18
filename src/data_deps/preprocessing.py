import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def prepare_dataset(df: pd.DataFrame, model_name: str, max_length: int = 128) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    label_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'hope']
    
    # Create labels column as list of floats (required for BCEWithLogitsLoss)
    df['labels'] = df[label_cols].values.astype(float).tolist()
    
    # Ensure text is string and handle NaNs
    df['text'] = df['text'].fillna("").astype(str)
    
    dataset = Dataset.from_pandas(df[['text', 'labels']])
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Ensure correct columns are kept for PyTorch
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset
