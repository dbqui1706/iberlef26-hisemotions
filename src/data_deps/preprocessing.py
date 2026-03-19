import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def prepare_dataset(df: pd.DataFrame, model_name: str, max_length: int = 128) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

def clean_dataframe(df: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    """
    Cleans raw DataFrame before dataset preparation.
    1. Removes rows with NaN in label columns or text.
    2. Drops texts shorter than 3 characters after stripping.
    3. Merges duplicate texts by performing logical OR (max) on labels.
    """
    initial_len = len(df)
    
    # 1. Drop NaN
    df = df.dropna(subset=label_cols + ["text"]).copy()
    
    # Ensure text is string (avoid errors with len)
    df['text'] = df['text'].astype(str)
    
    # 2. Drop short text
    mask_short = df['text'].str.strip().str.len() >= 3
    df = df[mask_short]
    
    # 3. Merge duplicates (OR logic via max())
    df = df.groupby('text', as_index=False)[label_cols].max()
    
    final_len = len(df)
    print(f"  [Data Pipeline] Cleaned data: {initial_len} -> {final_len} rows. "
          f"Removed/merged {initial_len - final_len} rows.")
          
    return df
