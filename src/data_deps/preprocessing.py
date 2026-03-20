import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def prepare_dataset(df: pd.DataFrame, model_name: str, label_cols: list, max_length: int = 128) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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

def round_robin_balance(df: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    """
    Implements a two-level (Intra-class) balancing strategy for multi-label data.
    1. Rarest-Label-First Grouping: Assigns each sample to a disjoint group based on its rarest positive label.
    2. Scale Factors: Upsamples minority groups to match the size of the largest group.
    3. Round-Robin Scheduling: Interleaves samples from all scaled groups to avoid catastrophic forgetting.
    """
    initial_len = len(df)
    
    # 1. Rarest-Label-First Grouping
    # Calculate frequencies of each label
    label_freqs = df[label_cols].sum().sort_values() # ascending
    rarest_order = label_freqs.index.tolist()
    
    # Assign each row to a group
    group_assignments = []
    for _, row in df.iterrows():
        assigned = False
        # Check rarest labels first
        for label in rarest_order:
            if row[label] == 1:
                group_assignments.append(label)
                assigned = True
                break
        if not assigned:
            group_assignments.append("neutral")
            
    df['__group__'] = group_assignments
    
    # 2. Scale Factors
    group_sizes = df['__group__'].value_counts()
    max_size = group_sizes.max()
    
    scaled_groups = []
    for group_name in group_sizes.index:
        group_df = df[df['__group__'] == group_name].copy()
        
        # If group is too small, duplicate it to reach max_size
        scale_factor = max(1, max_size // len(group_df))
        
        # Duplicate dataframe
        scaled_df = pd.concat([group_df] * scale_factor, ignore_index=True)
        # Randomize internally
        scaled_df = scaled_df.sample(frac=1.0).reset_index(drop=True)
        scaled_groups.append(scaled_df)
        
    print(f"  [Balancer] Scale factors applied. Sub-datasets sizes: {[len(g) for g in scaled_groups]}")
        
    # 3. Round-Robin Scheduling
    # Interleave samples
    interleaved_records = []
    
    # Find the max length among all scaled groups to know how many rounds
    max_scaled_len = max(len(g) for g in scaled_groups)
    
    for i in range(max_scaled_len):
        for g_df in scaled_groups:
            if i < len(g_df):
                interleaved_records.append(g_df.iloc[i:i+1])
                
    balanced_df = pd.concat(interleaved_records, ignore_index=True)
    balanced_df = balanced_df.drop(columns=['__group__'])
    
    final_len = len(balanced_df)
    print(f"  [Balancer] Round-Robin balanced data from {initial_len} -> {final_len} rows.")
    return balanced_df

