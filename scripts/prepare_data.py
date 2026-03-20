import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.data_deps.preprocessing import clean_dataframe

def prepare_and_save_data(raw_train_path, raw_dev_path, out_train_path, out_dev_path):
    label_cols = ["anger", "fear", "joy", "sadness", "surprise", "hope"]
    
    print("Loading raw data...")
    train_df = pd.read_csv(raw_train_path)
    dev_df = pd.read_csv(raw_dev_path)
    
    print("\nCleaning Train Data:")
    train_df = clean_dataframe(train_df, label_cols)
    print("Cleaning Dev Data:")
    dev_df = clean_dataframe(dev_df, label_cols)
    
    print("\nAdding 'neutral' label to samples with no emotions...")
    # Add 'neutral' column initialized to 0
    train_df['neutral'] = 0
    dev_df['neutral'] = 0
    
    # If the sum of original labels is 0, set neutral to 1
    train_df.loc[train_df[label_cols].sum(axis=1) == 0, 'neutral'] = 1
    dev_df.loc[dev_df[label_cols].sum(axis=1) == 0, 'neutral'] = 1
    
    print(f"Train Neutral Samples: {train_df['neutral'].sum()} / {len(train_df)}")
    print(f"Dev Neutral Samples:   {dev_df['neutral'].sum()} / {len(dev_df)}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)
    
    # Save processed data
    train_df.to_csv(out_train_path, index=False)
    dev_df.to_csv(out_dev_path, index=False)
    print(f"\nSaved processed data to:\n- {out_train_path}\n- {out_dev_path}")

if __name__ == "__main__":
    raw_train = "data/raw/train.csv"
    raw_dev = "data/raw/dev.csv"
    out_train = "data/processed/train.csv"
    out_dev = "data/processed/dev.csv"
    prepare_and_save_data(raw_train, raw_dev, out_train, out_dev)
