import pandas as pd
from src.data_deps.preprocessing import prepare_dataset

def test_prepare_dataset():
    df = pd.DataFrame({
        'fragment': ['1'],
        'text': ['texto de prueba'],
        'anger': [0], 'fear': [0], 'joy': [1], 'sadness': [0], 'surprise': [0], 'hope': [1]
    })
    dataset = prepare_dataset(df, "PlanTL-GOB-ES/roberta-base-bne", max_length=16)
    assert 'labels' in dataset.features
    assert dataset[0]['labels'].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert 'input_ids' in dataset.features

def test_prepare_dataset_handles_nan():
    # Provide a dataframe with a NaN text value
    import numpy as np
    df = pd.DataFrame({
        'fragment': ['1'],
        'text': [np.nan],
        'anger': [0], 'fear': [0], 'joy': [1], 'sadness': [0], 'surprise': [0], 'hope': [0]
    })
    
    # Should not raise TypeError during map/tokenize
    dataset = prepare_dataset(df, "PlanTL-GOB-ES/roberta-base-bne", max_length=16)
    
    # After preprocessing, the nan text should be converted to an empty string,
    # so input_ids should just be special tokens ([CLS], [SEP], padding)
    assert 'input_ids' in dataset.features
    
    # Verify the label is still processed
    assert dataset[0]['labels'].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

from src.data_deps.preprocessing import clean_dataframe

def test_clean_dataframe_removes_nans_and_short_texts():
    data = {
        'text': ["valid text here", "a", "   ", None, "another valid text"],
        'anger': [0.0, 0.0, 0.0, 0.0, float('nan')],
        'fear': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    label_cols = ['anger', 'fear']
    
    cleaned = clean_dataframe(df, label_cols)
    
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['text'] == "valid text here"

def test_clean_dataframe_merges_duplicates_with_or_logic():
    data = {
        'text': ["duplicate text", "duplicate text", "Duplicate Text", "unique text"],
        'anger': [1.0, 0.0, 0.0, 1.0],
        'fear':  [0.0, 1.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    label_cols = ['anger', 'fear']
    
    cleaned = clean_dataframe(df, label_cols)
    
    assert len(cleaned) == 3
    # Grouped exactly (case matters unless we lower() it, but sticking to simple exact match)
    dup_row = cleaned[cleaned['text'] == "duplicate text"].iloc[0]
    assert dup_row['anger'] == 1.0
    assert dup_row['fear'] == 1.0
    
    dup_row2 = cleaned[cleaned['text'] == "Duplicate Text"].iloc[0]
    assert dup_row2['anger'] == 0.0
    assert dup_row2['fear'] == 0.0

def test_round_robin_balance():
    from src.data_deps.preprocessing import round_robin_balance
    data = {
        'text': ["t1", "t2", "t3", "t4", "t5"],
        'sadness': [1.0, 1.0, 1.0, 0.0, 0.0],
        'fear':    [0.0, 0.0, 0.0, 1.0, 0.0],
        'anger':   [0.0, 0.0, 0.0, 0.0, 1.0]
    }
    df = pd.DataFrame(data)
    label_cols = ['sadness', 'fear', 'anger']
    # sadness: 3, fear: 1, anger: 1
    # Group rarity: anger(1) -> fear(1) -> sadness(3)
    # M = 3. scale_anger = 3/1=3. scale_fear = 3/1=3. scale_sadness = 3/3=1.
    # Total expected rows: 3 (anger) + 3 (fear) + 3 (sadness) = 9
    
    balanced = round_robin_balance(df, label_cols)
    assert len(balanced) == 9
    
    # Check if interleaving works (round robin)
    # The first 3 rows should be 3 different classes
    counts_first_3 = balanced.iloc[:3][label_cols].sum()
    assert counts_first_3['sadness'] >= 1
    assert counts_first_3['fear'] >= 1
    assert counts_first_3['anger'] >= 1

