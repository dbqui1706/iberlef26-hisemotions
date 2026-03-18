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
