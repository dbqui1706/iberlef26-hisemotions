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
    assert dataset[0]['labels'] == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert 'input_ids' in dataset.features
