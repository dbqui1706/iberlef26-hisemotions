import pandas as pd
from src.data_deps.augmentation import contextual_augment, random_ops_augment, augment_rare_classes

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "hope", "neutral"]


def test_random_ops_augment_produces_new_texts():
    texts = ["Estoy muy feliz hoy porque llegó mi carta"]
    results = random_ops_augment(texts, n_aug=2)
    assert len(results) == 2


def test_augment_rare_classes_preserves_schema():
    df = pd.DataFrame({
        "text": ["texto triste", "texto neutral", "texto sorpresa"],
        "anger": [0, 0, 0],
        "fear": [0, 0, 0],
        "joy": [0, 0, 0],
        "sadness": [1, 0, 0],
        "surprise": [0, 0, 1],
        "hope": [0, 0, 0],
        "neutral": [0, 1, 0],
    })
    result = augment_rare_classes(df, LABEL_COLS, target_min=3, methods=["random"])
    # Schema must match
    assert list(result.columns) == list(df.columns)
    # Must have more rows (surprise was augmented)
    assert len(result) >= len(df)
    # Labels must be preserved for augmented rows
    surprise_rows = result[result["surprise"] == 1]
    assert len(surprise_rows) >= 3
