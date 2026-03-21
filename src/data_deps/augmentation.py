"""
Data augmentation strategies for rare emotion classes in historical Spanish text.
Three techniques:
  1. contextual_augment — MLM-based word substitution (preserves semantics)
  2. random_ops_augment — random swap/delete/insert (adds noise diversity)
  3. back_translate     — ES→EN→ES round-trip translation
"""
import random
from typing import List, Optional

import pandas as pd


def contextual_augment(
    texts: List[str], n_aug: int = 1,
    model_name: str = "bert-base-multilingual-cased",
) -> List[str]:
    """Use masked-language-model to substitute words contextually."""
    import nlpaug.augmenter.word as naw

    aug = naw.ContextualWordEmbsAug(
        model_path=model_name,
        action="substitute",
        aug_min=1,
        aug_max=3,
        device="cuda",
    )
    results = []
    for text in texts:
        augmented = aug.augment(text, n=n_aug)
        if isinstance(augmented, str):
            augmented = [augmented]
        results.extend(augmented)
    return results


def random_ops_augment(texts: List[str], n_aug: int = 1) -> List[str]:
    """Random word-level operations: swap, delete, crop."""
    import nlpaug.augmenter.word as naw

    aug = naw.RandomWordAug(action="swap", aug_min=1, aug_max=2)
    results = []
    for text in texts:
        augmented = aug.augment(text, n=n_aug)
        if isinstance(augmented, str):
            augmented = [augmented]
        results.extend(augmented)
    return results


def back_translate(texts: List[str], n_aug: int = 1) -> List[str]:
    """Back-translate ES → EN → ES using Google Translate."""
    from deep_translator import GoogleTranslator

    es_to_en = GoogleTranslator(source="es", target="en")
    en_to_es = GoogleTranslator(source="en", target="es")
    results = []
    for text in texts:
        for _ in range(n_aug):
            try:
                en = es_to_en.translate(text)
                back = en_to_es.translate(en)
                if back and back != text:
                    results.append(back)
                else:
                    results.append(text)
            except Exception:
                results.append(text)
    return results


def augment_rare_classes(
    df: pd.DataFrame,
    label_cols: list,
    target_min: int = 100,
    methods: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment classes below target_min samples.
    methods: list of ["contextual", "random", "backtranslate"]
    """
    random.seed(seed)
    if methods is None:
        methods = ["random"]

    new_rows = []
    for label in label_cols:
        label_df = df[df[label] == 1]
        count = len(label_df)
        if count >= target_min or count == 0:
            continue

        needed = target_min - count
        texts = label_df["text"].tolist()
        aug_texts: List[str] = []

        for method in methods:
            n_per_method = max(1, needed // len(methods))
            src = [texts[i % len(texts)] for i in range(n_per_method)]
            if method == "contextual":
                aug_texts.extend(contextual_augment(src, n_aug=1))
            elif method == "random":
                aug_texts.extend(random_ops_augment(src, n_aug=1))
            elif method == "backtranslate":
                aug_texts.extend(back_translate(src, n_aug=1))

        for aug_text in aug_texts[:needed]:
            src_row = label_df.sample(1, random_state=random.randint(0, 99999)).iloc[0]
            new_row = src_row.copy()
            new_row["text"] = aug_text
            new_rows.append(new_row)

        print(f"  [Augment] {label}: {count} -> {count + min(len(aug_texts), needed)} samples")

    if new_rows:
        aug_df = pd.DataFrame(new_rows)
        result = pd.concat([df, aug_df], ignore_index=True)
    else:
        result = df.copy()

    return result
