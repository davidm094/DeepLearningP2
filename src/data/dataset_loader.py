"""
Utilities to load the Andalusian Hotels dataset and prepare stratified folds.

The goal is to keep preprocessing minimal, reproducible, and aligned with the
course restrictions (no pretrained embeddings, only classic RNN pipelines).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Raw labels in the dataset: 0 (negativo), 3 (neutral/mixto), 1 (positivo).
# For training we remap them to contiguous indices {0,1,2}.
ORIGINAL_LABELS = [0, 3, 1]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(ORIGINAL_LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_path: str
    text_column: str = "review_text"
    label_column: str = "label"
    delimiter: str = ","
    encoding: str = "utf-8"
    lowercase: bool = True
    strip_text: bool = True
    max_len: int = 256
    vocab_size: int = 30000
    oov_token: str = "<OOV>"
    num_folds: int = 5
    seed: int = 42
    dropna: bool = True


@dataclass
class FoldDataset:
    """Container for a single fold (train/validation) split."""

    fold_index: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    tokenizer: Tokenizer


def load_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """
    Read the raw CSV and return a clean DataFrame with text and label columns.
    """

    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path, delimiter=cfg.delimiter, encoding=cfg.encoding)

    missing_cols = {cfg.text_column, cfg.label_column} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in dataset: {missing_cols}. "
            "Check the config text/label column names."
        )

    if cfg.dropna:
        df = df.dropna(subset=[cfg.text_column, cfg.label_column])

    df = df.copy()
    df[cfg.text_column] = df[cfg.text_column].astype(str)
    if cfg.lowercase:
        df[cfg.text_column] = df[cfg.text_column].str.lower()
    if cfg.strip_text:
        df[cfg.text_column] = df[cfg.text_column].str.strip()

    df[cfg.label_column] = (
        df[cfg.label_column]
        .astype(int)
        .map(LABEL_TO_INDEX)
    )
    df.reset_index(drop=True, inplace=True)
    return df


def _fit_tokenizer(texts: Iterable[str], cfg: DataConfig) -> Tokenizer:
    tokenizer = Tokenizer(
        num_words=cfg.vocab_size,
        oov_token=cfg.oov_token,
        lower=False,  # already handled above if requested
        filters=""  # minimal filtering; handled manually if needed
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def _vectorize(
    tokenizer: Tokenizer, texts: Iterable[str], cfg: DataConfig
) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=cfg.max_len, padding="post", truncating="post")


def generate_stratified_folds(
    cfg: DataConfig,
) -> Generator[FoldDataset, None, None]:
    """
    Yield StratifiedKFold splits with tokenization fitted on each training fold.

    Each fold returns padded sequences for train/validation sets plus the
    tokenizer (needed downstream for inference or test preprocessing).
    """

    df = load_dataframe(cfg)
    texts: List[str] = df[cfg.text_column].tolist()
    labels: np.ndarray = df[cfg.label_column].to_numpy()

    skf = StratifiedKFold(
        n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed
    )

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        tokenizer = _fit_tokenizer(train_texts, cfg)
        x_train = _vectorize(tokenizer, train_texts, cfg)
        x_val = _vectorize(tokenizer, val_texts, cfg)

        yield FoldDataset(
            fold_index=fold_index,
            x_train=x_train,
            y_train=labels[train_idx],
            x_val=x_val,
            y_val=labels[val_idx],
            tokenizer=tokenizer,
        )


def compute_class_weights(labels: Iterable[int]) -> dict[int, float]:
    """
    Compute balanced class weights compatible with Keras.
    """

    labels_array = np.array(labels)
    classes, counts = np.unique(labels_array, return_counts=True)
    total = labels_array.shape[0]
    weights = {int(cls): total / (len(classes) * count) for cls, count in zip(classes, counts)}
    return weights


def decode_labels(indices: Iterable[int]) -> np.ndarray:
    """Restore original label values {0,3,1} from contiguous indices."""

    indices_array = np.array(indices)
    vectorized = np.vectorize(INDEX_TO_LABEL.get)
    return vectorized(indices_array)


__all__ = [
    "DataConfig",
    "FoldDataset",
    "load_dataframe",
    "generate_stratified_folds",
    "compute_class_weights",
    "decode_labels",
    "ORIGINAL_LABELS",
    "LABEL_TO_INDEX",
]

