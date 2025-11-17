"""
Factories for SimpleRNN, LSTM and GRU models built with Keras.

The models follow the course restrictions:
- Embeddings are learned from scratch.
- Architectures stay within classic RNN variants (no pretrained components).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    SimpleRNN,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
)
from tensorflow.keras.optimizers import Adam, Optimizer

RNNType = Literal["simple_rnn", "lstm", "gru"]


@dataclass
class RNNModelConfig:
    """High-level configuration to build a classic RNN classifier."""

    vocab_size: int
    max_len: int
    embedding_dim: int = 128
    rnn_units: int = 128
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    post_rnn_dropout: float = 0.0
    rnn_type: RNNType = "lstm"
    output_classes: int = 3
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = "sparse_categorical_crossentropy"
    metrics: Optional[Iterable[str]] = None
    compile_model: bool = True
    bidirectional: bool = False


def _embedding_layer(cfg: RNNModelConfig) -> Embedding:
    return Embedding(
        input_dim=cfg.vocab_size,
        output_dim=cfg.embedding_dim,
        input_length=cfg.max_len,
        name="embedding",
    )


def _rnn_layer(cfg: RNNModelConfig):
    common_kwargs = {
        "units": cfg.rnn_units,
        "dropout": cfg.dropout,
        "recurrent_dropout": cfg.recurrent_dropout,
        "return_sequences": False,
        "name": cfg.rnn_type,
    }
    if cfg.rnn_type == "simple_rnn":
        return SimpleRNN(**common_kwargs)
    if cfg.rnn_type == "lstm":
        return LSTM(**common_kwargs)
    if cfg.rnn_type == "gru":
        return GRU(**common_kwargs)
    raise ValueError(f"Unsupported rnn_type={cfg.rnn_type}")


def _optimizer_from_cfg(cfg: RNNModelConfig) -> Optimizer:
    if cfg.optimizer.lower() == "adam":
        return Adam(learning_rate=cfg.learning_rate)
    raise ValueError(f"Unsupported optimizer {cfg.optimizer}")


def build_rnn_model(cfg: RNNModelConfig) -> Model:
    """
    Build (and optionally compile) a Sequential RNN classifier.
    """

    model = Sequential(name=f"{cfg.rnn_type}_classifier")
    model.add(_embedding_layer(cfg))
    rnn_layer = _rnn_layer(cfg)
    if cfg.bidirectional:
        rnn_layer = Bidirectional(rnn_layer, name=f"bi_{cfg.rnn_type}")
    model.add(rnn_layer)
    if cfg.post_rnn_dropout > 0:
        model.add(Dropout(cfg.post_rnn_dropout, name="post_rnn_dropout"))
    model.add(Dense(cfg.output_classes, activation="softmax", name="classifier"))

    if cfg.compile_model:
        optimizer = _optimizer_from_cfg(cfg)
        metrics = list(cfg.metrics) if cfg.metrics is not None else [
            "accuracy"
        ]
        model.compile(optimizer=optimizer, loss=cfg.loss, metrics=metrics)
    return model


__all__ = ["RNNModelConfig", "build_rnn_model"]

