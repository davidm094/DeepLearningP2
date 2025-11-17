"""
Training utilities for classic Keras RNNs using stratified k-fold CV.

Usage (from repo root):
    python -m src.train.train_rnn --config config/phase2.yaml --model-type lstm
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from src.data.dataset_loader import (
    DataConfig,
    compute_class_weights,
    generate_stratified_folds,
)
from src.eval.metrics import evaluate_predictions
from src.models.rnn_keras import RNNModelConfig, build_rnn_model


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _build_data_config(raw_cfg: Dict[str, Any]) -> DataConfig:
    return DataConfig(
        data_path=raw_cfg["path"],
        text_column=raw_cfg.get("text_column", "review_text"),
        label_column=raw_cfg.get("label_column", "label"),
        delimiter=raw_cfg.get("delimiter", ","),
        encoding=raw_cfg.get("encoding", "utf-8"),
        lowercase=raw_cfg.get("lowercase", True),
        strip_text=raw_cfg.get("strip_text", True),
        max_len=raw_cfg.get("max_len", 256),
        vocab_size=raw_cfg.get("vocab_size", 30000),
        oov_token=raw_cfg.get("oov_token", "<OOV>"),
        num_folds=raw_cfg.get("num_folds", 5),
        seed=raw_cfg.get("seed", 42),
    )


def _build_model_config(
    raw_cfg: Dict[str, Any], model_type: str, data_cfg: DataConfig, training_cfg: Dict[str, Any]
) -> RNNModelConfig:
    model_section = raw_cfg.get(model_type, {})
    embedding_dim = raw_cfg.get("embedding", {}).get("output_dim", 128)
    return RNNModelConfig(
        vocab_size=data_cfg.vocab_size,
        max_len=data_cfg.max_len,
        embedding_dim=embedding_dim,
        rnn_units=model_section.get("units", 128),
        dropout=model_section.get("dropout", 0.2),
        recurrent_dropout=model_section.get("recurrent_dropout", 0.0),
        rnn_type=model_type,
        output_classes=training_cfg.get("output_classes", 3),
        optimizer=training_cfg.get("optimizer", "adam"),
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        loss=training_cfg.get("loss", "sparse_categorical_crossentropy"),
        metrics=training_cfg.get("keras_metrics", ["accuracy"]),
        compile_model=True,
    )


def _build_callbacks(
    output_dir: Path,
    fold_idx: int,
    monitor: str,
    patience: int,
) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = []
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode="max" if "acc" in monitor or "f1" in monitor else "min",
            restore_best_weights=True,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / f"fold_{fold_idx}.weights.h5",
            monitor=monitor,
            mode="max" if "acc" in monitor or "f1" in monitor else "min",
            save_best_only=True,
            save_weights_only=True,
        )
    )
    return callbacks


def train_cross_validation(
    config_path: Path,
    model_type: str,
    output_dir: Path,
    selected_folds: Optional[Iterable[int]] = None,
    show_summary: bool = False,
) -> List[Dict[str, Any]]:
    cfg = _load_yaml(config_path)
    data_cfg = _build_data_config(cfg["data"])
    training_cfg = cfg["training"]
    metrics_cfg = cfg.get("metrics", [])
    model_cfg = _build_model_config(cfg["models"], model_type, data_cfg, training_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: List[Dict[str, Any]] = []
    selected_set = set(selected_folds) if selected_folds else None
    summary_printed = False

    for fold in generate_stratified_folds(data_cfg):
        if selected_set and fold.fold_index not in selected_set:
            continue
        print(f"[Fold {fold.fold_index}] Training {model_type.upper()} with {len(fold.y_train)} samples")
        model = build_rnn_model(model_cfg)
        if show_summary and not summary_printed:
            model.summary()
            summary_printed = True

        callbacks = _build_callbacks(
            output_dir=output_dir,
            fold_idx=fold.fold_index,
            monitor=training_cfg["callbacks"]["early_stopping"]["monitor"],
            patience=training_cfg["callbacks"]["early_stopping"].get("patience", 3),
        )

        class_weight = None
        if training_cfg.get("class_weight", False):
            class_weight = compute_class_weights(fold.y_train)

        history = model.fit(
            fold.x_train,
            fold.y_train,
            validation_data=(fold.x_val, fold.y_val),
            epochs=training_cfg.get("epochs", 20),
            batch_size=training_cfg.get("batch_size", 128),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        history_path = output_dir / f"fold_{fold.fold_index}_history.json"
        with history_path.open("w", encoding="utf-8") as fp:
            json.dump(history.history, fp, ensure_ascii=False, indent=2)

        tokenizer_path = output_dir / f"fold_{fold.fold_index}_tokenizer.json"
        with tokenizer_path.open("w", encoding="utf-8") as fp:
            fp.write(fold.tokenizer.to_json())

        y_proba = model.predict(fold.x_val, batch_size=training_cfg.get("batch_size", 128))
        fold_metric_values = evaluate_predictions(
            y_true=fold.y_val,
            y_proba=y_proba,
            metric_names=metrics_cfg,
        )
        fold_metrics.append({"fold": fold.fold_index, **fold_metric_values})

    metrics_path = output_dir / f"{model_type}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(fold_metrics, fp, ensure_ascii=False, indent=2)
    return fold_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classic RNNs with cross-validation.")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument("--model-type", type=str, choices=["simple_rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2"))
    parser.add_argument(
        "--folds",
        type=str,
        default="",
        help="Comma-separated list of fold indices to run (e.g., 1,3,5). Empty means all.",
    )
    parser.add_argument("--show-summary", action="store_true", help="Print Keras model.summary() once.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_filter = None
    if args.folds:
        fold_filter = [int(tok.strip()) for tok in args.folds.split(",") if tok.strip()]
    fold_scores = train_cross_validation(
        config_path=args.config,
        model_type=args.model_type,
        output_dir=args.output_dir,
        selected_folds=fold_filter,
        show_summary=args.show_summary,
    )
    print("Fold metrics:")
    for entry in fold_scores:
        print(entry)


if __name__ == "__main__":
    main()

