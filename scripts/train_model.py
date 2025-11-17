#!/usr/bin/env python
"""
Train a specific RNN model on cached folds, with explicit progress per fold.

Example:
  PYTHONUNBUFFERED=1 python scripts/train_model.py \
      --config config/phase2.yaml \
      --data-cache artifacts/data \
      --model lstm \
      --folds 1 \
      --output artifacts/phase2/lstm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from src.data.dataset_loader import DataConfig, compute_class_weights
from src.eval.metrics import evaluate_predictions
from src.models.rnn_keras import build_rnn_model
from src.train.train_rnn import (
    _build_callbacks,
    _build_data_config,
    _build_model_config,
    _load_yaml,
)


def parse_folds(value: str, available: List[int]) -> List[int]:
    if not value or value.lower() == "all":
        return available
    requested: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx not in available:
            raise ValueError(f"Fold {idx} not found in prepared cache (available={available}).")
        requested.append(idx)
    return requested


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not results:
        return summary
    metric_keys = [k for k in results[0].keys() if k != "fold"]
    for key in metric_keys:
        values = np.array([entry[key] for entry in results], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cached folds for a chosen RNN model.")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument("--data-cache", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--model", type=str, choices=["simple_rnn", "lstm", "gru"], default="lstm")
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Comma-separated fold indices (e.g. 1,3,5). Defaults to all cached folds.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2/lstm"))
    parser.add_argument("--show-summary", action="store_true", help="Print model.summary() once.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    return parser.parse_args()


def load_metadata(cache_dir: Path) -> Dict[str, object]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata.json found in {cache_dir}. Run scripts/prepare_folds.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_fold_arrays(cache_dir: Path, fold_index: int) -> Dict[str, np.ndarray]:
    fold_dir = cache_dir / f"fold_{fold_index}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory {fold_dir} not found. Re-run prepare_folds.")
    data_path = fold_dir / "data.npz"
    with np.load(data_path) as data:
        return {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "x_val": data["x_val"],
            "y_val": data["y_val"],
        }


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    metadata = load_metadata(args.data_cache)

    data_cfg = _build_data_config(cfg["data"])
    training_cfg = cfg["training"]
    metrics_cfg = cfg.get("metrics", [])
    model_cfg = _build_model_config(cfg["models"], args.model, data_cfg, training_cfg)

    available_folds = list(metadata.get("folds", []))
    requested_folds = parse_folds(args.folds, available_folds)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    printed_summary = False
    all_metrics: List[Dict[str, float]] = []

    epochs = args.epochs or training_cfg.get("epochs", 20)
    batch_size = args.batch_size or training_cfg.get("batch_size", 128)

    for fold_idx in requested_folds:
        arrays = load_fold_arrays(args.data_cache, fold_idx)
        x_train = arrays["x_train"]
        y_train = arrays["y_train"]
        x_val = arrays["x_val"]
        y_val = arrays["y_val"]

        print(
            f"[train_model] Fold {fold_idx} | "
            f"train={x_train.shape} val={x_val.shape} | model={args.model.upper()}"
        )

        model = build_rnn_model(model_cfg)
        if args.show_summary and not printed_summary:
            model.summary()
            printed_summary = True

        callbacks = _build_callbacks(
            output_dir=output_dir,
            fold_idx=fold_idx,
            monitor=training_cfg["callbacks"]["early_stopping"]["monitor"],
            patience=training_cfg["callbacks"]["early_stopping"].get("patience", 3),
        )

        class_weight = None
        if training_cfg.get("class_weight", False):
            class_weight = compute_class_weights(y_train)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight,
        )

        history_path = output_dir / f"fold_{fold_idx}_history.json"
        with history_path.open("w", encoding="utf-8") as fp:
            json.dump(history.history, fp, ensure_ascii=False, indent=2)

        print(f"[train_model] Fold {fold_idx} training complete. Evaluating metrics...")
        y_proba = model.predict(x_val, batch_size=batch_size, verbose=0)
        fold_metrics = evaluate_predictions(y_true=y_val, y_proba=y_proba, metric_names=metrics_cfg)
        fold_metrics["fold"] = fold_idx
        all_metrics.append(fold_metrics)

        metrics_path = output_dir / f"fold_{fold_idx}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(fold_metrics, fp, ensure_ascii=False, indent=2)
        print(f"[train_model] Fold {fold_idx} metrics: {json.dumps(fold_metrics, indent=2)}")

    summary = aggregate_metrics(all_metrics)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print("[train_model] Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

