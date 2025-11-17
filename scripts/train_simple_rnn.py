#!/usr/bin/env python
"""Train SimpleRNN using cached folds."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.dataset_loader import compute_class_weights
from src.eval.metrics import evaluate_predictions
from src.models.rnn_keras import RNNModelConfig, build_rnn_model
from src.train.train_rnn import _build_data_config, _build_model_config, _load_yaml
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

REGISTRY_PATH = Path("artifacts/experiments.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena SimpleRNN usando folds cacheados.")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--folds", type=str, default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dropout", type=float, default=None, help="Sobrescribe el dropout del modelo RNN.")
    parser.add_argument("--show-summary", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--bidirectional", action="store_true")
    return parser.parse_args()


def load_metadata(cache_dir: Path) -> Dict[str, object]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json en {cache_dir}")
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_fold(cache_dir: Path, fold_idx: int) -> Dict[str, np.ndarray]:
    fold_dir = cache_dir / f"fold_{fold_idx}"
    data_path = fold_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró {data_path}")
    with np.load(data_path) as data:
        return {"x_train": data["x_train"], "y_train": data["y_train"], "x_val": data["x_val"], "y_val": data["y_val"]}


def select_folds(request: str, available: List[int]) -> List[int]:
    if request.lower() == "all":
        return available
    folds = []
    for token in request.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx not in available:
            raise ValueError(f"Fold {idx} no existe; disponibles {available}")
        folds.append(idx)
    return folds


def append_registry(row: Dict[str, object]) -> None:
    write_header = not REGISTRY_PATH.exists()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_callbacks(training_cfg: Dict[str, object]) -> List[object]:
    callbacks_cfg = training_cfg.get("callbacks", {})
    callbacks: List[object] = []

    es_cfg = callbacks_cfg.get("early_stopping")
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 3),
                mode=es_cfg.get("mode", "min"),
                restore_best_weights=es_cfg.get("restore_best_weights", True),
            )
        )

    rl_cfg = callbacks_cfg.get("reduce_lr_on_plateau")
    if rl_cfg:
        callbacks.append(
            ReduceLROnPlateau(
                monitor=rl_cfg.get("monitor", "val_loss"),
                factor=rl_cfg.get("factor", 0.5),
                patience=rl_cfg.get("patience", 2),
                min_lr=rl_cfg.get("min_lr", 1e-6),
                mode=rl_cfg.get("mode", "min"),
            )
        )
    return callbacks


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    metadata = load_metadata(args.data_cache)

    folds = select_folds(args.folds, metadata["folds"])
    training_cfg = cfg["training"]
    data_cfg = _build_data_config(cfg["data"])
    model_cfg = _build_model_config(cfg["models"], "simple_rnn", data_cfg, training_cfg)
    if args.dropout is not None:
        print(f"[train_simple_rnn] Override dropout -> {args.dropout}")
        model_cfg.dropout = args.dropout
    if args.bidirectional:
        print("[train_simple_rnn] Activando modo bidireccional")
        model_cfg.bidirectional = True

    epochs = args.epochs or training_cfg.get("epochs", 5)
    batch_size = args.batch_size or training_cfg.get("batch_size", 128)

    args.output.mkdir(parents=True, exist_ok=True)
    summary_printed = False
    fold_reports = []

    for fold_idx in folds:
        data = load_fold(args.data_cache, fold_idx)
        x_train, y_train = data["x_train"], data["y_train"]
        x_val, y_val = data["x_val"], data["y_val"]

        print(f"[train_simple_rnn] Fold {fold_idx} | train={x_train.shape} val={x_val.shape} | epochs={epochs}")
        model = build_rnn_model(model_cfg)
        if args.show_summary and not summary_printed:
            model.summary()
            summary_printed = True

        callbacks = _build_callbacks(training_cfg)
        class_weight = compute_class_weights(y_train) if training_cfg.get("class_weight") else None
        multipliers = training_cfg.get("class_weight_multipliers")
        if class_weight and multipliers:
            for cls, mult in multipliers.items():
                cls_idx = int(cls)
                class_weight[cls_idx] = class_weight.get(cls_idx, 1.0) * float(mult)

        start = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            class_weight=class_weight,
            callbacks=callbacks if callbacks else None,
        )
        duration = time.perf_counter() - start
        epochs_ran = len(history.history.get("loss", []))

        y_proba = model.predict(x_val, batch_size=batch_size, verbose=0)
        metrics = evaluate_predictions(y_true=y_val, y_proba=y_proba, metric_names=cfg.get("metrics", []))
        metrics["fold"] = fold_idx
        metrics["train_time_sec"] = duration
        metrics["epochs_ran"] = epochs_ran
        fold_reports.append(metrics)

        with (args.output / f"fold_{fold_idx}_history.json").open("w", encoding="utf-8") as fp:
            json.dump(history.history, fp, ensure_ascii=False, indent=2)
        with (args.output / f"fold_{fold_idx}_metrics.json").open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, ensure_ascii=False, indent=2)

        append_registry(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": metadata["experiment_id"],
                "tag": args.tag,
                "model": "simple_rnn",
                "fold": fold_idx,
                "cleaning": metadata["cleaning"],
                "nlp_method": metadata["nlp_method"],
                "embedding": metadata["embedding"],
                "epochs": epochs,
                "batch_size": batch_size,
                "train_time_sec": duration,
                "f1_macro": metrics.get("f1_macro"),
                "recall_neg": metrics.get("recall_neg"),
                "precision_pos": metrics.get("precision_pos"),
                "notes": args.notes,
            }
        )

    summary = {
        "folds": fold_reports,
        "mean_f1_macro": float(np.mean([m.get("f1_macro", 0.0) for m in fold_reports])) if fold_reports else None,
        "mean_recall_neg": float(np.mean([m.get("recall_neg", 0.0) for m in fold_reports])) if fold_reports else None,
        "mean_precision_pos": float(np.mean([m.get("precision_pos", 0.0) for m in fold_reports])) if fold_reports else None,
        "mean_epochs_ran": float(np.mean([m.get("epochs_ran", epochs) for m in fold_reports])) if fold_reports else None,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print("[train_simple_rnn] Resumen:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
