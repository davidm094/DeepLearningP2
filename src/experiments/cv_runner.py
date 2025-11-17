"""
Command-line utility to run k-fold experiments for the supported RNN models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.train.train_rnn import train_cross_validation


def aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std for each metric across folds."""

    summary: Dict[str, Dict[str, float]] = {}
    metric_keys = [key for key in fold_metrics[0].keys() if key != "fold"]
    for metric in metric_keys:
        values = np.array([entry[metric] for entry in fold_metrics], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def run_all_models(config: Path, output_root: Path, models: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    output_root.mkdir(parents=True, exist_ok=True)
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_type in models:
        model_dir = output_root / model_type
        fold_metrics = train_cross_validation(
            config_path=config,
            model_type=model_type,
            output_dir=model_dir,
        )
        aggregated[model_type] = aggregate_metrics(fold_metrics)

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(aggregated, fp, ensure_ascii=False, indent=2)
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-fold CV for SimpleRNN/LSTM/GRU baselines.")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/phase2"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["simple_rnn", "lstm", "gru"],
        choices=["simple_rnn", "lstm", "gru"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all_models(
        config=args.config,
        output_root=args.output_root,
        models=args.models,
    )
    print("Aggregated metrics:")
    for model, metrics in summary.items():
        print(model)
        for metric_name, stats in metrics.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()

