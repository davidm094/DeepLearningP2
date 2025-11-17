#!/usr/bin/env python
"""
Precompute tokenized folds once so that training scripts can reuse them quickly.

Usage:
  python scripts/prepare_folds.py --config config/phase2.yaml --output artifacts/data
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.dataset_loader import (
    DataConfig,
    LABEL_TO_INDEX,
    generate_stratified_folds,
)
from src.train.train_rnn import _build_data_config, _load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached folds (tokenized arrays + tokenizer).")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data"),
        help="Directory where fold_* folders and metadata.json will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)
    data_cfg = _build_data_config(cfg["data"])

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_ids: List[int] = []
    for fold in generate_stratified_folds(data_cfg):
        fold_ids.append(fold.fold_index)
        fold_dir = output_dir / f"fold_{fold.fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            fold_dir / "data.npz",
            x_train=fold.x_train,
            y_train=fold.y_train,
            x_val=fold.x_val,
            y_val=fold.y_val,
        )
        with (fold_dir / "tokenizer.json").open("w", encoding="utf-8") as fp:
            fp.write(fold.tokenizer.to_json())

        print(f"[prepare_folds] Saved fold {fold.fold_index} to {fold_dir}")

    metadata: Dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "data_path": data_cfg.data_path,
        "num_folds": len(fold_ids),
        "folds": fold_ids,
        "max_len": data_cfg.max_len,
        "vocab_size": data_cfg.vocab_size,
        "label_map": LABEL_TO_INDEX,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    print(f"[prepare_folds] Metadata stored at {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

