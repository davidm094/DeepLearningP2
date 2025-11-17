#!/usr/bin/env python
"""
Dataset preparation pipeline:
  1. Cleans text (baseline / lemmatize / stem).
  2. Applies NLP feature extraction (keras_tokenizer or tfidf).
  3. (Optional) trains embeddings (learned vs word2vec).
  4. Generates stratified folds and persists them to disk.

Usage example:
PYTHONPATH=. python scripts/prepare_dataset.py \
    --config config/phase2.yaml \
    --output artifacts/data/EXP_BASE_LEM \
    --experiment-id EXP_BASE_LEM \
    --cleaning lemmatize \
    --nlp keras_tokenizer \
    --embedding word2vec
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.data.dataset_loader import LABEL_TO_INDEX

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover
    Word2Vec = None

import nltk
from nltk.stem import SnowballStemmer
import joblib
import shutil


CLEANING_CHOICES = ["baseline", "lemmatize", "stem"]
NLP_CHOICES = ["keras_tokenizer", "tfidf"]
EMBEDDING_CHOICES = ["learned", "word2vec"]


def ensure_output_dir(path: Path, force: bool) -> bool:
    if path.exists():
        if not force:
            print(
                f"[prepare_dataset] Directorio {path} ya existe. Usa --force para regenerar o reutiliza este cache.",
                flush=True,
            )
            return False
        print(f"[prepare_dataset] --force habilitado. Eliminando {path}...", flush=True)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return True


def get_clean_cache_path(cache_dir: Path, cleaning: str) -> Path:
    return cache_dir / f"clean_{cleaning}.joblib"


def load_or_build_clean_texts(
    cleaning: str, texts: Iterable[str], cache_path: Path
) -> Tuple[List[str], List[List[str]]]:
    if cache_path.exists():
        data = joblib.load(cache_path)
        print(f"[prepare_dataset] Cache de limpieza reutilizado desde {cache_path}", flush=True)
        return data["cleaned_texts"], data["token_lists"]

    print(f"[prepare_dataset] Generando limpieza '{cleaning}' (sin cache)...", flush=True)
    if cleaning == "baseline":
        cleaned_texts, token_lists = baseline_clean(texts)
    elif cleaning == "lemmatize":
        cleaned_texts, token_lists = lemmatize_clean(texts)
    else:
        cleaned_texts, token_lists = stem_clean(texts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cleaned_texts": cleaned_texts, "token_lists": token_lists}, cache_path)
    print(f"[prepare_dataset] Cache de limpieza guardado en {cache_path}", flush=True)
    return cleaned_texts, token_lists


def get_fold_splits_path(cache_dir: Path, path_arg: Path | None, folds: int, seed: int) -> Path:
    if path_arg is not None:
        return path_arg
    return cache_dir / f"folds_seed{seed}_k{folds}.json"


def load_or_build_fold_indices(
    path: Path, labels: np.ndarray, folds: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        print(f"[prepare_dataset] Reutilizando índices de folds desde {path}", flush=True)
        return [
            (np.array(entry["train"], dtype=np.int64), np.array(entry["val"], dtype=np.int64))
            for entry in data["folds"]
        ]

    print(f"[prepare_dataset] Generando índices de folds (k={folds}, seed={seed})...", flush=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_entries = []
    dummy_features = np.zeros_like(labels)
    for train_idx, val_idx in skf.split(dummy_features, labels):
        fold_entries.append({"train": train_idx.tolist(), "val": val_idx.tolist()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump({"folds": fold_entries, "seed": seed, "k": folds}, fp, ensure_ascii=False, indent=2)
    print(f"[prepare_dataset] Índices guardados en {path}", flush=True)
    return [
        (np.array(entry["train"], dtype=np.int64), np.array(entry["val"], dtype=np.int64))
        for entry in fold_entries
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset folds with configurable preprocessing.")
    parser.add_argument("--config", type=Path, default=Path("config/phase2.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for folds/metadata.")
    parser.add_argument("--experiment-id", type=str, required=True, help="Unique identifier for this experiment.")
    parser.add_argument("--cleaning", choices=CLEANING_CHOICES, default="baseline")
    parser.add_argument("--nlp", choices=NLP_CHOICES, default="keras_tokenizer")
    parser.add_argument("--embedding", choices=EMBEDDING_CHOICES, default="learned")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Dimension for embeddings/Word2Vec.")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds (<=5 recomendado).")
    parser.add_argument("--max-len", type=int, default=None, help="Sobrescribe max_len del config si se indica.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Sobrescribe vocab_size del config si se indica.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"), help="Ruta para reutilizar textos limpios e índices.")
    parser.add_argument(
        "--fold-splits-path",
        type=Path,
        help="Ruta para guardar/reusar los índices de folds. Por defecto usa <cache-dir>/folds_seedK.json",
    )
    parser.add_argument("--force", action="store_true", help="Regenera la salida aunque ya exista.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def baseline_clean(texts: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    cleaned = []
    tokens = []
    for txt in texts:
        txt = str(txt).lower().strip()
        toks = re.findall(r"\b\w+\b", txt, flags=re.UNICODE)
        cleaned.append(" ".join(toks))
        tokens.append(toks)
    return cleaned, tokens


def load_spacy_model() -> "spacy.language.Language":
    if spacy is None:
        raise ImportError("spaCy no está instalado. Ejecuta `pip install spacy`.")
    try:
        return spacy.load("es_core_news_sm")
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            "No se encontró el modelo spaCy 'es_core_news_sm'. Instálalo con "
            "`python -m spacy download es_core_news_sm`."
        ) from exc


def lemmatize_clean(texts: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    nlp = load_spacy_model()
    cleaned = []
    tokens = []
    for txt in texts:
        doc = nlp(str(txt))
        toks = [token.lemma_.lower() for token in doc if token.is_alpha]
        cleaned.append(" ".join(toks))
        tokens.append(toks)
    return cleaned, tokens


def stem_clean(texts: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    stemmer = SnowballStemmer("spanish")
    cleaned = []
    tokens = []
    for txt in texts:
        base_tokens = re.findall(r"\b\w+\b", str(txt).lower(), flags=re.UNICODE)
        stems = [stemmer.stem(tok) for tok in base_tokens]
        cleaned.append(" ".join(stems))
        tokens.append(stems)
    return cleaned, tokens


def build_tokenizer(texts: List[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", filters="")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def apply_nlp(
    texts: List[str],
    tokens: List[List[str]],
    method: str,
    max_len: int,
    vocab_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    metadata: Dict[str, object] = {}
    if method == "keras_tokenizer":
        tokenizer = build_tokenizer(texts, vocab_size)
        sequences = tokenizer.texts_to_sequences(texts)
        array = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        metadata["tokenizer"] = tokenizer
        metadata["vocab_size"] = min(vocab_size, len(tokenizer.word_index) + 1)
        return array, metadata
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=vocab_size)
        matrix = vectorizer.fit_transform(texts)
        return matrix.astype(np.float32).toarray(), {"vectorizer": vectorizer, "vocab_size": matrix.shape[1]}
    raise ValueError(f"NLP method {method} no soportado.")


def train_word2vec(tokens: List[List[str]], embedding_dim: int) -> "Word2Vec":
    if Word2Vec is None:
        raise ImportError("gensim no está instalado. Ejecuta `pip install gensim` para usar embedding word2vec.")
    model = Word2Vec(
        sentences=tokens,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=4,
        epochs=10,
    )
    return model


def build_embedding_matrix(
    tokenizer: Tokenizer,
    w2v: "Word2Vec",
    embedding_dim: int,
    vocab_size: int,
) -> np.ndarray:
    matrix = np.random.normal(size=(vocab_size, embedding_dim)).astype(np.float32)
    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        if word in w2v.wv:
            matrix[idx] = w2v.wv[word]
    return matrix


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    text_col = data_cfg.get("text_column", "review_text")
    label_col = data_cfg.get("label_column", "label")
    max_len = args.max_len or data_cfg.get("max_len", 256)
    base_vocab_size = args.vocab_size or data_cfg.get("vocab_size", 30000)
    vocab_size = base_vocab_size + 1  # +1 for OOV/padding

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    clean_cache_path = get_clean_cache_path(args.cache_dir, args.cleaning)
    fold_splits_path = get_fold_splits_path(args.cache_dir, args.fold_splits_path, args.folds, args.seed)

    if not ensure_output_dir(args.output, args.force):
        return

    print(f"[prepare_dataset] Cargando dataset desde {data_cfg['path']}...", flush=True)
    df = pd.read_csv(
        data_cfg["path"],
        delimiter=data_cfg.get("delimiter", ","),
        encoding=data_cfg.get("encoding", "utf-8"),
    )
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"No se encontraron columnas {text_col}/{label_col} en el dataset.")
    print(f"[prepare_dataset] Dataset cargado con {len(df)} filas.", flush=True)

    texts_raw = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].astype(int).map(LABEL_TO_INDEX).to_numpy()
    fold_count = min(args.folds, len(np.unique(labels_raw)))

    fold_pairs = load_or_build_fold_indices(fold_splits_path, labels_raw, fold_count, args.seed)

    print(f"[prepare_dataset] Inicio limpieza '{args.cleaning}'...", flush=True)
    cleaned_texts, token_lists = load_or_build_clean_texts(args.cleaning, texts_raw, clean_cache_path)
    print(f"[prepare_dataset] Limpieza completada. Ejemplo: {cleaned_texts[0][:80]}...", flush=True)

    print(f"[prepare_dataset] Aplicando NLP '{args.nlp}'...", flush=True)
    features, nlp_metadata = apply_nlp(
        cleaned_texts,
        token_lists,
        method=args.nlp,
        max_len=max_len,
        vocab_size=vocab_size,
    )
    print(f"[prepare_dataset] NLP completado. Features shape={features.shape}.", flush=True)

    tokenizer = nlp_metadata.get("tokenizer")
    vectorizer = nlp_metadata.get("vectorizer")
    final_vocab_size = nlp_metadata.get("vocab_size", vocab_size)

    embedding_matrix_path = None
    if args.embedding == "word2vec":
        if tokenizer is None:
            raise RuntimeError("word2vec solo está soportado con `nlp=keras_tokenizer`.")
        print(f"[prepare_dataset] Entrenando Word2Vec (dim={args.embedding_dim})...", flush=True)
        w2v = train_word2vec(token_lists, args.embedding_dim)
        matrix = build_embedding_matrix(tokenizer, w2v, args.embedding_dim, final_vocab_size)
        embedding_matrix_path = args.output / "embedding_matrix.npy"
        np.save(embedding_matrix_path, matrix)
        print(f"[prepare_dataset] Embedding Word2Vec guardado en {embedding_matrix_path}.", flush=True)

    fold_indices: List[int] = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_pairs, start=1):
        fold_indices.append(fold_idx)
        fold_dir = args.output / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            fold_dir / "data.npz",
            x_train=features[train_idx],
            y_train=labels_raw[train_idx],
            x_val=features[val_idx],
            y_val=labels_raw[val_idx],
        )
        print(f"[prepare_dataset] Fold {fold_idx} guardado en {fold_dir}")

    if tokenizer is not None:
        with (args.output / "tokenizer.json").open("w", encoding="utf-8") as fp:
            fp.write(tokenizer.to_json())
    if vectorizer is not None:
        joblib.dump(vectorizer, args.output / "vectorizer.joblib")

    metadata = {
        "experiment_id": args.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": data_cfg["path"],
        "cleaning": args.cleaning,
        "nlp_method": args.nlp,
        "embedding": args.embedding,
        "embedding_dim": args.embedding_dim,
        "max_len": max_len,
        "vocab_size": int(final_vocab_size),
        "num_folds": len(fold_indices),
        "folds": fold_indices,
        "tokenizer_path": str(args.output / "tokenizer.json") if tokenizer is not None else None,
        "vectorizer_path": str(args.output / "vectorizer.joblib") if vectorizer is not None else None,
        "embedding_matrix": str(embedding_matrix_path) if embedding_matrix_path else None,
        "clean_cache": str(clean_cache_path),
        "fold_indices_path": str(fold_splits_path),
        "notes": args.notes,
        "seed": args.seed,
    }
    with (args.output / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    print(f"[prepare_dataset] Metadata almacenada en {args.output / 'metadata.json'}")


if __name__ == "__main__":
    nltk.download("punkt", quiet=True)  # ensure tokenizer resources
    main()

