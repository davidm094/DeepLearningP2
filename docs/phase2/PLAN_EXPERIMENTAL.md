## Plan Experimental (DoE) – Proyecto RNN Sentimientos

Inspirado en la guía `@experimentos.mdc`, este documento define cómo diseñaremos, ejecutaremos y documentaremos cada experimento. El objetivo es poder comparar de forma justa los efectos de:

1. Técnicas de **limpieza/preprocesamiento** (`baseline`, `lemmatize`, `stem`).
2. Técnicas **NLP / embeddings** (`keras_tokenizer + learned`, `keras_tokenizer + word2vec`, `tfidf`).
3. **Modelos** (SimpleRNN, LSTM, GRU y sus variantes).
4. **Hiperparámetros** (unidades, dropout, epochs, batch size).

Cada combinación de factores constituye un **tratamiento** y recibe un ID único `EXP_<CLEAN>_<NLP>_<EMB>_<MODEL>_<VAR>`.

---

### 1. Flujo Experimental

| Paso | Acción | Scripts |
| ---- | ------ | ------- |
| 1 | Definir factores/valores en `experiments/plan.yaml` | Manual |
| 2 | Preparar dataset (limpieza + NLP + embedding + folds) | `scripts/prepare_dataset.py` |
| 3 | Entrenar modelo por folds (time per fold, métricas) | `scripts/train_<modelo>.py` |
| 4 | Registrar resultados en `artifacts/experiments.csv` + `BITACORA.md` | Automático + `scripts/bitacora_add.sh` |
| 5 | Comparar resultados (pivot por factores) | `scripts/compare_results.py` (próximo) |

---

### 2. Variables de Diseño

| Factor | Tipo | Niveles iniciales | Notas |
| ------ | ---- | ----------------- | ----- |
| Limpieza (`--cleaning`) | Cualitativo | baseline, lemmatize, stem | Uso de spaCy o NLTK |
| NLP (`--nlp`) | Cualitativo | keras_tokenizer, tfidf | Incluye dimensiones `vocab_size`, `max_len` |
| Embedding (`--embedding`) | Cualitativo | learned, word2vec | `word2vec` entrena sobre tokens del paso anterior |
| Modelo | Cualitativo | SimpleRNN, LSTM, GRU | Cada uno tendrá su script dedicado |
| Hiperparámetros | Cuantitativo | unidades, dropout, epochs, batch size | Se controlan vía `config/*.yaml` y flags |

Variables de respuesta (por fold):
- **F1 Macro** (Caso 3 – dashboard estratégico).
- **Recall clase 0** (Caso 1 – alerta).
- **Precisión clase 1** (Caso 2 – testimonios).
- **Tiempo de entrenamiento** total y por fold.

---

### 3. Preparación de Dataset (Paso 2)

```bash
PYTHONPATH=. python scripts/prepare_dataset.py \
    --config config/phase2.yaml \
    --output artifacts/data/EXP_BASE_LEM_W2V \
    --experiment-id EXP_BASE_LEM_W2V \
    --cleaning lemmatize \
    --nlp keras_tokenizer \
    --embedding word2vec \
    --folds 5 \
    --notes \"Lema + tokenizer + word2vec\"
```

Salidas:
- `fold_1 … fold_5 / data.npz`
- `tokenizer.json` o `vectorizer.joblib`
- `embedding_matrix.npy` (si aplica)
- `metadata.json` con todo el contexto (semilla, técnicas, timestamp).

---

### 4. Entrenamiento por Modelo (Paso 3)

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_simple_rnn.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/EXP_BASE_LEM_W2V \
    --folds 1,2,3,4,5 \
    --output artifacts/phase2/EXP_BASE_LEM_W2V/simple_rnn \
    --tag BASE_LEM_W2V \
    --show-summary
```

- Logs muestran cada `epoch` y el tiempo se calcula con `time.perf_counter()`.
- Cada fold genera `history`, `metrics` y pesos (opcional).
- Se añade una fila a `artifacts/experiments.csv` con:
  - ID de experimento, tag, modelo, fold, técnicas usadas.
  - Hiperparámetros efectivos (epochs, batch size, unidades).
  - Métricas (F1 macro, recall_neg, precision_pos).
  - Tiempos: `train_time_sec`.
- Se recomienda registrar el hito en `BITACORA.md`:
  ```bash
  bash scripts/bitacora_add.sh \"EXP_BASE_LEM_W2V fold 1 completado en 00:12:34\" EXP
  ```

---

### 5. Comparación y análisis (Pasos 4 y 5)

1. `artifacts/experiments.csv` es la tabla maestra (DoE):
   - Se puede agrupar por limpieza, NLP, embedding, modelo, hiperparámetros.
   - Permite ANOVA o análisis factorial (según `@experimentos.mdc`).
2. `scripts/compare_results.py` (por implementar) leerá ese CSV y generará:
   - Tablas pivote (media y std de métricas por factor).
   - Gráficas para observar interacciones.
3. Conclusiones y próximos experimentos se documentan en `docs/phase2/RESULTADOS.md` y `BITACORA.md`.

---

### 6. Buenas prácticas de DoE aplicadas

- **Aleatorización:** `StratifiedKFold` con `shuffle=True` y semilla fija.
- **Repetición:** Cada combinación se evalúa en 5 folds → promediamos la variabilidad.
- **Control de ruido:** Semillas fijadas (`seed`), misma GPU, misma versión del entorno (`03_setup`).
- **Interacciones:** Al cruzar factores (limpieza × embedding × modelo) podremos identificar interacciones significativas siguiendo la metodología de `experimentos.mdc`.

---

### 7. Checklist por experimento

1. Definir ID y factores en `plan.yaml`.
2. Ejecutar `scripts/prepare_dataset.py`.
3. Entrenar modelos deseados (`train_simple_rnn.py`, etc.) fold a fold.
4. Registrar métricas en `experiments.csv` y `BITACORA.md`.
5. Analizar resultados y proponer siguiente iteración.

Con esto garantizamos reproducibilidad, visibilidad del progreso y documentación completa de cada variación, cumpliendo la expectativa de un flujo experimental riguroso.

