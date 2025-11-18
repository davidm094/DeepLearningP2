## Pipeline experimental (limpieza → NLP → folds → modelos)

1. **Preparación de dataset**
   ```bash
   PYTHONPATH=. python scripts/prepare_dataset.py \
       --config config/phase2.yaml \
       --output artifacts/data/EXP_BASE_LEM \
       --experiment-id EXP_BASE_LEM \
       --cleaning lemmatize \
       --nlp keras_tokenizer \
       --embedding word2vec \
       --folds 5
   ```
   - `--cleaning`: `baseline`, `lemmatize`, `stem`.
   - `--nlp`: `keras_tokenizer` (secuencias) o `tfidf`.
   - `--embedding`: `learned` (Embedding estándar) o `word2vec` (entrenado sobre los tokens).
   - Salida: `artifacts/data/<EXP>/fold_*/data.npz`, `tokenizer.json`, `metadata.json` (incluye técnica aplicada, vocabulario, rutas, timestamps).

2. **Entrenamiento por modelo**
   - SimpleRNN:
     ```bash
     PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_simple_rnn.py \
         --config config/phase2.yaml \
         --data-cache artifacts/data/EXP_BASE_LEM \
         --folds 1,2,3,4,5 \
         --output artifacts/phase2/EXP_BASE_LEM/simple_rnn \
         --tag CLN_LEM_W2V \
         --show-summary
     ```
   - (Análogos `train_lstm.py`, `train_gru.py` se generarán siguiendo la misma estructura.)
   - Cada fold imprime el progreso epoch a epoch y guarda:
     - `fold_<n>_history.json`
     - `fold_<n>_metrics.json` (F1 macro, recall negativa, precisión positiva, tiempo)
   - Se agrega o actualiza `artifacts/experiments.csv` con una fila por fold (ID de experimento, técnicas, hiperparámetros, tiempos).

3. **Comparación**
   - `artifacts/experiments.csv` permite filtrar por:
     - Técnica de limpieza (`cleaning`)
     - Técnica NLP (`nlp_method`)
     - Embedding (`embedding`)
     - Modelo (`model`)
   - Posteriormente `scripts/compare_results.py` podrá pivotear resultados (pendiente de implementación).

### Variables experimentales
1. **Limpieza / dataset** (`--cleaning` + parámetros opcionales).
2. **Representación NLP** (`--nlp`: tokenizer vs. tfidf; `--embedding`: learned, word2vec, etc.).
3. **Modelo** (`train_simple_rnn.py`, `train_lstm.py`, `train_gru.py`) y variantes/hyperparams desde `config/phase2.yaml`.
4. **Hiperparámetros** (epochs, batch size, unidades). Cada corrida anota los valores usados.

### Recomendaciones
- Asignar un `experiment-id` único para cada combinación de variables (`EXP_CLN1_NLP2_LSTM_A`).
- Ejecutar `scripts/prepare_dataset.py` una sola vez por combinación `cleaning + nlp + embedding`.
- Entrenar modelo por modelo reutilizando los mismos folds (sin re-preprocesar).
- Registrar en `BITACORA.md` cada experimento usando `scripts/bitacora_add.sh`.

