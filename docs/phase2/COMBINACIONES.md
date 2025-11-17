## Tabla de combinaciones experimentales

| ID | Limpieza | NLP | Embedding | Notas |
|----|----------|-----|-----------|-------|
| C01 | baseline | keras_tokenizer | learned | Embedding entrenada junto al modelo |
| C02 | baseline | keras_tokenizer | word2vec | word2vec entrenado sobre baseline |
| C03 | lemmatize | keras_tokenizer | learned | Requiere spaCy `es_core_news_sm` |
| C04 | lemmatize | keras_tokenizer | word2vec | word2vec sobre textos lematizados |
| C05 | stem | keras_tokenizer | learned | Usa SnowballStemmer |
| C06 | stem | keras_tokenizer | word2vec | |
| C07 | baseline | keras_tokenizer | learned (max_len=384) | Secuencias más largas |
| C08 | baseline | keras_tokenizer | word2vec (dim=256) | Embedding 256 dims |
| C09 | baseline | keras_tokenizer | learned (vocab=50k) | |
| C10 | lemmatize | keras_tokenizer | learned (dropout=0.3) | Explorar regularización |
| C11 | stem | keras_tokenizer | learned (dropout=0.3) | |

> Las combinaciones con TF-IDF se eliminaron porque las RNN requieren entradas secuenciales. TF-IDF se reservará para modelos clásicos en otra etapa.

> Los IDs se usan para definir experimentos reproducibles. Si ajustas hiperparámetros específicos (ej. vocab, max_len), hazlo dentro de la configuración correspondiente al ID.

### Identificador de corrida
Cada entrenamiento debe usar:
```
RUN_ID = <COMBO_ID>_<MODEL>_<YYYYMMDD-HHMMSS>
```
Ejemplo: `C04_LSTM_20251115-1830`.

Se recomienda:
- Usar `--tag C04_LSTM` (o incluir el RUN_ID completo) al ejecutar `train_<modelo>.py`.
- Guardar los artefactos en `artifacts/phase2/<RUN_ID>/`.
- Registrar el RUN_ID en `BITACORA.md` y en `artifacts/experiments.csv`.

Así, cada corrida queda trazable: sabemos qué combinación (tabla) se usó, qué modelo se entrenó y cuándo.

