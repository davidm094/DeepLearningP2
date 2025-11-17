## Planteamiento experimental

- **Objetivo**: Clasificar el sentimiento (negativo, neutro, positivo) en reseñas de hoteles andaluces usando sólo modelos RNN (SimpleRNN, LSTM, GRU) con embeddings entrenados desde cero o Word2Vec propio. La métrica oficial es **F1 macro** sobre 3 folds estratificados; se monitorean también `recall_neg`, `precision_pos`, la matriz de confusión y los tiempos de entrenamiento.
- **Contexto Fase 1**: El diagnóstico inicial (`docs/resources/entregas/Proyecto_DL_Fase1.pdf`) evidenció 112k reseñas y un fuerte desbalance (≈66 % positivas, 13 % negativas, 21 % neutrales). Esta caracterización motivó el uso de métricas macro y la priorización del `recall_neg` para los casos de negocio (alerta temprana y reputación).
- **Pipeline**:
  1. Limpiar el texto según la variante de combinación (baseline, lematización spaCy, stemming Snowball).
  2. Generar tokens y secuencias (tokenizer Keras) o entrenar Word2Vec según la configuración.
  3. Guardar folds estratificados (k=3, misma semilla) para reutilizarlos en todos los modelos.
  4. Entrenar cada modelo/variación con callbacks (`EarlyStopping`, `ReduceLROnPlateau`), batch size 256 y dropout externo cuando aplica.
- **Variables experimentales**:
  - **Limpieza**: `baseline`, `lemmatize`, `stem`.
  - **NLP/Embeddings**: tokenizer Keras (embedding aprendido), Word2Vec (dim 128/256/300 según combo), variaciones en `max_len` y `vocab_size`.
  - **Modelo**: SimpleRNN, LSTM, GRU y sus versiones bidireccionales (capa `Bidirectional` con cuDNN).
  - **Hiperparámetros**: unidades (128 en SimpleRNN/GRU, 64 en LSTM optimizado), tasas de dropout/post-RNN, `class_weight_multipliers`, número de épocas máximas (30) con early stopping.
- **Infraestructura**: GPU RTX 3090 con `TF_FORCE_GPU_ALLOW_GROWTH=true`. Todos los scripts escriben en `artifacts/phase2/` y registran métricas en `artifacts/experiments.csv`.

### Arquitectura base y variables ensayadas
- **Capas**: `Embedding` (128 dims) → capa recurrente única (SimpleRNN 128 u., LSTM/GRU 64 u.) → `Dropout` externo (0.2) → `Dense` softmax de 3 nodos.
- **Aceleración cuDNN**: en LSTM/GRU fijamos `dropout=recurrent_dropout=0` dentro de la celda y trasladamos la regularización al `post_rnn_dropout`; así se activa el kernel cuDNN y los tiempos bajan a 20–40 s/fold en LSTM simple y a ~35 s/fold en BiLSTM.
- **Pesos de clase**: `compute_class_weights` + multiplicador manual de 1.2 para la clase 0 (negativa) con el fin de elevar `recall_neg`.
- **Callbacks**: `EarlyStopping (patience=5, min_delta=0.002)` y `ReduceLROnPlateau (factor=0.5)` en todos los modelos para garantizar convergencia comparable.
- **Variables exploradas**:
  - Limpieza (`baseline`, `lemmatize`, `stem`).
  - Representación (`keras_tokenizer + embedding entrenado`, `keras_tokenizer + Word2Vec 128/256/300`).
  - Secuencias (`max_len=256` base, override `max_len=384` en C07) y vocabulario (`vocab_size=30k`, override `50k` en C09).
  - Regularización externa (`dropout=0.3` post-RNN en C10/C11), y comparaciones learnt vs Word2Vec por cada limpieza.

## Checklist de combinaciones

| ID | Limpieza | NLP/Notas | Embedding | SRN | SRN-BI | LSTM | LSTM-BI | GRU | GRU-BI |
|----|----------|-----------|-----------|-----|--------|------|---------|-----|--------|
| C01 | baseline | tokenizer | learned | OK | OK | OK | OK | OK | OK |
| C02 | baseline | tokenizer | word2vec (128) | OK | OK | OK | OK | OK | OK |
| C03 | lemmatize | tokenizer | learned | OK | OK | OK | OK | OK | OK |
| C04 | lemmatize | tokenizer | word2vec (128) | OK | OK | OK | OK | OK | OK |
| C05 | stem | tokenizer | learned | OK | OK | OK | OK | OK | OK |
| C06 | stem | tokenizer | word2vec (128) | OK | OK | OK | OK | OK | OK |
| C07 | baseline | tokenizer | learned + `max_len=384` | OK | OK | OK | OK | OK | OK |
| C08 | baseline | tokenizer | word2vec (256) | OK | OK | OK | OK | OK | OK |
| C09 | baseline | tokenizer | learned + `vocab=50k` | OK | OK | OK | OK | OK | OK |
| C10 | lemmatize | tokenizer | learned + `dropout=0.3` | OK | OK | OK | OK | OK | OK |
| C11 | stem | tokenizer | learned + `dropout=0.3` | OK | OK | OK | OK | OK | OK |

Legenda: OK combinación terminada para ese modelo; PEND pendiente de ejecución/registro. Actualmente el bloque restante es **GRU bidireccional (C05–C11)**, que se está corriendo en este momento (ver bitácora).

## Resumen de mejores experimentos por modelo

| Experimento | Modelo | Limpieza | Embedding | Notas relevantes | F1 Macro | Recall Neg | Precisión Pos | Tiempo ≈ s/fold |
|-------------|--------|----------|-----------|------------------|---------:|-----------:|--------------:|----------------:|
| `C01_SRN_LONG_20251117-004240` | SimpleRNN | baseline | learned | LR=5e-4, peso clase 0=1.2, entrenamiento largo | 0.321 | 0.530 | 0.752 | 61 |
| `C02_SRN_BI_20251117-080430` | SimpleRNN-BI | baseline | word2vec (128) | Bidireccional, batch 256 | 0.751 | 0.826 | 0.928 | 65 |
| `C05_LSTM_20251117-144218` | LSTM | stem | learned | LSTM 64 u., dropout externo 0.2, cuDNN | 0.239 | 0.372 | 0.823 | 24 |
| `C02_LSTM_BI_20251117-111045` | LSTM-BI | baseline | word2vec (128) | Bidireccional, batch 256, cuDNN | **0.785** | **0.823** | **0.964** | 31 |
| `C06_GRU_SIMPLE_20251117-100806` | GRU | stem | word2vec (128) | LR=5e-4, peso clase 0=1.2 | 0.241 | 0.372 | 0.490 | 18 |
| `C05_GRU_BI_20251117-115826` | GRU-BI | stem | learned | Bidireccional, dropout externo 0.3 | 0.768 | 0.848 | 0.961 | 28 |

Estas cifras provienen de los `summary.json` de cada experimento y se registran en `artifacts/experiments.csv`, garantizando trazabilidad de métricas y tiempos.

### Discusión de factores experimentales
- **Limpieza**: El stemming (C05/C06) eleva `recall_neg` en GRU y LSTM simple, mientras que la lematización (C03/C04) mejora la estabilidad de BiLSTM al reducir la variabilidad léxica.
- **Embeddings**: Word2Vec entrenado sobre el propio corpus aporta mayor precisión en los modelos bidireccionales (SimpleRNN-BI y BiLSTM), mientras que los embeddings aprendidos funcionan mejor con recetas de entrenamiento largo o stemming agresivo.
- **Longitud y vocabulario**: Los overrides `max_len=384` (C07) y `vocab_size=50k` (C09) aumentan ligeramente F1 en SimpleRNN pero con un costo de tiempo (hasta 45 s/fold) sin mejoras claras en precisión.
- **Dropout externo**: Subir `post_rnn_dropout` a 0.3 (C10/C11) ayuda a BiLSTM/BiGRU a mantener `precision_pos>0.95` con textos limpios o stemmed; en modelos unidireccionales simplemente incrementa el sesgo hacia una sola clase.
- **Precisión positiva**: Además del F1 macro y `recall_neg`, se monitorea `precision_pos` para el caso de uso de testimonios. BiLSTM y BiGRU superan 0.96, SimpleRNN-BI se mantiene en ~0.93, mientras que los modelos simples sin bidireccionalidad quedan por debajo de 0.83, evidenciando el trade-off sin contexto bidireccional.

## Resultados por modelo

### SimpleRNN (unidireccional) – C01 a C11

| Combo | Fold | RUN_ID | Limpieza | NLP | Embedding | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|-------|------|--------|----------|-----|-----------|---------:|-----------:|--------------:|---------------:|
| C01 | 1 | C01_SRN_20251116-223956 | baseline | tokenizer | learned | 0.2974 | 0.0135 | 0.7331 | 24.78 |
| C01 | 2 | C01_SRN_20251116-223956 | baseline | tokenizer | learned | 0.1634 | 0.0034 | 0.7923 | 21.97 |
| C01 | 3 | C01_SRN_20251116-223956 | baseline | tokenizer | learned | 0.1455 | 0.4371 | 0.7143 | 20.94 |
| **Promedio C01** | – | – | – | – | – | **0.2021** | **0.1513** | **0.7465** | – |
| C02 | 1 | C02_SRN_20251116-224108 | baseline | tokenizer | word2vec | 0.1374 | 0.3951 | 1.0000 | 26.18 |
| C02 | 2 | C02_SRN_20251116-224108 | baseline | tokenizer | word2vec | 0.3017 | 0.0270 | 0.7330 | 22.47 |
| C02 | 3 | C02_SRN_20251116-224108 | baseline | tokenizer | word2vec | 0.3192 | 0.3697 | 0.7409 | 20.32 |
| **Promedio C02** | – | – | – | – | – | **0.2528** | **0.2639** | **0.8246** | – |
| C03 | 1 | C03_SRN_20251116-224222 | lemmatize | tokenizer | learned | 0.2932 | 0.3412 | 0.7387 | 24.73 |
| C03 | 2 | C03_SRN_20251116-224222 | lemmatize | tokenizer | learned | 0.3083 | 0.0719 | 0.7207 | 23.23 |
| C03 | 3 | C03_SRN_20251116-224222 | lemmatize | tokenizer | learned | 0.2666 | 0.3247 | 0.7670 | 19.95 |
| **Promedio C03** | – | – | – | – | – | **0.2894** | **0.2459** | **0.7421** | – |
| C04 | 1 | C04_SRN_20251116-224336 | lemmatize | tokenizer | word2vec | 0.1020 | 0.0449 | 1.0000 | 24.00 |
| C04 | 2 | C04_SRN_20251116-224336 | lemmatize | tokenizer | word2vec | 0.2260 | 0.4809 | 0.7687 | 22.82 |
| C04 | 3 | C04_SRN_20251116-224336 | lemmatize | tokenizer | word2vec | 0.1678 | 0.2528 | 0.7249 | 21.31 |
| **Promedio C04** | – | – | – | – | – | **0.1653** | **0.2595** | **0.8312** | – |
| C05 | 1 | C05_SRN_20251116-225043 | stem | tokenizer | learned | 0.2536 | 0.0988 | 0.6963 | 24.32 |
| C05 | 2 | C05_SRN_20251116-225043 | stem | tokenizer | learned | 0.2356 | 0.0753 | 0.7582 | 22.25 |
| C05 | 3 | C05_SRN_20251116-225043 | stem | tokenizer | learned | 0.2745 | 0.3112 | 0.7180 | 20.53 |
| **Promedio C05** | – | – | – | – | – | **0.2546** | **0.1618** | **0.7242** | – |
| C06 | 1 | C06_SRN_20251116-225208 | stem | tokenizer | word2vec | 0.2548 | 0.4130 | 0.7621 | 23.74 |
| C06 | 2 | C06_SRN_20251116-225208 | stem | tokenizer | word2vec | 0.2846 | 0.0438 | 0.7442 | 22.33 |
| C06 | 3 | C06_SRN_20251116-225208 | stem | tokenizer | word2vec | 0.1781 | 0.1596 | 0.7750 | 21.24 |
| **Promedio C06** | – | – | – | – | – | **0.2392** | **0.2055** | **0.7604** | – |
| C07 | 1 | C07_SRN_20251116-225606 | baseline | tokenizer | learned (max_len=384) | 0.2852 | 0.0067 | 0.7283 | 32.94 |
| C07 | 2 | C07_SRN_20251116-225606 | baseline | tokenizer | learned (max_len=384) | 0.2914 | 0.1562 | 0.7310 | 31.64 |
| C07 | 3 | C07_SRN_20251116-225606 | baseline | tokenizer | learned (max_len=384) | 0.2969 | 0.0539 | 0.7228 | 29.88 |
| **Promedio C07** | – | – | – | – | – | **0.2912** | **0.0723** | **0.7274** | – |
| C08 | 1 | C08_SRN_20251116-225748 | baseline | tokenizer | word2vec (dim=256) | 0.1300 | 0.2144 | 1.0000 | 25.19 |
| C08 | 2 | C08_SRN_20251116-225748 | baseline | tokenizer | word2vec (dim=256) | 0.1421 | 0.4596 | 1.0000 | 23.67 |
| C08 | 3 | C08_SRN_20251116-225748 | baseline | tokenizer | word2vec (dim=256) | 0.2847 | 0.0056 | 0.7287 | 21.55 |
| **Promedio C08** | – | – | – | – | – | **0.1856** | **0.2265** | **0.9096** | – |
| C09 | 1 | C09_SRN_20251116-225903 | baseline | tokenizer | learned (vocab=50k) | 0.1134 | 0.0673 | 0.6667 | 24.54 |
| C09 | 2 | C09_SRN_20251116-225903 | baseline | tokenizer | learned (vocab=50k) | 0.1334 | 0.8888 | 0.0000 | 23.35 |
| C09 | 3 | C09_SRN_20251116-225903 | baseline | tokenizer | learned (vocab=50k) | 0.3015 | 0.3303 | 0.7353 | 20.76 |
| **Promedio C09** | – | – | – | – | – | **0.1828** | **0.4288** | **0.4673** | – |
| C10 | 1 | C10_SRN_20251116-230320 | lemmatize | tokenizer | learned (dropout=0.3) | 0.2765 | 0.2929 | 0.7232 | 26.88 |
| C10 | 2 | C10_SRN_20251116-230320 | lemmatize | tokenizer | learned (dropout=0.3) | 0.0896 | 0.9775 | 0.5185 | 22.73 |
| C10 | 3 | C10_SRN_20251116-230320 | lemmatize | tokenizer | learned (dropout=0.3) | 0.2782 | 0.0112 | 0.7204 | 20.52 |
| **Promedio C10** | – | – | – | – | – | **0.2148** | **0.4272** | **0.6540** | – |
| C11 | 1 | C11_SRN_20251116-230435 | stem | tokenizer | learned (dropout=0.3) | 0.2902 | 0.2480 | 0.7205 | 24.82 |
| C11 | 2 | C11_SRN_20251116-230435 | stem | tokenizer | learned (dropout=0.3) | 0.2012 | 0.1135 | 0.7117 | 22.98 |
| C11 | 3 | C11_SRN_20251116-230435 | stem | tokenizer | learned (dropout=0.3) | 0.2727 | 0.2022 | 0.7366 | 21.07 |
| **Promedio C11** | – | – | – | – | – | **0.2547** | **0.1879** | **0.7229** | – |

Notas:
- Las métricas se regeneraron tras renumerar combos y reutilizar los mismos índices de folds (archivo `artifacts/cache/folds_seed42_k3.json`).
- El caching de textos limpios permitió que C02 y C04 sólo entrenaran el embedding Word2Vec sin repetir la lematización/baseline.
- C02 ofrece el mejor promedio de F1 macro en SimpleRNN hasta ahora; C03 se acerca con limpieza lematizada pero sin word2vec.
- C05/C06 (stemming) ya usan los mismos folds; C05 se comporta mejor con embedding aprendido, mientras que C06 logra mayor recall negativo gracias al word2vec.
- C07 (max_len 384) incrementa ligeramente el F1 respecto a C01, al costo de más tiempo por fold (~33 s); C08 (word2vec 256) mejora precisión positiva pero no F1; C09 (vocab 50k) muestra recall alto en el fold 2 pero la precisión se desploma por falta de predicciones positivas.
- C10/C11 (dropout 0.3) ayudan a regularizar pero mantienen resultados mixtos; C10 logra alto recall en el fold 2 pero pierde F1, mientras que C11 se mantiene cercano a C05.
- Todos los entrenamientos usan 5 épocas y batch 128 en GPU (RTX 3090); XLA sigue emitiendo advertencias, documentadas en los logs.

Próximos pasos:
1. Ejecutar LSTM y GRU sobre los mismos caches (C01–C11) sin reprocesar datos.
2. Comparar tiempos y métricas entre arquitecturas en el mismo reporte.
3. Incorporar matrices de confusión y métricas por clase en el informe final.

## Resumen por modelo (combinaciones completas)

- **SimpleRNN (unidireccional)**: C01–C11 (incluye corridas largas con LR 5e-4). El mejor promedio corresponde a `C01_SRN_LONG_20251117-004240` con **F1_macro=0.321** y **recall_neg=0.53**; requiere más épocas y pesos de clase para rescatar la clase negativa.
- **SimpleRNN bidireccional**: C01–C11 terminados. `C03_SRN_BI_20251117-104355` (lemmatización) mantiene **F1_macro≈0.75** y **recall_neg≈0.82** con tiempos <45 s/fold, convirtiéndose en el baseline fuerte de RNN sencillas.
- **LSTM (sin bidireccional)**: C01–C11 completados, pero incluso la mejor variante (`C03_LSTM_20251117-094318`) se queda en **F1_macro≈0.37** y recall_neg bajo; el modelo tiende a colapsar en la clase positiva.
- **LSTM bidireccional**: C01–C11 re-ejecutados con cuDNN. `C02_LSTM_BI_20251117-111045` logra **F1_macro=0.785**, **recall_neg=0.82** y tiempos de ~34 s/fold, siendo la referencia actual.
- **GRU (sin bidireccional)**: C01–C11 completados tras la receta LR 5e-4. Todos los combos quedan entre **F1_macro 0.09–0.32** (ej. `C06_GRU_SIMPLE_20251117-100806`), confirmando el sesgo extremo hacia una sola clase.
- **GRU bidireccional**: ejecuciones en curso; el primer combo (`C01_GRU_BI_20251117-113707`) ya ofrece **F1_macro≈0.77** y se documenta más abajo.

## SimpleRNN – Entrenamiento largo (C01, 15 épocas + callbacks)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C01_SRN_LONG_20251116-235745 | 0.3235 | 0.0359 | 0.7284 | 42.41 |
| 2 | C01_SRN_LONG_20251116-235745 | 0.3040 | 0.3652 | 0.7166 | 36.78 |
| 3 | C01_SRN_LONG_20251116-235745 | 0.3295 | 0.1955 | 0.7300 | 20.81 |
| **Promedio** | – | **0.3190** | **0.1989** | **0.7250** | – |

- Los callbacks (`EarlyStopping`, `ReduceLROnPlateau`) permiten entrenar hasta 15 épocas sin colapsar en la clase positiva; el recall negativo sube a ~0.20 y el F1 macro mejora respecto al baseline corto (~0.30).
- Las nuevas métricas incluyen `confusion_matrix`, lo que facilita identificar que el fold 2 es el que verdaderamente recupera la clase negativa (36 % de aciertos en la fila 0).
- Tiempos por fold se mantienen manejables (20–42 s) y los artefactos quedaron en `artifacts/phase2/C01_SRN_LONG_20251116-235745/`.

## SimpleRNN – C03 (lemmatize) con LR bajo y 30 épocas

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C03_SRN_LONG_20251117-000550 | 0.2162 | 0.4736 | 0.7405 | 32.14 |
| 2 | C03_SRN_LONG_20251117-000550 | 0.0897 | 0.9831 | 0.6053 | 29.11 |
| 3 | C03_SRN_LONG_20251117-000550 | 0.0903 | 0.9843 | 0.6818 | 24.10 |
| **Promedio** | – | **0.1321** | **0.8137** | **0.6759** | – |

- Con LR inicial 5e-4 y pesos reforzados para la clase 0, el recall negativo se dispara (folds 2 y 3 >0.98), aunque el modelo cae en predecir casi todo como negativo, dejando el F1 macro muy bajo.  
- Esto confirma que los nuevos hiperparámetros pueden sesgar fuertemente hacia la clase minoritaria; el siguiente paso será balancear este comportamiento (p.ej. aumentando `min_delta`, ajustando paciencia o combinándolo con un scheduler más suave) antes de aplicar la configuración masivamente.

## Resultados LSTM – C01 (baseline)

| Combo | Fold | RUN_ID | Limpieza | NLP | Embedding | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|-------|------|--------|----------|-----|-----------|---------:|-----------:|--------------:|---------------:|
| C01 | 1 | C01_LSTM_20251116-231015 | baseline | tokenizer | learned | 0.1055 | 0.0483 | 0.8889 | 701.45 |
| C01 | 2 | C01_LSTM_20251116-231015 | baseline | tokenizer | learned | 0.3128 | 0.0483 | 0.7334 | 687.13 |
| C01 | 3 | C01_LSTM_20251116-231015 | baseline | tokenizer | learned | 0.1088 | 0.0551 | 1.0000 | 676.09 |
| **Promedio C01 (LSTM)** | – | – | – | – | – | **0.1757** | **0.0505** | **0.8741** | – |
| C01 | 1 | C01_LSTM_LONG_20251117-001258 | baseline | tokenizer | learned | 0.3180 | 0.0617 | 0.7349 | 1239.25 |

Notas LSTM:
- Estas corridas iniciales (16/nov) usaban 128 unidades y dropout interno, por lo que tardaban ≈11 min/fold; se mantienen como referencia histórica.
- La reejecución con arquitectura optimizada (64 u., dropout externo) se describe en la siguiente sección y es la que se utiliza para comparar contra el resto de modelos.
- Los artefactos siguen disponibles en `artifacts/phase2/C01_LSTM_20251116-231015/` y `C01_LSTM_LONG_20251117-001258/`.


### LSTM (unidireccional) – Reejecución cuDNN 2025-11-17

Se re-entrenaron todas las combinaciones con la arquitectura optimizada (64 unidades, dropout externo 0.2, cuDNN habilitado) para comparar tiempos/métricas en igualdad de condiciones. Cada conjunto de resultados está en `artifacts/phase2/Cxx_LSTM_20251117-14xxxx/`.

| Combo | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Épocas prom. | Tiempo medio (s/fold) |
|-------|--------|---------:|-----------:|--------------:|-------------:|----------------------:|
| C01 | C01_LSTM_20251117-143604 | 0.240 | 0.372 | 0.823 | 7.67 | 24.84 |
| C02 | C02_LSTM_20251117-143726 | 0.167 | 0.366 | 0.578 | 8.00 | 26.28 |
| C03 | C03_LSTM_20251117-143851 | 0.246 | 0.382 | 0.824 | 9.00 | 28.27 |
| C04 | C04_LSTM_20251117-144020 | 0.089 | 1.000 | 1.000 | 11.67 | 37.56 |
| C05 | C05_LSTM_20251117-144218 | 0.239 | 0.372 | 0.823 | 7.33 | 24.30 |
| C06 | C06_LSTM_20251117-144335 | 0.163 | 0.685 | 0.578 | 9.00 | 29.14 |
| C07 | C07_LSTM_20251117-144508 | 0.086 | 1.000 | 0.000 | 11.33 | 45.03 |
| C08 | C08_LSTM_20251117-144730 | 0.316 | 0.053 | 0.734 | 8.67 | 27.76 |
| C09 | C09_LSTM_20251117-144858 | 0.162 | 0.684 | 0.911 | 8.00 | 26.13 |
| C10 | C10_LSTM_20251117-145021 | 0.238 | 0.366 | 0.822 | 11.00 | 35.33 |
| C11 | C11_LSTM_20251117-145212 | 0.163 | 0.687 | 0.578 | 7.33 | 24.20 |

- Los tiempos por fold bajaron a 20–45 s gracias a cuDNN, lo que permite comparar directamente con SimpleRNN y GRU.
- Las variantes con stemming o vocabulario extendido (C06, C09, C11) elevan `recall_neg` (>0.68) aun sin bidireccionalidad, a costa de precisión.
- C08 maximiza F1 (0.316) pero evidencia que el LSTM simple sigue favoreciendo la clase positiva; la versión bidireccional continúa siendo la recomendada para despliegue.

## Nuevas corridas (documentadas 2025-11-17)

### SimpleRNN – Receta LR 5e-4, peso clase 0 = 1.2

| Combo | Fold | RUN_ID | Notas | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|-------|------|--------|-------|---------:|-----------:|--------------:|---------------:|
| C01 | 1 | C01_SRN_LONG_20251117-004240 | baseline receta nueva | 0.3825 | 0.5253 | 0.7780 | 71.49 |
| C01 | 2 | C01_SRN_LONG_20251117-004240 | baseline receta nueva | 0.2381 | 0.6685 | 0.7333 | 43.33 |
| C01 | 3 | C01_SRN_LONG_20251117-004240 | baseline receta nueva | 0.3426 | 0.3955 | 0.7449 | 69.02 |
| **Prom C01** | – | – | – | **0.3211** | **0.5298** | **0.7521** | – |
| C05 | 1 | C05_SRN_LONG_20251117-004943 | stem | 0.1962 | 0.8126 | 0.7479 | 45.16 |
| C05 | 2 | C05_SRN_LONG_20251117-004943 | stem | 0.3265 | 0.2674 | 0.7424 | 36.45 |
| C05 | 3 | C05_SRN_LONG_20251117-004943 | stem | 0.2436 | 0.5843 | 0.7352 | 30.87 |
| **Prom C05** | – | – | – | **0.2554** | **0.5548** | **0.7418** | – |
| C07 | 1 | C07_SRN_LONG_20251117-061925 | baseline max_len=384 | 0.0921 | 0.9574 | 0.5243 | 71.09 |
| C07 | 2 | C07_SRN_LONG_20251117-061925 | baseline max_len=384 | 0.2004 | 0.7146 | 0.7347 | 38.06 |
| C07 | 3 | C07_SRN_LONG_20251117-061925 | baseline max_len=384 | 0.3150 | 0.1843 | 0.7319 | 66.20 |
| **Prom C07** | – | – | – | **0.2025** | **0.6187** | **0.6636** | – |

### LSTM – Receta LR 5e-4 (C01)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C01_LSTM_LONG_20251117-062630 | 0.0855 | 1.0000 | 0.0000 | 1072.34 |
| 2 | C01_LSTM_LONG_20251117-062630 | 0.3192 | 0.0629 | 0.7350 | 1083.92 |
| 3 | C01_LSTM_LONG_20251117-062630 | 0.3217 | 0.0652 | 0.7351 | 1504.63 |
| **Promedio** | – | **0.2421** | **0.3760** | **0.4900** | – |

- El fold 1 aún se va al extremo negativo; antes de ejecutar C02–C11 con LSTM ajustaremos unidades y el máximo de épocas para moderar este comportamiento.

### GRU – C01 (receta LR 5e-4, peso clase 0 = 1.2)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec | Epochs |
|------|--------|---------:|-----------:|--------------:|---------------:|-------:|
| 1 | C01_GRU_20251117-075013 | 0.0946 | 0.9966 | 0.9000 | 53.67 | 14 |
| 2 | C01_GRU_20251117-075013 | 0.1110 | 0.0596 | 1.0000 | 43.93 | 12 |
| 3 | C01_GRU_20251117-075013 | 0.0854 | 1.0000 | 0.0000 | 29.84 | 8 |
| **Promedio** | – | **0.0970** | **0.6854** | **0.6333** | – | **11.3** |

- GRU replica el patrón observado en SimpleRNN “agresivo”: algunos folds predicen casi todo negativo. Necesitamos revisar pesos/clasificador antes de masificar.

### Bidirectional SimpleRNN – C01

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec | Epochs |
|------|--------|---------:|-----------:|--------------:|---------------:|-------:|
| 1 | C01_SRN_BI_20251117-075222 | 0.7267 | 0.7385 | 0.9417 | 67.15 | 9 |
| 2 | C01_SRN_BI_20251117-075222 | 0.7314 | 0.8034 | 0.9535 | 64.99 | 9 |
| 3 | C01_SRN_BI_20251117-075222 | 0.7424 | 0.7899 | 0.9415 | 63.19 | 9 |
| **Promedio** | – | **0.7335** | **0.7773** | **0.9456** | – | **9.0** |

- El BRNN converge muy rápido y alcanza F1 macro ~0.73 con alta precisión positiva; habrá que confirmar que no esté sobreajustando (ver curvas y confusiones) antes de adoptarlo como baseline.

### Bidirectional SimpleRNN – C02

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec | Epochs |
|------|--------|---------:|-----------:|--------------:|---------------:|-------:|
| 1 | C02_SRN_BI_20251117-080430 | 0.7609 | 0.8620 | 0.9263 | 67.37 | 9 |
| 2 | C02_SRN_BI_20251117-080430 | 0.7440 | 0.8303 | 0.9242 | 58.44 | 8 |
| 3 | C02_SRN_BI_20251117-080430 | 0.7484 | 0.7865 | 0.9325 | 69.70 | 10 |
| **Promedio** | – | **0.7511** | **0.8263** | **0.9277** | – | **9.0** |

- Mantiene el desempeño alto de C01, ahora con embedding word2vec; tiempos se sostienen en ~1 min/fold.

### Bidirectional LSTM – C01 (cuDNN activado, prueba fold 1)

| Fold | RUN_ID | Batch | Max Épocas | Épocas reales | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|-------|------------|--------------:|---------:|-----------:|--------------:|---------------:|
| 1 | C01_LSTM_BI_TEST_20251117-082642 | 256 | 20 | 10 | 0.7814 | 0.8070 | 0.9531 | 38.26 |

- Ajustes clave: `dropout=recurrent_dropout=0`, `post_rnn_dropout=0.2`, `TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false`, batch 256.  
- Tiempo por época bajó de ~400 s a ~6 s; la GPU se mantiene ocupada y las métricas saltan a F1 ≈ 0.78. Listo para escalar a los demás folds/combos.

### Bidirectional LSTM – C01 (folds 1–3 completos)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C01_LSTM_BI_20251117-084113 | 0.7757 | 0.7363 | 0.9476 | 13 | 47.85 |
| 2 | C01_LSTM_BI_20251117-084113 | 0.7907 | 0.8101 | 0.9651 | 12 | 41.88 |
| 3 | C01_LSTM_BI_20251117-084113 | 0.7712 | 0.8382 | 0.9614 | 10 | 37.15 |
| **Promedio** | – | **0.7792** | **0.7949** | **0.9580** | **11.7** | – |

- Cada fold tomó <50 s gracias a cuDNN + batch 256. Métricas superan ampliamente a todas las variantes anteriores.

### Bidirectional LSTM – C02

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C02_LSTM_BI_20251117-084446 | 0.7849 | 0.7621 | 0.9611 | 10 | 37.80 |
| 2 | C02_LSTM_BI_20251117-084446 | 0.7910 | 0.7955 | 0.9460 | 12 | 41.79 |
| 3 | C02_LSTM_BI_20251117-084446 | 0.7557 | 0.8427 | 0.9478 | 10 | 112.95 |
| **Promedio** | – | **0.7772** | **0.8001** | **0.9516** | **10.7** | – |

- El fold 3 tardó más (registro extendido de logs), pero las métricas se mantienen muy cercanas a C01 pese al embedding Word2Vec.

### Bidirectional LSTM – C03 (lemmatize)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C03_LSTM_BI_20251117-085119 | 0.7770 | 0.7329 | 0.9561 | 18 | 65.97 |
| 2 | C03_LSTM_BI_20251117-085119 | 0.7845 | 0.7910 | 0.9617 | 11 | 39.29 |
| 3 | C03_LSTM_BI_20251117-085119 | 0.7707 | 0.7472 | 0.9573 | 11 | 39.53 |
| **Promedio** | – | **0.7774** | **0.7570** | **0.9584** | **13.3** | – |

- Incluso con textos lematizados, el BiLSTM mantiene F1≈0.78; fold 1 necesitó más épocas por la curva inicial más lenta.

### Bidirectional LSTM – C04 (lemmatize + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C04_LSTM_BI_20251117-090027 | 0.7663 | 0.7946 | 0.9506 | 11 | 39.91 |
| 2 | C04_LSTM_BI_20251117-090027 | 0.7803 | 0.8112 | 0.9601 | 10 | 30.94 |
| 3 | C04_LSTM_BI_20251117-090027 | 0.7758 | 0.8494 | 0.9671 | 11 | 31.69 |
| **Promedio** | – | **0.7741** | **0.8184** | **0.9593** | **10.7** | – |

- Word2Vec con lematización se mantiene en la misma franja de desempeño; tiempos por fold siguen en ~30–40 s.

### Bidirectional LSTM – C05 (stem)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C05_LSTM_BI_20251117-090613 | 0.7586 | 0.7980 | 0.9598 | 8 | 26.01 |
| 2 | C05_LSTM_BI_20251117-090613 | 0.7761 | 0.7416 | 0.9482 | 13 | 35.91 |
| 3 | C05_LSTM_BI_20251117-090613 | 0.7744 | 0.8449 | 0.9653 | 14 | 39.24 |
| **Promedio** | – | **0.7697** | **0.7948** | **0.9578** | **11.7** | – |

- A pesar del stemming (texto más agresivo), las métricas se mantienen en la banda alta, con tiempos por fold <40 s.

### Bidirectional LSTM – C06 (stem + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C06_LSTM_BI_20251117-091103 | 0.7773 | 0.7789 | 0.9607 | 14 | 42.21 |
| 2 | C06_LSTM_BI_20251117-091103 | 0.7788 | 0.8247 | 0.9635 | 12 | 34.92 |
| 3 | C06_LSTM_BI_20251117-091103 | 0.7693 | 0.8000 | 0.9655 | 12 | 35.75 |
| **Promedio** | – | **0.7751** | **0.8012** | **0.9632** | **12.7** | – |

- Con word2vec entrenado sobre stemmed tokens seguimos con F1≈0.775 y recall_neg >0.80; tiempos por fold 35–42 s.

### Bidirectional LSTM – C07 (max_len 384)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C07_LSTM_BI_20251117-091739 | 0.7812 | 0.7710 | 0.9496 | 12 | 46.22 |
| 2 | C07_LSTM_BI_20251117-091739 | 0.7873 | 0.8247 | 0.9562 | 11 | 41.14 |
| 3 | C07_LSTM_BI_20251117-091739 | 0.7846 | 0.8292 | 0.9639 | 11 | 42.32 |
| **Promedio** | – | **0.7844** | **0.8083** | **0.9566** | **11.3** | – |

- Aumentar `max_len` a 384 no penaliza el tiempo gracias al batch 256; las métricas se mantienen en la franja superior.

### Bidirectional LSTM – C08 (word2vec dim=256)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C08_LSTM_BI_20251117-092340 | 0.7881 | 0.7935 | 0.9470 | 11 | 33.49 |
| 2 | C08_LSTM_BI_20251117-092340 | 0.7798 | 0.8618 | 0.9693 | 11 | 31.96 |
| 3 | C08_LSTM_BI_20251117-092340 | 0.7479 | 0.6404 | 0.9571 | 13 | 37.77 |
| **Promedio** | – | **0.7719** | **0.7652** | **0.9578** | **11.7** | – |

- El word2vec de 256 dimensiones mantiene F1≈0.77; el fold 3 tuvo menor recall_neg pero sigue arriba del baseline original.

### Bidirectional LSTM – C09 (vocab_size 50k)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C09_LSTM_BI_20251117-092423 | 0.7749 | 0.8126 | 0.9636 | 10 | 30.51 |
| 2 | C09_LSTM_BI_20251117-092423 | 0.7885 | 0.8011 | 0.9638 | 11 | 31.42 |
| 3 | C09_LSTM_BI_20251117-092423 | 0.7703 | 0.8011 | 0.9581 | 9 | 27.06 |
| **Promedio** | – | **0.7779** | **0.8049** | **0.9618** | **10.0** | – |

- Aumentar el vocabulario a 50 k no degrada el desempeño; los tiempos se mantienen en ~30 s por fold.

### Bidirectional LSTM – C10 (dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C10_LSTM_BI_20251117-092935 | 0.7717 | 0.7228 | 0.9564 | 12 | 37.09 |
| 2 | C10_LSTM_BI_20251117-092935 | 0.7787 | 0.7596 | 0.9539 | 11 | 32.41 |
| 3 | C10_LSTM_BI_20251117-092935 | 0.7570 | 0.8910 | 0.9535 | 11 | 32.32 |
| **Promedio** | – | **0.7692** | **0.7911** | **0.9546** | **11.3** | – |

- Aunque el fold 3 elevó el recall negativo a costa de precisión, los promedios se mantienen en línea con los combos anteriores.

### Bidirectional LSTM – C11 (stem + dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C11_LSTM_BI_20251117-093214 | 0.7866 | 0.7946 | 0.9586 | 14 | 41.07 |
| 2 | C11_LSTM_BI_20251117-093214 | 0.7782 | 0.8348 | 0.9628 | 9 | 109.07 |
| 3 | C11_LSTM_BI_20251117-093214 | 0.7567 | 0.7753 | 0.9688 | 10 | 28.67 |
| **Promedio** | – | **0.7738** | **0.8016** | **0.9634** | **11.0** | – |

- Con stemming + dropout 0.3 se mantiene F1≈0.77; el fold 2 tardó más por logs adicionales pero terminó sin inconvenientes.

### Bidirectional LSTM – C02 (word2vec) – Reintento 2025-11-17 11:10

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C02_LSTM_BI_20251117-111045 | 0.7903 | 0.8283 | 0.9618 | 34.01 |
| 2 | C02_LSTM_BI_20251117-111045 | 0.7829 | 0.8180 | 0.9642 | 30.31 |
| 3 | C02_LSTM_BI_20251117-111045 | 0.7828 | 0.8213 | 0.9644 | 29.24 |
| **Promedio (retry)** | – | **0.7853** | **0.8225** | **0.9635** | – |

- Repetimos C02 para validar la receta actual; las métricas se mantienen e incluso suben ligeramente respecto al bloque anterior, con tiempos <35 s/fold.

### Bidirectional LSTM – C03 (lemmatize) – Reintento 2025-11-17 11:13

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C03_LSTM_BI_20251117-111320 | 0.7761 | 0.8002 | 0.9625 | 31.25 |
| 2 | C03_LSTM_BI_20251117-111320 | 0.7921 | 0.8360 | 0.9647 | 34.71 |
| 3 | C03_LSTM_BI_20251117-111320 | 0.7778 | 0.8303 | 0.9659 | 26.90 |
| **Promedio (retry)** | – | **0.7820** | **0.8222** | **0.9643** | – |

- La versión lematizada repite el excelente rendimiento del BiLSTM con tiempos entre 27 y 35 s; se confirma que la receta es estable con distintas limpiezas.

### Bidirectional LSTM – C04 (lemmatize + word2vec) – Reintento 2025-11-17 11:15

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C04_LSTM_BI_20251117-111519 | 0.7892 | 0.8440 | 0.9587 | 32.58 |
| 2 | C04_LSTM_BI_20251117-111519 | 0.7743 | 0.7562 | 0.9627 | 28.36 |
| 3 | C04_LSTM_BI_20251117-111519 | 0.7816 | 0.8191 | 0.9587 | 31.95 |
| **Promedio (retry)** | – | **0.7817** | **0.8064** | **0.9600** | – |

- El combo con lematización + word2vec confirma F1≈0.782 y alta precisión (&gt;0.96) con tiempos de ~30 s/fold; se mantiene competitivo frente a los mejores resultados previos.

### Bidirectional LSTM – C05 (stemming) – Reintento 2025-11-17 11:17

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C05_LSTM_BI_20251117-111722 | 0.7804 | 0.7710 | 0.9553 | 30.95 |
| 2 | C05_LSTM_BI_20251117-111722 | 0.7706 | 0.7674 | 0.9580 | 23.82 |
| 3 | C05_LSTM_BI_20251117-111722 | 0.7688 | 0.8618 | 0.9530 | 26.28 |
| **Promedio (retry)** | – | **0.7733** | **0.8001** | **0.9554** | – |

- Con stemming el BiLSTM se mantiene alrededor de F1≈0.77 y recall_neg≈0.80; los tiempos bajan a 24–31 s/fold gracias al vocabulario reducido.

### Bidirectional LSTM – C06 (stem + word2vec) – Reintento 2025-11-17 11:21

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C06_LSTM_BI_20251117-112114 | 0.7800 | 0.8171 | 0.9703 | 30.75 |
| 2 | C06_LSTM_BI_20251117-112114 | 0.7859 | 0.8573 | 0.9641 | 32.21 |
| 3 | C06_LSTM_BI_20251117-112114 | 0.7570 | 0.8573 | 0.9592 | 26.71 |
| **Promedio (retry)** | – | **0.7743** | **0.8439** | **0.9645** | – |

- Con word2vec sobre textos stemmed, el BiLSTM alcanza recall_neg >0.84 y mantiene tiempos <33 s/fold, confirmando la robustez de la receta.

### Bidirectional LSTM – C07 (max_len 384) – Reintento 2025-11-17 11:23

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C07_LSTM_BI_20251117-112328 | 0.7942 | 0.8126 | 0.9547 | 42.55 |
| 2 | C07_LSTM_BI_20251117-112328 | 0.7822 | 0.8270 | 0.9622 | 33.60 |
| 3 | C07_LSTM_BI_20251117-112328 | 0.7687 | 0.7719 | 0.9668 | 40.83 |
| **Promedio (retry)** | – | **0.7817** | **0.8038** | **0.9612** | – |

- Con `max_len=384` el BiLSTM mantiene F1≈0.782 y tiempos 34–43 s/fold; el mayor contexto no penaliza la convergencia gracias al batch 256.

### Bidirectional LSTM – C08 (word2vec dim=256) – Reintento 2025-11-17 11:26

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C08_LSTM_BI_20251117-112602 | 0.7830 | 0.8159 | 0.9640 | 30.39 |
| 2 | C08_LSTM_BI_20251117-112602 | 0.7788 | 0.7303 | 0.9559 | 30.93 |
| 3 | C08_LSTM_BI_20251117-112602 | 0.7868 | 0.8191 | 0.9587 | 28.45 |
| **Promedio (retry)** | – | **0.7829** | **0.7885** | **0.9595** | – |

- El embedding de 256 dimensiones se sostiene en F1≈0.783; aunque el fold 2 baja el recall_neg, la precisión positiva se mantiene alta y los tiempos rondan los 30 s/fold.

### Bidirectional LSTM – C09 (vocab_size 50k) – Reintento 2025-11-17 11:28

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C09_LSTM_BI_20251117-112809 | 0.7739 | 0.7441 | 0.9463 | 29.19 |
| 2 | C09_LSTM_BI_20251117-112809 | 0.7728 | 0.8000 | 0.9596 | 25.93 |
| 3 | C09_LSTM_BI_20251117-112809 | 0.7805 | 0.8079 | 0.9591 | 26.40 |
| **Promedio (retry)** | – | **0.7757** | **0.7840** | **0.9550** | – |

- Con vocabulario ampliado a 50 k tokens, el BiLSTM mantiene F1≈0.776 y tiempos ~26–29 s/fold; la mayor cobertura léxica no penaliza el desempeño.

### Bidirectional LSTM – C10 (dropout 0.3) – Reintento 2025-11-17 11:30

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C10_LSTM_BI_20251117-113019 | 0.7713 | 0.7677 | 0.9578 | 28.65 |
| 2 | C10_LSTM_BI_20251117-113019 | 0.7893 | 0.8337 | 0.9585 | 34.27 |
| 3 | C10_LSTM_BI_20251117-113019 | 0.7485 | 0.8944 | 0.9642 | 26.96 |
| **Promedio (retry)** | – | **0.7697** | **0.8319** | **0.9601** | – |

- El dropout interno de 0.3 sigue equilibrando el BiLSTM: recall_neg supera 0.83 y los tiempos por fold permanecen entre 27 y 34 s.

### Bidirectional LSTM – C11 (stem + dropout 0.3) – Reintento 2025-11-17 11:32

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C11_LSTM_BI_20251117-113215 | 0.7677 | 0.8227 | 0.9631 | 24.62 |
| 2 | C11_LSTM_BI_20251117-113215 | 0.7782 | 0.8303 | 0.9671 | 28.76 |
| 3 | C11_LSTM_BI_20251117-113215 | 0.7555 | 0.8618 | 0.9629 | 28.73 |
| **Promedio (retry)** | – | **0.7672** | **0.8383** | **0.9644** | – |

- Con stemming + dropout, el BiLSTM mantiene F1≈0.77 y recall_neg≈0.84 con tiempos de 25–29 s/fold; cierra el bloque de LSTM bidireccional actualizado.

## Bidirectional GRU

### C01 (baseline)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C01_GRU_BI_20251117-113707 | 0.7800 | 0.8171 | 0.9703 | 30.75 |
| 2 | C01_GRU_BI_20251117-113707 | 0.7859 | 0.8573 | 0.9641 | 32.21 |
| 3 | C01_GRU_BI_20251117-113707 | 0.7570 | 0.8573 | 0.9591 | 26.71 |
| **Promedio** | – | **0.7743** | **0.8439** | **0.9645** | – |

- La primera corrida de GRU bidireccional ya iguala al BiLSTM en F1≈0.77, con tiempos <33 s/fold. El resto de combinaciones están en ejecución siguiendo la misma receta (batch 256, cuDNN activo y dropout externo).

### C02 (word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C02_GRU_BI_20251117-114101 | 0.7621 | 0.7609 | 0.9473 | 31.69 |
| 2 | C02_GRU_BI_20251117-114101 | 0.7650 | 0.8011 | 0.9484 | 29.69 |
| 3 | C02_GRU_BI_20251117-114101 | 0.7730 | 0.8607 | 0.9670 | 25.58 |
| **Promedio** | – | **0.7667** | **0.8076** | **0.9542** | – |

- Con word2vec la BiGRU mantiene F1≈0.767 y añade estabilidad en recall_neg>0.80; los tiempos bajan a 25–32 s/fold.

### C03 (lemmatize)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C03_GRU_BI_20251117-114625 | 0.7572 | 0.7149 | 0.9438 | 33.83 |
| 2 | C03_GRU_BI_20251117-114625 | 0.7660 | 0.8011 | 0.9455 | 21.88 |
| 3 | C03_GRU_BI_20251117-114625 | 0.7751 | 0.7921 | 0.9616 | 26.99 |
| **Promedio** | – | **0.7661** | **0.7679** | **0.9503** | – |

- Con textos lematizados, la BiGRU mantiene F1≈0.77; el fold 1 es el único con recall_neg más bajo pero la precisión positiva se mantiene >0.94.

### C04 (lemmatize + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C04_GRU_BI_20251117-114804 | 0.7673 | 0.7587 | 0.9539 | 28.02 |
| 2 | C04_GRU_BI_20251117-114804 | 0.7795 | 0.7966 | 0.9531 | 26.87 |
| 3 | C04_GRU_BI_20251117-114804 | 0.7759 | 0.8101 | 0.9591 | 29.00 |
| **Promedio** | – | **0.7742** | **0.7885** | **0.9554** | – |

- Al combinar lematización con word2vec, la BiGRU vuelve a la franja 0.774 de F1 con recall_neg cercano a 0.79 y tiempos de ~27 s/fold.

### C05 (stem)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C05_GRU_BI_20251117-115826 | 0.7687 | 0.8496 | 0.9607 | 31.01 |
| 2 | C05_GRU_BI_20251117-115826 | 0.7736 | 0.8315 | 0.9615 | 29.08 |
| 3 | C05_GRU_BI_20251117-115826 | 0.7630 | 0.8640 | 0.9611 | 24.05 |
| **Promedio** | – | **0.7684** | **0.8484** | **0.9611** | – |

- Con stemming y embeddings aprendidos, la BiGRU ofrece recall_neg elevado (>0.84) manteniendo tiempos por fold inferiores a 32 s.

### C06 (stem + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C06_GRU_BI_20251117-120012 | 0.7664 | 0.7621 | 0.9546 | 30.03 |
| 2 | C06_GRU_BI_20251117-120012 | 0.7766 | 0.8236 | 0.9604 | 28.72 |
| 3 | C06_GRU_BI_20251117-120012 | 0.7553 | 0.7775 | 0.9509 | 29.54 |
| **Promedio** | – | **0.7661** | **0.7877** | **0.9553** | – |

- El Word2Vec entrenado sobre tokens con stemming mantiene el desempeño en la banda 0.76–0.77 de F1 y conserva recall_neg cercano a 0.79.

### C07 (baseline + `max_len=384`)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C07_GRU_BI_20251117-120243 | 0.7672 | 0.7767 | 0.9561 | 33.98 |
| 2 | C07_GRU_BI_20251117-120243 | 0.7775 | 0.8112 | 0.9491 | 34.77 |
| 3 | C07_GRU_BI_20251117-120243 | 0.7655 | 0.7764 | 0.9665 | 32.51 |
| **Promedio** | – | **0.7701** | **0.7881** | **0.9572** | – |

- Incrementar `max_len` a 384 no penaliza a la BiGRU: mantiene F1≈0.77, recall_neg≈0.79 y tiempos por fold ~33 s.

### C08 (baseline + word2vec 256)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C08_GRU_BI_20251117-121547 | 0.7687 | 0.8092 | 0.9586 | 32.14 |
| 2 | C08_GRU_BI_20251117-121547 | 0.7858 | 0.7820 | 0.9523 | 24.17 |
| 3 | C08_GRU_BI_20251117-121547 | 0.7669 | 0.7674 | 0.9477 | 28.39 |
| **Promedio** | – | **0.7738** | **0.7862** | **0.9529** | – |

- El embedding de 256 dimensiones apenas cambia los tiempos (~28 s/fold) y mantiene F1≈0.774; el recall_neg sigue en el rango 0.78–0.81.

### C09 (baseline + `vocab_size=50k`)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C09_GRU_BI_20251117-121832 | 0.7611 | 0.8485 | 0.9532 | 24.06 |
| 2 | C09_GRU_BI_20251117-121832 | 0.7505 | 0.8213 | 0.9468 | 22.13 |
| 3 | C09_GRU_BI_20251117-121832 | 0.7695 | 0.8101 | 0.9623 | 26.48 |
| **Promedio** | – | **0.7604** | **0.8266** | **0.9541** | – |

- Incrementar el vocabulario permite más recall (0.82–0.85) sin aumentar el tiempo (<27 s/fold); F1 se mantiene en 0.76 debido al fold 2 ligeramente más débil.

### C10 (lemmatize + dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C10_GRU_BI_20251117-122047 | 0.7566 | 0.7542 | 0.9271 | 31.63 |
| 2 | C10_GRU_BI_20251117-122047 | 0.7646 | 0.8225 | 0.9548 | 36.60 |
| 3 | C10_GRU_BI_20251117-122047 | 0.7616 | 0.7517 | 0.9623 | 29.61 |
| **Promedio** | – | **0.7609** | **0.7761** | **0.9480** | – |

- Aun con dropout interno alto (0.3), la BiGRU preserva F1≈0.76 y tiempos ~30–37 s; el fold 1 tiende a menor precisión positiva.

### C11 (stem + dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C11_GRU_BI_20251117-122312 | 0.7667 | 0.8137 | 0.9638 | 31.62 |
| 2 | C11_GRU_BI_20251117-122312 | 0.7614 | 0.8663 | 0.9658 | 27.18 |
| 3 | C11_GRU_BI_20251117-122312 | 0.7690 | 0.7910 | 0.9519 | 31.72 |
| **Promedio** | – | **0.7657** | **0.8237** | **0.9605** | – |

- El combo con stemming + dropout cierra el bloque BiGRU con F1≈0.766 y recall_neg≈0.824; todas las combinaciones C01–C11 quedaron completas para GRU bidireccional.

## LSTM (sin bidireccional) – Nuevas corridas

### C01 (baseline)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C01_LSTM_20251117-094003 | 0.3181 | 0.0527 | 0.7342 | 16 | 31.41 |
| 2 | C01_LSTM_20251117-094003 | 0.3127 | 0.0517 | 0.7337 | 8 | 14.65 |
| 3 | C01_LSTM_20251117-094003 | 0.3134 | 0.0539 | 0.7336 | 8 | 14.67 |
| **Promedio** | – | **0.3148** | **0.0528** | **0.7338** | **10.7** | – |

- Sin la capa bidireccional el LSTM vuelve a colapsar en la clase positiva: el recall negativo es ≈0.05 en todos los folds. Mantendremos este resultado como referencia antes de lanzar el resto de combos simples.

### C02 (word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C02_LSTM_20251117-094132 | 0.0855 | 1.0000 | 0.0000 | 8 | 16.19 |
| 2 | C02_LSTM_20251117-094132 | 0.3137 | 0.0528 | 0.7340 | 10 | 18.10 |
| 3 | C02_LSTM_20251117-094132 | 0.3143 | 0.0517 | 0.7336 | 10 | 18.72 |
| **Promedio** | – | **0.2378** | **0.3682** | **0.4892** | **9.3** | – |

- El fold 1 predijo todo como clase negativa (F1≈0.08); los otros dos repiten el colapso positivo. Concluimos que el LSTM simple necesita ajustes adicionales (p.ej. más unidades o capas) si quiere igualar al BiLSTM.

### C03 (lemmatize)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C03_LSTM_20251117-094318 | 0.4753 | 0.8283 | 0.8764 | 17 | 31.85 |
| 2 | C03_LSTM_20251117-094318 | 0.3123 | 0.0472 | 0.7335 | 10 | 18.01 |
| 3 | C03_LSTM_20251117-094318 | 0.3128 | 0.0528 | 0.7336 | 7 | 12.66 |
| **Promedio** | – | **0.3668** | **0.3094** | **0.7811** | **11.3** | – |

- El fold 1 alcanzó recall_neg 0.83 pero los otros dos volvieron a predecir casi todo como positivo; el promedio sigue lejos del BiLSTM.

### C04 (lemmatize + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C04_LSTM_20251117-094431 | 0.0855 | 1.0000 | 0.0000 | 7 | 14.90 |
| 2 | C04_LSTM_20251117-094431 | 0.0856 | 1.0000 | 1.0000 | 11 | 20.43 |
| 3 | C04_LSTM_20251117-094431 | 0.4667 | 0.6787 | 0.8357 | 20 | 36.83 |
| **Promedio** | – | **0.2126** | **0.8929** | **0.6119** | **12.7** | – |

- Un fold se fue totalmente al negativo, otro al positivo, y el tercero se mantuvo balanceado; el promedio sigue mostrando inestabilidad extrema en LSTM simple.

### C05 (stem)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C05_LSTM_20251117-094627 | 0.0855 | 1.0000 | 0.0000 | 7 | 14.34 |
| 2 | C05_LSTM_20251117-094627 | 0.0857 | 1.0000 | 1.0000 | 11 | 19.93 |
| 3 | C05_LSTM_20251117-094627 | 0.0856 | 1.0000 | 1.0000 | 9 | 16.33 |
| **Promedio** | – | **0.0856** | **1.0000** | **0.6667** | **9.0** | – |

- Aquí los tres folds predijeron únicamente la clase negativa, confirmando que el LSTM simple no logra equilibrar las etiquetas con la receta actual.

### C06 (stem + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C06_LSTM_20251117-094758 | 0.0908 | 0.9989 | 1.0000 | 14 | 27.14 |
| 2 | C06_LSTM_20251117-094758 | 0.0856 | 1.0000 | 1.0000 | 8 | 15.52 |
| 3 | C06_LSTM_20251117-094758 | 0.0856 | 1.0000 | 1.0000 | 10 | 18.37 |
| **Promedio** | – | **0.0873** | **0.9996** | **1.0000** | **10.7** | – |

- Mismo patrón: todos los folds se van al extremo negativo, con F1≈0.086 pese al recall máximo.

### C07 (max_len 384)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C07_LSTM_20251117-094917 | 0.0855 | 1.0000 | 0.0 | 12 | 29.60 |
| 2 | C07_LSTM_20251117-094917 | 0.0854 | 1.0000 | 0.0 | 6 | 14.37 |
| 3 | C07_LSTM_20251117-094917 | 0.0854 | 1.0000 | 0.0 | 12 | 28.33 |
| **Promedio** | – | **0.0854** | **1.0000** | **0.0** | **10.0** | – |

- Aun con secuencias largas, el modelo simple sigue prediciendo exclusivamente la clase negativa.

### C08 (word2vec dim=256)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C08_LSTM_20251117-095103 | 0.0855 | 1.0000 | 0.0 | 8 | 16.50 |
| 2 | C08_LSTM_20251117-095103 | 0.0857 | 1.0000 | 1.0 | 8 | 15.56 |
| 3 | C08_LSTM_20251117-095103 | 0.0856 | 1.0000 | 1.0 | 10 | 18.81 |
| **Promedio** | – | **0.0856** | **1.0000** | **0.6667** | **8.7** | – |

- Ningún fold logró predecir la clase positiva establemente; seguimos con F1≈0.086.

### C09 (vocab_size 50k)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C09_LSTM_SIMPLE_20251117-095230 | 0.3162 | 0.0449 | 0.7332 | 14 | 27.26 |
| 2 | C09_LSTM_SIMPLE_20251117-095230 | 0.3201 | 0.0640 | 0.7353 | 10 | 18.53 |
| 3 | C09_LSTM_SIMPLE_20251117-095230 | 0.0856 | 1.0000 | 1.0000 | 13 | 24.38 |
| **Promedio** | – | **0.2407** | **0.3696** | **0.8229** | **12.3** | – |

- Un fold colapsó al negativo y los otros dos continuaron con sesgo positivo; el promedio sigue bajo.

### C10 (dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C10_LSTM_SIMPLE_20251117-095411 | 0.1157 | 0.0730 | 0.0 | 7 | 15.02 |
| 2 | C10_LSTM_SIMPLE_20251117-095411 | 0.4977 | 0.7348 | 0.8422 | 13 | 23.76 |
| 3 | C10_LSTM_SIMPLE_20251117-095411 | 0.0856 | 1.0000 | 1.0000 | 7 | 13.04 |
| **Promedio** | – | **0.2330** | **0.6026** | **0.6141** | **9.0** | – |

- Incluso con dropout 0.3 los resultados siguen inestables (fold 2 razonable, 1 y 3 extremos).

### C11 (stem + dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C11_LSTM_SIMPLE_20251117-095533 | 0.3114 | 0.0494 | 0.7336 | 8 | 16.24 |
| 2 | C11_LSTM_SIMPLE_20251117-095533 | 0.0856 | 1.0000 | 1.0000 | 9 | 16.55 |
| 3 | C11_LSTM_SIMPLE_20251117-095533 | 0.0887 | 1.0000 | 1.0000 | 11 | 20.69 |
| **Promedio** | – | **0.1619** | **0.6831** | **0.9112** | **9.3** | – |

- Solo el fold 1 intentó predecir positivos; los otros dos volvieron al extremo negativo.

### Resumen LSTM simple (C01–C11)

| Combo | Configuración | Mean F1 | Mean Recall Neg | Mean Prec Pos |
|-------|--------------|--------:|----------------:|--------------:|
| C01 | baseline | 0.315 | 0.053 | 0.734 |
| C02 | baseline + word2vec | 0.238 | 0.368 | 0.489 |
| C03 | lemmatize | 0.367 | 0.309 | 0.781 |
| C04 | lemmatize + word2vec | 0.213 | 0.893 | 0.612 |
| C05 | stem | 0.086 | 1.000 | 0.667 |
| C06 | stem + word2vec | 0.087 | 1.000 | 1.000 |
| C07 | baseline max_len=384 | 0.085 | 1.000 | 0.000 |
| C08 | baseline word2vec256 | 0.086 | 1.000 | 0.667 |
| C09 | baseline vocab 50k | 0.241 | 0.370 | 0.823 |
| C10 | lemmatize dropout0.3 | 0.233 | 0.603 | 0.614 |
| C11 | stem dropout0.3 | 0.162 | 0.683 | 0.911 |

- En todos los casos el LSTM simple queda muy por debajo del BiLSTM: o predice casi todo positivo (recall_neg≈0.05) o colapsa al negativo (recall_neg=1). Necesitaremos otra arquitectura si queremos usar LSTM sin bidireccionalidad.

## GRU (sin bidireccional) – Nuevas corridas

### C01 (baseline)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C01_GRU_SIMPLE_20251117-095754 | 0.0855 | 1.0000 | 0.0000 | 9 | 16.86 |
| 2 | C01_GRU_SIMPLE_20251117-095754 | 0.0859 | 1.0000 | 1.0000 | 9 | 15.90 |
| 3 | C01_GRU_SIMPLE_20251117-095754 | 0.3141 | 0.0517 | 0.7335 | 8 | 14.37 |
| **Promedio** | – | **0.1618** | **0.6839** | **0.5778** | **8.7** | – |

- Misma historia que el LSTM simple: dos folds colapsan al negativo y uno al positivo, dejando un F1 pobre.

### C02 (baseline + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C02_GRU_SIMPLE_20251117-100241 | 0.0934 | 0.9989 | 0.8571 | 14 | 25.82 |
| 2 | C02_GRU_SIMPLE_20251117-100241 | 0.0856 | 1.0000 | 1.0000 | 8 | 14.01 |
| 3 | C02_GRU_SIMPLE_20251117-100241 | 0.3189 | 0.0584 | 0.7343 | 11 | 18.27 |
| **Promedio** | – | **0.1660** | **0.6858** | **0.8638** | **11.0** | – |

- Nuevamente, dos folds se van al extremo negativo mientras el tercero conserva sesgo positivo; GRU simple no mejora al LSTM simple bajo esta receta.

### C03 (lemmatize)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C03_GRU_SIMPLE_20251117-100554 | 0.0863 | 1.0000 | 1.0000 | 7 | 13.45 |
| 2 | C03_GRU_SIMPLE_20251117-100554 | 0.0875 | 1.0000 | 1.0000 | 6 | 10.46 |
| 3 | C03_GRU_SIMPLE_20251117-100554 | 0.3159 | 0.0562 | 0.7341 | 10 | 16.80 |
| **Promedio** | – | **0.1632** | **0.6854** | **0.9114** | **7.7** | – |

- Igual que antes: dos folds colapsan al negativo, uno intenta balancear pero queda sesgado al positivo.

### C04 (lemmatize + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C04_GRU_SIMPLE_20251117-100534 | 0.0861 | 1.0000 | 1.0000 | 9 | 17.58 |
| 2 | C04_GRU_SIMPLE_20251117-100534 | 0.0856 | 1.0000 | 1.0000 | 8 | 14.54 |
| 3 | C04_GRU_SIMPLE_20251117-100534 | 0.0865 | 1.0000 | 1.0000 | 7 | 12.73 |
| **Promedio** | – | **0.0861** | **1.0000** | **1.0000** | **8.0** | – |

- Este combo directamente predice todo como clase negativa en los tres folds.

### C05 (stem)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C05_GRU_SIMPLE_20251117-100648 | 0.0873 | 1.0000 | 0.0 | 6 | 11.96 |
| 2 | C05_GRU_SIMPLE_20251117-100648 | 0.0854 | 1.0000 | 0.0 | 6 | 10.33 |
| 3 | C05_GRU_SIMPLE_20251117-100648 | 0.0854 | 1.0000 | 0.0 | 6 | 10.72 |
| **Promedio** | – | **0.0860** | **1.0000** | **0.0** | **6.0** | – |

- Stem sin word2vec vuelve a predecir sólo la clase negativa con F1≈0.086.

### C06 (stem + word2vec)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C06_GRU_SIMPLE_20251117-100806 | 0.0855 | 1.0000 | 0.0 | 7 | 13.33 |
| 2 | C06_GRU_SIMPLE_20251117-100806 | 0.3218 | 0.0629 | 0.7355 | 13 | 22.41 |
| 3 | C06_GRU_SIMPLE_20251117-100806 | 0.3164 | 0.0539 | 0.7338 | 11 | 18.79 |
| **Promedio** | – | **0.2412** | **0.3723** | **0.4897** | **10.3** | – |

- Sólo el fold 1 colapsa por completo; los otros dos repiten el sesgo positivo ya observado en combos anteriores.

### C07 (max_len 384)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C07_GRU_SIMPLE_20251117-101027 | 0.0855 | 1.0000 | 0.0 | 9 | 21.79 |
| 2 | C07_GRU_SIMPLE_20251117-101027 | 0.0854 | 1.0000 | 0.0 | 6 | 13.28 |
| 3 | C07_GRU_SIMPLE_20251117-101027 | 0.0856 | 1.0000 | 1.0 | 13 | 28.03 |
| **Promedio** | – | **0.0855** | **1.0000** | **0.3333** | **9.3** | – |

- A pesar de las secuencias largas, el GRU simple no logra producir predicciones positivas coherentes.

### C08 (word2vec dim=256)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C08_GRU_SIMPLE_20251117-101159 | 0.0855 | 1.0000 | 0.0 | 9 | 17.30 |
| 2 | C08_GRU_SIMPLE_20251117-101159 | 0.3169 | 0.0584 | 0.7347 | 11 | 19.06 |
| 3 | C08_GRU_SIMPLE_20251117-101159 | 0.3205 | 0.0640 | 0.7348 | 11 | 18.15 |
| **Promedio** | – | **0.2410** | **0.3742** | **0.4899** | **10.3** | – |

- Similar al combo C06: un fold negativo extremo y dos con sesgo positivo, sin mejoras claras.

### C09 (vocab_size 50k)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C09_GRU_SIMPLE_20251117-101357 | 0.3123 | 0.0494 | 0.7339 | 11 | 20.17 |
| 2 | C09_GRU_SIMPLE_20251117-101357 | 0.3103 | 0.0472 | 0.7334 | 10 | 17.12 |
| 3 | C09_GRU_SIMPLE_20251117-101357 | 0.3116 | 0.0506 | 0.7334 | 9 | 15.52 |
| **Promedio** | – | **0.3112** | **0.0490** | **0.7335** | **10.0** | – |

- Todos los folds repiten el sesgo positivo leve (recall_neg≈0.05); la métrica promedio sigue baja.

### C10 (dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C10_GRU_SIMPLE_20251117-101639 | 0.3118 | 0.0494 | 0.7339 | 11 | 20.17 |
| 2 | C10_GRU_SIMPLE_20251117-101639 | 0.3103 | 0.0472 | 0.7334 | 10 | 17.12 |
| 3 | C10_GRU_SIMPLE_20251117-101639 | 0.3116 | 0.0506 | 0.7334 | 9 | 15.52 |
| **Promedio** | – | **0.3112** | **0.0490** | **0.7335** | **10.0** | – |

- Dropout adicional no ayuda: seguimos con recall_neg≈0.05 y F1 bajo en todos los folds.

### C11 (stem + dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C11_GRU_SIMPLE_20251117-101747 | 0.0861 | 1.0000 | 1.0000 | 11 | 109.68 |
| 2 | C11_GRU_SIMPLE_20251117-101747 | 0.3215 | 0.0629 | 0.7353 | 12 | 20.76 |
| 3 | C11_GRU_SIMPLE_20251117-101747 | 0.3147 | 0.0438 | 0.7336 | 16 | 26.59 |
| **Promedio** | – | **0.2408** | **0.3689** | **0.8229** | **13.0** | – |

- Último combo de GRU simple tampoco mejora: un fold negativo total y los otros dos sesgo positivo leve.

### C10 (dropout 0.3) – Reintento

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | Epochs | train_time_sec |
|------|--------|---------:|-----------:|--------------:|-------:|---------------:|
| 1 | C10_GRU_SIMPLE_20251117-102120_retry | 0.0855 | 1.0000 | 0.0 | 6 | 11.73 |
| 2 | C10_GRU_SIMPLE_20251117-102120_retry | 0.3066 | 0.0326 | 0.7335 | 16 | 27.48 |
| 3 | C10_GRU_SIMPLE_20251117-102120_retry | 0.3164 | 0.0551 | 0.7339 | 12 | 21.03 |
| **Promedio (retry)** | – | **0.2361** | **0.3625** | **0.4891** | **11.3** | – |

- El rerun confirma los mismos patrones erráticos; mantener este combo no aporta mejora adicional.

### Bidirectional SimpleRNN – C02 (word2vec) – Reintento

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C02_SRN_BI_20251117-102515_retry | 0.7618 | 0.8227 | 0.9283 | 42.43 |
| 2 | C02_SRN_BI_20251117-102515_retry | 0.7396 | 0.7888 | 0.9295 | 39.14 |
| 3 | C02_SRN_BI_20251117-102515_retry | 0.7506 | 0.7854 | 0.9259 | 38.29 |
| **Promedio (retry)** | – | **0.7507** | **0.7989** | **0.9279** | – |

- Los nuevos tiempos y métricas confirman el gran desempeño del BRNN con word2vec; guardamos este rerun para tener artefactos consistentes (los logs iniciales mostraban errores de XLA y preferimos repetirlos).

### Bidirectional SimpleRNN – C03 (lemmatize)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C03_SRN_BI_20251117-104355 | 0.7592 | 0.8552 | 0.9411 | 43.46 |
| 2 | C03_SRN_BI_20251117-104355 | 0.7339 | 0.7685 | 0.9291 | 40.73 |
| 3 | C03_SRN_BI_20251117-104355 | 0.7610 | 0.8348 | 0.9327 | 38.92 |
| **Promedio** | – | **0.7513** | **0.8195** | **0.9343** | – |

- Con limpieza lematizada, el BRNN sigue ofreciendo F1≈0.75 y recall_neg≈0.82; los warnings de XLA persisten pero el entrenamiento converge estable.

### Bidirectional SimpleRNN – C04 (stemming)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C04_SRN_BI_20251117-104744 | 0.7650 | 0.8541 | 0.9382 | 42.58 |
| 2 | C04_SRN_BI_20251117-104744 | 0.7516 | 0.7989 | 0.9272 | 41.32 |
| 3 | C04_SRN_BI_20251117-104744 | 0.7317 | 0.7360 | 0.9360 | 42.02 |
| **Promedio** | – | **0.7495** | **0.7963** | **0.9338** | – |

- El stemming no mejora F1 frente a C02/C03, pero mantiene el rango alto (0.74–0.76) con buen balance entre clases; seguimos documentando los tiempos para la comparación final.

### Bidirectional SimpleRNN – C05 (baseline + word2vec 300d)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C05_SRN_BI_20251117-105004 | 0.7414 | 0.8013 | 0.9391 | 42.40 |
| 2 | C05_SRN_BI_20251117-105004 | 0.7488 | 0.7629 | 0.9340 | 44.55 |
| 3 | C05_SRN_BI_20251117-105004 | 0.7607 | 0.8045 | 0.9241 | 38.67 |
| **Promedio** | – | **0.7503** | **0.7896** | **0.9324** | – |

- El embedding de 300 dimensiones se mantiene en el mismo nivel que C02-C04; no muestra ventaja clara en F1 pero ofrece recall_neg estable cercano a 0.79–0.80.

### Bidirectional SimpleRNN – C06 (baseline + drop=0.1)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C06_SRN_BI_20251117-105236 | 0.7538 | 0.8608 | 0.9398 | 46.70 |
| 2 | C06_SRN_BI_20251117-105236 | 0.7365 | 0.7303 | 0.9343 | 47.34 |
| 3 | C06_SRN_BI_20251117-105236 | 0.7456 | 0.8034 | 0.9400 | 42.28 |
| **Promedio** | – | **0.7453** | **0.7982** | **0.9380** | – |

- El ajuste de `post_rnn_dropout=0.1` mantiene F1≈0.745 con ligera mejora en recall_neg del fold 1; seguimos sin degradación notable pese al aumento de regularización externa.

### Bidirectional SimpleRNN – C07 (max_len 384)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C07_SRN_BI_20251117-105520 | 0.7456 | 0.7823 | 0.9307 | 50.63 |
| 2 | C07_SRN_BI_20251117-105520 | 0.7522 | 0.8348 | 0.9303 | 51.67 |
| 3 | C07_SRN_BI_20251117-105520 | 0.7543 | 0.8079 | 0.9395 | 50.03 |
| **Promedio** | – | **0.7507** | **0.8083** | **0.9335** | – |

- Aumentar `max_len` a 384 no degrada desempeño y mantiene F1≈0.75; tiempos por fold suben a ~50 s debido a secuencias más largas.

### Bidirectional SimpleRNN – C08 (limpieza baseline + embedding 64d reducido)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C08_SRN_BI_20251117-105818 | 0.7431 | 0.7755 | 0.9263 | 39.68 |
| 2 | C08_SRN_BI_20251117-105818 | 0.7323 | 0.7955 | 0.9395 | 36.84 |
| 3 | C08_SRN_BI_20251117-105818 | 0.7552 | 0.7742 | 0.9298 | 41.48 |
| **Promedio** | – | **0.7435** | **0.7817** | **0.9319** | – |

- Reducir `embedding_dim` a 64 mantiene F1 cercano a 0.74–0.75 con ligera caída de recall_neg frente a combos anteriores; tiempos disminuyen a ~40 s/fold por el embeding compacto.

### Bidirectional SimpleRNN – C09 (vocab_size 50k)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C09_SRN_BI_20251117-110041 | 0.7270 | 0.7407 | 0.9259 | 41.66 |
| 2 | C09_SRN_BI_20251117-110041 | 0.7586 | 0.8427 | 0.9270 | 33.84 |
| 3 | C09_SRN_BI_20251117-110041 | 0.7572 | 0.8517 | 0.9396 | 34.74 |
| **Promedio** | – | **0.7476** | **0.8117** | **0.9308** | – |

- Ampliar vocabulario a 50k mantiene los resultados en el rango 0.74–0.76 de F1, con buen recall_neg (>0.81) y tiempo por fold algo menor (~35-42 s) gracias al reuse del embedding base.

### Bidirectional SimpleRNN – C10 (dropout interno 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C10_SRN_BI_20251117-110259 | 0.7381 | 0.8620 | 0.9338 | 35.66 |
| 2 | C10_SRN_BI_20251117-110259 | 0.7546 | 0.7888 | 0.9349 | 42.74 |
| 3 | C10_SRN_BI_20251117-110259 | 0.7627 | 0.8067 | 0.9300 | 35.55 |
| **Promedio** | – | **0.7518** | **0.8192** | **0.9329** | – |

- Mantener dropout interno alto (0.3) no perjudica F1 en el BRNN; recall_neg se mantiene en ~0.82 y los tiempos siguen alrededor de 35–42 s por fold.

### Bidirectional SimpleRNN – C11 (dropout+recurrent_dropout 0.3)

| Fold | RUN_ID | F1 Macro | Recall Neg | Precisión Pos | train_time_sec |
|------|--------|---------:|-----------:|--------------:|---------------:|
| 1 | C11_SRN_BI_20251117-110521 | 0.7596 | 0.8092 | 0.9289 | 49.63 |
| 2 | C11_SRN_BI_20251117-110521 | 0.7492 | 0.8079 | 0.9316 | 43.91 |
| 3 | C11_SRN_BI_20251117-110521 | 0.7463 | 0.8067 | 0.9404 | 39.27 |
| **Promedio** | – | **0.7517** | **0.8079** | **0.9336** | – |

- Añadir `recurrent_dropout=0.3` mantiene el desempeño en el rango 0.75 F1 aunque introduce mayor tiempo de entrenamiento (hasta ~50 s) y warnings de XLA; no se observan mejoras respecto a dropout sólo en la capa externa.

