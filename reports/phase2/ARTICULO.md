# Bidirectional Recurrent Networks for Sentiment Analysis of Andalusian Hotel Reviews

## Resumen
Presentamos un framework reproducible para clasificar el sentimiento en 112 mil reseñas hoteleras andaluzas empleando únicamente RNNs (SimpleRNN, LSTM, GRU) con embeddings aprendidos o Word2Vec propio. El pipeline separa la preparación (limpieza, lematización o stemming, tokenización y entrenamiento Word2Vec) del entrenamiento por arquitectura, permitiendo depurar experimentos sin dependencias cruzadas. Ejecutamos 66 combinaciones (C01–C11 × modelos, con y sin bidireccionalidad) y registramos F1 macro, `recall_neg`, `precision_pos`, matrices de confusión y tiempo de entrenamiento por fold. Las variantes bidireccionales son claramente superiores: BiLSTM alcanza F1=0.785 (`recall_neg=0.823`) y BiGRU 0.770 (`recall_neg=0.848`) con menos de 45 s por fold gracias a cuDNN.

## Impacto
El monitoreo del sentimiento turístico en Andalucía requiere modelos que prioricen la detección de reseñas negativas pese al fuerte desbalance. Demostramos que una plataforma basada en RNNs bidireccionales con embeddings propios satisface este requisito con tiempos compatibles con una sola GPU de escritorio. El pipeline garantiza trazabilidad (RUN_ID único, caches compartidos, bitácora) y permite que observatorios locales o pymes repliquen/ajusten experimentos sin depender de transformadores pesados.

## Palabras clave
Sentiment analysis, Spanish NLP, recurrent neural networks, bidirectional LSTM, Word2Vec, tourism analytics.

## 1. Introducción
Durante la Fase 1 del proyecto (ver `docs/resources/entregas/Proyecto_DL_Fase1.pdf`) consolidamos el dataset **Big_AHR**, compuesto por 112 mil reseñas de hoteles andaluces, con etiquetas originales $\{0, 3, 1\}$ (negativo, neutro, positivo) y un fuerte desbalance (≈66 % positivas, 13 % negativas, 21 % neutrales). Este análisis inicial dejó dos lecciones principales: (i) cualquier métrica debe ser macro-promediada para no favorecer la clase mayoritaria y (ii) la detección de reseñas negativas es prioritaria para los casos de uso de negocio (alerta temprana y reputación).

La Fase 2 se diseñó como un plan experimental controlado donde sólo se permiten **redes recurrentes clásicas** (SimpleRNN, LSTM, GRU, con o sin bidireccionalidad) y embeddings aprendidos desde cero o entrenados in situ mediante Word2Vec; se excluyen transformadores o embeddings preentrenados externos para respetar las restricciones de la asignatura. Además, se exige que cada experimento sea trazable (RUN_ID único, cachés reproducibles, métricas completas y tiempos por fold) y que los resultados se documenten en informes y artículos de manera similar a una entrega académica formal (IEEE).

El presente artículo sintetiza ese flujo completo: describe cómo preparamos los datos, qué variables experimentales se exploraron (limpieza, representaciones, hiperparámetros), cómo se estructuró la arquitectura y cuáles fueron los aprendizajes clave al comparar decenas de combinaciones. La intención es que el lector pueda replicar la investigación con los scripts proporcionados o extenderla con nuevas variantes (p. ej. mecanismos de atención o pérdidas sensibles al costo).

## 2. Preparación del conjunto de datos
- `scripts/prepare_dataset.py` es el eje de la fase de datos. Lee `data/Big_AHR.csv`, aplica la técnica de limpieza solicitada (baseline, lematización con spaCy `es_core_news_sm`, stemming con Snowball), genera las representaciones (tokenizer de Keras con vocabularios de 30k o 50k palabras y, opcionalmente, un modelo Word2Vec entrenado sobre el mismo corpus) y produce folds estratificados con `seed=42`.
- Para asegurar reproducibilidad, la limpieza y los índices de folds se almacenan en `artifacts/cache/` (`clean_<modo>.joblib`, `folds_seed42_k3.json`), lo que evita recalcular procesos costosos cada vez que se lanza un experimento.
- Cada combinación C01–C11 genera un directorio en `artifacts/data/<combo>/` con `data.npz` por fold, `tokenizer.json`, `embedding_matrix.npy` (cuando aplica) y `metadata.json` donde se registran `max_len`, `vocab_size`, notas y RUN_ID.
- Las decisiones de esta fase se sustentan en la primera entrega: las longitudes de 256 tokens cubren el percentil 95 del corpus y las variantes `max_len=384` se reservaron para explorar si ampliar la ventana temporal agrega información relevante.

## 3. Diseño experimental
- El plan experimental, descrito originalmente en `docs/phase2/PLAN_EXPERIMENTAL.md`, se articula en tres etapas: (1) preparación única de datos/folds, (2) entrenamiento independiente por arquitectura y (3) registro/análisis de resultados. Cada etapa corresponde a un script (`prepare_dataset.py`, `train_<modelo>.py`, utilidades de bitácora) y se puede ejecutar de manera aislada para facilitar troubleshooting.
- Se definieron 11 **experimentos base (C01–C11)** que cruzan las variables clave: técnica de limpieza (baseline/lemma/stem), representación (embedding aprendido vs. Word2Vec 128/256), ajustes estructurales (`max_len=384`, `vocab_size=50k`) y regularización (`dropout` post-RNN de 0.2 o 0.3). Todas las arquitecturas se entrenaron sobre los mismos folds para garantizar comparabilidad.
- Los scripts de entrenamiento (`train_simple_rnn.py`, `train_lstm.py`, `train_gru.py`) reciben parámetros como `--bidirectional`, `--dropout`, `--folds` y `--tag`. Cada corrida produce los historiales (`fold_*_history.json`), métricas detalladas (`fold_*_metrics.json` con F1 macro, `recall_neg`, `precision_pos`, matriz de confusión, tiempo y épocas efectivas) y una entrada en `artifacts/experiments.csv`.
- La **bitácora** (`BITACORA.md`) funciona como cuaderno de laboratorio: anota fechas, RUN_ID, métricas destacadas y decisiones tomadas (p. ej., cuándo se adoptó la arquitectura optimizada de LSTM para reducir tiempos de 11 min/fold a 30 s/fold). Esto replica la narrativa de la primera entrega, donde cada hallazgo se documentaba junto a su impacto en el siguiente ciclo.

## 4. Arquitecturas
- **Secuencia de capas**: `Embedding (dim=128)` → capa recurrente única (SimpleRNN 128 u., LSTM/GRU 64 u.) → `Dropout` externo (`post_rnn_dropout=0.2`) → `Dense` softmax (3 clases).
- **Bidireccionalidad**: opcional mediante `Bidirectional` para capturar contexto en ambas direcciones (SimpleRNN-BI, LSTM-BI, GRU-BI).
- **Regularización/cuDNN**: en LSTM/GRU fijamos `dropout=recurrent_dropout=0` dentro de la celda y trasladamos la regularización al `post_rnn_dropout`. Esto habilita cuDNN y reduce los tiempos por fold a 20–45 s (vs. >10 min en la versión previa).
- **Pesos y callbacks**: `compute_class_weights` con multiplicador 1.2 para la clase 0, `EarlyStopping (patience=5, min_delta=0.002)` y `ReduceLROnPlateau (factor=0.5)`.
- **Variables exploradas**: limpieza (`baseline`, `lemmatize`, `stem`), embeddings (aprendidos vs Word2Vec 128/256/300), longitud (`max_len=256/384`), vocabulario (`30k/50k`), dropout externo elevado (0.3) y batch size (128 vs 256 en variantes bidireccionales).
- **Justificación**: este diseño minimalista replica el pipeline descrito en la primera entrega (Embedding + RNN + Dense) pero introduce mejoras prácticas (dropout externo, pesos de clase, entrenamiento separado por arquitectura) que permiten medir con precisión el impacto de cada factor sin mezclarlo con cambios en la infraestructura.

## 5. Resultados y discusión
| Modelo | Experimento | Limpieza | Embedding | Notas | F1 Macro | `recall_neg` | `precision_pos` | Tiempo ≈ s/fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleRNN | `C01_SRN_LONG_20251117-004240` | baseline | learned | LR=5e-4, peso clase 0=1.2 | 0.321 | 0.530 | 0.752 | 61 |
| Bi-SimpleRNN | `C02_SRN_BI_20251117-080430` | baseline | word2vec 128 | Bidireccional, batch 256 | 0.751 | 0.826 | 0.928 | 65 |
| LSTM | `C05_LSTM_20251117-144218` | stem | learned | LSTM 64 u., dropout externo 0.2 | 0.239 | 0.372 | 0.823 | **24** |
| Bi-LSTM | `C02_LSTM_BI_20251117-111045` | baseline | word2vec 128 | Bidireccional, cuDNN | **0.785** | **0.823** | **0.964** | **31** |
| GRU | `C06_GRU_SIMPLE_20251117-100806` | stem | word2vec 128 | LR=5e-4, peso clase 0=1.2 | 0.241 | 0.372 | 0.490 | 18 |
| Bi-GRU | `C05_GRU_BI_20251117-115826` | stem | learned | Bidireccional, dropout externo 0.3 | 0.768 | 0.848 | 0.961 | 28 |

- Las versiones simples fallan al cubrir la clase negativa a pesar de los pesos; las bidireccionales equilibran recall sin sacrificar precisión y mantienen `precision_pos` >0.92.
- Word2Vec entrenado sobre los textos objetivo mejora estabilidad en BiLSTM; stemming favorece a BiGRU, especialmente para `recall_neg`.
- Registrar tiempos reales permitió descartar configuraciones inviables (LSTM simple a 11 min/fold) y repetirlas con la arquitectura optimizada (64 u., dropout externo), dejando tiempos de 25–45 s/fold en toda la familia LSTM.

### 5.1 Análisis de variables
- **Limpieza**: stemming (C05/C06) elevó `recall_neg` tanto en LSTM como en BiGRU, mientras que la lematización estabilizó las predicciones de BiLSTM ante vocabularios grandes.
- **Embeddings**: Word2Vec de 128/256 dims aportó estabilidad en BiLSTM (C02, C08) sin penalizar tiempos. En LSTM simple, los embeddings aprendidos rindieron mejor cuando se fijó `post_rnn_dropout` y se mantuvo el vocabulario en 30k.
- **Longitud/vocabulario**: Extender las secuencias a 384 tokens (C07) o el vocabulario a 50k (C09) incrementó el costo en SimpleRNN/GRU pero permitió diagnósticos específicos; en LSTM simple el efecto principal fue sobre tiempos (45 s/fold en C07) sin mejora tangible en F1.
- **Dropout externo**: Subirlo a 0.3 (C10/C11) ayudó a BiLSTM a evitar sobreajuste con textos limpios o stemmed; en LSTM simple, la agresividad extra redujo aún más la precisión positiva, reforzando la necesidad de la versión bidireccional.

## 6. Conclusiones
- BiLSTM (C02) es el modelo recomendado para despliegue; BiGRU es la alternativa cuando se prioriza `recall_neg`.
- El pipeline modular permite agregar nuevas variantes (ej. atención, otros embeddings) sin repetir la preparación de datos.
- Documentamos cada experimento (RUN_ID, métricas, tiempo) para trazabilidad completa.

## 7. Trabajo futuro
1. Incorporar mecanismos de atención ligera para elevar `recall_neg` >0.85 sin aumentar latencia.
2. Implementar `scripts/compare_results.py` que genere automáticamente tablas y gráficas desde `artifacts/experiments.csv`.
3. Evaluar pérdidas sensibles al costo (focal loss, class-balanced loss) en los modelos bidireccionales.

## Referencias
1. D. Martínez y A. Flores, “Dataset resumen fase 1: Reseñas hoteleras andaluzas,” DeepLearningP2, 2025.
2. M. Honnibal et al., “spaCy: Industrial-strength NLP in Python,” Zenodo, 2020.
3. T. Mikolov et al., “Efficient estimation of word representations in vector space,” Proc. ICLR Workshops, 2013.
4. F. Chollet et al., “Keras,” https://keras.io, 2015.
