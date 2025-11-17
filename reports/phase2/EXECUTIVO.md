# Resumen Ejecutivo – Fase 2 Sentiment RNNs

## Objetivo
Construir un pipeline reproducible para clasificar el sentimiento (negativo, neutro, positivo) en las 112 k reseñas de hoteles andaluces descritas en la Fase 1 (`docs/resources/entregas/Proyecto_DL_Fase1.pdf`). El dataset presenta un desbalance pronunciado (≈66 % positivas, 13 % negativas, 21 % neutrales), por lo que la métrica oficial es F1 macro; complementamos con `recall_neg`, `precision_pos`, matrices de confusión y tiempos reales de entrenamiento.

## Enfoque
- **Preparación modular**: `scripts/prepare_dataset.py` limpia (baseline, lematización, stemming), tokeniza, entrena Word2Vec y genera folds estratificados reutilizables.
- **Entrenamiento independiente por modelo**: scripts dedicados para SimpleRNN, LSTM y GRU (con bandera `--bidirectional`). Todos usan callbacks, batch 256 y registro automático en `artifacts/experiments.csv` + `BITACORA.md`.
- **Cobertura experimental**: 11 combinaciones (C01–C11) que atraviesan limpieza, embedding, `max_len`, `vocab_size` y dropout. Cada corrida adopta `RUN_ID = <combo>_<modelo>_<timestamp>`.

## Hallazgos principales
- Las variantes **bidireccionales** son imprescindibles. SimpleRNN/LSTM/GRU unidireccionales colapsan hacia la clase positiva pese a los pesos de clase.
- **BiLSTM + Word2Vec (C02)** es el baseline recomendado: F1 macro promedio **0.785**, `recall_neg=0.823`, `precision_pos=0.964`, con ~35 s por fold tras habilitar cuDNN (dropout externo).
- **BiGRU (C05/C06)** ofrece desempeño similar con `recall_neg` ligeramente superior (≈0.84) y ~33 s por fold, ideal cuando se prioriza detectar críticas negativas.
- El uso de **dropout externo** y batch 256 redujo el tiempo de LSTM de 400 s/época a 6 s/época. Sin ello, los experimentos eran inviables.
- Lematización y stemming impactan de forma diferente según el modelo: stemming ayuda a BiGRU, lematización estabiliza BiLSTM combinado con Word2Vec.
- Mejores experimentos registrados:
  - SimpleRNN `C01_SRN_LONG_20251117-004240` (F1=0.321, `recall_neg`=0.53, `precision_pos`=0.75, ~61 s/fold).
  - SimpleRNN-BI `C02_SRN_BI_20251117-080430` (F1=0.751, `recall_neg`=0.83, `precision_pos`=0.93, ~65 s/fold).
  - LSTM `C05_LSTM_20251117-144218` (F1=0.239, `recall_neg`=0.37, `precision_pos`=0.82, 24 s/fold).
  - LSTM-BI `C02_LSTM_BI_20251117-111045` (F1=0.785, `recall_neg`=0.82, `precision_pos`=0.96, 31 s/fold).
  - GRU `C06_GRU_SIMPLE_20251117-100806` (F1=0.241, `recall_neg`=0.37, `precision_pos`=0.49, 18 s/fold).
  - GRU-BI `C05_GRU_BI_20251117-115826` (F1=0.768, `recall_neg`=0.85, `precision_pos`=0.96, 28 s/fold).

## Arquitectura y variables exploradas
- **Capas**: `Embedding` (128 dims) → RNN (SimpleRNN 128 u., LSTM/GRU 64 u.) → `Dropout` externo (0.2) → `Dense` softmax de 3 salidas.
- **CuDNN**: forzamos `dropout=recurrent_dropout=0` dentro de LSTM/GRU para activar el kernel acelerado y trasladamos la regularización al `post_rnn_dropout`.
- **Pesos y callbacks**: `class_weight` con multiplicador 1.2 para la clase 0, `EarlyStopping` (`patience=5`, `min_delta=0.002`) y `ReduceLROnPlateau`.
- **Variables probadas**: limpieza (`baseline`, `lemmatize`, `stem`), embeddings (aprendidos vs Word2Vec 128/256/300), longitud (`max_len=256/384`), vocabulario (30k/50k), dropout externo (0.2 / 0.3).
- **Actualización LSTM**: todas las combinaciones C01–C11 se re-corrieron el 17/11/2025 con esta arquitectura, dejando tiempos por fold de 20–45 s y métricas consistentes con los demás modelos.

## Entregables
- `reports/phase2/INFORME_COMPLETO.md`: tablas fold a fold, checklist de combinaciones, narrativas por modelo.
- `reports/phase2/ARTICULO.md`: artículo tipo IEEE en Markdown con pipeline, resultados comparativos y discusión.
- `BITACORA.md`: registro cronológico de corridas, ajustes y tiempos.

## Próximos pasos
1. Automatizar gráficas y tablas desde `artifacts/experiments.csv` (script de comparación pendiente).
2. Incorporar capas de atención ligera o pérdidas focales para elevar `recall_neg` sin sacrificar precisión.
3. Empaquetar scripts en plantillas (`Makefile`/`tox`) que permitan lanzar bloques completos de combinaciones con un solo comando.
