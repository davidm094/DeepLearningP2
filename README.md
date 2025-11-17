# DeepLearningP2 – Sentiment RNN Experiments

Proyecto de clasificación de sentimiento para reseñas de hoteles andaluces utilizando únicamente redes recurrentes (SimpleRNN, LSTM, GRU y variantes bidireccionales) con embeddings entrenados desde cero o Word2Vec propio. El repositorio contiene el pipeline modular para preparar datos, entrenar cada arquitectura de forma independiente, registrar métricas (F1 macro, `recall_neg`, `precision_pos`, matrices de confusión) y documentar resultados.

## Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| `src/` | Código fuente principal (`data`, `models`, `train`, `eval`). |
| `scripts/` | Entrypoints operativos (`prepare_dataset.py`, `train_simple_rnn.py`, `train_lstm.py`, `train_gru.py`, utilidades de bitácora). |
| `config/` | Configuraciones YAML (por ejemplo `phase2.yaml` con hiperparámetros y rutas). |
| `data/` | Datos locales (ej. `Big_AHR.csv`). No se versionan resultados intermedios. |
| `artifacts/` | Salidas generadas (folds preparados, pesos, métricas, `experiments.csv`). Ignorados en control de versiones. |
| `docs/phase2/` | Documentación técnica de la fase actual (dataset, plan experimental, pipeline, combinaciones, modelos). |
| `docs/resources/` | Material de referencia reorganizado:<br>• `entregas/Proyecto_DL_Fase1.pdf`<br>• `instrucciones/Proyecto 2025-*.pdf`<br>• `templates/01_IEEE_TEMPLATE`<br>• `referencias/Casosdeuso.md`. |
| `reports/` | Resultados consolidados (`phase2/RESULTADOS.md`, `phase2/ieee_paper/`). |
| `notebooks/` | Exploraciones y prototipos (si aplica). |
| `BITACORA.md` | Registro cronológico de corridas y decisiones. |
| `03_setup/` | Scripts/locks para recrear el entorno `dl_project` con CUDA 12.6. |

## Requisitos

1. **Entorno**: Python 3.10+, CUDA toolkit 12.6 con `cuda-nvcc` disponible. Recomendado usar el entorno `dl_project` definido en `03_setup/environment.lock.yml`.
2. **Dependencias adicionales**: spaCy (`es_core_news_sm`), NLTK, gensim, TensorFlow/Keras, scikit-learn, pandas, numpy.
3. **Datos**: colocar `Big_AHR.csv` en `data/` (ruta controlada desde `config/phase2.yaml`).
4. **GPU**: exportar `TF_FORCE_GPU_ALLOW_GROWTH=true` y `TF_GPU_ALLOCATOR=cuda_malloc_async` para aprovechar la RTX 3090 (o equivalente).

## Flujo de trabajo

1. **Preparar datos y folds**
   ```bash
   PYTHONPATH=. python scripts/prepare_dataset.py \
       --config config/phase2.yaml \
       --output artifacts/data/C01 \
       --experiment-id C01 \
       --cleaning baseline \
       --nlp keras_tokenizer \
       --embedding learned \
       --folds 3
   ```
   - Reutiliza caches en `artifacts/cache/` (textos limpios, índices de folds) para cualquier combinación.

2. **Entrenar modelos** (cada script acepta overrides de hiperparámetros, `--dropout`, `--bidirectional`, etc.).
   ```bash
   PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_lstm.py \
       --config config/phase2.yaml \
       --data-cache artifacts/data/C01 \
       --output artifacts/phase2/C01_LSTM \
       --folds 1,2,3 \
       --tag C01_LSTM \
       --bidirectional
   ```
   - Registro automático en `artifacts/experiments.csv` y `BITACORA.md` (usando `scripts/bitacora_add.py` si se requiere).

3. **Analizar resultados**
   - `reports/phase2/RESULTADOS.md` centraliza tablas por modelo/combinación y checklist de progreso.
   - `reports/phase2/ieee_paper/paper.tex` contiene el artículo IEEE listo para compilar con `pdflatex`.

## Documentación clave

- `docs/phase2/DATASET_RESUMEN.md`: resumen del dataset y restricciones técnicas.
- `docs/phase2/PLAN_EXPERIMENTAL.md`: diseño de experimentos y métricas de respuesta.
- `docs/phase2/PIPELINE.md`: instrucciones paso a paso del pipeline modular.
- `docs/phase2/COMBINACIONES.md`: tabla oficial de C01–C11 y formato de `RUN_ID`.
- `reports/phase2/RESULTADOS.md`: reporte vivo con métricas fold a fold y conclusiones.

## Próximos pasos sugeridos

- Automatizar `scripts/compare_results.py` para generar tablas y gráficas directamente desde `artifacts/experiments.csv`.
- Añadir pruebas unitarias mínimas para los módulos de datos y evaluación en `src/`.
- Versionar plantillas de comandos (por ejemplo en `Makefile` o `tox`/`nox`) para ejecutar bloques completos de combinaciones.
