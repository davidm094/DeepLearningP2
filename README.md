# DeepLearningP2 – Sentiment RNN Experiments

Proyecto de clasificación de sentimiento para reseñas de hoteles andaluces utilizando únicamente redes recurrentes (SimpleRNN, LSTM, GRU y variantes bidireccionales) con embeddings entrenados desde cero o Word2Vec propio. El repositorio contiene el pipeline modular para preparar datos, entrenar cada arquitectura de forma independiente, registrar métricas (F1 macro, `recall_neg`, `precision_pos`, matrices de confusión) y documentar resultados.

## Estructura del repositorio

| Carpeta | Contenido |
| --- | --- |
| `src/` | Código fuente principal (`data`, `models`, `train`, `eval`). |
| `scripts/` | Entrypoints operativos (`prepare_dataset.py`, `train_simple_rnn.py`, `train_lstm.py`, `train_gru.py`, `generate_figures.py`). |
| `config/` | Configuraciones YAML (por ejemplo `phase2.yaml` con hiperparámetros y rutas). |
| `data/` | Datos locales (ej. `Big_AHR.csv`). No se versionan resultados intermedios. |
| `artifacts/` | Salidas generadas (folds preparados, pesos, métricas, `experiments.csv`). Ignorados en control de versiones. |
| `notebooks/` | **Nuevo**: Notebooks para reproducción interactiva (`C02_bilstm_pipeline.ipynb`). |
| `docs/phase2/` | Documentación técnica: `01_plan_experimental.md`, `02_combinaciones.md`, `03_arquitectura_modelos.md`, `04_pipeline.md`, `05_dataset.md`. |
| `docs/resources/` | Material de referencia reorganizado (entregas previas, instrucciones, templates IEEE, casos de uso). |
| `reports/phase2/` | **Documentos de entrega y resultados**:<br>• `entregas/` - Documentos oficiales numerados<br>• `figuras/` - 8 figuras de alta calidad (PDF + PNG)<br>• `latex/` - Archivos LaTeX del artículo IEEE<br>• `resultados_experimentales.md` - Resultados detallados |
| `backups/` | Respaldos automáticos antes de reorganizaciones. |

## 📦 Documentos de Entrega (Phase 2)

Los documentos oficiales están en `reports/phase2/entregas/`:

| Archivo | Descripción | Páginas |
|---------|-------------|---------|
| `01_resumen_ejecutivo.md` | Resumen ejecutivo del proyecto | ~3 |
| `02_articulo_ieee.tex` | Artículo en formato IEEE Conference (LaTeX) | 5-6 |
| `02_articulo_ieee.md` | Artículo en formato Markdown | ~6 |
| `03_reporte_tecnico.md` | Reporte técnico completo con EDA, metodología, resultados | ~25 |
| `04_bitacora_proyecto.md` | Bitácora cronológica del proyecto | ~4 |

### 📊 Figuras

8 figuras de alta calidad (300 DPI) en `reports/phase2/figuras/`:

1. `fig01_distribucion_clases` - Distribución de clases en el dataset
2. `fig02_longitud_resenas` - Longitud de reseñas por clase
3. `fig03_comparacion_f1` - Comparación de F1-macro por arquitectura
4. `fig04_unidireccional_vs_bidireccional` - Impacto de bidireccionalidad
5. `fig05_eficiencia` - Trade-off F1 vs tiempo de entrenamiento
6. `fig06_matriz_confusion` - Matriz de confusión del mejor modelo
7. `fig07_impacto_preprocesamiento` - Impacto de técnicas de limpieza
8. `fig08_optimizacion_cudnn` - Aceleración con cuDNN

## Requisitos

1. **Entorno**: Python 3.11+, CUDA toolkit 12.6 con `cuda-nvcc` disponible.
2. **Dependencias**: TensorFlow 2.19.0, spaCy (`es_core_news_sm`), NLTK, gensim, scikit-learn, pandas, numpy, matplotlib, seaborn.
3. **Datos**: colocar `Big_AHR.csv` en `data/` (112,408 reseñas).
4. **GPU**: RTX 3090 o equivalente con 24 GB VRAM. Exportar `TF_FORCE_GPU_ALLOW_GROWTH=true` y `TF_GPU_ALLOCATOR=cuda_malloc_async`.

### Instalación (macOS / Apple Silicon)

Para usuarios de Mac con chips M1/M2/M3, se ha provisto un entorno compatible (`tensorflow` + `metal`, `torch` CPU/MPS).

```bash
conda env create -f 03_setup/environment.mac.yml
conda activate dl_project
python -m spacy download es_core_news_sm
```

## Flujo de trabajo

### 0. Reproducción rápida (Notebook)

Para replicar interactivamente el mejor modelo (C02 - BiLSTM), abrir:
`notebooks/C02_bilstm_pipeline.ipynb`

### 1. Preparar datos y folds

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

- Reutiliza caches en `artifacts/cache/` (textos limpios, índices de folds).

### 2. Entrenar modelos

```bash
# BiLSTM (mejor F1-macro: 0.785)
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_lstm.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C02 \
    --output artifacts/phase2/C02_LSTM_BI \
    --folds 1,2,3 \
    --tag C02_LSTM_BI \
    --bidirectional \
    --batch-size 256

# BiGRU (mejor recall_neg: 0.848)
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_gru.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C05 \
    --output artifacts/phase2/C05_GRU_BI \
    --folds 1,2,3 \
    --tag C05_GRU_BI \
    --bidirectional \
    --batch-size 256
```

- Registro automático en `artifacts/experiments.csv`.

### 3. Generar figuras

```bash
PYTHONPATH=. python scripts/generate_figures.py
```

- Genera las 8 figuras en `reports/phase2/figuras/`.

### 4. Compilar artículo IEEE

```bash
cd reports/phase2/latex
pdflatex articulo_ieee.tex
pdflatex articulo_ieee.tex  # Segunda pasada para referencias
```

## Resultados Principales

| Modelo | Config | F1-Macro | Recall_Neg | Precision_Pos | Tiempo (s/fold) |
|--------|--------|----------|------------|---------------|-----------------|
| **BiLSTM** | C02 | **0.785** | 0.823 | **0.964** | 31 |
| **BiGRU** | C05 | 0.768 | **0.848** | 0.961 | 28 |
| SimpleRNN-BI | C03 | 0.751 | 0.820 | 0.934 | 41 |

**Hallazgos clave**:
- Bidireccionalidad mejora F1-macro en **204%** (0.25 → 0.76, p<0.001)
- Preprocesamiento mínimo (baseline) es óptimo con embeddings densos
- Optimización cuDNN acelera entrenamiento **28-112×** sin pérdida de desempeño

## Documentación clave

### Documentación Técnica (`docs/phase2/`)
- `01_plan_experimental.md` - Diseño de experimentos (DoE) y métricas
- `02_combinaciones.md` - Tabla de 11 combinaciones (C01-C11)
- `03_arquitectura_modelos.md` - Detalles de arquitecturas RNN
- `04_pipeline.md` - Pipeline modular paso a paso
- `05_dataset.md` - Resumen del dataset y restricciones

### Reportes (`reports/phase2/`)
- `entregas/` - Documentos oficiales de entrega
- `resultados_experimentales.md` - Resultados detallados por experimento
- `latex/` - Artículo IEEE en LaTeX

## Reproducibilidad

1. **Clonar repositorio**:
   ```bash
   git clone https://github.com/davidm094/DeepLearningP2.git
   cd DeepLearningP2
   ```

2. **Crear entorno**:
   ```bash
   # Opción A: Linux/CUDA (Recomendado para producción)
   conda env create -f 03_setup/environment.lock.yml
   
   # Opción B: macOS (Apple Silicon)
   conda env create -f 03_setup/environment.mac.yml
   
   conda activate dl_project
   python -m spacy download es_core_news_sm
   ```

3. **Verificar GPU**:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

4. **Ejecutar experimentos** (ver secciones anteriores)

## Contacto

**Repositorio**: https://github.com/davidm094/DeepLearningP2  
**Proyecto**: DeepLearningP2 - Análisis de Sentimientos  
**Fecha**: Noviembre 2025
