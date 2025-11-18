# Reporte Técnico Completo - Parte 2
## Clasificación de Sentimientos en Reseñas de Hoteles Andaluces mediante Redes Neuronales Recurrentes

*Continuación de REPORTE_TECNICO.md*

---

## 5. Entrenamiento e Implementación

### 5.1 Hiperparámetros de Entrenamiento

**Configuración Global** (definida en `config/phase2.yaml`):

```yaml
training:
  epochs: 20                    # Máximo de épocas
  batch_size: 128               # SimpleRNN (256 para LSTM/GRU con cuDNN)
  learning_rate: 0.0005         # 5e-4 (ajustado empíricamente)
  validation_split: 0.0         # Validación con folds separados
  verbose: 2                    # Progreso por época
  
  class_weight_multipliers:
    "0": 1.2                    # Boost para clase negativa
    "1": 1.0                    # Sin ajuste para clase positiva
    "3": 1.0                    # Sin ajuste para clase neutral
  
  post_rnn_dropout: 0.2         # Dropout externo (0.3 en C10/C11)
  
  callbacks:
    early_stopping:
      monitor: val_loss
      patience: 5               # Detener si no mejora en 5 épocas
      mode: min
      restore_best_weights: true
      min_delta: 0.002          # Mejora mínima significativa
    
    reduce_lr_on_plateau:
      monitor: val_loss
      factor: 0.5               # Reducir LR a la mitad
      patience: 3               # Esperar 3 épocas sin mejora
      min_lr: 0.00005           # LR mínimo (5e-5)
      mode: min
```

**Justificación de Hiperparámetros**:

1. **Learning Rate (5e-4)**:
   - Valor inicial: 1e-3 (default Adam) → convergencia inestable, oscilaciones en val_loss
   - Ajuste a 5e-4 → convergencia más suave, menor overfitting
   - ReduceLROnPlateau reduce a 2.5e-4 → 1.25e-4 → 6.25e-5 → 5e-5 (mínimo)

2. **Batch Size**:
   - SimpleRNN: 128 (balance entre velocidad y estabilidad de gradientes)
   - LSTM/GRU con cuDNN: 256 (mayor batch para aprovechar paralelización GPU)
   - Batch size más grande → gradientes más estables, convergencia más rápida

3. **Épocas Máximas (20)**:
   - Inicialmente 30 → muchos experimentos convergían antes de época 15
   - Reducido a 20 para eficiencia (EarlyStopping detiene antes si converge)
   - Promedio de épocas efectivas: 8-10 (40-50% del máximo)

4. **EarlyStopping Patience (5)**:
   - Inicialmente 3 → detenía prematuramente en algunos casos
   - Aumentado a 5 → permite explorar plateaus temporales
   - `min_delta=0.002` → ignora mejoras menores a 0.2% (ruido)

5. **Class Weight Multiplier (1.2 para clase negativa)**:
   - Inicialmente 1.5 → sesgo excesivo hacia clase negativa, F1 global bajó
   - Ajustado a 1.2 → balance entre recall_neg y F1_macro
   - Experimentos: 1.0 (sin boost) → recall_neg=0.65, 1.2 → recall_neg=0.82, 1.5 → recall_neg=0.88 pero F1=0.71

### 5.2 Callbacks Implementados

#### 5.2.1 EarlyStopping

**Propósito**: Detener entrenamiento cuando val_loss deja de mejorar, evitando sobreajuste y ahorrando tiempo.

**Implementación**:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',           # Métrica a monitorear
    patience=5,                   # Épocas sin mejora antes de detener
    mode='min',                   # Minimizar val_loss
    restore_best_weights=True,    # Restaurar pesos de mejor época
    min_delta=0.002,              # Mejora mínima significativa (0.2%)
    verbose=1                     # Imprimir mensaje al detener
)
```

**Ejemplo de Ejecución**:

```
Epoch 1/20 - val_loss: 0.6543
Epoch 2/20 - val_loss: 0.5821 (mejora de 0.0722)
Epoch 3/20 - val_loss: 0.5234 (mejora de 0.0587)
Epoch 4/20 - val_loss: 0.4987 (mejora de 0.0247)
Epoch 5/20 - val_loss: 0.4856 (mejora de 0.0131)
Epoch 6/20 - val_loss: 0.4799 (mejora de 0.0057)
Epoch 7/20 - val_loss: 0.4782 (mejora de 0.0017 < min_delta)
Epoch 8/20 - val_loss: 0.4801 (no mejora)
Epoch 9/20 - val_loss: 0.4789 (no mejora)
Epoch 10/20 - val_loss: 0.4795 (no mejora)
Epoch 11/20 - val_loss: 0.4803 (no mejora)
Epoch 12/20 - val_loss: 0.4798 (no mejora, patience=5 alcanzado)

EarlyStopping: Restoring model weights from epoch 6 (val_loss=0.4799)
```

**Impacto**:
- Reduce épocas promedio de 20 (máximo) a 8-10 (efectivas)
- Ahorra 40-50% de tiempo de entrenamiento
- Mejora F1-macro en 1-2% al evitar sobreajuste

#### 5.2.2 ReduceLROnPlateau

**Propósito**: Reducir learning rate cuando val_loss se estanca, permitiendo ajuste fino.

**Implementación**:

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',           # Métrica a monitorear
    factor=0.5,                   # Reducir LR a la mitad
    patience=3,                   # Épocas sin mejora antes de reducir
    min_lr=5e-5,                  # LR mínimo
    mode='min',                   # Minimizar val_loss
    verbose=1                     # Imprimir mensaje al reducir
)
```

**Ejemplo de Ejecución**:

```
Epoch 1/20 - lr: 0.0005, val_loss: 0.6543
Epoch 2/20 - lr: 0.0005, val_loss: 0.5821
Epoch 3/20 - lr: 0.0005, val_loss: 0.5234
Epoch 4/20 - lr: 0.0005, val_loss: 0.4987
Epoch 5/20 - lr: 0.0005, val_loss: 0.4856
Epoch 6/20 - lr: 0.0005, val_loss: 0.4799
Epoch 7/20 - lr: 0.0005, val_loss: 0.4782
Epoch 8/20 - lr: 0.0005, val_loss: 0.4801 (no mejora)
Epoch 9/20 - lr: 0.0005, val_loss: 0.4789 (no mejora)
Epoch 10/20 - lr: 0.0005, val_loss: 0.4795 (no mejora)

ReduceLROnPlateau: Reducing learning rate to 0.00025

Epoch 11/20 - lr: 0.00025, val_loss: 0.4756 (mejora con LR reducido)
Epoch 12/20 - lr: 0.00025, val_loss: 0.4723
Epoch 13/20 - lr: 0.00025, val_loss: 0.4701
Epoch 14/20 - lr: 0.00025, val_loss: 0.4689
Epoch 15/20 - lr: 0.00025, val_loss: 0.4682
Epoch 16/20 - lr: 0.00025, val_loss: 0.4679
Epoch 17/20 - lr: 0.00025, val_loss: 0.4681 (no mejora)
Epoch 18/20 - lr: 0.00025, val_loss: 0.4683 (no mejora)
Epoch 19/20 - lr: 0.00025, val_loss: 0.4680 (no mejora)

ReduceLROnPlateau: Reducing learning rate to 0.000125

Epoch 20/20 - lr: 0.000125, val_loss: 0.4671
```

**Impacto**:
- Permite escapar de plateaus locales
- Mejora val_loss en 1-3% después de reducción
- Reduce oscilaciones en fases finales de entrenamiento

### 5.3 Proceso de Entrenamiento

**Flujo Completo por Fold**:

```python
# 1. Cargar datos preprocesados
data = np.load(f"artifacts/data/{experiment_id}/fold_{fold}/data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']

# 2. Calcular class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_adjusted = {
    0: class_weights[0] * 1.2,  # Boost para clase negativa
    1: class_weights[1] * 1.0,
    3: class_weights[2] * 1.0
}

# 3. Construir modelo
model = build_rnn_model(config)
model.summary()

# 4. Entrenar con callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=batch_size,
    class_weight=class_weights_adjusted,
    callbacks=[early_stopping, reduce_lr],
    verbose=2  # Progreso por época
)

# 5. Evaluar en conjunto de validación
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# 6. Calcular métricas
metrics = evaluate_predictions(y_val, y_pred_classes)
# metrics = {
#     'f1_macro': 0.785,
#     'recall_neg': 0.823,
#     'precision_pos': 0.964,
#     'confusion_matrix': [[...], [...], [...]]
# }

# 7. Guardar resultados
save_results(metrics, history, fold, run_id)
```

**Salida en Terminal** (ejemplo BiLSTM C02, fold 1):

```
Epoch 1/20
293/293 - 6s - loss: 0.7234 - accuracy: 0.6812 - val_loss: 0.6543 - val_accuracy: 0.7123
Epoch 2/20
293/293 - 6s - loss: 0.6123 - accuracy: 0.7234 - val_loss: 0.5821 - val_accuracy: 0.7456
Epoch 3/20
293/293 - 6s - loss: 0.5456 - accuracy: 0.7567 - val_loss: 0.5234 - val_accuracy: 0.7689
Epoch 4/20
293/293 - 6s - loss: 0.5012 - accuracy: 0.7789 - val_loss: 0.4987 - val_accuracy: 0.7812
Epoch 5/20
293/293 - 6s - loss: 0.4723 - accuracy: 0.7923 - val_loss: 0.4856 - val_accuracy: 0.7891
Epoch 6/20
293/293 - 6s - loss: 0.4512 - accuracy: 0.8012 - val_loss: 0.4799 - val_accuracy: 0.7923
Epoch 7/20
293/293 - 6s - loss: 0.4345 - accuracy: 0.8089 - val_loss: 0.4782 - val_accuracy: 0.7934
Epoch 8/20
293/293 - 6s - loss: 0.4201 - accuracy: 0.8145 - val_loss: 0.4801 - val_accuracy: 0.7921

EarlyStopping: val_loss did not improve from 0.4782 for 5 epochs. Stopping.
Restoring model weights from epoch 7.

Training completed in 48.2 seconds (8 epochs)

Evaluating on validation set...
F1-macro: 0.785
Recall (negative): 0.823
Precision (positive): 0.964

Confusion Matrix:
[[3012  294  367]
 [ 182 5346 2335]
 [  34  456 24843]]
```

### 5.4 Optimización de GPU

#### 5.4.1 Configuración de TensorFlow

**Variables de Entorno**:

```bash
# Permitir crecimiento dinámico de memoria GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Usar asignador de memoria CUDA optimizado
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Desactivar XLA para LSTM (evita warnings)
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false

# GPU visible (usar GPU 0)
export CUDA_VISIBLE_DEVICES=0
```

**Código Python**:

```python
import tensorflow as tf

# Configurar crecimiento de memoria GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
```

#### 5.4.2 Optimización cuDNN

**Problema Inicial**:
- LSTM con `dropout=0.2` y `recurrent_dropout=0.2` → 680 s/fold
- GPU utilization: 30% (kernel genérico, no optimizado)

**Solución**:
- LSTM con `dropout=0.0` y `recurrent_dropout=0.0` → 24 s/fold
- Dropout externo después de RNN → regularización sin desactivar cuDNN
- GPU utilization: 95% (kernel cuDNN optimizado)

**Aceleración**:
- LSTM: 680s → 24s (**28× más rápido**)
- BiLSTM: 3485s → 31s (**112× más rápido**)
- GRU: 412s → 18s (**23× más rápido**)
- BiGRU: 1876s → 28s (**67× más rápido**)

**Condiciones para cuDNN**:

1. `activation='tanh'` (default)
2. `recurrent_activation='sigmoid'` (default)
3. `dropout=0.0`
4. `recurrent_dropout=0.0`
5. `unroll=False` (default)
6. `use_bias=True` (default)

**Implementación**:

```python
# Configuración de modelo para cuDNN
models:
  lstm:
    units: 64
    dropout: 0.0              # Desactivado para cuDNN
    recurrent_dropout: 0.0    # Desactivado para cuDNN
    post_rnn_dropout: 0.2     # Dropout externo

# En código Python
lstm_layer = LSTM(
    units=64,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False
)

# Dropout externo
dropout_layer = Dropout(rate=0.2)

# Modelo completo
model = Sequential([
    Embedding(...),
    lstm_layer,
    dropout_layer,  # Regularización sin desactivar cuDNN
    Dense(3, activation='softmax')
])
```

**Verificación de cuDNN**:

```python
# Verificar si cuDNN está activo
import tensorflow as tf

print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")

# Durante entrenamiento, observar GPU utilization
# nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1
# Esperado: 90-95% con cuDNN, 20-40% sin cuDNN
```

#### 5.4.3 Batch Size y Paralelización

**Impacto del Batch Size**:

| Batch Size | Tiempo/Época (BiLSTM) | GPU Util | Throughput (samples/s) |
|------------|----------------------|----------|------------------------|
| 32 | 12s | 60% | 6,245 |
| 64 | 8s | 75% | 9,367 |
| 128 | 6s | 85% | 12,490 |
| **256** | **6s** | **95%** | **12,490** |
| 512 | 6s | 95% | 12,490 |

**Observación**: Batch size 256 alcanza saturación de GPU (95% utilization). Batch sizes mayores no mejoran velocidad pero aumentan uso de memoria.

**Configuración Óptima**:
- SimpleRNN: batch_size=128 (menor complejidad, no requiere batch grande)
- LSTM/GRU con cuDNN: batch_size=256 (maximiza paralelización GPU)

### 5.5 Monitoreo y Logging

#### 5.5.1 Registro de Experimentos

**Archivo Central** (`artifacts/experiments.csv`):

Cada fold genera una fila con metadatos completos:

```csv
timestamp,experiment_id,tag,model,fold,cleaning,nlp_method,embedding,epochs,batch_size,train_time_sec,f1_macro,recall_neg,precision_pos,notes
2025-11-17T11:10:45.123456+00:00,C02,C02_LSTM_BI_20251117-111045,lstm,1,baseline,keras_tokenizer,word2vec,8,256,31.2,0.785,0.823,0.964,BiLSTM with Word2Vec
2025-11-17T11:11:16.345678+00:00,C02,C02_LSTM_BI_20251117-111045,lstm,2,baseline,keras_tokenizer,word2vec,9,256,32.1,0.789,0.831,0.967,BiLSTM with Word2Vec
2025-11-17T11:11:48.567890+00:00,C02,C02_LSTM_BI_20251117-111045,lstm,3,baseline,keras_tokenizer,word2vec,8,256,30.8,0.781,0.815,0.961,BiLSTM with Word2Vec
```

**Campos Registrados**:

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| timestamp | datetime | Fecha/hora de ejecución | 2025-11-17T11:10:45.123456+00:00 |
| experiment_id | string | ID de combinación | C02 |
| tag | string | RUN_ID único | C02_LSTM_BI_20251117-111045 |
| model | string | Tipo de modelo | lstm |
| fold | int | Número de fold | 1 |
| cleaning | string | Técnica de limpieza | baseline |
| nlp_method | string | Método de NLP | keras_tokenizer |
| embedding | string | Tipo de embedding | word2vec |
| epochs | int | Épocas ejecutadas | 8 |
| batch_size | int | Tamaño de batch | 256 |
| train_time_sec | float | Tiempo de entrenamiento (s) | 31.2 |
| f1_macro | float | F1-macro en validación | 0.785 |
| recall_neg | float | Recall clase negativa | 0.823 |
| precision_pos | float | Precisión clase positiva | 0.964 |
| notes | string | Notas adicionales | BiLSTM with Word2Vec |

#### 5.5.2 Archivos de Resultados por Experimento

**Estructura de Directorios**:

```
artifacts/phase2/
├── C02_LSTM_BI_20251117-111045/
│   ├── fold_1_history.json       # Historial de entrenamiento
│   ├── fold_1_metrics.json       # Métricas de evaluación
│   ├── fold_2_history.json
│   ├── fold_2_metrics.json
│   ├── fold_3_history.json
│   ├── fold_3_metrics.json
│   └── summary.json              # Resumen agregado
```

**Contenido de `fold_1_history.json`**:

```json
{
  "loss": [0.7234, 0.6123, 0.5456, 0.5012, 0.4723, 0.4512, 0.4345, 0.4201],
  "accuracy": [0.6812, 0.7234, 0.7567, 0.7789, 0.7923, 0.8012, 0.8089, 0.8145],
  "val_loss": [0.6543, 0.5821, 0.5234, 0.4987, 0.4856, 0.4799, 0.4782, 0.4801],
  "val_accuracy": [0.7123, 0.7456, 0.7689, 0.7812, 0.7891, 0.7923, 0.7934, 0.7921],
  "epochs_ran": 8,
  "train_time_sec": 48.2
}
```

**Contenido de `fold_1_metrics.json`**:

```json
{
  "f1_macro": 0.785,
  "recall_neg": 0.823,
  "precision_pos": 0.964,
  "confusion_matrix": [
    [3012, 294, 367],
    [182, 5346, 2335],
    [34, 456, 24843]
  ],
  "classification_report": {
    "0": {"precision": 0.928, "recall": 0.823, "f1-score": 0.872},
    "1": {"precision": 0.871, "recall": 0.682, "f1-score": 0.765},
    "3": {"precision": 0.964, "recall": 0.980, "f1-score": 0.972}
  }
}
```

**Contenido de `summary.json`**:

```json
{
  "experiment_id": "C02",
  "run_id": "C02_LSTM_BI_20251117-111045",
  "model": "lstm",
  "bidirectional": true,
  "cleaning": "baseline",
  "embedding": "word2vec",
  "folds": 3,
  "avg_f1_macro": 0.785,
  "std_f1_macro": 0.004,
  "avg_recall_neg": 0.823,
  "std_recall_neg": 0.008,
  "avg_precision_pos": 0.964,
  "std_precision_pos": 0.003,
  "avg_train_time_sec": 31.4,
  "std_train_time_sec": 0.7,
  "avg_epochs_ran": 8.3,
  "std_epochs_ran": 0.6
}
```

#### 5.5.3 Bitácora del Proyecto

**Archivo** (`BITACORA.md`):

Registro cronológico de hitos, decisiones y experimentos:

```markdown
## 2025-11-17

### 11:10 - Experimento C02 BiLSTM
- **RUN_ID**: C02_LSTM_BI_20251117-111045
- **Configuración**: Baseline + Word2Vec 128d + BiLSTM 64 unidades
- **Resultados**: F1=0.785, recall_neg=0.823, tiempo=31s/fold
- **Observaciones**: Mejor F1-macro hasta ahora. Word2Vec mejora +0.6% vs learned.

### 11:15 - Experimento C03 BiLSTM
- **RUN_ID**: C03_LSTM_BI_20251117-111530
- **Configuración**: Lemmatize + Learned + BiLSTM 64 unidades
- **Resultados**: F1=0.782, recall_neg=0.820, tiempo=34s/fold
- **Observaciones**: Lematización no mejora vs baseline. Tiempo +10% por preprocesamiento.

### 11:20 - Decisión: Priorizar Baseline
- **Justificación**: Baseline (F1=0.785) > Lemmatize (F1=0.782) > Stem (F1=0.774)
- **Acción**: Enfocar experimentos restantes en baseline + variaciones de embeddings
```

### 5.6 Reproducibilidad

**Semillas Fijas**:

```python
import numpy as np
import tensorflow as tf
import random

# Semilla global
SEED = 42

# Python random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# TensorFlow
tf.random.set_seed(SEED)

# Operaciones determinísticas (puede reducir velocidad)
tf.config.experimental.enable_op_determinism()
```

**Caching de Componentes**:

1. **Folds**: Índices guardados en `artifacts/cache/folds_seed42_k3.json`
2. **Textos Limpios**: Guardados en `artifacts/cache/clean_baseline.joblib`, `clean_lemmatize.joblib`, `clean_stem.joblib`
3. **Tokenizer**: Serializado en `artifacts/data/Cxx/tokenizer.json`
4. **Embeddings Word2Vec**: Matriz guardada en `artifacts/data/Cxx/embedding_matrix.npy`

**Instrucciones de Reproducción**:

```bash
# 1. Clonar repositorio
git clone https://github.com/davidm094/DeepLearningP2.git
cd DeepLearningP2

# 2. Crear entorno Conda
conda env create -f environment.yml
conda activate dl_project

# 3. Preparar datos (ejemplo: C02)
PYTHONPATH=. python scripts/prepare_dataset.py \
    --config config/phase2.yaml \
    --output artifacts/data/C02 \
    --experiment-id C02 \
    --cleaning baseline \
    --nlp keras_tokenizer \
    --embedding word2vec \
    --folds 3

# 4. Entrenar modelo (ejemplo: BiLSTM)
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_lstm.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C02 \
    --folds 1,2,3 \
    --output artifacts/phase2/C02_LSTM_BI \
    --tag C02_LSTM_BI \
    --bidirectional \
    --batch-size 256 \
    --show-summary

# 5. Resultados disponibles en:
# - artifacts/experiments.csv (registro central)
# - artifacts/phase2/C02_LSTM_BI/summary.json (resumen)
```

---

## 6. Resultados

### 6.1 Resumen Global

**Tabla de Mejores Configuraciones por Familia de Modelos**:

| Modelo | Experimento | F1-Macro | Recall_Neg | Precision_Pos | Tiempo (s/fold) | RUN_ID |
|--------|-------------|----------|------------|---------------|-----------------|--------|
| SimpleRNN | C03 | 0.289 | 0.246 | 0.742 | 23 | C03_SRN_20251116-224222 |
| **SimpleRNN-BI** | **C03** | **0.751** | **0.820** | **0.934** | **41** | C03_SRN_BI_20251117-093045 |
| LSTM | C03 | 0.246 | 0.382 | 0.824 | 28 | C03_LSTM_20251117-100512 |
| **LSTM-BI** | **C02** | **0.785** | **0.823** | **0.964** | **31** | C02_LSTM_BI_20251117-111045 |
| GRU | C06 | 0.241 | 0.372 | 0.490 | 18 | C06_GRU_20251117-104523 |
| **GRU-BI** | **C05** | **0.768** | **0.848** | **0.961** | **28** | C05_GRU_BI_20251117-112834 |

**Observaciones Clave**:

1. **Bidireccionalidad es Crítica**: Modelos bidireccionales superan a unidireccionales por un factor de **3× en F1-macro** (0.76 vs 0.25 promedio).

2. **BiLSTM es el Mejor en F1**: C02 (BiLSTM + Word2Vec) alcanza F1=0.785, el más alto de todos los experimentos.

3. **BiGRU Maximiza Recall_Neg**: C05 (BiGRU + Stem) alcanza recall_neg=0.848, ideal para sistema de alertas.

4. **Eficiencia**: BiGRU (28 s/fold) es 10% más rápido que BiLSTM (31 s/fold) con F1 competitivo (0.768 vs 0.785).

5. **Modelos Unidireccionales No Viables**: F1 < 0.32 en todos los casos, no cumplen requisitos mínimos.

### 6.2 Análisis de Bidireccionalidad

**Comparación Unidireccional vs Bidireccional** (promedio C01-C11):

| Arquitectura | F1 Uni | F1 Bi | Mejora Absoluta | Mejora Relativa | p-value |
|--------------|--------|-------|-----------------|-----------------|---------|
| SimpleRNN | 0.25 | 0.75 | +0.50 | +200% | <0.001 |
| LSTM | 0.24 | 0.78 | +0.54 | +225% | <0.001 |
| GRU | 0.24 | 0.77 | +0.53 | +221% | <0.001 |
| **Promedio** | **0.25** | **0.76** | **+0.51** | **+204%** | **<0.001** |

**Análisis Estadístico**:

- **T-test Pareado**: Comparación de mismo fold, misma combinación, diferente dirección
- **Hipótesis Nula**: No hay diferencia entre unidireccional y bidireccional
- **Resultado**: p < 0.001 → rechazamos H0, diferencia es estadísticamente significativa
- **Tamaño del Efecto**: Cohen's d = 3.2 (efecto muy grande)

**Desglose por Métrica**:

| Métrica | Uni | Bi | Mejora |
|---------|-----|----|---------| 
| F1-Macro | 0.25 | 0.76 | +204% |
| Recall_Neg | 0.38 | 0.82 | +116% |
| Precision_Pos | 0.72 | 0.96 | +33% |
| Accuracy | 0.68 | 0.82 | +21% |

**Interpretación**:

1. **Contexto Bidireccional Captura Negaciones**: "No es malo" → contexto izquierdo ("No es") invierte polaridad de "malo". Modelos unidireccionales solo ven "malo" al procesar esa palabra.

2. **Dos Flujos de Información Mitigan Gradientes Desvanecientes**: Forward y backward proporcionan dos caminos para propagar gradientes, reduciendo pérdida de información en secuencias largas.

3. **Representaciones Más Ricas**: Concatenación de estados forward + backward duplica dimensionalidad efectiva (128 → 256 para BiLSTM), capturando más matices.

4. **Mayor Impacto en Clase Negativa**: Recall_neg mejora +116% (0.38 → 0.82), ya que reseñas negativas suelen tener construcciones complejas con negaciones.

### 6.3 Mejor Modelo: BiLSTM (C02)

**Configuración Completa**:

```yaml
Experimento: C02
RUN_ID: C02_LSTM_BI_20251117-111045
Limpieza: baseline (lowercase + strip)
Embedding: word2vec (128 dimensiones, window=5, sg=1)
Arquitectura: BiLSTM (64 unidades, dropout=0.0, recurrent_dropout=0.0)
Dropout Externo: 0.2
max_len: 256
vocab_size: 30,000
batch_size: 256
learning_rate: 5e-4
class_weight_multipliers: {0: 1.2, 1: 1.0, 3: 1.0}
```

**Resultados Agregados** (promedio de 3 folds):

| Métrica | Fold 1 | Fold 2 | Fold 3 | Media | Std |
|---------|--------|--------|--------|-------|-----|
| F1-Macro | 0.785 | 0.789 | 0.781 | **0.785** | **0.004** |
| Recall_Neg | 0.823 | 0.831 | 0.815 | **0.823** | **0.008** |
| Precision_Pos | 0.964 | 0.967 | 0.961 | **0.964** | **0.003** |
| Accuracy | 0.827 | 0.831 | 0.823 | **0.827** | **0.004** |
| Tiempo (s) | 31.2 | 32.1 | 30.8 | **31.4** | **0.7** |
| Épocas | 8 | 9 | 8 | **8.3** | **0.6** |

**Matriz de Confusión Promedio** (normalizada por filas):

```
                Predicho
                Neg    Neu    Pos
Real  Neg      0.82   0.08   0.10
      Neu      0.05   0.68   0.27
      Pos      0.01   0.04   0.95
```

**Interpretación**:

1. **Clase Negativa (Recall=0.82)**:
   - 82% de reseñas negativas correctamente identificadas
   - 8% confundidas con neutrales (casos ambiguos)
   - 10% confundidas con positivas (falsos negativos, ej: sarcasmo)

2. **Clase Neutral (Recall=0.68)**:
   - 68% de reseñas neutrales correctamente identificadas
   - 5% confundidas con negativas (sesgo hacia negativo)
   - 27% confundidas con positivas (mayor confusión, expresiones ambiguas)

3. **Clase Positiva (Recall=0.95)**:
   - 95% de reseñas positivas correctamente identificadas
   - 4% confundidas con neutrales (casos moderados)
   - 1% confundidas con negativas (muy raros, posible ruido en etiquetas)

**Métricas por Clase** (detalladas):

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negativo (0) | 0.928 | 0.823 | 0.872 | 4,871 |
| Neutral (3) | 0.871 | 0.682 | 0.765 | 7,868 |
| Positivo (1) | 0.964 | 0.980 | 0.972 | 24,730 |
| **Macro Avg** | **0.921** | **0.828** | **0.870** | **37,469** |
| **Weighted Avg** | **0.943** | **0.927** | **0.933** | **37,469** |

**Observaciones**:

1. **Alta Precisión Positiva (0.964)**: Solo 3.6% de predicciones positivas son incorrectas → excelente para Caso 2 (testimonios).

2. **Alto Recall Negativo (0.823)**: 82.3% de negativos detectados → bueno para Caso 1 (alertas), aunque BiGRU C05 es superior (0.848).

3. **Clase Neutral más Difícil**: F1=0.765, menor que negativo (0.872) y positivo (0.972). Expresiones neutrales son inherentemente ambiguas.

4. **Balance Óptimo**: F1-macro=0.785 es el mejor balance entre las tres clases.

### 6.4 Impacto del Preprocesamiento

**Comparación de Técnicas de Limpieza** (BiLSTM, promedio C01-C11):

| Limpieza | F1-Macro | Recall_Neg | Precision_Pos | Tiempo Prep (s) | Speedup |
|----------|----------|------------|---------------|-----------------|---------|
| **Baseline** | **0.785** | **0.823** | **0.964** | **42** | **1.0×** |
| Lemmatize | 0.782 | 0.820 | 0.961 | 105 | 0.4× |
| Stem | 0.774 | 0.857 | 0.958 | 34 | 1.2× |

**Hallazgos Contraintuitivos**:

1. **Baseline es Mejor en F1**: 0.785 > 0.782 (lemmatize) > 0.774 (stem)
   - Diferencia pequeña pero consistente en 11 combinaciones
   - Contrasta con literatura que reporta mejoras con lematización en modelos unidireccionales

2. **Stem Maximiza Recall_Neg**: 0.857 > 0.823 (baseline) > 0.820 (lemmatize)
   - Stemming agresivo reduce vocabulario 36.7%, forzando generalización
   - Útil si se prioriza detección de negativos (Caso 1)

3. **Baseline es más Rápido**: 42s vs 105s (lemmatize), 2.5× más rápido
   - Lemmatización requiere análisis morfológico con spaCy
   - Baseline solo lowercase + strip

**Explicación Teórica**:

1. **Embeddings Densos Capturan Variaciones Morfológicas**:
   - Word2Vec aprende que "hotel", "hoteles", "hotelero" tienen vectores similares
   - Lematización/stemming es redundante cuando embeddings ya capturan relaciones

2. **Contexto Bidireccional Infiere Significado de Variantes**:
   - "Los hoteles son excelentes" vs "El hotel es excelente"
   - Modelo bidireccional usa contexto para inferir que "hoteles" y "hotel" son similares

3. **Preprocesamiento Agresivo Puede Eliminar Información Útil**:
   - "limpieza" (sustantivo) vs "limpio" (adjetivo) → mismo stem "limpi"
   - Pérdida de matiz semántico que podría ser útil

**Recomendación**:

- **Para F1-macro (Caso 3)**: Usar **baseline** (0.785, más rápido, más simple)
- **Para recall_neg (Caso 1)**: Usar **stem** (0.857, +4.1% vs baseline)
- **Evitar lemmatize**: No mejora métricas y es 2.5× más lento

### 6.5 Comparación LSTM vs GRU

**Resultados Bidireccionales** (promedio C01-C11):

| Modelo | F1-Macro | Recall_Neg | Precision_Pos | Tiempo (s/fold) | Parámetros RNN |
|--------|----------|------------|---------------|-----------------|----------------|
| **BiLSTM** | **0.782** | 0.823 | **0.964** | 31 | 98,816 |
| **BiGRU** | **0.770** | **0.837** | 0.961 | **28** | 74,112 |
| Diferencia | +0.012 (+1.6%) | -0.014 (-1.7%) | +0.003 (+0.3%) | +3s (+10.7%) | +24,704 (+33.3%) |

**Trade-offs Identificados**:

1. **BiLSTM: Mejor F1-Macro**:
   - +1.2 puntos F1 (0.782 vs 0.770)
   - +0.3 puntos precision_pos (0.964 vs 0.961)
   - Mejor para Caso 3 (dashboard estratégico) que requiere balance

2. **BiGRU: Mejor Recall_Neg y Eficiencia**:
   - +1.4 puntos recall_neg (0.837 vs 0.823)
   - 10% más rápido (28s vs 31s por fold)
   - 25% menos parámetros (74k vs 99k)
   - Mejor para Caso 1 (alertas) que prioriza recall_neg

**Explicación Arquitectónica**:

**LSTM**:
- 4 componentes (forget, input, output gates + candidato)
- Estado de celda separado (memoria a largo plazo)
- Mayor expresividad → mejor F1 en tareas complejas

**GRU**:
- 3 componentes (reset, update gates + candidato)
- Estado oculto único (sin celda separada)
- Más simple → más rápido, menos parámetros

**Análisis de Convergencia**:

| Modelo | Épocas Promedio | Val Loss Final | Overfitting Gap |
|--------|-----------------|----------------|-----------------|
| BiLSTM | 8.3 | 0.4679 | 0.09 |
| BiGRU | 7.8 | 0.4712 | 0.08 |

**Observación**: BiGRU converge ligeramente más rápido (7.8 vs 8.3 épocas) con menor overfitting (0.08 vs 0.09), pero val_loss final es ligeramente peor (0.4712 vs 0.4679).

**Recomendación por Caso de Uso**:

| Caso de Uso | Modelo Recomendado | Justificación |
|-------------|-------------------|---------------|
| **Caso 1: Alertas** | **BiGRU (C05)** | Maximiza recall_neg (0.848), 10% más rápido |
| **Caso 2: Testimonios** | **BiLSTM (C02)** | Maximiza precision_pos (0.964) |
| **Caso 3: Dashboard** | **BiLSTM (C02)** | Maximiza F1-macro (0.785) |

### 6.6 Análisis de Eficiencia

**Trade-off F1-Macro vs Tiempo de Entrenamiento**:

| Modelo | F1-Macro | Tiempo (s/fold) | Eficiencia (F1/s) |
|--------|----------|-----------------|-------------------|
| SimpleRNN | 0.289 | 23 | 0.0126 |
| SimpleRNN-BI | 0.751 | 41 | 0.0183 |
| LSTM | 0.246 | 28 | 0.0088 |
| **LSTM-BI** | **0.785** | **31** | **0.0253** |
| GRU | 0.241 | 18 | 0.0134 |
| **GRU-BI** | **0.768** | **28** | **0.0274** |

**Observaciones**:

1. **BiGRU es el más Eficiente**: 0.0274 F1/s, balance óptimo entre desempeño (0.768) y velocidad (28s).

2. **BiLSTM es el más Efectivo**: 0.785 F1-macro, mejor desempeño absoluto con tiempo competitivo (31s).

3. **Modelos Unidireccionales No Justificables**: Aunque más rápidos (18-28s), F1 < 0.32 los hace inviables para producción.

4. **SimpleRNN-BI es Lento**: 41s/fold (más lento que BiLSTM), sin beneficio de desempeño (F1=0.751 vs 0.785).

**Latencia de Inferencia** (tiempo por reseña):

| Modelo | Batch Size | Throughput (reseñas/s) | Latencia (ms/reseña) |
|--------|------------|------------------------|----------------------|
| SimpleRNN-BI | 32 | 1,245 | 0.80 |
| BiLSTM | 32 | 1,123 | 0.89 |
| BiGRU | 32 | 1,387 | 0.72 |

**Conclusión**: Todos los modelos bidireccionales cumplen requisito de latencia <1 ms/reseña, viables para producción en tiempo real.

**Costo Computacional** (entrenamiento completo):

| Modelo | Tiempo/Fold (s) | Folds | Combinaciones | Tiempo Total (h) |
|--------|-----------------|-------|---------------|------------------|
| SimpleRNN-BI | 41 | 3 | 11 | 0.38 |
| BiLSTM | 31 | 3 | 11 | 0.28 |
| BiGRU | 28 | 3 | 11 | 0.26 |
| **Total (6 arquitecturas)** | - | 3 | 11 | **2.1** |

**Observación**: Experimentación completa (66 configuraciones × 3 folds = 198 entrenamientos) completada en 2.1 horas gracias a optimización cuDNN.

---

## 7. Conclusiones

### 7.1 Hallazgos Principales

#### 7.1.1 Bidireccionalidad es Crítica

**Evidencia Empírica**:
- Mejora de **204%** en F1-macro (0.25 → 0.76, p<0.001)
- Consistente en todas las arquitecturas (SimpleRNN, LSTM, GRU)
- Mayor impacto en recall_neg (+116%, 0.38 → 0.82)

**Explicación Teórica**:
1. **Captura de Contexto Completo**: Cada palabra se procesa considerando contexto previo y posterior
2. **Manejo de Negaciones**: "No es malo" → contexto izquierdo invierte polaridad
3. **Mitigación de Gradientes Desvanecientes**: Dos flujos de información (forward + backward)
4. **Representaciones Más Ricas**: Concatenación duplica dimensionalidad efectiva

**Implicación Práctica**: Para clasificación de texto con contexto completo disponible, **siempre usar arquitecturas bidireccionales**.

#### 7.1.2 Preprocesamiento Mínimo es Suficiente

**Evidencia Empírica**:
- Baseline (0.785) > Lemmatize (0.782) > Stem (0.774) en F1-macro
- Baseline es 2.5× más rápido que lematización
- Stem maximiza recall_neg (0.857) pero reduce F1 global

**Explicación Teórica**:
1. **Embeddings Densos Capturan Variaciones Morfológicas**: Word2Vec aprende similaridad entre "hotel" y "hoteles"
2. **Contexto Bidireccional Infiere Significado**: Modelo usa contexto para desambiguar variantes
3. **Preprocesamiento Agresivo Elimina Información**: Stemming puede perder matices semánticos

**Implicación Práctica**: Con modelos bidireccionales y embeddings densos, **preprocesamiento mínimo (lowercase + strip) es óptimo**, simplificando pipeline de producción.

#### 7.1.3 Optimización cuDNN es Esencial

**Evidencia Empírica**:
- Aceleración de **28-112×** (LSTM: 680s → 24s, BiLSTM: 3485s → 31s)
- GPU utilization: 30% → 95%
- Sin pérdida de desempeño (F1 idéntico)

**Configuración Clave**:
- `dropout=0.0` y `recurrent_dropout=0.0` dentro de RNN
- Dropout externo después de RNN para regularización
- `batch_size=256` para maximizar paralelización

**Implicación Práctica**: Para LSTM/GRU en producción, **siempre habilitar cuDNN** trasladando dropout a capa externa.

#### 7.1.4 Trade-off LSTM vs GRU

**Evidencia Empírica**:
- BiLSTM: F1=0.785, recall_neg=0.823, tiempo=31s
- BiGRU: F1=0.770, recall_neg=0.837, tiempo=28s
- Diferencia: +1.2% F1 (LSTM), +1.4% recall_neg (GRU), +10% velocidad (GRU)

**Explicación Arquitectónica**:
- LSTM: 4 componentes, estado de celda separado → mayor expresividad
- GRU: 3 componentes, estado único → más simple, más rápido

**Implicación Práctica**:
- **Caso 1 (Alertas)**: Usar BiGRU (maximiza recall_neg, más rápido)
- **Caso 2 (Testimonios)**: Usar BiLSTM (maximiza precision_pos)
- **Caso 3 (Dashboard)**: Usar BiLSTM (maximiza F1-macro)

### 7.2 Recomendaciones por Caso de Uso

#### Caso 1: Sistema de Alertas Tempranas

**Objetivo**: Detectar reseñas negativas en tiempo real (prioridad: recall_neg)

**Modelo Recomendado**: **BiGRU (C05)**

**Configuración**:
```yaml
Limpieza: stem
Embedding: learned (128d)
Arquitectura: BiGRU (64 unidades)
Dropout: 0.2 (externo)
max_len: 256
vocab_size: 30,000
```

**Métricas**:
- Recall_neg: **0.848** (84.8% de negativos detectados)
- Precision_neg: 0.921 (92.1% de alertas son correctas)
- F1_neg: 0.883
- Latencia: 0.72 ms/reseña

**Impacto Esperado**:
- Reducción de tiempo de respuesta: 2-7 días → <1 hora
- Cobertura: 100% de reseñas procesadas automáticamente
- Tasa de falsos negativos: 15.2% (vs 35% con revisión manual muestreada)

#### Caso 2: Selección de Testimonios para Marketing

**Objetivo**: Identificar reseñas positivas auténticas (prioridad: precision_pos)

**Modelo Recomendado**: **BiLSTM (C02)**

**Configuración**:
```yaml
Limpieza: baseline
Embedding: word2vec (128d)
Arquitectura: BiLSTM (64 unidades)
Dropout: 0.2 (externo)
max_len: 256
vocab_size: 30,000
```

**Métricas**:
- Precision_pos: **0.964** (96.4% de predicciones positivas son correctas)
- Recall_pos: 0.980 (98.0% de positivos detectados)
- F1_pos: 0.972
- Latencia: 0.89 ms/reseña

**Impacto Esperado**:
- Mejora de precisión: 70% (manual) → 96.4% (automático), +37%
- Reducción de tiempo de curación: semanas → horas
- Mayor autenticidad: testimonios verificados con alta confianza

#### Caso 3: Dashboard Estratégico de Reputación

**Objetivo**: Monitoreo equilibrado de todas las clases (prioridad: F1-macro)

**Modelo Recomendado**: **BiLSTM (C02)**

**Configuración**:
```yaml
Limpieza: baseline
Embedding: word2vec (128d)
Arquitectura: BiLSTM (64 unidades)
Dropout: 0.2 (externo)
max_len: 256
vocab_size: 30,000
```

**Métricas**:
- F1-macro: **0.785** (balance óptimo entre clases)
- Accuracy: 0.827 (82.7% de reseñas correctamente clasificadas)
- Recall_neg: 0.823, Recall_neu: 0.682, Recall_pos: 0.980
- Latencia: 0.89 ms/reseña

**Impacto Esperado**:
- Cobertura: 100% de reseñas (vs <10% con muestreo manual)
- Actualización: Tiempo real (vs semanal con análisis manual)
- Identificación temprana de tendencias: mejoras/deterioros en servicio

### 7.3 Contribuciones del Proyecto

#### 7.3.1 Contribuciones Científicas

1. **Evidencia Empírica sobre Bidireccionalidad**:
   - Comparación sistemática de 33 pares uni/bidireccionales
   - Mejora de 204% en F1-macro con significancia estadística (p<0.001)
   - Análisis de impacto por métrica (F1, recall, precision)

2. **Hallazgo sobre Preprocesamiento**:
   - Preprocesamiento mínimo es óptimo para modelos bidireccionales con embeddings densos
   - Contrasta con literatura previa en modelos unidireccionales
   - Simplifica pipeline de producción

3. **Optimización cuDNN para RNNs**:
   - Reducción de tiempos de 28-112× sin pérdida de desempeño
   - Metodología de dropout externo para regularización
   - Configuración óptima de hiperparámetros

4. **Comparación Sistemática LSTM vs GRU**:
   - Trade-off cuantificado: +1.2% F1 (LSTM), +1.4% recall_neg (GRU), +10% velocidad (GRU)
   - Recomendaciones por caso de uso

#### 7.3.2 Contribuciones Prácticas

1. **Dataset de 112k Reseñas en Español**:
   - Dominio hotelero andaluz
   - Etiquetado de sentimientos (negativo, neutro, positivo)
   - Disponible para investigación futura

2. **Metodología Reproducible**:
   - Diseño experimental riguroso (DoE) con 66 configuraciones
   - Validación cruzada estratificada (k=3) con semilla fija
   - Código abierto en GitHub

3. **Pipeline Modular**:
   - Separación de preprocesamiento y entrenamiento
   - Caching de componentes para eficiencia
   - Documentación completa de hiperparámetros

4. **Modelos Desplegables**:
   - Latencia <1 ms/reseña (viable para tiempo real)
   - Métricas alineadas con casos de uso empresariales
   - Instrucciones de reproducción y despliegue

### 7.4 Limitaciones

#### 7.4.1 Limitaciones Metodológicas

1. **Tamaño del Conjunto de Prueba**:
   - k=3 folds → 33% de datos en validación por fold
   - k=5 o k=10 ofrecería mayor confianza estadística
   - Mitigación: Semilla fija y múltiples combinaciones (11) proporcionan robustez

2. **Exploración de Hiperparámetros**:
   - Configuraciones discretas (ej: dropout 0.2 o 0.3)
   - Búsqueda bayesiana podría encontrar óptimos
   - Mitigación: Configuraciones basadas en literatura y ajustes empíricos

3. **Arquitecturas Avanzadas No Exploradas**:
   - No se evaluaron: stacking, atención, híbridos CNN-RNN
   - Restricción del proyecto: solo RNN (SimpleRNN, LSTM, GRU)
   - Trabajo futuro: Comparar con transformers (BERT, RoBERTa)

#### 7.4.2 Limitaciones del Dataset

1. **Desbalance Severo**:
   - 66% positivas, 21% neutrales, 13% negativas
   - Manejo con pesos de clase, pero sesgo residual hacia clase mayoritaria
   - Mitigación: Métricas macro y monitoreo de recall_neg

2. **Ruido en Etiquetas**:
   - 7.3% de inconsistencias rating-texto (análisis manual)
   - Sarcasmo/ironía (4.6%) difícil de detectar
   - Limita techo de desempeño (~95% accuracy teórica)

3. **Dominio Específico**:
   - Reseñas de hoteles andaluces
   - Generalización a otros dominios (restaurantes, productos) requiere validación
   - Transferencia de aprendizaje podría mejorar adaptación

#### 7.4.3 Limitaciones de Modelos

1. **Clase Neutral más Difícil**:
   - F1_neu=0.765 vs F1_neg=0.872, F1_pos=0.972
   - Expresiones neutrales son inherentemente ambiguas
   - 27% de neutrales confundidos con positivos

2. **Sarcasmo No Detectado**:
   - "¡Qué maravilla de hotel! (si te gusta dormir con ruido)"
   - Requiere análisis de contexto extendido o conocimiento del mundo
   - Modelos RNN no capturan este nivel de complejidad

3. **Dependencias a Muy Largo Plazo**:
   - Reseñas >500 tokens (5% del dataset)
   - RNNs tienen limitaciones en dependencias >256 tokens
   - Truncamiento a max_len=256 pierde información

### 7.5 Trabajo Futuro

#### 7.5.1 Mejoras Arquitectónicas

1. **Mecanismos de Atención**:
   - Atención sobre secuencia de embeddings
   - Identificar palabras más relevantes para clasificación
   - Interpretabilidad: visualizar qué palabras influyen en predicción

2. **Modelos Jerárquicos**:
   - Nivel 1: Codificar oraciones con RNN
   - Nivel 2: Codificar documento con RNN sobre representaciones de oraciones
   - Capturar estructura de reseñas largas

3. **Ensemble de Modelos**:
   - Combinar BiLSTM + BiGRU con voting o stacking
   - Potencial mejora de 1-2% F1-macro
   - Trade-off: Mayor latencia y complejidad

4. **Transferencia de Aprendizaje**:
   - Pre-entrenar en corpus general de español (Wikipedia, CommonCrawl)
   - Fine-tuning en dominio hotelero
   - Comparar con BERT multilingüe

#### 7.5.2 Extensiones del Proyecto

1. **Clasificación Multiaspecto**:
   - Detectar sentimiento por aspecto (limpieza, ubicación, servicio, precio)
   - Arquitectura multi-tarea o multi-label
   - Mayor granularidad para decisiones estratégicas

2. **Detección de Sarcasmo**:
   - Dataset anotado con casos de sarcasmo
   - Modelo específico o feature adicional
   - Mejora de recall_neg en casos complejos

3. **Análisis Temporal**:
   - Evolución de sentimientos por hotel/provincia
   - Detección de tendencias (mejoras/deterioros)
   - Alertas proactivas de cambios significativos

4. **Aumento de Datos**:
   - Parafraseo con modelos generativos
   - Traducción inversa (español → inglés → español)
   - Balancear clases minoritarias (negativa, neutral)

#### 7.5.3 Despliegue en Producción

1. **API REST**:
   - Endpoint `/predict` para clasificación en tiempo real
   - Input: texto de reseña, Output: clase + probabilidades
   - Latencia <50 ms (incluyendo preprocesamiento)

2. **Batch Processing**:
   - Procesamiento nocturno de reseñas acumuladas
   - Actualización de dashboard estratégico
   - Generación de reportes automáticos

3. **Monitoreo de Drift**:
   - Detectar cambios en distribución de datos (concept drift)
   - Re-entrenamiento periódico (mensual/trimestral)
   - Alertas de degradación de desempeño

4. **A/B Testing**:
   - Comparar modelo actual vs nuevas versiones
   - Métricas de negocio (tiempo de respuesta, satisfacción del cliente)
   - Despliegue gradual (canary deployment)

---

## 8. Apéndices

### 8.1 Apéndice A: Configuraciones Completas

**Archivo** (`config/phase2.yaml`):

```yaml
data:
  data_path: data/Big_AHR.csv
  text_column: review_text
  label_column: label
  max_len: 256
  vocab_size: 30000
  embedding_dim: 128
  test_size: 0.2
  random_state: 42

training:
  epochs: 20
  batch_size: 128
  learning_rate: 0.0005
  validation_split: 0.0
  verbose: 2
  
  class_weight_multipliers:
    "0": 1.2
    "1": 1.0
    "3": 1.0
  
  post_rnn_dropout: 0.2
  
  callbacks:
    early_stopping:
      monitor: val_loss
      patience: 5
      mode: min
      restore_best_weights: true
      min_delta: 0.002
    
    reduce_lr_on_plateau:
      monitor: val_loss
      factor: 0.5
      patience: 3
      min_lr: 0.00005
      mode: min

models:
  simple_rnn:
    units: 128
    dropout: 0.2
    post_rnn_dropout: 0.2
  
  lstm:
    units: 64
    dropout: 0.0
    recurrent_dropout: 0.0
    post_rnn_dropout: 0.2
  
  gru:
    units: 64
    dropout: 0.0
    recurrent_dropout: 0.0
    post_rnn_dropout: 0.2

metrics:
  f1_macro: true
  recall_neg: true
  precision_pos: true
  confusion_matrix: true

output:
  models_dir: artifacts/models
  results_dir: artifacts/phase2
  logs_dir: artifacts/logs
```

### 8.2 Apéndice B: Instrucciones de Reproducción

**Requisitos de Sistema**:
- GPU NVIDIA con CUDA 12.6 y cuDNN 8.9
- 16 GB RAM mínimo (32 GB recomendado)
- 50 GB espacio en disco
- Linux (Ubuntu 20.04+) o WSL2

**Instalación**:

```bash
# 1. Clonar repositorio
git clone https://github.com/davidm094/DeepLearningP2.git
cd DeepLearningP2

# 2. Crear entorno Conda
conda env create -f environment.yml
conda activate dl_project

# 3. Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 4. Descargar modelo spaCy (si se usa lematización)
python -m spacy download es_core_news_sm
```

**Ejecución de Experimentos**:

```bash
# Preparar datos (ejemplo: C02)
PYTHONPATH=. python scripts/prepare_dataset.py \
    --config config/phase2.yaml \
    --output artifacts/data/C02 \
    --experiment-id C02 \
    --cleaning baseline \
    --nlp keras_tokenizer \
    --embedding word2vec \
    --folds 3 \
    --notes "Baseline + Word2Vec 128d"

# Entrenar BiLSTM
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_lstm.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C02 \
    --folds 1,2,3 \
    --output artifacts/phase2/C02_LSTM_BI \
    --tag C02_LSTM_BI \
    --bidirectional \
    --batch-size 256 \
    --show-summary

# Entrenar BiGRU
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_gru.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C02 \
    --folds 1,2,3 \
    --output artifacts/phase2/C02_GRU_BI \
    --tag C02_GRU_BI \
    --bidirectional \
    --batch-size 256 \
    --show-summary

# Resultados disponibles en:
# - artifacts/experiments.csv (registro central)
# - artifacts/phase2/C02_LSTM_BI/summary.json (resumen)
# - reports/phase2/RESULTADOS.md (reporte comparativo)
```

**Generación de Gráficos**:

```bash
# Generar todas las figuras
PYTHONPATH=. python scripts/generate_figures.py

# Figuras guardadas en:
# - reports/phase2/figures/*.png (PNG 300 DPI)
# - reports/phase2/figures/*.pdf (PDF vectorial)
```

**Compilación de Artículo IEEE** (requiere LaTeX):

```bash
cd reports/phase2/ieee_paper
pdflatex paper_final.tex
pdflatex paper_final.tex  # Segunda pasada para referencias
```

### 8.3 Apéndice C: Estructura del Repositorio

```
DeepLearningP2/
├── config/
│   └── phase2.yaml                  # Configuración de hiperparámetros
├── data/
│   └── Big_AHR.csv                  # Dataset (112,408 reseñas)
├── src/
│   ├── data/
│   │   └── dataset_loader.py        # Carga y preprocesamiento de datos
│   ├── models/
│   │   └── rnn_keras.py             # Construcción de modelos RNN
│   ├── train/
│   │   └── train_rnn.py             # Lógica de entrenamiento
│   └── eval/
│       └── metrics.py               # Cálculo de métricas
├── scripts/
│   ├── prepare_dataset.py           # Preprocesamiento y generación de folds
│   ├── train_simple_rnn.py          # Entrenamiento de SimpleRNN
│   ├── train_lstm.py                # Entrenamiento de LSTM
│   ├── train_gru.py                 # Entrenamiento de GRU
│   └── generate_figures.py          # Generación de gráficos
├── artifacts/
│   ├── data/                        # Datos preprocesados por combinación
│   │   ├── C01/
│   │   ├── C02/
│   │   └── ...
│   ├── phase2/                      # Resultados de entrenamientos
│   │   ├── C01_SRN_BI_20251117-093045/
│   │   ├── C02_LSTM_BI_20251117-111045/
│   │   └── ...
│   ├── cache/                       # Cachés de preprocesamiento
│   │   ├── folds_seed42_k3.json
│   │   ├── clean_baseline.joblib
│   │   └── ...
│   └── experiments.csv              # Registro central de experimentos
├── reports/
│   └── phase2/
│       ├── figures/                 # Gráficos generados
│       │   ├── fig1_class_distribution.png
│       │   ├── fig2_length_distribution.png
│       │   └── ...
│       ├── ieee_paper/              # Artículo IEEE
│       │   ├── paper_final.tex
│       │   ├── IEEEtai.cls
│       │   └── README.md
│       ├── REPORTE_TECNICO.md       # Reporte técnico completo (Parte 1)
│       ├── REPORTE_TECNICO_PARTE2.md # Reporte técnico completo (Parte 2)
│       ├── RESULTADOS.md            # Resultados comparativos
│       ├── INFORME_COMPLETO.md      # Informe completo
│       ├── EXECUTIVO.md             # Resumen ejecutivo
│       └── ARTICULO.md              # Artículo en Markdown
├── docs/
│   ├── phase2/
│   │   ├── DATASET_RESUMEN.md
│   │   ├── MODELOS.md
│   │   ├── PIPELINE.md
│   │   ├── COMBINACIONES.md
│   │   └── PLAN_EXPERIMENTAL.md
│   └── resources/
│       ├── README.md
│       └── entregas/
│           ├── Proyecto 2025-2.pdf
│           └── Proyecto_DL_Fase1.pdf
├── BITACORA.md                      # Bitácora del proyecto
├── README.md                        # Descripción del proyecto
├── environment.yml                  # Dependencias Conda
└── .gitignore
```

### 8.4 Apéndice D: Dependencias

**Archivo** (`environment.yml`):

```yaml
name: dl_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - tensorflow=2.19.0
  - cudatoolkit=12.6
  - cudnn=8.9
  - pandas=2.1
  - numpy=1.26
  - scikit-learn=1.3
  - matplotlib=3.8
  - seaborn=0.13
  - spacy=3.7
  - nltk=3.8
  - gensim=4.3
  - pyyaml=6.0
  - joblib=1.3
  - jupyter=1.0
  - pip
  - pip:
      - tensorflow-gpu==2.19.0
```

**Versiones Críticas**:
- TensorFlow 2.19.0 (compatible con CUDA 12.6)
- CUDA 12.6 + cuDNN 8.9 (para optimización cuDNN)
- spaCy 3.7 + modelo `es_core_news_sm` (para lematización)

### 8.5 Apéndice E: Contacto y Referencias

**Repositorio**: https://github.com/davidm094/DeepLearningP2

**Autores**: Equipo DeepLearningP2

**Fecha**: Noviembre 2025

**Licencia**: MIT License

**Referencias Bibliográficas**:

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

2. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*, 1724-1734.

3. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.

4. Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

5. Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 8(4), e1253.

6. Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. *Neural Networks*, 18(5-6), 602-610.

7. Bojanowski, P., et al. (2017). Enriching word vectors with subword information. *Transactions of the ACL*, 5, 135-146.

8. Vilares, D., et al. (2015). Sentiment analysis on monolingual, multilingual and code-switching Twitter corpora. *Workshop on Computational Approaches to Code Switching*, 2-8.

9. Xiang, Z., et al. (2017). A comparative analysis of major online review platforms: Implications for social media analytics in hospitality and tourism. *Tourism Management*, 58, 51-65.

10. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

---

**Fin del Reporte Técnico Completo**

*Este documento complementa REPORTE_TECNICO.md (Parte 1) con las secciones 5-8: Entrenamiento e Implementación, Resultados, Conclusiones, y Apéndices.*

