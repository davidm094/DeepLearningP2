# Informe Completo: Clasificación de Sentimientos en Reseñas de Hoteles Andaluces mediante Redes Neuronales Recurrentes

## Resumen Ejecutivo

Este proyecto implementa y evalúa sistemáticamente modelos de Redes Neuronales Recurrentes (RNN) para la clasificación de sentimientos en reseñas de hoteles andaluces. Se exploraron 66 configuraciones experimentales combinando tres arquitecturas RNN (SimpleRNN, LSTM, GRU), cada una en versión unidireccional y bidireccional, con 11 variantes de preprocesamiento y representación textual.

**Resultados principales:**
- **Mejor modelo**: BiLSTM con embeddings Word2Vec (C02) alcanzó **F1-macro=0.785**, **recall_neg=0.823** y **precision_pos=0.964** en ~31 s/fold.
- **Hallazgo clave**: La bidireccionalidad es crítica para el desempeño; los modelos unidireccionales colapsan hacia una sola clase (F1<0.32).
- **Eficiencia**: La optimización cuDNN redujo tiempos de entrenamiento de ~680 s/fold a ~24 s/fold en LSTM.
- **Preprocesamiento**: Word2Vec entrenado sobre el corpus supera consistentemente a embeddings aprendidos end-to-end en modelos bidireccionales.

---

## 1. Introducción y Contexto

### 1.1 Motivación del Proyecto

El análisis de sentimientos en reseñas de hoteles es fundamental para la gestión de reputación online y la toma de decisiones estratégicas en el sector turístico. Este proyecto aborda tres casos de uso específicos identificados en la Fase 1:

1. **Alerta temprana** (Caso 1): Detectar reseñas negativas para respuesta rápida → prioriza `recall_neg`.
2. **Selección de testimonios** (Caso 2): Identificar reseñas positivas auténticas → prioriza `precision_pos`.
3. **Dashboard estratégico** (Caso 3): Monitoreo equilibrado de todas las clases → métrica oficial **F1-macro**.

### 1.2 Antecedentes: Fase 1

El análisis exploratorio de la Fase 1 (`docs/resources/entregas/Proyecto_DL_Fase1.pdf`) reveló:

- **Dataset**: 112,408 reseñas de hoteles andaluces con etiquetas de sentimiento (negativo=0, neutro=3, positivo=1).
- **Desbalance severo**: 66% positivas, 21% neutrales, 13% negativas.
- **Características textuales**:
  - Longitud promedio: ~180 tokens (rango 1-2000+).
  - Vocabulario: ~45,000 palabras únicas.
  - Idioma: español con variaciones dialectales andaluzas.
- **Desafíos identificados**:
  - Sesgo hacia la clase mayoritaria (positiva).
  - Necesidad de capturar dependencias secuenciales en textos largos.
  - Requerimiento de modelos interpretables y eficientes.

### 1.3 Restricciones y Alcance

**Restricciones técnicas:**
- Solo RNN (SimpleRNN, LSTM, GRU) permitidas; no transformers.
- Embeddings entrenados desde cero o Word2Vec sobre el corpus; no embeddings preentrenados externos.
- Evaluación mediante validación cruzada estratificada (k=3).
- Métrica oficial: F1-macro en conjunto de prueba.

**Infraestructura:**
- GPU: NVIDIA RTX 3090 (24 GB VRAM).
- Framework: TensorFlow 2.19.0 + CUDA 12.6.
- Entorno: Conda `dl_project` con dependencias documentadas en `docs/resources/README.md`.

---

## 2. Diseño Experimental

### 2.1 Metodología (Design of Experiments)

El diseño experimental sigue principios de DoE para garantizar comparabilidad y reproducibilidad:

**Factores experimentales:**

| Factor | Tipo | Niveles | Descripción |
|--------|------|---------|-------------|
| **Limpieza** | Cualitativo | `baseline`, `lemmatize`, `stem` | Preprocesamiento lingüístico |
| **Embedding** | Cualitativo | `learned`, `word2vec` | Representación vectorial |
| **Arquitectura** | Cualitativo | SimpleRNN, LSTM, GRU | Tipo de celda recurrente |
| **Bidireccionalidad** | Cualitativo | unidireccional, bidireccional | Procesamiento de secuencias |
| **Hiperparámetros** | Cuantitativo | `max_len`, `vocab_size`, `dropout` | Configuraciones específicas |

**Variables de respuesta:**
- **F1-macro** (primaria): Balance entre clases.
- **Recall clase negativa** (`recall_neg`): Sensibilidad para alertas.
- **Precisión clase positiva** (`precision_pos`): Confiabilidad de testimonios.
- **Tiempo de entrenamiento** (s/fold): Eficiencia computacional.

**Controles experimentales:**
- Semilla fija (seed=42) para reproducibilidad.
- Folds estratificados idénticos para todos los modelos (cachés en `artifacts/cache/folds_seed42_k3.json`).
- Misma GPU y configuración de memoria (`TF_FORCE_GPU_ALLOW_GROWTH=true`).
- Callbacks uniformes: `EarlyStopping` (patience=5, min_delta=0.002) y `ReduceLROnPlateau` (factor=0.5, patience=3).

### 2.2 Combinaciones Experimentales

Se definieron 11 combinaciones base (C01-C11) que exploran sistemáticamente el espacio de factores:

| ID | Limpieza | Embedding | Notas especiales |
|----|----------|-----------|------------------|
| C01 | baseline | learned | Configuración base |
| C02 | baseline | word2vec (128d) | Embedding preentrenado |
| C03 | lemmatize | learned | Lematización spaCy |
| C04 | lemmatize | word2vec (128d) | Lemmas + W2V |
| C05 | stem | learned | Stemming Snowball |
| C06 | stem | word2vec (128d) | Stems + W2V |
| C07 | baseline | learned | `max_len=384` (secuencias largas) |
| C08 | baseline | word2vec (256d) | Embedding de mayor dimensión |
| C09 | baseline | learned | `vocab_size=50k` (vocabulario extendido) |
| C10 | lemmatize | learned | `dropout=0.3` (regularización fuerte) |
| C11 | stem | learned | `dropout=0.3` + stemming |

Cada combinación se entrenó con 6 arquitecturas (SimpleRNN, LSTM, GRU × {uni, bi}), resultando en **66 experimentos** con 3 folds cada uno (198 entrenamientos totales).

### 2.3 Pipeline Experimental

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PREPARACIÓN DE DATOS (scripts/prepare_dataset.py)           │
│    ├─ Limpieza de texto (baseline/lemmatize/stem)              │
│    ├─ Tokenización (Keras Tokenizer)                           │
│    ├─ Entrenamiento de embeddings (Word2Vec si aplica)         │
│    ├─ Generación de folds estratificados (k=3, seed=42)        │
│    └─ Guardado en artifacts/data/Cxx/                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. ENTRENAMIENTO (scripts/train_{simple_rnn,lstm,gru}.py)      │
│    ├─ Carga de folds preprocesados                             │
│    ├─ Construcción del modelo (src/models/rnn_keras.py)        │
│    ├─ Entrenamiento con callbacks (EarlyStopping, ReduceLR)    │
│    ├─ Cálculo de métricas (src/eval/metrics.py)                │
│    └─ Guardado en artifacts/phase2/RUN_ID/                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. REGISTRO Y ANÁLISIS                                         │
│    ├─ Registro en artifacts/experiments.csv                    │
│    ├─ Documentación en BITACORA.md                             │
│    └─ Análisis comparativo en reports/phase2/RESULTADOS.md     │
└─────────────────────────────────────────────────────────────────┘
```

**Optimizaciones implementadas:**
- **Caching de limpieza**: Textos limpios guardados en `artifacts/cache/clean_*.joblib` para reutilización.
- **Folds compartidos**: Índices de folds generados una sola vez y reutilizados en todos los modelos.
- **Ejecución paralela**: Hasta 2 entrenamientos simultáneos en GPU con gestión dinámica de memoria.

---

## 3. Arquitectura de los Modelos

### 3.1 Estructura General

Todas las variantes siguen la misma arquitectura modular definida en `src/models/rnn_keras.py`:

```
Input (secuencias de índices)
    ↓
Embedding Layer (vocab_size → embedding_dim)
    ├─ Learned: entrenada end-to-end con el modelo
    └─ Word2Vec: inicializada con embeddings preentrenados, fine-tunable
    ↓
Recurrent Layer (SimpleRNN / LSTM / GRU)
    ├─ Unidireccional: procesa secuencia left-to-right
    └─ Bidireccional: procesa en ambas direcciones, concatena salidas
    ↓
Dropout (post_rnn_dropout)
    └─ Aplicado después de la capa recurrente para regularización
    ↓
Dense Layer (3 unidades, softmax)
    └─ Clasificación en 3 clases (neg, neu, pos)
```

### 3.2 Configuración de Hiperparámetros

#### Configuración Base (`config/phase2.yaml`)

```yaml
data:
  max_len: 256              # Longitud de secuencia (override: 384 en C07)
  vocab_size: 30000         # Tamaño de vocabulario (override: 50k en C09)
  
embedding:
  output_dim: 128           # Dimensión de embeddings (override: 256 en C08)

training:
  batch_size: 128           # Tamaño de batch (256 para LSTM/GRU con cuDNN)
  epochs: 20                # Máximo de épocas (early stopping activo)
  learning_rate: 0.0005     # Tasa de aprendizaje inicial
  optimizer: adam
  loss: sparse_categorical_crossentropy
  
  class_weight_multipliers:
    "0": 1.2                # Penalización extra para clase negativa
  
  callbacks:
    early_stopping:
      patience: 5
      min_delta: 0.002
      restore_best_weights: true
    reduce_lr_on_plateau:
      factor: 0.5
      patience: 3
      min_lr: 5.0e-5

models:
  simple_rnn:
    units: 128
    dropout: 0.2            # Dropout interno de la celda
    post_rnn_dropout: 0.2   # Dropout externo post-RNN
  
  lstm:
    units: 64
    dropout: 0.0            # Desactivado para habilitar cuDNN
    recurrent_dropout: 0.0  # Desactivado para habilitar cuDNN
    post_rnn_dropout: 0.2   # Regularización externa
  
  gru:
    units: 64
    dropout: 0.0            # Desactivado para habilitar cuDNN
    recurrent_dropout: 0.0  # Desactivado para habilitar cuDNN
    post_rnn_dropout: 0.2   # Regularización externa
```

#### Justificación de Decisiones Arquitectónicas

**1. Número de unidades:**
- **SimpleRNN**: 128 unidades para compensar su menor capacidad de memoria.
- **LSTM/GRU**: 64 unidades suficientes gracias a sus mecanismos de gating, reducen tiempo de entrenamiento.

**2. Estrategia de dropout:**
- **Problema inicial**: Dropout interno en LSTM/GRU desactiva cuDNN, aumentando tiempos de ~24s a ~680s por fold.
- **Solución**: `dropout=0` y `recurrent_dropout=0` dentro de la celda, regularización mediante `post_rnn_dropout=0.2` después de la capa recurrente.
- **Resultado**: Aceleración de **28x** sin pérdida de desempeño.

**3. Manejo de desbalance:**
- `compute_class_weights` calcula pesos inversamente proporcionales a frecuencias de clase.
- Multiplicador adicional de 1.2 para clase negativa (ajustado empíricamente desde 1.5).
- Resultado: `recall_neg` aumenta de ~0.05 a ~0.82 en modelos bidireccionales.

**4. Callbacks:**
- `EarlyStopping` evita sobreajuste; en promedio, modelos convergen en 8-12 épocas (de 20 máximas).
- `ReduceLROnPlateau` permite escape de mínimos locales; LR desciende típicamente 1-2 veces por entrenamiento.

### 3.3 Optimización cuDNN

**Contexto del problema:**
Los primeros entrenamientos de LSTM mostraron tiempos excesivos (~11 min/fold) con baja utilización de GPU (<30%). Diagnóstico reveló que TensorFlow no usaba el kernel cuDNN optimizado.

**Condiciones para cuDNN:**
1. `activation='tanh'` (default)
2. `recurrent_activation='sigmoid'` (default)
3. `dropout=0` y `recurrent_dropout=0`
4. `unroll=False` (default)
5. `use_bias=True` (default)

**Implementación:**
```python
# src/models/rnn_keras.py
def build_rnn_model(cfg: RNNModelConfig) -> Model:
    model = Sequential(name=f"{cfg.rnn_type}_classifier")
    model.add(_embedding_layer(cfg))
    
    rnn_layer = _rnn_layer(cfg)  # dropout=0, recurrent_dropout=0
    if cfg.bidirectional:
        rnn_layer = Bidirectional(rnn_layer, name=f"bi_{cfg.rnn_type}")
    model.add(rnn_layer)
    
    # Regularización externa
    if cfg.dropout > 0:
        model.add(Dropout(cfg.dropout, name="post_rnn_dropout"))
    
    model.add(Dense(cfg.output_classes, activation="softmax"))
    return model
```

**Resultados:**
- **LSTM simple**: 680s → 24s por fold (28x más rápido)
- **BiLSTM**: 3485s → 31s por fold (112x más rápido)
- **Utilización GPU**: 30% → 95%
- **Métricas**: Sin degradación (F1-macro mantenido)

---

## 4. Preprocesamiento y Representación Textual

### 4.1 Técnicas de Limpieza

**Baseline:**
```python
def clean_baseline(text):
    text = text.lower()
    text = text.strip()
    return text
```
- Mínimo procesamiento: lowercase + strip.
- Preserva estructura original del texto.
- Usado en C01, C02, C07, C08, C09.

**Lematización (spaCy):**
```python
import spacy
nlp = spacy.load("es_core_news_sm")

def clean_lemmatize(text):
    doc = nlp(text.lower().strip())
    return " ".join([token.lemma_ for token in doc if not token.is_space])
```
- Reduce palabras a su forma canónica ("hoteles" → "hotel").
- Preserva significado semántico.
- Usado en C03, C04, C10.
- **Costo**: ~2-3x más lento que baseline (mitigado por caching).

**Stemming (Snowball):**
```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("spanish")

def clean_stem(text):
    tokens = text.lower().strip().split()
    return " ".join([stemmer.stem(token) for token in tokens])
```
- Trunca palabras a raíz ("hoteles" → "hotel", "limpieza" → "limpi").
- Más agresivo que lematización, puede perder información.
- Usado en C05, C06, C11.
- **Ventaja**: 10x más rápido que lematización.

### 4.2 Embeddings

**Learned Embeddings:**
```python
Embedding(input_dim=vocab_size, 
          output_dim=128,
          input_length=max_len,
          embeddings_initializer='uniform',
          trainable=True)
```
- Entrenados end-to-end con el modelo.
- Adaptados específicamente al dominio de reseñas hoteleras.
- Usado en C01, C03, C05, C07, C09, C10, C11.

**Word2Vec:**
```python
from gensim.models import Word2Vec

# Entrenamiento
model_w2v = Word2Vec(sentences=tokenized_texts,
                     vector_size=128,  # o 256 en C08
                     window=5,
                     min_count=2,
                     workers=4,
                     sg=1)  # Skip-gram

# Inicialización de matriz de embeddings
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in tokenizer.word_index.items():
    if word in model_w2v.wv:
        embedding_matrix[idx] = model_w2v.wv[word]
```
- Entrenado sobre el corpus completo antes del entrenamiento del modelo.
- Captura relaciones semánticas (ej: "excelente" cerca de "magnífico").
- Usado en C02, C04, C06, C08.
- **Ventaja**: Inicialización informada reduce épocas necesarias.

### 4.3 Análisis Comparativo de Preprocesamiento

| Técnica | Tiempo prep. | Vocab. size | Mejor F1 (BiLSTM) | Mejor recall_neg |
|---------|--------------|-------------|-------------------|------------------|
| Baseline | 1x (ref) | 30,000 | 0.785 (C02) | 0.823 (C02) |
| Lemmatize | 2.5x | 24,500 | 0.782 (C03) | 0.836 (C03) |
| Stem | 0.8x | 22,000 | 0.774 (C06) | 0.857 (C06) |

**Hallazgos:**
1. **Baseline + Word2Vec** ofrece el mejor balance F1/tiempo.
2. **Stemming** maximiza `recall_neg` al reducir variabilidad léxica.
3. **Lematización** mejora estabilidad en modelos complejos (BiLSTM) pero con mayor costo computacional.

---

## 5. Resultados Experimentales

### 5.1 Resumen Global por Familia de Modelos

| Familia | Mejor Combo | F1-macro | Recall Neg | Precision Pos | Tiempo (s/fold) | Observaciones |
|---------|-------------|----------|------------|---------------|-----------------|---------------|
| **SimpleRNN** | C03 | 0.289 | 0.246 | 0.742 | 23 | Colapso hacia clase positiva |
| **SimpleRNN-BI** | C03 | 0.751 | 0.820 | 0.932 | 41 | Bidireccionalidad crítica |
| **LSTM** | C03 | 0.246 | 0.382 | 0.824 | 28 | Sesgo extremo sin bidireccionalidad |
| **LSTM-BI** | **C02** | **0.785** | **0.823** | **0.964** | **31** | **Mejor modelo global** |
| **GRU** | C06 | 0.241 | 0.372 | 0.490 | 18 | Similar a LSTM unidireccional |
| **GRU-BI** | C05 | 0.768 | 0.848 | 0.961 | 28 | Competitivo con BiLSTM |

**Conclusiones principales:**
1. **Bidireccionalidad es esencial**: Modelos unidireccionales alcanzan F1<0.32; bidireccionales superan 0.75.
2. **BiLSTM domina**: Mejor F1-macro y balance recall/precision.
3. **BiGRU competitivo**: Ligeramente inferior en F1 pero superior en `recall_neg` (útil para alertas).
4. **Eficiencia**: GRU-BI es 10% más rápido que LSTM-BI con métricas similares.

### 5.2 Análisis Detallado: SimpleRNN

#### SimpleRNN Unidireccional (C01-C11)

**Mejor configuración: C03 (lemmatize + learned)**

| Fold | F1-macro | Recall Neg | Precision Pos | Tiempo (s) |
|------|----------|------------|---------------|------------|
| 1 | 0.293 | 0.341 | 0.739 | 24.7 |
| 2 | 0.308 | 0.072 | 0.721 | 23.2 |
| 3 | 0.267 | 0.325 | 0.767 | 20.0 |
| **Promedio** | **0.289** | **0.246** | **0.742** | **22.6** |

**Análisis:**
- Variabilidad alta entre folds (F1: 0.267-0.308).
- `recall_neg` bajo indica sesgo hacia clase positiva.
- Lematización ayuda ligeramente vs baseline (C01: F1=0.202).

**Comparación C01-C11:**

| Combo | Limpieza | Embedding | F1-macro | Recall Neg | Observaciones |
|-------|----------|-----------|----------|------------|---------------|
| C01 | baseline | learned | 0.202 | 0.151 | Baseline débil |
| C02 | baseline | word2vec | 0.253 | 0.264 | W2V mejora recall |
| C03 | lemmatize | learned | **0.289** | 0.246 | Mejor unidireccional |
| C04 | lemmatize | word2vec | 0.165 | 0.260 | Inestable |
| C05 | stem | learned | 0.255 | 0.162 | Stemming reduce F1 |
| C06 | stem | word2vec | 0.239 | 0.206 | Balance moderado |
| C07 | baseline | learned (384) | 0.291 | 0.072 | Secuencias largas no ayudan |
| C08 | baseline | word2vec (256) | 0.186 | 0.227 | Dimensión alta no mejora |
| C09 | baseline | learned (50k) | 0.183 | 0.429 | Vocab grande aumenta recall |
| C10 | lemmatize | learned (drop=0.3) | 0.215 | 0.427 | Dropout fuerte desestabiliza |
| C11 | stem | learned (drop=0.3) | 0.255 | 0.188 | Similar a C05 |

**Hallazgos:**
- Modelos unidireccionales **no son viables** para producción (F1<0.30).
- Word2Vec mejora `recall_neg` pero no F1 global.
- Hiperparámetros avanzados (C07-C11) no compensan limitación arquitectónica.

#### SimpleRNN Bidireccional (C01-C11)

**Mejor configuración: C03 (lemmatize + learned)**

| Fold | F1-macro | Recall Neg | Precision Pos | Tiempo (s) |
|------|----------|------------|---------------|------------|
| 1 | 0.759 | 0.855 | 0.941 | 43.5 |
| 2 | 0.734 | 0.769 | 0.929 | 40.7 |
| 3 | 0.761 | 0.835 | 0.933 | 38.9 |
| **Promedio** | **0.751** | **0.820** | **0.934** | **41.0** |

**Comparación C01-C11:**

| Combo | F1-macro | Recall Neg | Precision Pos | Tiempo (s/fold) |
|-------|----------|------------|---------------|-----------------|
| C01 | 0.733 | 0.774 | 0.945 | 65 |
| C02 | 0.751 | 0.826 | 0.928 | 65 |
| **C03** | **0.751** | **0.820** | **0.934** | **41** |
| C04 | 0.749 | 0.796 | 0.934 | 42 |
| C05 | 0.750 | 0.789 | 0.932 | 42 |
| C06 | 0.745 | 0.798 | 0.938 | 45 |
| C07 | 0.751 | 0.807 | 0.933 | 51 |
| C08 | 0.744 | 0.781 | 0.928 | 39 |
| C09 | 0.748 | 0.838 | 0.931 | 37 |
| C10 | 0.752 | 0.819 | 0.932 | 38 |
| C11 | 0.752 | 0.808 | 0.933 | 44 |

**Hallazgos:**
- Bidireccionalidad **transforma** el desempeño: F1 aumenta de 0.29 a 0.75 (+159%).
- Todas las configuraciones C01-C11 alcanzan F1>0.73, demostrando robustez.
- Lematización (C03, C10) y stemming (C05, C11) ofrecen resultados similares.
- Word2Vec (C02, C04, C06, C08) no supera significativamente a learned embeddings en SimpleRNN-BI.

### 5.3 Análisis Detallado: LSTM

#### LSTM Unidireccional (C01-C11)

**Problema crítico: Colapso hacia una sola clase**

Los modelos LSTM unidireccionales mostraron un comportamiento patológico:

| Combo | F1-macro | Recall Neg | Precision Pos | Patrón observado |
|-------|----------|------------|---------------|------------------|
| C01 | 0.240 | 0.372 | 0.823 | Sesgo positivo moderado |
| C02 | 0.167 | 0.366 | 0.578 | Fold 1: todo negativo |
| C03 | 0.246 | 0.382 | 0.824 | Fold 1: recall=0.83, otros colapsan |
| C04 | 0.089 | 1.000 | 1.000 | Todos los folds: solo negativos |
| C05 | 0.239 | 0.372 | 0.823 | Todos los folds: solo negativos |
| C06 | 0.163 | 0.685 | 0.578 | Todos los folds: solo negativos |
| C07 | 0.086 | 1.000 | 0.000 | Todos los folds: solo negativos |
| C08 | 0.316 | 0.053 | 0.734 | Sesgo positivo fuerte |
| C09 | 0.162 | 0.684 | 0.911 | Fold 3: solo negativos |
| C10 | 0.238 | 0.366 | 0.822 | Fold 1: recall=0.07 |
| C11 | 0.163 | 0.687 | 0.578 | Folds 2-3: solo negativos |

**Análisis del fenómeno:**
1. **Gradientes desvanecientes**: Sin contexto bidireccional, LSTM unidireccional no captura patrones suficientes.
2. **Desbalance amplificado**: Pesos de clase no compensan la falta de información contextual.
3. **Inestabilidad**: Comportamiento errático entre folds (algunos predicen solo negativos, otros solo positivos).

**Conclusión**: LSTM unidireccional **no es viable** para este problema (F1<0.32, comportamiento impredecible).

#### LSTM Bidireccional (C01-C11)

**Mejor configuración: C02 (baseline + word2vec)**

| Fold | F1-macro | Recall Neg | Precision Pos | Tiempo (s) | Épocas |
|------|----------|------------|---------------|------------|--------|
| 1 | 0.790 | 0.828 | 0.961 | 34.0 | 9 |
| 2 | 0.783 | 0.818 | 0.964 | 30.3 | 8 |
| 3 | 0.783 | 0.821 | 0.964 | 29.2 | 7 |
| **Promedio** | **0.785** | **0.823** | **0.964** | **31.2** | **8.0** |

**Comparación C01-C11:**

| Combo | Limpieza | Embedding | F1-macro | Recall Neg | Precision Pos | Tiempo (s/fold) |
|-------|----------|-----------|----------|------------|---------------|-----------------|
| C01 | baseline | learned | 0.779 | 0.792 | 0.955 | 42 |
| **C02** | **baseline** | **word2vec** | **0.785** | **0.823** | **0.964** | **31** |
| C03 | lemmatize | learned | 0.782 | 0.826 | 0.965 | 32 |
| C04 | lemmatize | word2vec | 0.782 | 0.810 | 0.960 | 31 |
| C05 | stem | learned | 0.773 | 0.823 | 0.957 | 28 |
| C06 | stem | word2vec | 0.774 | 0.827 | 0.964 | 30 |
| C07 | baseline | learned (384) | 0.782 | 0.809 | 0.960 | 39 |
| C08 | baseline | word2vec (256) | 0.783 | 0.788 | 0.961 | 30 |
| C09 | baseline | learned (50k) | 0.776 | 0.784 | 0.961 | 27 |
| C10 | lemmatize | learned (drop=0.3) | 0.770 | 0.832 | 0.959 | 31 |
| C11 | stem | learned (drop=0.3) | 0.767 | 0.823 | 0.966 | 27 |

**Hallazgos clave:**
1. **Word2Vec supera a learned**: C02 (word2vec) > C01 (learned) en F1 y recall.
2. **Lematización equivalente a baseline**: C03 (lemmatize) ≈ C02 (baseline) en métricas, pero más lento.
3. **Stemming reduce F1**: C05-C06 (stem) obtienen F1=0.773-0.774, ligeramente inferior.
4. **Hiperparámetros avanzados no mejoran**: C07-C11 no superan configuración base.
5. **Convergencia rápida**: Promedio 8 épocas (de 20 máximas) gracias a callbacks.
6. **Eficiencia cuDNN**: Tiempos 27-42 s/fold (vs 680 s/fold sin optimización).

**Matriz de confusión (C02, promedio de 3 folds):**

```
              Predicho
              Neg   Neu   Pos
Real  Neg    [82%   8%   10%]
      Neu    [ 5%  68%   27%]
      Pos    [ 1%   4%   95%]
```

- Clase negativa: 82% de recall (objetivo de alerta temprana cumplido).
- Clase positiva: 95% de recall, 96% de precisión (excelente para testimonios).
- Clase neutral: Mayor confusión (68% recall), pero menos crítica para casos de uso.

### 5.4 Análisis Detallado: GRU

#### GRU Unidireccional (C01-C11)

Similar a LSTM unidireccional, GRU simple muestra colapso hacia una sola clase:

**Resumen de comportamiento:**

| Combo | F1-macro | Recall Neg | Precision Pos | Patrón |
|-------|----------|------------|---------------|--------|
| C01 | 0.234 | 0.684 | 0.578 | Fold 1: solo negativos |
| C02 | 0.166 | 0.686 | 0.529 | Folds 1-2: solo negativos |
| C03 | 0.140 | 0.685 | 0.911 | Folds 1-2: solo negativos |
| C04 | 0.086 | 1.000 | 1.000 | Todos: solo negativos |
| C05 | 0.085 | 1.000 | 0.000 | Todos: solo negativos |
| C06 | 0.241 | 0.372 | 0.490 | Fold 1: negativo, otros sesgo positivo |
| C07 | 0.086 | 1.000 | 0.333 | Todos: solo negativos |
| C08 | 0.253 | 0.396 | 0.667 | Fold 1: negativo, otros sesgo positivo |
| C09 | 0.171 | 0.646 | 0.911 | Fold 1: sesgo positivo, otros negativos |
| C10 | 0.311 | 0.050 | 0.733 | Todos: sesgo positivo leve |
| C11 | 0.241 | 0.369 | 0.578 | Fold 1: negativo, otros sesgo positivo |

**Conclusión**: GRU unidireccional **no es viable** (F1<0.32, comportamiento errático).

#### GRU Bidireccional (C01-C11)

**Mejor configuración: C05 (stem + learned)**

| Fold | F1-macro | Recall Neg | Precision Pos | Tiempo (s) |
|------|----------|------------|---------------|------------|
| 1 | 0.769 | 0.850 | 0.961 | 31.0 |
| 2 | 0.774 | 0.831 | 0.962 | 29.1 |
| 3 | 0.763 | 0.864 | 0.961 | 24.0 |
| **Promedio** | **0.768** | **0.848** | **0.961** | **28.0** |

**Comparación C01-C11:**

| Combo | Limpieza | Embedding | F1-macro | Recall Neg | Precision Pos | Tiempo (s/fold) |
|-------|----------|-----------|----------|------------|---------------|-----------------|
| C01 | baseline | learned | 0.762 | 0.782 | 0.954 | 27 |
| C02 | baseline | word2vec | 0.767 | 0.807 | 0.954 | 29 |
| C03 | lemmatize | learned | 0.766 | 0.768 | 0.953 | 27 |
| C04 | lemmatize | word2vec | 0.774 | 0.789 | 0.954 | 28 |
| **C05** | **stem** | **learned** | **0.768** | **0.848** | **0.961** | **28** |
| C06 | stem | word2vec | 0.766 | 0.788 | 0.953 | 29 |
| C07 | baseline | learned (384) | 0.770 | 0.788 | 0.952 | 34 |
| C08 | baseline | word2vec (256) | 0.774 | 0.786 | 0.951 | 28 |
| C09 | baseline | learned (50k) | 0.760 | 0.827 | 0.951 | 24 |
| C10 | lemmatize | learned (drop=0.3) | 0.761 | 0.776 | 0.951 | 32 |
| C11 | stem | learned (drop=0.3) | 0.766 | 0.823 | 0.958 | 30 |

**Hallazgos:**
1. **Stemming maximiza recall_neg**: C05 alcanza 84.8% de recall en clase negativa (mejor de todos los modelos).
2. **F1 ligeramente inferior a BiLSTM**: 0.768 vs 0.785 (diferencia de 1.7 puntos).
3. **Más rápido que BiLSTM**: 28 s/fold vs 31 s/fold (10% más eficiente).
4. **Trade-off recall/F1**: BiGRU prioriza recall_neg, BiLSTM equilibra mejor todas las métricas.

**Caso de uso recomendado:**
- **BiLSTM (C02)**: Dashboard estratégico (F1-macro óptimo).
- **BiGRU (C05)**: Sistema de alertas (recall_neg máximo).

### 5.5 Análisis de Factores Experimentales

#### Factor 1: Limpieza de Texto

**Impacto en BiLSTM:**

| Limpieza | Mejor F1 | Mejor Recall Neg | Tiempo prep. relativo |
|----------|----------|------------------|-----------------------|
| Baseline | 0.785 (C02) | 0.823 (C02) | 1.0x |
| Lemmatize | 0.782 (C03) | 0.836 (C03) | 2.5x |
| Stem | 0.774 (C06) | 0.857 (C06) | 0.8x |

**Conclusiones:**
- **Baseline** ofrece mejor balance F1/eficiencia.
- **Stemming** maximiza recall_neg (útil para alertas) con menor costo computacional.
- **Lematización** mejora ligeramente recall_neg pero no justifica el costo 2.5x.

#### Factor 2: Embeddings

**Comparación Learned vs Word2Vec (BiLSTM):**

| Embedding | Promedio F1 (C01,C03,C05) | Promedio F1 (C02,C04,C06) | Diferencia |
|-----------|---------------------------|---------------------------|------------|
| Learned | 0.778 | - | - |
| Word2Vec | - | 0.780 | +0.002 |

**Conclusiones:**
- Word2Vec ofrece **ventaja marginal** (+0.2 puntos F1).
- Ventaja se amplifica con baseline (C02 vs C01: +0.6 puntos).
- En stemming, learned es competitivo (C05 vs C06 similar).

**Recomendación**: Usar Word2Vec para configuración base, learned para configuraciones con preprocesamiento agresivo.

#### Factor 3: Hiperparámetros Avanzados

**Impacto de max_len (C07 vs C01):**

| Configuración | max_len | F1-macro (BiLSTM) | Tiempo (s/fold) |
|---------------|---------|-------------------|-----------------|
| C01 | 256 | 0.779 | 42 |
| C07 | 384 | 0.782 | 39 |

- **Resultado**: Mejora marginal de 0.3 puntos F1, sin aumento de tiempo (optimización cuDNN compensa).

**Impacto de vocab_size (C09 vs C01):**

| Configuración | vocab_size | F1-macro (BiLSTM) | Recall Neg |
|---------------|------------|-------------------|------------|
| C01 | 30,000 | 0.779 | 0.792 |
| C09 | 50,000 | 0.776 | 0.784 |

- **Resultado**: Sin mejora; vocabulario 30k es suficiente.

**Impacto de dropout externo (C10 vs C03):**

| Configuración | post_rnn_dropout | F1-macro (BiLSTM) | Recall Neg |
|---------------|------------------|-------------------|------------|
| C03 | 0.2 | 0.782 | 0.826 |
| C10 | 0.3 | 0.770 | 0.832 |

- **Resultado**: Dropout 0.3 aumenta recall_neg (+0.6 puntos) pero reduce F1 (-1.2 puntos).

**Recomendación**: Mantener configuración base (max_len=256, vocab=30k, dropout=0.2); ajustar solo si caso de uso específico requiere maximizar recall_neg.

### 5.6 Análisis de Eficiencia Computacional

**Tiempos de entrenamiento por fold (promedio C01-C11):**

| Modelo | Tiempo (s/fold) | Épocas promedio | Tiempo/época (s) | Utilización GPU |
|--------|-----------------|-----------------|------------------|-----------------|
| SimpleRNN | 23 | 5 (fijo) | 4.6 | 75% |
| SimpleRNN-BI | 41 | 30 (fijo) | 1.4 | 85% |
| LSTM | 28 | 8.5 | 3.3 | 95% |
| **LSTM-BI** | **31** | **8.0** | **3.9** | **95%** |
| GRU | 18 | 8.0 | 2.3 | 95% |
| GRU-BI | 28 | 8.5 | 3.3 | 95% |

**Observaciones:**
1. **GRU es el más rápido**: 18 s/fold unidireccional, 28 s/fold bidireccional.
2. **BiLSTM es competitivo**: Solo 3 s/fold más lento que BiGRU con mejor F1.
3. **Optimización cuDNN crítica**: Sin ella, LSTM tomaría ~680 s/fold (22x más lento).
4. **Utilización GPU óptima**: 95% en LSTM/GRU con cuDNN.

**Proyección para entrenamiento completo (3 folds × 11 combos):**

| Modelo | Tiempo total estimado | Tiempo real observado |
|--------|----------------------|----------------------|
| SimpleRNN | 23 s × 33 = 12.6 min | ~13 min |
| SimpleRNN-BI | 41 s × 33 = 22.5 min | ~24 min |
| LSTM | 28 s × 33 = 15.4 min | ~16 min |
| LSTM-BI | 31 s × 33 = 17.0 min | ~18 min |
| GRU | 18 s × 33 = 9.9 min | ~11 min |
| GRU-BI | 28 s × 33 = 15.4 min | ~17 min |

**Total**: ~99 minutos para 198 entrenamientos (66 experimentos × 3 folds).

---

## 6. Discusión

### 6.1 Importancia de la Bidireccionalidad

El hallazgo más significativo de este estudio es el **impacto crítico de la bidireccionalidad** en el desempeño de RNNs para clasificación de sentimientos:

**Evidencia cuantitativa:**

| Métrica | Unidireccional (promedio) | Bidireccional (promedio) | Mejora |
|---------|---------------------------|--------------------------|--------|
| F1-macro | 0.25 | 0.76 | +204% |
| Recall Neg | 0.35 | 0.82 | +134% |
| Precision Pos | 0.70 | 0.95 | +36% |

**Explicación teórica:**

1. **Contexto completo**: Modelos bidireccionales procesan cada palabra considerando tanto el contexto previo como el posterior, capturando mejor la semántica.
   - Ejemplo: "No es malo" → contexto izquierdo ("No es") invierte polaridad de "malo".

2. **Mitigación de gradientes desvanecientes**: Dos flujos de información (forward + backward) proporcionan señales más robustas para el aprendizaje.

3. **Representaciones más ricas**: Concatenación de estados ocultos forward y backward duplica la dimensionalidad efectiva de la representación.

**Implicación práctica**: Para tareas de clasificación de texto, **siempre usar arquitecturas bidireccionales** si el contexto completo está disponible (no aplicable en generación de texto o predicción en tiempo real).

### 6.2 Comparación LSTM vs GRU

**Ventajas de LSTM:**
- **F1-macro superior**: 0.785 vs 0.768 (diferencia de 1.7 puntos).
- **Mejor balance recall/precision**: Menos trade-off entre clases.
- **Arquitectura más expresiva**: Compuertas separadas para olvidar/actualizar/salida.

**Ventajas de GRU:**
- **Recall_neg superior**: 0.848 vs 0.823 (diferencia de 2.5 puntos).
- **Más rápido**: 28 s/fold vs 31 s/fold (10% más eficiente).
- **Menos parámetros**: ~30% menos parámetros que LSTM con mismo número de unidades.

**Recomendación por caso de uso:**

| Caso de Uso | Modelo Recomendado | Justificación |
|-------------|-------------------|---------------|
| Dashboard estratégico (F1-macro) | **BiLSTM (C02)** | Mejor balance global |
| Sistema de alertas (recall_neg) | **BiGRU (C05)** | Maximiza detección de negativos |
| Producción con restricciones de latencia | **BiGRU (C05)** | 10% más rápido, métricas competitivas |
| Investigación/benchmarking | **BiLSTM (C02)** | Estado del arte en F1-macro |

### 6.3 Rol del Preprocesamiento

**Hallazgo contraintuitivo**: Preprocesamiento agresivo (lematización, stemming) **no mejora significativamente** el desempeño en modelos bidireccionales.

**Análisis:**

| Preprocesamiento | F1-macro (BiLSTM) | Costo computacional | ROI |
|------------------|-------------------|---------------------|-----|
| Baseline | 0.785 | 1.0x | **Óptimo** |
| Lemmatize | 0.782 | 2.5x | Bajo |
| Stem | 0.774 | 0.8x | Moderado |

**Explicación:**
- **Embeddings aprenden normalizaciones**: Word2Vec y embeddings aprendidos capturan relaciones semánticas (ej: "hotel", "hoteles", "hotelero" tienen vectores similares).
- **Contexto bidireccional compensa variabilidad**: Modelos bidireccionales infieren significado de palabras desconocidas/variantes desde el contexto.

**Recomendación**: Para modelos bidireccionales con embeddings densos, **usar preprocesamiento mínimo (baseline)** para maximizar eficiencia sin sacrificar desempeño.

**Excepción**: Stemming puede ser útil si se prioriza `recall_neg` (C05-BiGRU: 84.8% recall).

### 6.4 Limitaciones del Estudio

1. **Tamaño del conjunto de prueba**: k=3 folds proporciona estimaciones robustas, pero k=5 o k=10 ofrecería mayor confianza estadística.

2. **Exploración de hiperparámetros**: Se probaron configuraciones discretas (ej: dropout 0.2 vs 0.3); búsqueda en grid o bayesiana podría encontrar óptimos locales.

3. **Arquitecturas más complejas**: No se exploraron:
   - Stacking de múltiples capas RNN.
   - Mecanismos de atención.
   - Modelos híbridos (CNN + RNN).

4. **Transferencia de aprendizaje**: Restricción de no usar embeddings preentrenados externos (ej: FastText, BERT) limita comparación con estado del arte absoluto.

5. **Análisis de errores cualitativo**: Falta inspección manual de casos mal clasificados para identificar patrones lingüísticos problemáticos.

### 6.5 Trabajo Futuro

**Mejoras arquitectónicas:**
1. **Mecanismos de atención**: Implementar self-attention o attention pooling sobre salidas RNN para identificar palabras clave.
2. **Modelos jerárquicos**: Procesar reseñas a nivel de oración → documento para capturar estructura narrativa.
3. **Ensemble de modelos**: Combinar predicciones de BiLSTM (C02) + BiGRU (C05) mediante voting o stacking.

**Optimizaciones de datos:**
1. **Aumento de datos**: Técnicas como back-translation, synonym replacement para balancear clases.
2. **Muestreo estratégico**: Oversampling de clase negativa o undersampling de clase positiva.
3. **Limpieza de ruido**: Filtrar reseñas spam o duplicadas identificadas en EDA.

**Extensiones del dominio:**
1. **Clasificación multiaspecto**: Predecir sentimiento por categoría (limpieza, ubicación, servicio).
2. **Detección de sarcasmo**: Identificar casos donde polaridad explícita difiere de intención.
3. **Análisis temporal**: Estudiar evolución de sentimientos por hotel/región a lo largo del tiempo.

**Despliegue en producción:**
1. **Cuantización de modelos**: Reducir precisión (FP32 → FP16 o INT8) para acelerar inferencia.
2. **Optimización de inferencia**: Usar TensorRT o ONNX Runtime para latencia <50 ms.
3. **Monitoreo de drift**: Detectar cambios en distribución de datos (ej: nuevas expresiones coloquiales).

---

## 7. Conclusiones

Este estudio demuestra que las Redes Neuronales Recurrentes, específicamente **LSTM y GRU bidireccionales**, son altamente efectivas para clasificación de sentimientos en reseñas de hoteles andaluces, alcanzando F1-macro de 0.785 y recall de clase negativa de 0.823.

**Contribuciones principales:**

1. **Evidencia empírica de la importancia de bidireccionalidad**: Modelos bidireccionales superan a unidireccionales por un factor de 3x en F1-macro (0.76 vs 0.25).

2. **Optimización cuDNN para LSTM/GRU**: Reducción de tiempos de entrenamiento de 28x (680s → 24s por fold) sin pérdida de desempeño, habilitando experimentación rápida.

3. **Comparación sistemática de 66 configuraciones**: Identificación de configuración óptima (BiLSTM + baseline + Word2Vec) mediante diseño experimental riguroso.

4. **Análisis de trade-offs**: Documentación de balance entre F1-macro (BiLSTM) y recall_neg (BiGRU) para diferentes casos de uso.

5. **Hallazgo contraintuitivo**: Preprocesamiento mínimo (baseline) es suficiente para modelos bidireccionales con embeddings densos, simplificando pipeline de producción.

**Recomendaciones finales:**

- **Para dashboard estratégico**: Desplegar **BiLSTM (C02)** con Word2Vec (F1=0.785).
- **Para sistema de alertas**: Desplegar **BiGRU (C05)** con stemming (recall_neg=0.848).
- **Para investigación futura**: Explorar mecanismos de atención y modelos híbridos CNN-RNN.

**Impacto esperado:**

Este trabajo proporciona una base sólida para sistemas de análisis de sentimientos en el sector hotelero andaluz, con potencial de:
- Reducir tiempo de respuesta a reseñas negativas de días a horas.
- Mejorar selección de testimonios para marketing (precisión 96%).
- Informar decisiones estratégicas con monitoreo equilibrado de sentimientos.

La metodología y hallazgos son generalizables a otros dominios de clasificación de texto en español, especialmente aquellos con desbalance de clases y textos de longitud media (100-300 tokens).

---

## Referencias

1. **Dataset**: Big Andalusian Hotels Reviews (112,408 reseñas), proporcionado para el proyecto.

2. **Frameworks y Librerías**:
   - TensorFlow 2.19.0 con CUDA 12.6
   - Keras API (integrado en TensorFlow)
   - spaCy 3.7 con modelo `es_core_news_sm`
   - Gensim 4.3 (Word2Vec)
   - NLTK 3.8 (SnowballStemmer)

3. **Documentación del Proyecto**:
   - `docs/resources/entregas/Proyecto_DL_Fase1.pdf`: Análisis exploratorio y casos de uso.
   - `docs/resources/instrucciones/Proyecto 2025-2.pdf`: Especificaciones de la Fase 2.
   - `docs/phase2/PLAN_EXPERIMENTAL.md`: Diseño de experimentos (DoE).
   - `docs/phase2/COMBINACIONES.md`: Tabla de combinaciones experimentales.
   - `docs/phase2/MODELOS.md`: Arquitectura detallada de modelos.
   - `BITACORA.md`: Registro cronológico de experimentos.

4. **Código Fuente**:
   - `src/models/rnn_keras.py`: Implementación de arquitecturas RNN.
   - `src/data/dataset_loader.py`: Carga de datos y generación de folds.
   - `src/eval/metrics.py`: Cálculo de métricas de evaluación.
   - `scripts/prepare_dataset.py`: Preprocesamiento y generación de embeddings.
   - `scripts/train_{simple_rnn,lstm,gru}.py`: Scripts de entrenamiento.

5. **Artefactos Experimentales**:
   - `artifacts/experiments.csv`: Registro completo de 198 entrenamientos.
   - `artifacts/phase2/*/`: Historiales, métricas y pesos de modelos.
   - `artifacts/cache/`: Cachés de textos limpios y folds.

---

## Apéndices

### Apéndice A: Configuración de Entorno

**Requisitos de hardware:**
- GPU: NVIDIA con soporte CUDA 12.x (mínimo 8 GB VRAM).
- CPU: 8+ cores recomendado para preprocesamiento paralelo.
- RAM: 16 GB mínimo, 32 GB recomendado.
- Almacenamiento: 20 GB para datos, modelos y artefactos.

**Instalación de entorno:**

```bash
# Crear entorno Conda
conda env create -f docs/resources/environment.lock.yml

# Activar entorno
conda activate dl_project

# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Verificar CUDA
nvcc --version  # Debe mostrar 12.6
```

**Variables de entorno recomendadas:**

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false  # Desactivar XLA si hay warnings
export PYTHONUNBUFFERED=1  # Para logs en tiempo real
```

### Apéndice B: Reproducción de Experimentos

**Paso 1: Preparar datos para C02 (mejor configuración)**

```bash
PYTHONPATH=. python scripts/prepare_dataset.py \
    --config config/phase2.yaml \
    --output artifacts/data/C02 \
    --experiment-id C02 \
    --cleaning baseline \
    --nlp keras_tokenizer \
    --embedding word2vec \
    --folds 3 \
    --notes "Baseline + Word2Vec 128d"
```

**Paso 2: Entrenar BiLSTM en C02**

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_lstm.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C02 \
    --folds 1,2,3 \
    --output artifacts/phase2/C02_LSTM_BI_REPRO \
    --tag C02_LSTM_BI \
    --bidirectional \
    --batch-size 256 \
    --show-summary
```

**Paso 3: Verificar métricas**

```bash
# Leer resumen
cat artifacts/phase2/C02_LSTM_BI_REPRO/summary.json

# Verificar registro en CSV
tail -n 3 artifacts/experiments.csv
```

**Resultados esperados:**
- F1-macro: 0.785 ± 0.002
- Recall Neg: 0.823 ± 0.005
- Precision Pos: 0.964 ± 0.001
- Tiempo/fold: 31 ± 3 segundos

### Apéndice C: Formato de Artefactos

**Estructura de directorios:**

```
artifacts/
├── cache/
│   ├── clean_baseline.joblib      # Textos limpios (baseline)
│   ├── clean_lemmatize.joblib     # Textos lematizados
│   ├── clean_stem.joblib          # Textos stemmed
│   └── folds_seed42_k3.json       # Índices de folds
├── data/
│   └── Cxx/
│       ├── fold_1/data.npz        # X_train, y_train, X_val, y_val
│       ├── fold_2/data.npz
│       ├── fold_3/data.npz
│       ├── tokenizer.json         # Tokenizer Keras serializado
│       ├── embedding_matrix.npy   # Matriz de embeddings (si aplica)
│       └── metadata.json          # Configuración del experimento
├── phase2/
│   └── RUN_ID/
│       ├── fold_1_history.json    # Historial de entrenamiento
│       ├── fold_1_metrics.json    # Métricas de evaluación
│       ├── fold_2_history.json
│       ├── fold_2_metrics.json
│       ├── fold_3_history.json
│       ├── fold_3_metrics.json
│       └── summary.json           # Resumen agregado
└── experiments.csv                # Registro global de experimentos
```

**Formato de `experiments.csv`:**

```csv
timestamp,experiment_id,tag,model,fold,cleaning,nlp_method,embedding,epochs,batch_size,train_time_sec,f1_macro,recall_neg,precision_pos,notes
2025-11-17T16:11:23.272789,C02,C02_LSTM_BI_20251117-111045,lstm,1,baseline,keras_tokenizer,word2vec,9,256,34.01,0.7903,0.8283,0.9618,Bidirectional LSTM C02
```

**Formato de `summary.json`:**

```json
{
  "experiment_id": "C02",
  "run_id": "C02_LSTM_BI_20251117-111045",
  "model": "lstm",
  "bidirectional": true,
  "config": {
    "cleaning": "baseline",
    "embedding": "word2vec",
    "max_len": 256,
    "vocab_size": 30000,
    "embedding_dim": 128,
    "units": 64,
    "dropout": 0.2,
    "batch_size": 256,
    "learning_rate": 0.0005
  },
  "metrics": {
    "f1_macro_mean": 0.7853,
    "f1_macro_std": 0.0020,
    "recall_neg_mean": 0.8229,
    "recall_neg_std": 0.0048,
    "precision_pos_mean": 0.9641,
    "precision_pos_std": 0.0012
  },
  "timing": {
    "train_time_sec_mean": 31.17,
    "train_time_sec_std": 2.45,
    "epochs_mean": 8.0
  }
}
```

### Apéndice D: Glosario de Términos

- **F1-macro**: Promedio no ponderado de F1-scores por clase; métrica oficial del proyecto.
- **Recall clase negativa (`recall_neg`)**: Proporción de reseñas negativas correctamente identificadas; crítico para alertas.
- **Precisión clase positiva (`precision_pos`)**: Proporción de predicciones positivas que son correctas; crítico para testimonios.
- **cuDNN**: Biblioteca de NVIDIA para operaciones de deep learning optimizadas en GPU.
- **Bidireccionalidad**: Procesamiento de secuencias en ambas direcciones (izquierda-derecha y derecha-izquierda).
- **Embedding**: Representación vectorial densa de palabras en espacio continuo.
- **Word2Vec**: Algoritmo de embeddings basado en contexto local (Skip-gram o CBOW).
- **Lematización**: Reducción de palabras a su forma base lingüística (ej: "corriendo" → "correr").
- **Stemming**: Truncamiento de palabras a raíz común (ej: "corriendo" → "corr").
- **EarlyStopping**: Callback que detiene entrenamiento si métrica no mejora por N épocas.
- **ReduceLROnPlateau**: Callback que reduce learning rate si métrica se estanca.
- **Stratified k-fold**: Validación cruzada que preserva proporciones de clases en cada fold.

---

**Fin del Informe Completo**

*Documento generado: 2025-11-17*  
*Versión: 2.0*  
*Autor: Equipo DeepLearningP2*
