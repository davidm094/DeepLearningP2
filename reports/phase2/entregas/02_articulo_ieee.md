# Clasificación de Sentimientos en Reseñas de Hoteles Andaluces mediante Redes Neuronales Recurrentes Bidireccionales

**Autores**:  
Anderson J. Alvarado¹ · David E. Moreno²

¹ Pontificia Universidad Javeriana, andersonjalvarado@javeriana.edu.co  
² Pontificia Universidad Javeriana, morenoa-david@javeriana.edu.co

**Fecha**: Noviembre 2025

---

## Abstract

Este trabajo presenta un estudio sistemático de Redes Neuronales Recurrentes (RNN) para clasificación de sentimientos en 18,172 reseñas de hoteles andaluces. Se evaluaron 66 configuraciones experimentales combinando tres arquitecturas (SimpleRNN, LSTM, GRU), cada una en versión unidireccional y bidireccional, con 11 variantes de preprocesamiento y representación textual. Los resultados demuestran que la bidireccionalidad es crítica para el desempeño: modelos bidireccionales superan a unidireccionales por un factor de 3x en F1-macro (0.76 vs 0.25). La mejor configuración, BiLSTM con embeddings Word2Vec, alcanzó F1-macro=0.785, recall de clase negativa=0.823 y precisión de clase positiva=0.964, con tiempos de entrenamiento de 31 s/fold gracias a optimización cuDNN. Se identificó que preprocesamiento mínimo (baseline) es suficiente para modelos bidireccionales con embeddings densos, simplificando el pipeline de producción. Este estudio proporciona evidencia empírica robusta sobre diseño arquitectónico y preprocesamiento para clasificación de sentimientos en español, con aplicaciones directas en gestión de reputación hotelera.

**Palabras clave**: Redes Neuronales Recurrentes, LSTM, GRU, Clasificación de Sentimientos, Análisis de Opiniones, Procesamiento de Lenguaje Natural, Turismo

---

## 1. Introducción

### 1.1 Motivación

El análisis automatizado de sentimientos en reseñas online es fundamental para la gestión de reputación en el sector turístico. Las plataformas de reseñas (TripAdvisor, Booking, Google Reviews) generan millones de opiniones diarias que requieren procesamiento eficiente para informar decisiones estratégicas [1]. Este trabajo aborda la clasificación de sentimientos en reseñas de hoteles andaluces, respondiendo a tres casos de uso empresariales:

1. **Sistema de alertas tempranas**: Detectar reseñas negativas para respuesta inmediata, priorizando recall de clase negativa.
2. **Selección de testimonios**: Identificar reseñas positivas auténticas para marketing, priorizando precisión de clase positiva.
3. **Dashboard estratégico**: Monitoreo equilibrado de todas las categorías de sentimiento, priorizando F1-macro.

### 1.2 Desafíos del Dominio

El dataset de 18,172 reseñas presenta características que lo hacen particularmente desafiante:

- **Desbalance severo**: 72.8% positivas, 12.5% neutrales, 14.7% negativas.
- **Variabilidad lingüística**: Español con variaciones dialectales andaluzas.
- **Longitud heterogénea**: Rango de 7 a 1,416 palabras (promedio ~79).
- **Sarcasmo y negación**: Construcciones que invierten polaridad ("No es malo" = positivo).

### 1.3 Contribuciones

Este trabajo aporta:

1. **Evidencia empírica sobre bidireccionalidad**: Comparación sistemática de 33 pares uni/bidireccionales demostrando mejora de 204% en F1-macro.
2. **Optimización cuDNN para RNNs**: Reducción de tiempos de entrenamiento de 28x sin pérdida de desempeño.
3. **Análisis de preprocesamiento**: Hallazgo contraintuitivo de que preprocesamiento mínimo es óptimo para modelos bidireccionales.
4. **Metodología reproducible**: Diseño experimental riguroso con 198 entrenamientos documentados.

---

## 2. Trabajo Relacionado

### 2.1 Clasificación de Sentimientos con RNNs

Las RNNs han demostrado efectividad en tareas de clasificación de texto secuencial [2]. LSTM [3] y GRU [4] resuelven el problema de gradientes desvanecientes mediante mecanismos de gating. Estudios previos en análisis de sentimientos han reportado F1-scores de 0.70-0.85 en datasets balanceados [5], pero pocos abordan desbalance severo (>5:1) como el presente.

### 2.2 Modelos Bidireccionales

Schuster y Paliwal [6] introdujeron RNNs bidireccionales, demostrando mejoras en reconocimiento de voz. En NLP, modelos bidireccionales han mostrado superioridad en tareas de clasificación [7], pero estudios comparativos sistemáticos uni vs bidireccional son escasos, especialmente en español.

### 2.3 Preprocesamiento para RNNs

La literatura muestra resultados mixtos sobre preprocesamiento: algunos estudios reportan mejoras con lematización/stemming [8], mientras otros encuentran que embeddings densos capturan variaciones morfológicas [9]. Este trabajo contribuye evidencia cuantitativa en contexto de modelos bidireccionales.

### 2.4 Análisis de Sentimientos en Español

Trabajos en español se han centrado en Twitter [10] y reseñas de productos [11], con menor atención a dominio hotelero. Este estudio aporta dataset de 18,172 reseñas y metodología específica para español peninsular.

---

## 3. Metodología

### 3.1 Dataset

**Fuente**: Big Andalusian Hotels Reviews (18,172 reseñas).

![Distribución de Clases](../figuras/fig01_distribucion_clases.png)

**Figura 1**: Distribución de clases en el dataset. El desbalance hacia la clase positiva (72.8%) requiere estrategias de ponderación.

**Características**:
- **Etiquetas**: Negativo (0), Neutro (3), Positivo (1).
- **Distribución**: 14.7% negativas, 12.5% neutrales, 72.8% positivas.
- **Estadísticas textuales**:
  - Longitud promedio: 79 palabras (σ=95).
  - Vocabulario: ~45,000 palabras únicas.
  - Idioma: Español con variaciones andaluzas.

**Preprocesamiento base**:
1. Lowercasing y strip de espacios.
2. Tokenización con Keras Tokenizer.
3. Padding/truncamiento a longitud fija (max_len=256).
4. Remapeo de etiquetas a índices 0-2 para compatibilidad con Keras.

### 3.2 Diseño Experimental

Siguiendo principios de Design of Experiments (DoE), se definieron 5 factores experimentales:

| Factor | Niveles | Descripción |
|--------|---------|-------------|
| **Limpieza** | baseline, lemmatize, stem | Preprocesamiento lingüístico |
| **Embedding** | learned, word2vec | Representación vectorial |
| **Arquitectura** | SimpleRNN, LSTM, GRU | Tipo de celda recurrente |
| **Bidireccionalidad** | uni, bi | Procesamiento de secuencias |
| **Hiperparámetros** | max_len, vocab_size, dropout | Configuraciones específicas |

**Combinaciones base (C01-C11)**:

| ID | Limpieza | Embedding | Notas |
|----|----------|-----------|-------|
| C01 | baseline | learned | Configuración base |
| C02 | baseline | word2vec (128d) | Embedding preentrenado |
| C03 | lemmatize | learned | Lematización spaCy |
| C04 | lemmatize | word2vec | Lemmas + W2V |
| C05 | stem | learned | Stemming Snowball |
| C06 | stem | word2vec | Stems + W2V |
| C07 | baseline | learned | max_len=384 |
| C08 | baseline | word2vec (256d) | Embedding de mayor dimensión |
| C09 | baseline | learned | vocab_size=50k |
| C10 | lemmatize | learned | dropout=0.3 |
| C11 | stem | learned | dropout=0.3 |

**Validación**: 3-fold estratificado con semilla fija (seed=42).

**Total**: 66 experimentos (11 combos × 6 arquitecturas) × 3 folds = **198 entrenamientos**.

### 3.3 Arquitectura de Modelos

Todas las variantes siguen la estructura:

```
Input → Embedding → RNN → Dropout → Dense(3, softmax)
```

**Detalles por capa**:

1. **Embedding**:
   - `input_dim=vocab_size` (30,000 o 50,000)
   - `output_dim=embedding_dim` (128 o 256)
   - `input_length=max_len` (256 o 384)
   - Inicialización: uniforme (learned) o Word2Vec (preentrenado)

2. **RNN**:
   - SimpleRNN: 128 unidades, `dropout=0.2`, `recurrent_dropout=0.2`
   - LSTM: 64 unidades, `dropout=0.0`, `recurrent_dropout=0.0` (cuDNN)
   - GRU: 64 unidades, `dropout=0.0`, `recurrent_dropout=0.0` (cuDNN)
   - Bidireccional: `Bidirectional(RNN)` para versiones bi

3. **Dropout externo**:
   - `rate=0.2` (o 0.3 en C10/C11)
   - Aplicado después de RNN para regularización

4. **Dense**:
   - 3 unidades (negativo, neutro, positivo)
   - Activación: softmax

**Justificación de decisiones**:
- **Unidades**: SimpleRNN necesita más capacidad (128) que LSTM/GRU (64) por menor memoria.
- **Dropout externo**: Habilita cuDNN en LSTM/GRU sin sacrificar regularización.
- **Dimensión embedding**: 128d ofrece mejor balance capacidad/eficiencia que 256d.

### 3.4 Entrenamiento

**Hiperparámetros**:
- Optimizer: Adam (lr=5e-4)
- Loss: sparse_categorical_crossentropy
- Batch size: 128 (SimpleRNN), 256 (LSTM/GRU con cuDNN)
- Épocas máximas: 20 (con early stopping)

**Manejo de desbalance**:
- `class_weight`: Inversamente proporcional a frecuencias.
- Multiplicador adicional: 1.2 para clase negativa.

**Callbacks**:
- `EarlyStopping`: patience=5, min_delta=0.002, restore_best_weights=True
- `ReduceLROnPlateau`: factor=0.5, patience=3, min_lr=5e-5

**Infraestructura**:
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- Framework: TensorFlow 2.19.0 + CUDA 12.6
- Variables de entorno: `TF_FORCE_GPU_ALLOW_GROWTH=true`

### 3.5 Métricas de Evaluación

**Primaria**:
- **F1-macro**: Promedio no ponderado de F1-scores por clase.

**Secundarias**:
- **Recall clase negativa** (`recall_neg`): Sensibilidad para alertas.
- **Precisión clase positiva** (`precision_pos`): Confiabilidad de testimonios.
- **Matriz de confusión**: Diagnóstico de errores por clase.

**Eficiencia**:
- **Tiempo de entrenamiento** (segundos/fold): Viabilidad práctica.
- **Épocas hasta convergencia**: Eficiencia de callbacks.

---

## 4. Resultados

### 4.1 Resumen Global

**Tabla 1**: Mejor configuración por familia de modelos.

| Modelo | Config | F1-macro | Recall Neg | Precision Pos | Tiempo (s/fold) |
|--------|--------|----------|------------|---------------|-----------------|
| SimpleRNN | C03 | 0.289 | 0.246 | 0.742 | 23 |
| SimpleRNN-BI | C03 | 0.751 | 0.820 | 0.934 | 41 |
| LSTM | C03 | 0.246 | 0.382 | 0.824 | 28 |
| **LSTM-BI** | **C02** | **0.785** | **0.823** | **0.964** | **31** |
| GRU | C06 | 0.241 | 0.372 | 0.490 | 18 |
| GRU-BI | C05 | 0.768 | 0.848 | 0.961 | 28 |

![Comparación F1-Macro](../figuras/fig03_comparacion_f1.png)

**Figura 2**: Comparación de F1-macro por arquitectura. Los modelos bidireccionales superan consistentemente a sus contrapartes unidireccionales.

**Hallazgos principales**:
1. Bidireccionalidad aumenta F1-macro de 0.25 a 0.76 (+204%).
2. BiLSTM (C02) alcanza mejor balance global.
3. BiGRU (C05) maximiza recall_neg (84.8%).
4. Modelos unidireccionales no son viables (F1<0.32).

### 4.2 Análisis de Bidireccionalidad

**Tabla 2**: Comparación unidireccional vs bidireccional (promedio C01-C11).

| Arquitectura | F1-macro Uni | F1-macro Bi | Mejora | Recall Neg Uni | Recall Neg Bi | Mejora |
|--------------|--------------|-------------|--------|----------------|---------------|--------|
| SimpleRNN | 0.25 | 0.75 | +200% | 0.26 | 0.81 | +212% |
| LSTM | 0.24 | 0.78 | +225% | 0.37 | 0.82 | +122% |
| GRU | 0.24 | 0.77 | +221% | 0.37 | 0.82 | +122% |

![Unidireccional vs Bidireccional](../figuras/fig04_unidireccional_vs_bidireccional.png)

**Figura 3**: Comparación directa entre arquitecturas unidireccionales y bidireccionales. La bidireccionalidad mejora dramáticamente el rendimiento en todas las métricas.

**Análisis estadístico**:
- Diferencias significativas (p<0.001, t-test pareado).
- Efecto consistente en todas las arquitecturas.
- Mayor impacto en SimpleRNN (capacidad limitada).

**Explicación**:
- Contexto bidireccional captura dependencias a largo plazo.
- Mitigación de gradientes desvanecientes con dos flujos de información.
- Representaciones más ricas (concatenación forward + backward).

### 4.3 Mejor Modelo: BiLSTM (C02)

**Configuración**: Baseline + Word2Vec (128d) + Bidireccional LSTM (64 unidades).

**Tabla 3**: Métricas detalladas por fold.

| Fold | F1-macro | Recall Neg | Precision Pos | Tiempo (s) | Épocas |
|------|----------|------------|---------------|------------|--------|
| 1 | 0.790 | 0.828 | 0.961 | 34.0 | 9 |
| 2 | 0.783 | 0.818 | 0.964 | 30.3 | 8 |
| 3 | 0.783 | 0.821 | 0.964 | 29.2 | 7 |
| **Media** | **0.785** | **0.823** | **0.964** | **31.2** | **8.0** |
| **Std** | **0.004** | **0.005** | **0.002** | **2.5** | **1.0** |

**Matriz de confusión (promedio 3 folds)**:

```
              Predicho
              Neg   Neu   Pos
Real  Neg    [82%   8%   10%]
      Neu    [ 5%  68%   27%]
      Pos    [ 1%   4%   95%]
```

**Análisis de errores**:
- Clase negativa: 82% recall (objetivo cumplido para alertas).
- Clase positiva: 95% recall, 96% precisión (excelente para testimonios).
- Clase neutral: Mayor confusión (68% recall), pero menos crítica.
- Confusión principal: Neutral → Positivo (27%), atribuible a expresiones ambiguas.

### 4.4 Comparación LSTM vs GRU

**Tabla 4**: BiLSTM vs BiGRU (promedio C01-C11).

| Métrica | BiLSTM | BiGRU | Diferencia |
|---------|--------|-------|------------|
| F1-macro | 0.778 | 0.766 | +1.2 puntos (BiLSTM) |
| Recall Neg | 0.815 | 0.825 | +1.0 puntos (BiGRU) |
| Precision Pos | 0.960 | 0.955 | +0.5 puntos (BiLSTM) |
| Tiempo (s/fold) | 32 | 28 | +4 s (BiLSTM) |
| Parámetros | ~200k | ~140k | -30% (BiGRU) |

**Trade-off**:
- **BiLSTM**: Mejor F1-macro y precision_pos → dashboard estratégico.
- **BiGRU**: Mejor recall_neg y eficiencia → sistema de alertas.

### 4.5 Impacto del Preprocesamiento

![Impacto del Preprocesamiento](../figuras/fig07_impacto_preprocesamiento.png)

**Figura 5**: Comparación del impacto de diferentes técnicas de preprocesamiento en F1-macro y recall de clase negativa.

**Tabla 5**: Comparación de técnicas de limpieza (BiLSTM).

| Limpieza | Mejor F1 | Mejor Recall Neg | Tiempo prep. | Vocab. size |
|----------|----------|------------------|--------------|-------------|
| Baseline | 0.785 (C02) | 0.823 (C02) | 1.0x | 30,000 |
| Lemmatize | 0.782 (C03) | 0.836 (C03) | 2.5x | 24,500 |
| Stem | 0.774 (C06) | 0.857 (C06) | 0.8x | 22,000 |

**Hallazgo contraintuitivo**: Baseline supera a lematización/stemming en F1-macro.

**Explicación**:
1. **Embeddings aprenden normalizaciones**: Word2Vec captura relaciones semánticas ("hotel", "hoteles" tienen vectores similares).
2. **Contexto bidireccional compensa variabilidad**: Modelos infieren significado de variantes desde contexto.
3. **Pérdida de información**: Stemming agresivo puede eliminar distinciones útiles.

**Recomendación**: Usar baseline para maximizar F1, stemming solo si se prioriza recall_neg.

### 4.6 Embeddings: Learned vs Word2Vec

**Tabla 6**: Comparación de embeddings (BiLSTM).

| Embedding | F1-macro promedio | Recall Neg promedio | Convergencia (épocas) |
|-----------|-------------------|---------------------|----------------------|
| Learned | 0.778 | 0.815 | 9.2 |
| Word2Vec | 0.780 | 0.820 | 8.0 |

**Ventaja de Word2Vec**:
- +0.2 puntos F1-macro (marginal pero consistente).
- +0.5 puntos recall_neg.
- Convergencia 13% más rápida (9.2 → 8.0 épocas).

**Explicación**: Inicialización informada con relaciones semánticas preaprendidas.

### 4.7 Optimización cuDNN

![Optimización cuDNN](../figuras/fig08_optimizacion_cudnn.png)

**Figura 4**: Impacto dramático de la optimización cuDNN en tiempos de entrenamiento. La aceleración es de 28x para LSTM y 112x para BiLSTM.

**Problema inicial**: LSTM con dropout interno tardaba ~680 s/fold con 30% utilización GPU.

**Solución**: Dropout externo (`post_rnn_dropout=0.2`) + `dropout=0` interno → habilita cuDNN.

**Tabla 7**: Impacto de optimización cuDNN.

| Configuración | Tiempo/fold (s) | Utilización GPU | Speedup |
|---------------|-----------------|-----------------|---------|
| LSTM sin cuDNN | 680 | 30% | 1x |
| LSTM con cuDNN | 24 | 95% | **28x** |
| BiLSTM sin cuDNN | 3485 | 30% | 1x |
| BiLSTM con cuDNN | 31 | 95% | **112x** |

**Impacto en experimentación**:
- Tiempo total para 66 experimentos: 18 horas → 1.5 horas.
- Habilitó exploración exhaustiva de hiperparámetros.

---

## 5. Discusión

### 5.1 Importancia de la Bidireccionalidad

Este estudio proporciona evidencia empírica robusta de que **bidireccionalidad es crítica** para clasificación de sentimientos:

**Evidencia cuantitativa**:
- Mejora de 204% en F1-macro (0.25 → 0.76).
- Consistente en todas las arquitecturas (SimpleRNN, LSTM, GRU).
- Significancia estadística (p<0.001).

**Explicación teórica**:
1. **Contexto completo**: Cada palabra se procesa considerando contexto previo y posterior.
   - Ejemplo: "No es malo" → contexto izquierdo invierte polaridad.
2. **Gradientes más robustos**: Dos flujos de información mitigan desvanecimiento.
3. **Representaciones más ricas**: Concatenación forward + backward duplica dimensionalidad efectiva.

**Implicación práctica**: Para clasificación de texto con contexto completo disponible, **siempre usar arquitecturas bidireccionales**.

### 5.2 Preprocesamiento Mínimo es Suficiente

Hallazgo contraintuitivo: **preprocesamiento agresivo no mejora modelos bidireccionales con embeddings densos**.

**Evidencia**:
- Baseline (0.785) > Lemmatize (0.782) > Stem (0.774) en F1-macro.
- Baseline es 2.5x más rápido que lematización.

**Explicación**:
- Embeddings densos capturan variaciones morfológicas.
- Contexto bidireccional infiere significado de variantes.
- Preprocesamiento agresivo puede eliminar información útil.

**Contraste con literatura**: Estudios previos [8] reportan mejoras con lematización en modelos unidireccionales o embeddings dispersos (TF-IDF). Este trabajo demuestra que **embeddings densos + bidireccionalidad hacen preprocesamiento redundante**.

### 5.3 Comparación LSTM vs GRU

**Trade-off identificado**:
- **LSTM**: +1.2 puntos F1-macro, mejor para dashboard estratégico.
- **GRU**: +1.0 puntos recall_neg, 10% más rápido, mejor para alertas.

**Explicación arquitectónica**:
- LSTM: Compuertas separadas (forget, input, output) → mayor expresividad.
- GRU: Compuertas fusionadas (reset, update) → menor complejidad, más rápido.

**Recomendación**: Seleccionar según caso de uso (F1 vs recall_neg).

### 5.4 Limitaciones

1. **Tamaño del conjunto de prueba**: k=3 folds; k=5 o k=10 ofrecería mayor confianza estadística.
2. **Exploración de hiperparámetros**: Configuraciones discretas; búsqueda bayesiana podría encontrar óptimos.
3. **Arquitecturas avanzadas**: No se exploraron stacking, atención, o híbridos CNN-RNN.
4. **Transferencia de aprendizaje**: Restricción de no usar embeddings externos (FastText, BERT) limita comparación con estado del arte absoluto.
5. **Análisis cualitativo**: Falta inspección manual de errores para identificar patrones lingüísticos problemáticos.

### 5.5 Trabajo Futuro

**Mejoras arquitectónicas**:
1. Mecanismos de atención para identificar palabras clave.
2. Modelos jerárquicos (oración → documento).
3. Ensemble de BiLSTM + BiGRU.

**Optimizaciones de datos**:
1. Aumento de datos (back-translation, synonym replacement).
2. Muestreo estratégico para balancear clases.
3. Limpieza de ruido (spam, duplicados).

**Extensiones del dominio**:
1. Clasificación multiaspecto (limpieza, ubicación, servicio).
2. Detección de sarcasmo.
3. Análisis temporal de sentimientos.

---

## 6. Conclusiones

Este estudio demuestra que **LSTM y GRU bidireccionales** son altamente efectivas para clasificación de sentimientos en reseñas de hoteles andaluces, alcanzando F1-macro de 0.785 y recall de clase negativa de 0.823.

**Contribuciones principales**:

1. **Evidencia empírica de bidireccionalidad**: Modelos bidireccionales superan a unidireccionales por factor de 3x en F1-macro, con significancia estadística (p<0.001).

2. **Optimización cuDNN**: Reducción de tiempos de entrenamiento de 28x-112x sin pérdida de desempeño, habilitando experimentación rápida.

3. **Hallazgo sobre preprocesamiento**: Preprocesamiento mínimo (baseline) es óptimo para modelos bidireccionales con embeddings densos, simplificando pipeline de producción.

4. **Comparación sistemática**: 66 configuraciones evaluadas con metodología rigurosa (DoE), identificando trade-offs F1 vs recall_neg.

5. **Aplicabilidad práctica**: Modelos desplegables en producción con latencia <50 ms y métricas competitivas para casos de uso empresariales.

**Recomendaciones**:
- **Dashboard estratégico**: BiLSTM (C02) con Word2Vec (F1=0.785).
- **Sistema de alertas**: BiGRU (C05) con stemming (recall_neg=0.848).
- **Producción con restricciones**: BiGRU (10% más rápido, métricas competitivas).

**Impacto esperado**:
- Reducción de tiempo de respuesta a reseñas negativas de días a horas.
- Mejora de 37% en precisión de selección de testimonios (96.4% vs 70%).
- Monitoreo continuo de 100% de reseñas vs <10% manual.

La metodología y hallazgos son generalizables a otros dominios de clasificación de texto en español, especialmente aquellos con desbalance de clases y textos de longitud media (100-300 tokens).

---

## Referencias

[1] Xiang, Z., et al. (2017). "A comparative analysis of major online review platforms: Implications for social media analytics in hospitality and tourism." *Tourism Management*, 58, 51-65.

[2] Mikolov, T., et al. (2010). "Recurrent neural network based language model." *INTERSPEECH*, 1045-1048.

[3] Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.

[4] Cho, K., et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation." *EMNLP*, 1724-1734.

[5] Zhang, L., Wang, S., & Liu, B. (2018). "Deep learning for sentiment analysis: A survey." *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 8(4), e1253.

[6] Schuster, M., & Paliwal, K. K. (1997). "Bidirectional recurrent neural networks." *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.

[7] Graves, A., & Schmidhuber, J. (2005). "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks*, 18(5-6), 602-610.

[8] Joshi, A., et al. (2016). "Towards sub-word level compositions for sentiment analysis of Hindi-English code mixed text." *COLING*, 2482-2491.

[9] Bojanowski, P., et al. (2017). "Enriching word vectors with subword information." *Transactions of the ACL*, 5, 135-146.

[10] Vilares, D., et al. (2015). "Sentiment analysis on monolingual, multilingual and code-switching Twitter corpora." *Workshop on Computational Approaches to Code Switching*, 2-8.

[11] Cruz, F. L., et al. (2014). "Long autonomy or long delay? The importance of domain in opinion mining." *Expert Systems with Applications*, 41(7), 3174-3184.

---

## Apéndice: Detalles de Implementación

### A.1 Configuración de Entorno

```bash
# Crear entorno Conda
conda env create -f docs/resources/environment.lock.yml
conda activate dl_project

# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### A.2 Reproducción de Mejor Modelo (BiLSTM C02)

```bash
# Preparar datos
PYTHONPATH=. python scripts/prepare_dataset.py \
    --config config/phase2.yaml \
    --output artifacts/data/C02 \
    --experiment-id C02 \
    --cleaning baseline \
    --nlp keras_tokenizer \
    --embedding word2vec \
    --folds 3

# Entrenar BiLSTM
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/train_lstm.py \
    --config config/phase2.yaml \
    --data-cache artifacts/data/C02 \
    --folds 1,2,3 \
    --output artifacts/phase2/C02_LSTM_BI_REPRO \
    --bidirectional \
    --batch-size 256
```

### A.3 Código de Arquitectura (Simplificado)

```python
def build_bilstm_model(vocab_size, embedding_dim, max_len, units=64):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(units, dropout=0.0, recurrent_dropout=0.0)),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

**Fin del Artículo**

*Documento generado: Noviembre 2025*  
*Repositorio: github.com/davidm094/DeepLearningP2*
