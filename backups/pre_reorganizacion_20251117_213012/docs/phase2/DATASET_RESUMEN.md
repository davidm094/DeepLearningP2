## Resumen dataset y pautas (Fase 2)

### 1. Dataset oficial
- **Nombre**: *Andalusian Hotels Reviews Unbalanced* (18,172 reseñas en español).
- **Dominio**: reseñas de hoteles en Andalucía (715 hoteles, 33 ubicaciones).
- **Caracterización F1**:
  - Longitud media/madiana: 78.8 / 61 palabras (máx. 1,416).
  - Vocabulario: 62,639 tokens únicos; corpus 1.43 M palabras; diversidad léxica 0.0437.
  - Columnas originales: 7 (texto, título, ubicación, hotel, rating, label, índice).
  - Derivadas de EDA: longitud en caracteres/palabras/oraciones.

### 2. Variable objetivo
| Clase | Frecuencia | % | Interpretación |
| --- | --- | --- | --- |
| 1 | 13,227 | 72.79% | Sentimiento positivo |
| 0 | 2,671 | 14.70% | Sentimiento negativo |
| 3 | 2,274 | 12.51% | Sentimiento neutral/mixto |

Implicaciones: desbalance 5.82:1 (clase 1 vs 3). Las métricas macro y el ajuste de `class_weight` son obligatorios para evitar sesgos.

### 3. Pautas técnicas confirmadas
- **Modelado**: solo RNN clásicas Keras/TensorFlow (SimpleRNN, LSTM, GRU). Sin redes preentrenadas ni embeddings externos.
- **Embeddings**: capa `Embedding` inicializada aleatoriamente y entrenada en conjunto.
- **Entorno CUDA**: para entrenar en GPU se requiere `cuda-version=12.6` y `cuda-nvcc=12.6` dentro del entorno (proporciona `ptxas`/`libdevice`). Instalar vía `conda install -n dl_project -c conda-forge cuda-version=12.6 cuda-nvcc=12.6`.
- **Validación**: k-fold estratificado permitido (e.g., k=5). Nada de ajuste con el set de prueba final.
- **Framework**: Keras (preferencia del usuario) sobre TensorFlow.

### 4. Pipeline de datos permitido
1. **Limpieza ligera**: normalización mínima (lowercase, stripping). Evitar lematización/stemming agresivo salvo que la pauta lo autorice explícitamente.
2. **Tokenización**: `Tokenizer` de Keras con `num_words=vocab_size` fijado en config.
3. **Secuencias**: truncado/padding (`pad_sequences`) a `max_len` definido; mantener coherencia con análisis (ej. cubrir p95 de longitudes).
4. **Splits**:
   - Train/validation mediante `StratifiedKFold`.
   - Test final separado (mismo split usado en Fase 1) y reservado para el reporte definitivo.
5. **Reproducibilidad**: fijar `seed` en NumPy/Python/TF, registrar en `config/phase2.yaml`.

### 5. Métricas y relación con negocio (Casosdeuso.md)
- **Caso 3 (prioritario)**: Dashboard estratégico → **F1 Macro** (igual peso a las 3 clases). Resultado esperado: tabla por fold + promedio ± desviación.
- **Caso 1**: Sistema de alerta de crisis → **Recall clase 0**. Debe acompañar cada experimento.
- **Caso 2**: Motor de testimonios → **Precisión clase 1**. Reportar y discutir trade-offs.
- **Apoyos**: Matriz de confusión normalizada y soporte por clase para conectar errores con decisiones (ej. falsa alarma vs. omisión de crítica).

### 6. Referencias utilizadas
- `docs/resources/entregas/Proyecto_DL_Fase1.pdf`: EDA completo, estadísticas descriptivas y contexto del dominio.
- `docs/resources/instrucciones/Proyecto 2025-2.pdf`: reglas de modelado, entregables y evaluación.
- `docs/resources/referencias/Casosdeuso.md`: conexión negocio-métricas.

Este documento deberá actualizarse si la pauta introduce restricciones adicionales (p.ej., límite exacto de `max_len` o reglas específicas de limpieza). Guardar versión junto con los experimentos de Fase 2.

