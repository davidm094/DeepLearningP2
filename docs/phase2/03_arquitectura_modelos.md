## Arquitectura de los modelos RNN (Fase 2)

Todas las variantes comparten la misma estructura general definida en `src/models/rnn_keras.py`:

1. `Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)`  
   - Entrenada desde cero o inicializada con Word2Vec entrenado sobre el mismo corpus.
   - Parámetros controlados desde `config/phase2.yaml` (`embedding.output_dim`, `data.max_len`, `data.vocab_size`).
2. **Capa recurrente única** (`SimpleRNN`, `LSTM` o `GRU`):
   - SimpleRNN usa 128 unidades; LSTM/GRU usan 64 unidades para equilibrar capacidad y latencia.
   - Para LSTM/GRU se fija `dropout=recurrent_dropout=0` dentro de la celda y la regularización se traslada al `post_rnn_dropout`, habilitando cuDNN.
3. `Dropout` posterior (`post_rnn_dropout`) aplicado sólo si `dropout > 0`.
4. `Dense(output_classes, activation="softmax")`.

Características adicionales:
- La bidireccionalidad se activa desde los scripts (`--bidirectional`) envolviendo la capa recurrente con `keras.layers.Bidirectional`.
- Los pesos de clase pueden escalarse mediante `class_weight_multipliers` para reforzar la clase negativa.
- Los callbacks (`EarlyStopping`, `ReduceLROnPlateau`) se configuran en `training.callbacks`.

### Configuración actual (resumen)
```yaml
embedding:
  output_dim: 128

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
```

### Cómo ver el `model.summary()`
Cada vez que ejecutes:
```bash
python -m src.train.train_rnn --config config/phase2.yaml \
       --model-type lstm --folds 1 --show-summary
```
el script imprimirá la arquitectura (una única vez) antes de iniciar el entrenamiento del fold seleccionado. Esto permite inspeccionar rápidamente el número de parámetros y capas para cada variante (unidireccional o bidireccional).

