## Bitácora del Proyecto DeepLearningP2

**Autores**:  
Anderson J. Alvarado¹ · David E. Moreno²

¹ Pontificia Universidad Javeriana, andersonjalvarado@javeriana.edu.co  
² Pontificia Universidad Javeriana, morenoa-david@javeriana.edu.co

---

### 2025-11-15
- 17:15 — [GPU] CUDA runtime alineado a 12.6 e instalado cuda-nvcc; entrenamiento corto en GPU confirmado con LSTM 1 época
- 16:53 — [PLAN] Estructura Fase 2 creada (data loader, configs, modelos, runner). Pendiente correr CV por falta del CSV oficial.
- **Resumen**: Centralizamos el setup en `03_setup`, documentamos el entorno `ml001_env`, creamos el nuevo entorno `dl_project` con GPU funcionando, limpiamos archivos sueltos y realizamos commit.
- **Detalles**:
  - Generado `03_setup/` con: `environment.lock.yml`, `environment.template.yml`, scripts de creación, `gpu_check.py`, `EXPLICACION_DEPENDENCIAS.md` y `ENVIRONMENT_REPORT.md`.
  - Creado entorno `dl_project` desde plantilla; verificado Torch 2.7.0+cu126 y TensorFlow 2.19.0 con GPU disponible.
  - Eliminados `environment.yml`, `environment.ml001_env.lock.yml` y `test_gpu.py` en raíz para evitar duplicados.
  - Commit: “setup: centralize environment setup in 03_setup; add lock/template/scripts; move env report; remove root env files and test script”.
- **Próximos pasos**:
  - [ ] Definir alcance de las RNN (tareas, métricas, dataset).
  - [ ] Estructura base del código (carpetas, módulos de datos/modelo/entrenamiento).
  - [ ] Selección de dataset y baseline RNN (TF o PyTorch).
  - [ ] Configurar tracking de experimentos (nombres, seeds, logs).

### 2025-11-16
- 17:58 — [DATA] `prepare_dataset.py` actualizado con logs detallados; generados folds C04 (lemmatize + tokenizer + learned) en `artifacts/data/C04`.
- 18:01 — [TRAIN] `train_simple_rnn.py` corrido con RUN_ID `C04_SRN_20251116-220151`; métricas guardadas por fold y registradas en `artifacts/experiments.csv`.
- 18:14 — [DATA] Generados folds C05 (lemmatize + tokenizer + word2vec) con `prepare_dataset.py`; embedding guardado en `artifacts/data/C05/embedding_matrix.npy`.
- 18:17 — [TRAIN] RUN_ID `C05_SRN_20251116-221454` completó los tres folds; archivos en `artifacts/phase2/C05_SRN_20251116-221454/` y registro en `artifacts/experiments.csv`.
- 12:02 — [TRAIN] RUN_ID `C07_GRU_BI_20251117-120243` (BiGRU max_len 384) completado; F1≈0.77 y tiempos 32–35 s/fold.
- 12:15 — [TRAIN] RUN_ID `C08_GRU_BI_20251117-121547` (BiGRU word2vec 256) completado; F1≈0.774 y recall_neg≈0.79.
- 12:18 — [TRAIN] RUN_ID `C09_GRU_BI_20251117-121832` (BiGRU vocab_size 50k) completado; F1≈0.760 y recall_neg≈0.83.
- 12:20 — [TRAIN] RUN_ID `C10_GRU_BI_20251117-122047` (BiGRU dropout 0.3) completado; F1≈0.761 y recall_neg≈0.78.
- 12:23 — [TRAIN] RUN_ID `C11_GRU_BI_20251117-122312` (BiGRU stem + dropout 0.3) completado; F1≈0.766 y recall_neg≈0.82.
- 12:16 — [TRAIN] RUN_ID `C09_GRU_BI_20251117-1220XX` pendiente de lanzamiento.
- 14:55 — [TRAIN] Reejecución completa de LSTM C01–C11 con arquitectura 64 u. + dropout externo (`Cxx_LSTM_20251117-14xxxx`); tiempos 20–45 s/fold y métricas registradas en `artifacts/experiments.csv`.
- **Notas**:
  - Se documentó en `docs/phase2/COMBINACIONES.md` que TF-IDF se descarta para RNN.
  - `reports/phase2/RESULTADOS.md` incluye C01, C02, C04 y C05 con promedios comparables.

### 2025-11-17
- 00:35 — [PIPELINE] Renumeramos combinaciones (C01–C11 sin huecos) y añadimos cachés compartidos de limpieza/folds en `scripts/prepare_dataset.py`.
- 00:36 — [RESET] Se limpiaron `artifacts/data/`, `artifacts/phase2/` y `artifacts/experiments.csv` para re-ejecutar todo con los nuevos IDs.
- 00:40 — [DATA] Re-generados C01–C04 usando caches (`artifacts/cache/clean_*.joblib`, `folds_seed42_k3.json`) para garantizar folds idénticos.
- 00:44 — [TRAIN] SimpleRNN re-entrenado para C01–C04 (RUN_IDs `C0x_SRN_20251116-22xxxx`); resultados actualizados en `reports/phase2/RESULTADOS.md` y `artifacts/experiments.csv`.
- 00:50 — [DATA] Preparado C05 (stem + learned) creando cache `clean_stem.joblib`.
- 00:52 — [TRAIN] RUN_ID `C05_SRN_20251116-225043` completó los tres folds.
- 00:53 — [DATA] Preparado C06 (stem + word2vec) reutilizando tokens stem y entrenando solo el embedding.
- 00:55 — [TRAIN] RUN_ID `C06_SRN_20251116-225208` finalizado; métricas integradas al reporte y registry.
- 00:56 — [DATA] C07 generado (baseline + learned, max_len=384) usando caches compartidos.
- 00:58 — [TRAIN] RUN_ID `C07_SRN_20251116-225606` completó folds con secuencias largas (~33 s/fold).
- 00:59 — [DATA] C08 (baseline + word2vec dim=256) y C09 (baseline + learned, vocab=50k) generados sin repetir limpieza.
- 01:00 — [TRAIN] RUN_IDs `C08_SRN_20251116-225748` y `C09_SRN_20251116-225903` registrados; métricas agregadas a `reports/phase2/RESULTADOS.md`.
- 01:02 — [DATA] C10/C11 preparados con `prepare_dataset.py` reutilizando caches de lematizado y stemming (sin recomputar spaCy/Snowball).
- 01:04 — [TRAIN] RUN_ID `C10_SRN_20251116-230320` (dropout=0.3) ejecutado; registros añadidos a `artifacts/experiments.csv`.
- 01:05 — [TRAIN] RUN_ID `C11_SRN_20251116-230435` (dropout=0.3) completado; resultados resumen en `reports/phase2/RESULTADOS.md`.
- 01:10 — [TRAIN] Primera corrida LSTM (`C01_LSTM_20251116-231015`) sobre los mismos folds; tiempos ~11 min/fold y métricas documentadas en el reporte y `artifacts/experiments.csv`.
- 01:15 — [TRAIN] SimpleRNN “long” (`C01_SRN_LONG_20251116-235745`, 15 épocas + callbacks) elevó el recall negativo a ~0.20; matrices de confusión guardadas en los metrics individuales.
- 01:20 — [TRAIN] RUN_ID `C01_SRN_100E_20251117-000258` (límite 100 épocas con EarlyStopping) se detuvo en las épocas [4,4,6]; permitió verificar que el modelo corta temprano y que el recall negativo vuelve a caer cuando el entrenamiento insiste en secuencias largas.
- 01:25 — [TRAIN] RUN_ID `C03_SRN_LONG_20251117-000550` (lemmatize, LR 5e-4, 30 épocas) mostró recall negativo >0.8 pero F1 bajo; artefactos guardados para analizar el trade-off.
- 01:30 — [TRAIN] RUN_ID `C01_LSTM_LONG_20251117-001258` (solo fold 1, LR 5e-4, pesos 1.2) probado para evaluar los callbacks en LSTM; métricas añadidas al reporte.
- 06:20 — [TRAIN] RUN_ID `C01_SRN_LONG_20251117-004240` (baseline, receta LR 5e-4 + peso clase 0=1.2) completó folds 1–3; mejora de F1 a 0.32 con recall_neg promedio 0.53 documentada en `reports/phase2/RESULTADOS.md`.
- 06:25 — [TRAIN] RUN_ID `C05_SRN_LONG_20251117-004943` (stem + receta nueva) finalizado; métricas agregadas al reporte y `artifacts/experiments.csv`.
- 06:40 — [TRAIN] RUN_ID `C07_SRN_LONG_20251117-061925` (max_len=384 con receta nueva) ejecutado; se evidenciaron extremos en recall_neg, documentados en el reporte.
- 06:55 — [TRAIN] RUN_ID `C01_LSTM_LONG_20251117-062630` (folds 1–3, LR 5e-4) cerró después de ~18 min/fold; resultados añadidos a `reports/phase2/RESULTADOS.md` y `artifacts/experiments.csv`.
- 07:50 — [TRAIN] RUN_ID `C01_GRU_20251117-075013` ejecutado directamente con `python scripts/train_gru.py`; tiempos 30–54 s/fold y métricas añadidas al reporte.
- 07:52 — [TRAIN] RUN_ID `C01_SRN_BI_20251117-075222` (SimpleRNN bidireccional) completó 3 folds en ~65 s/fold, alcanzando F1≈0.73; documentado en el reporte.
- 08:04 — [TRAIN] RUN_ID `C02_SRN_BI_20251117-080430` (SimpleRNN bidireccional + word2vec) ejecutado; promedios F1≈0.75 con tiempos ~1 min/fold.
- 08:07 — [TRAIN] RUN_ID `C01_LSTM_BI_20251117-080742` iniciado (fold 1, 64 unidades, 20 épocas); usuario lo interrumpió tras ~2 épocas por duración (~400 s/época). Pendiente decidir si se relanza con ajustes adicionales.
- 08:26 — [TRAIN] RUN_ID `C01_LSTM_BI_TEST_20251117-082642` (BiLSTM cuDNN, dropout interno 0, batch 256, XLA off) completó fold 1 en ~38 s totales (10 épocas); confirmó aceleración masiva y F1≈0.78.
- 08:41 — [TRAIN] RUN_ID `C01_LSTM_BI_20251117-084113` (BiLSTM C01 folds 1–3 completos con receta cuDNN) finalizado; cada fold <50 s y F1 promedio 0.78.
- 08:44 — [TRAIN] RUN_ID `C02_LSTM_BI_20251117-084446` (BiLSTM C02 word2vec) completó folds 1–3; métricas similares a C01 y registradas en `RESULTADOS.md`.
- 08:51 — [TRAIN] RUN_ID `C03_LSTM_BI_20251117-085119` (BiLSTM lemmatize) ejecutado; fold 1 necesitó 18 épocas pero los promedios se mantuvieron en F1≈0.78.
- 09:00 — [TRAIN] RUN_ID `C04_LSTM_BI_20251117-090027` (BiLSTM lemmatize + word2vec) finalizado; tiempos por fold ~30–40 s y métricas similares a C03.
- 09:06 — [TRAIN] RUN_ID `C05_LSTM_BI_20251117-090613` (BiLSTM stem) completado; cada fold <40 s y F1 promedio 0.77.
- 09:11 — [TRAIN] RUN_ID `C06_LSTM_BI_20251117-091103` (BiLSTM stem + word2vec) terminado; mantiene F1≈0.775 con folds de 35–42 s.
- 09:17 — [TRAIN] RUN_ID `C07_LSTM_BI_20251117-091739` (BiLSTM max_len=384) completado; métricas similares (F1≈0.784) con tiempos ~40 s/fold.
- 09:23 — [TRAIN] RUN_ID `C08_LSTM_BI_20251117-092340` (BiLSTM word2vec 256) finalizado; fold 3 tuvo menor recall pero el promedio se mantuvo en F1≈0.77.
- 09:24 — [TRAIN] RUN_ID `C09_LSTM_BI_20251117-092423` (BiLSTM vocab_size=50k) completado; tiempos ~30 s/fold y F1 promedio 0.78.
- 09:29 — [TRAIN] RUN_ID `C10_LSTM_BI_20251117-092935` (BiLSTM dropout 0.3) ejecutado; promedios en la misma franja (~0.77 F1) con fold 3 privilegiando recall negativo.
- 09:32 — [TRAIN] RUN_ID `C11_LSTM_BI_20251117-093214` (BiLSTM stem + dropout 0.3) finalizado; fold 2 tardó más por logs extendidos pero las métricas fueron consistentes (F1≈0.77).
- 09:40 — [TRAIN] RUN_ID `C01_LSTM_20251117-094003` (LSTM simple C01 con receta cuDNN) ejecutado; el modelo sin bidireccional volvió a colapsar en la clase positiva (recall_neg≈0.05) pese al batch 256.
- 09:41 — [TRAIN] RUN_ID `C02_LSTM_20251117-094132` (LSTM simple word2vec) completado; fold 1 predijo todo como negativo y los otros dos repitieron el sesgo positivo → promedio F1=0.24.
- 09:43 — [TRAIN] RUN_ID `C03_LSTM_20251117-094318` (LSTM simple lemmatize) finalizado; fold 1 alcanzó recall 0.83, pero los otros dos volvieron a colapsar.
- 09:44 — [TRAIN] RUN_ID `C04_LSTM_20251117-094431` (LSTM simple lemmatize + word2vec) terminó; comportamiento errático (fold 1 todo negativo, fold 2 todo positivo, fold 3 balanceado).
- 09:46 — [TRAIN] RUN_ID `C05_LSTM_20251117-094627` (LSTM simple stem) completado; los tres folds predijeron únicamente la clase negativa (F1≈0.086).
- 09:47 — [TRAIN] RUN_ID `C06_LSTM_20251117-094758` (LSTM simple stem + word2vec) finalizado; nuevamente los tres folds se fueron al extremo negativo.
- 09:49 — [TRAIN] RUN_ID `C07_LSTM_20251117-094917` (LSTM simple max_len=384) completado; todos los folds predijeron solo la clase negativa.
- 09:51 — [TRAIN] RUN_ID `C08_LSTM_20251117-095103` (LSTM simple word2vec 256) finalizado; misma tendencia de predicciones negativas exclusivas.
- 09:52 — [TRAIN] RUN_ID `C09_LSTM_SIMPLE_20251117-095230` (LSTM simple vocab 50k) completado; fold 3 volvió a predecir sólo negativos y los otros dos mantuvieron sesgo positivo.
- 09:55 — [TRAIN] RUN_ID `C10_LSTM_SIMPLE_20251117-095411` (LSTM simple dropout 0.3) completado; fold 2 se balanceó pero los otros dos permanecieron extremos.
- 09:55 — [TRAIN] RUN_ID `C11_LSTM_SIMPLE_20251117-095533` (LSTM simple stem + dropout 0.3) completado; sólo el fold 1 intentó predecir positivos, los otros se quedaron en negativo.
- 09:57 — [TRAIN] RUN_ID `C01_GRU_SIMPLE_20251117-095754` iniciado para abrir la serie de GRU simples; los primeros resultados confirman el mismo sesgo extremo que en LSTM simple.
- 10:02 — [TRAIN] RUN_ID `C02_GRU_SIMPLE_20251117-100241` (GRU simple word2vec) completado; dos folds predijeron todo negativo y uno repitió el sesgo positivo (F1 promedio 0.166).
- 10:05 — [TRAIN] RUN_ID `C03_GRU_SIMPLE_20251117-100554` (GRU simple lemmatize) finalizado; patrón idéntico (dos folds negativos, uno sesgo positivo).
- 10:05 — [TRAIN] RUN_ID `C04_GRU_SIMPLE_20251117-100534` (GRU simple lemmatize + word2vec) completado; los tres folds se fueron al extremo negativo.
- 10:06 — [TRAIN] RUN_ID `C05_GRU_SIMPLE_20251117-100648` (GRU simple stem) completado; de nuevo todos los folds predicen sólo la clase negativa.
- 10:08 — [TRAIN] RUN_ID `C06_GRU_SIMPLE_20251117-100806` (GRU simple stem + word2vec) finalizado; fold 1 negativo y los otros dos con sesgo positivo, promedio F1≈0.24.
- 10:10 — [TRAIN] RUN_ID `C07_GRU_SIMPLE_20251117-101027` (GRU simple max_len=384) completado; nuevamente todos los folds se van al negativo.
- 10:12 — [TRAIN] RUN_ID `C08_GRU_SIMPLE_20251117-101159` (GRU simple word2vec 256) cerrado; fold 1 negativo y los otros dos sesgo positivo.
- 10:13 — [TRAIN] RUN_ID `C09_GRU_SIMPLE_20251117-101357` (GRU simple vocab 50k) completado; todos los folds repitieron el sesgo positivo leve (recall_neg≈0.05).
- 10:17 — [TRAIN] RUN_ID `C10_GRU_SIMPLE_20251117-101639` y `C11_GRU_SIMPLE_20251117-101747` (GRU simple dropout 0.3) completados; un fold negativo y otros con sesgo positivo en cada caso.
- 10:21 — [TRAIN] RUN_ID `C10_GRU_SIMPLE_20251117-102120_retry` y `C02_SRN_BI_20251117-102515_retry` ejecutados para obtener logs estables; C10_GRU confirmó el patrón (F1≈0.24) y el BRNN C02 reafirmó F1≈0.75 con buen recall.
- 10:44 — [TRAIN] RUN_ID `C03_SRN_BI_20251117-104355` (SimpleRNN bidireccional lemmatize) completado; F1≈0.75, recall_neg≈0.82.
- 10:48 — [TRAIN] RUN_ID `C04_SRN_BI_20251117-104744` (SimpleRNN bidireccional stemming) finalizado; F1≈0.75 con recall_neg≈0.80.
- 10:50 — [TRAIN] RUN_ID `C05_SRN_BI_20251117-105004` (SimpleRNN bidireccional w2v-300) cerrado; F1≈0.75 y recall_neg≈0.79.
- 10:53 — [TRAIN] RUN_ID `C06_SRN_BI_20251117-105236` (SimpleRNN bidireccional + dropout 0.1) terminado; F1≈0.745, recall_neg≈0.80.
- 10:56 — [TRAIN] RUN_ID `C07_SRN_BI_20251117-105520` (SimpleRNN bidireccional max_len 384) completado; F1≈0.75, recall_neg≈0.81.
- 10:59 — [TRAIN] RUN_ID `C08_SRN_BI_20251117-105818` (SimpleRNN bidireccional embedding 64d) finalizado; F1≈0.74, recall_neg≈0.78.
- 11:01 — [TRAIN] RUN_ID `C09_SRN_BI_20251117-110041` (SimpleRNN bidireccional vocab_size 50k) concluido; F1≈0.75, recall_neg≈0.81.
- 11:04 — [TRAIN] RUN_ID `C10_SRN_BI_20251117-110259` (SimpleRNN bidireccional dropout 0.3) finalizado; F1≈0.752, recall_neg≈0.82.
- 11:06 — [TRAIN] RUN_ID `C11_SRN_BI_20251117-110521` (SimpleRNN bidireccional dropout+recurrent_dropout 0.3) completado; F1≈0.752, recall_neg≈0.81.
- 11:10 — [TRAIN] RUN_ID `C02_LSTM_BI_20251117-111045` (BiLSTM word2vec rerun) ejecutado con cuDNN activo; F1≈0.785 y tiempos por fold <35 s.
- 11:13 — [TRAIN] RUN_ID `C03_LSTM_BI_20251117-111320` (BiLSTM lemmatize rerun) completado; F1≈0.782 con tiempos 27–35 s por fold.
- 11:15 — [TRAIN] RUN_ID `C04_LSTM_BI_20251117-111519` (BiLSTM lematize+word2vec rerun) finalizado; F1≈0.782 y tiempos ~30 s/fold.
- 11:17 — [TRAIN] RUN_ID `C05_LSTM_BI_20251117-111722` (BiLSTM stemming rerun) completado; F1≈0.773 y tiempos 24–31 s/fold.
- 11:21 — [TRAIN] RUN_ID `C06_LSTM_BI_20251117-112114` (BiLSTM stem + word2vec rerun) finalizado; F1≈0.774 con recall_neg>0.84.
- 11:23 — [TRAIN] RUN_ID `C07_LSTM_BI_20251117-112328` (BiLSTM max_len 384 rerun) completado; F1≈0.782 y tiempos 34–43 s/fold.
- 11:26 — [TRAIN] RUN_ID `C08_LSTM_BI_20251117-112602` (BiLSTM word2vec 256 rerun) finalizado; F1≈0.783, recall_neg≈0.79 y tiempos ~30 s/fold.
- 11:28 — [TRAIN] RUN_ID `C09_LSTM_BI_20251117-112809` (BiLSTM vocab_size 50k rerun) completado; F1≈0.776 y tiempos 26–29 s/fold.
- 11:30 — [TRAIN] RUN_ID `C10_LSTM_BI_20251117-113019` (BiLSTM dropout 0.3 rerun) finalizado; F1≈0.77 y recall_neg≈0.83.
- 11:32 — [TRAIN] RUN_ID `C11_LSTM_BI_20251117-113215` (BiLSTM stem + dropout 0.3 rerun) cerrado; F1≈0.77 y tiempos 25–29 s/fold.
- 11:37 — [TRAIN] RUN_ID `C01_GRU_BI_20251117-113707` (BiGRU baseline) ejecutado; F1≈0.77 y recall_neg≈0.84 con folds <33 s.
- 11:41 — [TRAIN] RUN_ID `C02_GRU_BI_20251117-114101` (BiGRU word2vec) completado; F1≈0.767 y tiempos 26–32 s/fold.
- 11:46 — [TRAIN] RUN_ID `C03_GRU_BI_20251117-114625` (BiGRU lemmatize) finalizado; F1≈0.77 y recall_neg≈0.77.
- 11:48 — [TRAIN] RUN_ID `C04_GRU_BI_20251117-114804` (BiGRU lemmatize+word2vec) completado; F1≈0.774 y recall_neg≈0.79.
- 11:58 — [TRAIN] RUN_ID `C05_GRU_BI_20251117-115826` (BiGRU stem) completado; F1≈0.768 y recall_neg≈0.85.
- 12:00 — [TRAIN] RUN_ID `C06_GRU_BI_20251117-120012` (BiGRU stem+word2vec) completado; F1≈0.766 y recall_neg≈0.79.
- 12:02 — [TRAIN] RUN_ID `C07_GRU_BI_20251117-120243` (BiGRU max_len 384) completado; F1≈0.770 y tiempos 32–35 s/fold.
- 12:15 — [TRAIN] RUN_ID `C08_GRU_BI_20251117-121547` (BiGRU word2vec 256) completado; F1≈0.774 y recall_neg≈0.79.
- **Notas**:
  - Caching reduce el tiempo de preparación en combos que comparten limpieza.
  - Los nuevos registros dejan listo el entorno para correr LSTM/GRU reutilizando los mismos folds.


