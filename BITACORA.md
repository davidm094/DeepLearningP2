## Bitácora del Proyecto DeepLearningP2

### 2025-11-15
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


