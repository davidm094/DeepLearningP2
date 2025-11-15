### Objetivo
Dejar todo listo para:
- Replicar exactamente el ambiente actual (`ml001_env`).
- Crear un ambiente nuevo base para proyectos de Deep Learning (RNNs).

### Requisitos
- Conda/Miniconda/Mamba instalado.
- GPU NVIDIA con drivers actualizados (detectado: RTX 3090, driver 581.57).
- WSL2/Linux OK. No es necesario `nvcc` para entrenar; solo si compilas kernels.

### Opción A (recomendado): Replicar EXACTO el ambiente actual
Esto utiliza un lockfile congelado de tu entorno actual.

1) Crear el entorno (puedes cambiar el nombre con el primer argumento):
```bash
./create_from_lock.sh ml001_env
```
2) Activar:
```bash
conda activate ml001_env
```
3) Verificar GPU y librerías:
```bash
python gpu_check.py
```

Archivos usados:
- `environment.lock.yml` (copiado del entorno actual)

### Opción B: Crear un entorno NUEVO desde la plantilla
La plantilla es opinada para RNNs, con PyTorch y TensorFlow vía pip.

1) Crear el entorno (elige un nombre):
```bash
./create_from_template.sh rnn_env
```
2) Activar:
```bash
conda activate rnn_env
```
3) Verificar GPU:
```bash
python gpu_check.py
```

Notas sobre GPU:
- PyTorch GPU por pip suele requerir usar el índice oficial de PyTorch para CUDA (si tu rueda no trae `+cuXXX` podrías tener CPU). En esta repo, la réplica exacta del entorno actual usa `torch 2.7.0+cu126`. Si la plantilla no toma la rueda CUDA automáticamente, sigue las indicaciones en consola o instala con índice de PyTorch.

### Troubleshooting rápido
- Si `torch.cuda.is_available()` es False:
  - Asegura drivers NVIDIA vigentes (ya tienes 581.57).
  - Reinstala PyTorch con rueda CUDA correspondiente (ver mensajes del script).
- Si TensorFlow no ve la GPU:
  - Asegura `tensorflow==2.19.0` y que no haya conflictos de CUDA/cuDNN.

### Archivos en esta carpeta
- `environment.lock.yml`: lockfile del entorno actual (replicación 1:1).
- `environment.template.yml`: plantilla comentada para proyectos RNN.
- `create_from_lock.sh`: crea/actualiza entorno desde el lockfile.
- `create_from_template.sh`: crea entorno desde la plantilla.
- `gpu_check.py`: verificación Torch/TensorFlow con GPU.
- `EXPLICACION_DEPENDENCIAS.md`: para qué sirve cada dependencia clave.
- `ENVIRONMENT_REPORT.md`: reporte del entorno actual y cómo replicarlo.


