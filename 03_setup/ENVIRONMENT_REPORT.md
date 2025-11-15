## ml001_env environment report

Este documento captura el estado del entorno activo `ml001_env` y cómo replicarlo. También resume diferencias respecto a un `environment.yml` previo del repositorio.

### Resumen
- Nombre: `ml001_env`
- Python: 3.11.12
- Canales efectivos: `conda-forge`, `defaults`
- GPU: NVIDIA GeForce RTX 3090 (24 GB), Driver 581.57
- CUDA (paquetes conda): 12.9 (runtime/tooling)
- PyTorch: 2.7.0+cu126 (CUDA 12.6), CUDA disponible: true
- TensorFlow: 2.19.0, compilado con CUDA: true, GPUs detectadas: 1
- JAX: no instalado

### Detalles clave
- Python executable: `/home/david/miniconda3/envs/ml001_env/bin/python`
- Torch CUDA version: 12.6 (`+cu126`)
- Stack CUDA conda (ejemplos): `cuda-version=12.9`, `libcublas`, `libcufft`, `libcurand`, `libcusolver`, `libcusparse`, `nccl`, etc.
- Paquetes GPU por pip:
  - `torch==2.7.0`, `triton==3.3.0`
  - `tensorflow==2.19.0`, `tensorflow-text==2.19.0`, `tensorflow-hub`
  - `cupy-cuda12x==13.4.1`

Nota: `nvcc` no está en PATH. No es necesario para entrenar/inferir; solo si compilas kernels.

### Diferencias respecto a un `environment.yml` previo
- Ese archivo incluía canales `rapidsai` y `nvidia` y el metapaquete `rapids` (no presentes en el entorno vivo).
- El entorno actual gestiona CUDA principalmente vía conda-forge (12.9) más ruedas NVIDIA por pip (cuDNN, cuBLAS, etc.).
- PyTorch está instalado por pip (`torch==2.7.0+cu126`), TensorFlow coincide con 2.19.0 y ve la GPU.

Implicaciones:
- Si necesitas RAPIDS (cuDF/cuML), no está incluido. Puedes añadirlo con cuidado de compatibilidades CUDA/ABI.
- Torch usa CUDA 12.6; las libs conda están en 12.9 (suele ser compatible hacia atrás).

### Cómo replicar exactamente este entorno
Opción 1 (script):
- `./03_setup/create_from_lock.sh ml001_env`
- `conda activate ml001_env`

Opción 2 (directo con conda):
- `conda env create -f 03_setup/environment.lock.yml`
- `conda activate ml001_env`

Verificación GPU:
- `conda run -n ml001_env python 03_setup/gpu_check.py`

Esto reproduce el set de paquetes (sin hashes de build) incluyendo pip.

### Fuentes
- Lockfile: `03_setup/environment.lock.yml`
- GPU: `nvidia-smi` → RTX 3090, driver 581.57
- Librerías dentro del entorno:
  - Python: 3.11.12
  - Torch: 2.7.0+cu126 (CUDA disponible: true)
  - TensorFlow: 2.19.0 (CUDA: true, GPU visible)
  - JAX: no instalado


