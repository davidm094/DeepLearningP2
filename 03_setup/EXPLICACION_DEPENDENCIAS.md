### Dependencias clave (enfocado a RNNs y DL)

- **python**: Intérprete base (3.11). Punto de partida para todo el stack científico.
- **numpy**: Tensores/arrays y operaciones vectorizadas. Capa base para cálculos numéricos.
- **pandas**: Manipulación/tablas y preprocesamiento de datasets.
- **scipy**: Rutinas científicas (optimizadores, estadística, señales).
- **scikit-learn**: Utilidades clásicas de ML (splits, métricas, pipelines, escalado).
- **matplotlib / seaborn**: Visualización exploratoria y de resultados/curvas de entrenamiento.
- **jupyterlab**: Notebooks para experimentación rápida y prototipado.
- **nltk / spacy**: Limpieza y tokenización de texto para RNNs de NLP.
- **xgboost / lightgbm**: Modelos de boosting como baseline o features complementarias.

#### Deep Learning (GPU)
- **tensorflow (2.19.0)** y complementos (**tensorflow-text, tensorflow-hub, tf-keras**): Framework de DL para definir/entrenar RNNs (Keras API), ops de texto y modelos preentrenados.
- **torch (2.7.0 + CUDA)**: Framework DL alternativo muy usado en investigación. RNNs con `torch.nn.RNN/LSTM/GRU` y ecosistema PyTorch.
- **triton (3.3.0)**: Kernel compiler usado por PyTorch para optimizaciones.

#### CUDA / Librerías de NVIDIA (vía conda y/o pip)
- **cuda-version (12.9)**: Metapaquete que alinea el runtime CUDA en el entorno.
- Librerías como **cublas**, **cufft**, **curand**, **cusolver**, **cusparse**, **nccl**, **cudnn**: Aceleración numérica, FFT, RNG, álgebra lineal, comunicación multi-GPU.

Notas:
- En pip, PyTorch GPU puede requerir instalarse desde el índice oficial de PyTorch para obtener la rueda `+cuXXX`. La réplica exacta del entorno actual ya incluye `torch 2.7.0+cu126`.
- No se necesita `nvcc` para entrenar/inferir. Solo es necesario si compilas kernels personalizados.


