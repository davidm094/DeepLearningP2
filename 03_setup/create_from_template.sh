#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 <nombre_entorno>" >&2
  exit 1
fi

ENV_NAME="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL_FILE="${HERE}/environment.template.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda no está en PATH." >&2
  exit 1
fi

echo "Creando entorno '${ENV_NAME}' desde ${TPL_FILE}..."
conda env create -n "${ENV_NAME}" -f "${TPL_FILE}"

echo "Entorno creado. Activa con: conda activate ${ENV_NAME}"
echo "Sugerencia: si PyTorch quedó en CPU, instala la rueda CUDA así:"
echo "  pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0"


