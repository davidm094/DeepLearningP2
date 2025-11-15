#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ml001_env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${HERE}/environment.lock.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda no está en PATH." >&2
  exit 1
fi

echo "Creando/actualizando entorno '${ENV_NAME}' desde ${LOCK_FILE}..."
if conda env list | grep -E "^${ENV_NAME}\s" >/dev/null 2>&1; then
  conda env update -n "${ENV_NAME}" -f "${LOCK_FILE}" --prune
else
  conda env create -n "${ENV_NAME}" -f "${LOCK_FILE}"
fi

echo "Listo. Activa con: conda activate ${ENV_NAME}"


