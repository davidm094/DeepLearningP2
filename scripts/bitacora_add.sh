#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 \"Mensaje del hito\" [TAG]" >&2
  exit 1
fi

MESSAGE="$1"
TAG="${2:-}"

if [[ -z "${TAG}" ]]; then
  "${PY}" "${HERE}/bitacora_add.py" -m "${MESSAGE}"
else
  "${PY}" "${HERE}/bitacora_add.py" -m "${MESSAGE}" --tag "${TAG}"
fi

echo "OK"


