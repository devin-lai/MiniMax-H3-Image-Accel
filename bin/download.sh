#!/usr/bin/env bash
set -euo pipefail
base=${H3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export H3_ROOT="$base"
models=${H3_MODELS_DIR:-$base/models}
argv=("$@")
for ((i=0; i<${#argv[@]}; i++)); do
  if [[ ${argv[i]} == --models-dir ]]; then
    models=${argv[i+1]:?Missing --models-dir value}
  elif [[ ${argv[i]} == --models-dir=* ]]; then
    models=${argv[i]#--models-dir=}
  fi
done
mkdir -p "$models"
exec "$base/.venv/bin/python" "$base/bin/disk_guard.py" --path "$models" -- \
  "$base/.venv/bin/python" "$base/bin/download_weights.py" "$@"
