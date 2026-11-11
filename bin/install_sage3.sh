#!/usr/bin/env bash
# Optional FP4 attention experiment; the default runtime remains Sage2.
set -euo pipefail
base=${H3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export H3_ROOT="$base"
python_bin="$base/.venv/bin/python"
uv_bin=${H3_UV:-$(command -v uv || true)}
[[ -n "$uv_bin" ]] || uv_bin="$base/.bootstrap/bin/uv"
export CUDA_HOME="$base/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" CUDA_VISIBLE_DEVICES=${H3_BUILD_GPU:-0} MAX_JOBS=${MAX_JOBS:-8} FAHOPPER_FORCE_BUILD=TRUE
mkdir -p "$base/cache/sage3-wheels"
cd "$base/vendor/SageAttention/sageattention3_blackwell"
"$python_bin" "$base/bin/disk_guard.py" --path "$base" --reserve-gib 3 -- \
  "$python_bin" setup.py bdist_wheel --dist-dir "$base/cache/sage3-wheels"
"$uv_bin" pip install --python "$python_bin" --no-deps "$base"/cache/sage3-wheels/sageattn3-*.whl
