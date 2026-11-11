#!/usr/bin/env bash
set -euo pipefail
base=${H3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH="$base/vendor/sglang/python" OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${H3_TEST_GPU:-0}
export CUDA_HOME="$base/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$base/.venv/bin:$PATH"
"$base/.venv/bin/python" "$base/tests/test_launch_profile.py"
"$base/.venv/bin/python" "$base/tests/test_quant_ulysses.py"
exec "$base/.venv/bin/python" "$base/tests/test_h3opt.py"
