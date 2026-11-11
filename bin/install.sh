#!/usr/bin/env bash
set -euo pipefail
base=${H3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export H3_ROOT="$base"
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'Install on the Linux x86_64 NVIDIA node; the Desktop copy is source code.' >&2; exit 1;
}
mkdir -p "$base"/{cache,tmp,logs,models,runs}
export UV_CACHE_DIR="$base/cache/uv" UV_PYTHON_INSTALL_DIR="$base/cache/python"
export TMPDIR="$base/tmp"
uv_bin=${H3_UV:-$(command -v uv || true)}
if [[ -z "$uv_bin" ]]; then
  python3 "$base/bin/disk_guard.py" --path "$base" --reserve-gib 2 -- \
    python3 -m pip install --target "$base/.bootstrap" uv==0.11.25
  uv_bin="$base/.bootstrap/bin/uv"
fi
if [[ ! -x "$base/.venv/bin/python" ]]; then
  "$uv_bin" venv --python 3.12 "$base/.venv"
fi
python_bin="$base/.venv/bin/python"
"$python_bin" "$base/bin/prepare_sglang.py"
index_args=(--index-url "${H3_PYPI_INDEX:-https://pypi.org/simple}")
# Keep all lock versions, including the tested CUDA wheel, without re-resolving
# optional SGLang packages that pin an older torch.
"$python_bin" "$base/bin/disk_guard.py" --path "$base" --reserve-gib 60 -- \
  "$uv_bin" pip install --python "$python_bin" --no-deps --index-strategy unsafe-best-match \
  "${index_args[@]}" \
  -r "$base/requirements.lock"
cuda_home=$($python_bin -c 'import pathlib,nvidia; print(pathlib.Path(next(iter(nvidia.__path__))) / "cu13")')
[[ -x "$cuda_home/bin/nvcc" ]] || { echo 'CUDA compiler missing from the pinned toolchain.' >&2; exit 1; }
[[ -e "$cuda_home/lib64" ]] || ln -s lib "$cuda_home/lib64"
[[ -e "$cuda_home/lib/libcudart.so" ]] || ln -s libcudart.so.13 "$cuda_home/lib/libcudart.so"
export CUDA_HOME="$cuda_home" PATH="$cuda_home/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-8}
if [[ -n ${H3_BUILD_GPU:-} ]]; then export CUDA_VISIBLE_DEVICES="$H3_BUILD_GPU"; fi
wheel="$base/wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
if [[ ${H3_BUILD_SAGE:-0} != 1 && -r "$wheel" ]] && \
  "$python_bin" -c 'import sys, torch; caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}; sys.exit(torch.__version__ != "2.14.1+cu130" or caps != {(12, 0)})'; then
  "$python_bin" -c 'import hashlib,sys; from pathlib import Path; assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == "a665ff7248eb9b2715c0e7665202db9d95d9874b72574ce4b0ad32ec922b26a6"' "$wheel"
  "$uv_bin" pip install --python "$python_bin" --no-deps "$wheel"
else
  "$python_bin" "$base/bin/disk_guard.py" --path "$base" --reserve-gib 3 -- \
    "$uv_bin" pip install --python "$python_bin" --no-deps --no-build-isolation \
    "${index_args[@]}" "$base/vendor/SageAttention"
fi
echo 'Runtime installed. Run bin/download.sh, then bin/start_profile.sh auto.'
