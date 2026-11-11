#!/usr/bin/env bash
set -euo pipefail
base=${H3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
export H3_ROOT="$base"
profile=${1:-native}; shift || true
port=${H3_PORT:-30000}
gpus=${H3_GPUS:-0,1,2,3}; tp=${H3_TP:-4}; ulysses=${H3_ULYSSES:-1}
replicas=${H3_DP:-1}
attention=${H3_ATTENTION:-fa}
models=${H3_MODELS_DIR:-$base/models}
mkdir -p "$base"/{logs,tmp,cache,runs}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpus"
export PATH="$base/.venv/bin:$PATH" PYTHONPATH="$base/vendor/sglang/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$base/cache/hf" XDG_CACHE_HOME="$base/cache" TMPDIR="$base/tmp"
export TORCHINDUCTOR_CACHE_DIR="$base/cache/inductor" TRITON_CACHE_DIR="$base/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$base/cache/flashinfer"
export SGLANG_CACHE_DIR="$base/cache/sglang" SGLANG_JIT_CACHE_DIR="$base/cache/sglang/jit"
export SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS=12
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
export H3_OPT_GPU_CONDITIONING=${H3_OPT_GPU_CONDITIONING:-1}
export H3_OPT_GPU_OUTPUT=${H3_OPT_GPU_OUTPUT:-1}
export H3_OPT_PREFIX_KV=${H3_OPT_PREFIX_KV:-0}
# This deployment's driver reports no GPU P2P. Override on a different topology.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
cuda_home="$base/.venv/lib/python3.12/site-packages/nvidia/cu13"
if [[ -x "$cuda_home/bin/nvcc" ]]; then
  export CUDA_HOME="$cuda_home" PATH="$cuda_home/bin:$PATH"
fi
"$base/.venv/bin/python" "$base/bin/disk_guard.py" --path "$base"
"$base/.venv/bin/python" "$base/bin/disk_guard.py" --path "$models"
[[ -r "$models/verified.json" ]] || {
  echo 'Run bin/download.sh (or --verify-only against existing weights) first.' >&2; exit 1;
}
IFS=, read -r -a gpu_array <<< "$gpus"
[[ "$replicas" =~ ^[1-9][0-9]*$ && ${#gpu_array[@]} -eq $((tp * ulysses * replicas)) ]] || {
  echo 'GPU count must equal TP × Ulysses × replicas.' >&2
  exit 1
}
if (( replicas > 1 )); then export H3_OPT_DP=1; fi
args=()
case "$profile" in
  native) ;;
  fp8-all) args+=(--quantization fp8 --component-quantizations.text_encoder fp8) ;;
  fp8-dit) args+=(--quantization fp8) ;;
  mxfp8-all) args+=(--quantization mxfp8 --component-quantizations.text_encoder mxfp8) ;;
  mxfp8-dit) args+=(--quantization mxfp8) ;;
  int8-mlp) args+=(--quantization convrot_int8 --quantization-ignored-layers attn token_refiner) ;;
  *) echo "Unknown quantization profile: $profile" >&2; exit 2 ;;
esac
if [[ "$attention" == sage_attn ]]; then
  export SGLANG_H3_IMAGE_EXPERIMENTAL_SAGE2=1
elif [[ "$attention" == sage_attn_3 ]]; then
  export H3_OPT_SAGE3_EXPERIMENTAL=1
fi
if [[ ${H3_NO_CONDITIONING_CACHE:-0} == 1 ]]; then args+=(--disable-conditioning-cache); fi
cd "$base"
# The guard also watches generated files and compilation caches during serving.
exec "$base/.venv/bin/python" "$base/bin/disk_guard.py" --path "$base" -- \
 "$base/.venv/bin/python" "$base/bin/sglang_diffusion_cli.py" serve \
 --model-path "$models/MiniMax-H3-Image" --text-encoder-path "$models/MiniMax-H3/text_encoder" \
 --backend sglang --host 127.0.0.1 --port "$port" \
 --num-gpus "${#gpu_array[@]}" --tp-size "$tp" --ulysses-degree "$ulysses" --ring-degree 1 --dp-size "$replicas" \
 --component-residency all=resident --encoder-parallel fold --minimax-h3-adaln-online true \
 --component-attention-backends "transformer=$attention" --warmup-mode off \
 --output-path "" --input-save-path "" --async-output-save "${args[@]}" "$@"
