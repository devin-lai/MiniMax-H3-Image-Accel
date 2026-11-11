# Runtime guide

[Project README](../README.md) · [Benchmarking](benchmarking.md) · [Licenses](../THIRD_PARTY_NOTICES.md)

## Requirements

- **Host**. Linux x86-64 with Bash, Git, system Python, and the Linux utilities `flock`, `setsid`, and `taskset`.
- **GPU**. NVIDIA L20 (48 GB) or GeForce RTX 5090 (32 GB) with a driver compatible with CUDA 13.0. Single-GPU profiles use pinned host memory for offloaded weights. Check host RAM as well as VRAM.
- **Build**. A C++20-capable compiler. The installer creates a Python 3.12 environment, installs the pinned PyTorch/CUDA packages, and builds SageAttention2 for the visible GPU architectures.
- **Storage**. Allow 126.08 GiB for 53 checkpoint files, plus the build, dependency and JIT caches, and any output images. The installer checks for a 60 GiB reserve. The disk guard uses a 78% filesystem utilization ceiling, including the space reserved for downloads.

`bin/install.sh` can bootstrap the pinned version of `uv` using `python3 -m pip`. Set `H3_UV` to use an existing binary, `H3_PYPI_INDEX` to select a PyPI mirror, or `H3_BUILD_GPU` to limit which visible GPU architectures are used for the SageAttention2 build. An optional compatible SM120 wheel under `wheels/` is checked by SHA-256 before use. Set `H3_BUILD_SAGE=1` to force a source build.

Installation fetches the pinned SGLang revision, applies the [H3 inference patch](../vendor/sglang-h3.patch), and caches the prepared runtime for offline serving.

## Checkpoints and storage

The downloader fetches the native-layout MiniMax-H3-Image checkpoint and the text encoder from MiniMax-H3. It verifies sizes and SHA-256 hashes against the pinned manifest. Review the upstream model agreements before downloading or using the files. [Checkpoint manifest](../configs/weights-manifest.json) · [Model agreements](../THIRD_PARTY_NOTICES.md#model-files-and-results)

```bash
python3 bin/download_weights.py --list
export H3_MODELS_DIR=/path/to/model-storage  # optional, shared by download and serve
bin/download.sh
# For an existing copy, verify the files and write the local verified.json marker.
bin/download.sh --verify-only
```

`HF_TOKEN` and `HF_ENDPOINT` use standard Hugging Face Hub behavior. The server runs in offline mode. `H3_ROOT` can override the checkout root. The runtime creates its environment, models, caches, logs, and run outputs separately from source files.

## GPU profiles

Server profiles select hardware, precision, and attention. Request profiles select the denoising schedule. All server profiles start with six native passes unless the request explicitly changes the schedule.

| Server profile | GPU layout | Precision and behavior |
| --- | --- | --- |
| `auto`, `accurate` | TP4 by default, or one idle L20 if TP4 cannot fit | BF16 and dense FlashAttention. L20 uses PCIe-aware placement and calibrated host transport for eligible collectives. RTX 5090 uses NCCL. |
| `native` | TP4 by default | Native BF16 and dense FlashAttention with NCCL by default. |
| `single` | 1 L20 or RTX 5090 | BF16 and dense FlashAttention. L20 keeps the VAE and part of the encoder resident and streams the DiT. RTX 5090 streams the encoder and part of the DiT. |
| `single-reference` | 1 RTX 5090 | BF16 streaming control with zero permanently resident DiT blocks. |
| `single-fp8` | 1 RTX 5090 | MXFP8 DiT, BF16 encoder, dense FlashAttention. |
| `single-fast` | 1 RTX 5090 | MXFP8 DiT and encoder, SageAttention2. |
| `single-int8-mlp` | 1 RTX 5090 | Experimental ConvRot INT8 in DiT MLPs, with BF16 encoder and attention. |
| `fast` | 2 or 4 L20s, or 4 RTX 5090s | L20 uses E4M3 FP8 with Ulysses and quantized exchange. RTX 5090 uses MXFP8 with TP2 × Ulysses2. Both use SageAttention2. |
| `latency` | Up to 8 L20s | One quantized Ulysses replica across the largest compatible group of 8, 4, or 2 GPUs. |
| `throughput` | L20 replicas, 2 GPUs each by default | Multiple FP8/SageAttention2 replicas behind one endpoint. Use `H3_ULYSSES=4` for four GPUs per replica. |

Use `bin/start_profile.sh --help` for the full list, including the optional `fp8-dit`, `compiled`, `graph`, and `sage3` profiles. SageAttention3 on RTX 5090 requires a separate build with `bin/install_sage3.sh`.

```bash
H3_GPUS=0 bin/start_profile.sh single --print-config
bin/start_profile.sh accurate --print-config
bin/h3ctl status
bin/h3ctl stop
```

`H3_GPUS` accepts comma-separated PCI-ordered CUDA ordinals or GPU UUIDs. Without it, the launcher uses `CUDA_VISIBLE_DEVICES`, available memory, and topology to select a group. `H3_TP`, `H3_ULYSSES`, and `H3_DP` adjust the layout where supported. `H3_CPUS` overrides CPU affinity. An empty value disables that override and leaves the process unpinned.

Stop the current instance before switching profiles. The default server binds to `127.0.0.1:30000`. Use the same `H3_PORT` for launch, status, and stop. For concurrent instances, use disjoint GPU groups and distinct HTTP, master, and scheduler ports. Reserve `H3_PORT+1` for each instance’s offline broker. First inference can spend time compiling CUDA kernels. Review `logs/current.log` for the default port or `logs/port-<port>/current.log` for a custom port.

## Generate and edit

The CLI defaults to `--profile six --quality exact`. The single-L20 server also defaults to native `exact`. Set the quality explicitly when comparing configurations. Upstream `quality=lossless` enables quality-gated fusion and tile batching and can change the pixels of a seeded image. Both settings can return losslessly encoded PNGs.

```bash
.venv/bin/python bin/generate.py "$(cat examples/edit-prompt.txt)" \
  --image examples/edit-reference.png --profile six --quality exact \
  --size 2048x2048 --seed 42 --output generated/coral-circle.png
```

The example input contains an orange circle and a blue arch. The edit prompt requests a color change while keeping the layout. [Reference PNG](../examples/edit-reference.png) · [Prompt](../examples/edit-prompt.txt)

The OpenAI-compatible generation endpoint accepts the same settings as JSON.

```bash
curl -fsS http://127.0.0.1:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d @configs/single-gpu-request.json
```

The response contains the PNG as `data[0].b64_json`. [`example-request.json`](../configs/example-request.json) shows the upstream `lossless` preset. The `/v1/images/edits` endpoint accepts a multipart `image` field and optional JSON in the `h3_opt` field. The CLI accepts `--url` when using a different local port.

### Request profiles

| Request profile | Denoising schedule |
| --- | --- |
| `six` | All six native passes. |
| `balanced` | Four passes using native heads at indices `[0, 1, 3, 5]` and merged velocity projections. |
| `extreme` | Three merged passes at `[0, 2, 5]` for previews. |
| `sparse-preview` | Three original heads without merging. |
| `tea`, `tea-conservative`, `tea-extreme` | Adaptive residual reuse with different thresholds and skip limits. |

Keep `num_inference_steps=6` for API requests. `h3_opt` selects passes from the six native heads. Merged heads and residual reuse are runtime approximations and can change details such as typography and texture. [Request profile definitions](../configs/request-profiles.json)

## Runtime controls

- **L20 BF16 transport**. The TP4 host all-reduce calibrates and checks its reduction order against NCCL before use for a supported buffer size. `H3_OPT_HOST_TP=0` returns all reductions to NCCL. `H3_PCIE_PROBE=0` skips probing PCIe uplinks.
- **Single L20**. `H3_SINGLE_ENCODER_RESIDENT` sets the resident encoder fraction (default `0.8`). `H3_OPT_EXACT_VAE_GRAPH=0` restores eager execution for the native VAE path.
- **Single RTX 5090**. `H3_SINGLE_RESIDENT_LAYERS` sets the resident BF16 DiT block count (default `24`). `H3_OPT_ENCODER_MMAP=1` borrows compatible checkpoint storage before pinned staging. Online MXFP8 conversion stages one linear layer at a time.
- **Conditioning**. A bounded cache can reuse embeddings for repeated prompts. Set `H3_NO_CONDITIONING_CACHE=1` to measure fresh conditioning. Experimental prefix KV reuse is off by default and can be selected with `--prefix-kv` for text-only prompts.
- **Output**. `H3_OPT_GPU_OUTPUT=1` packs the pixels on the GPU before transferring to the host. PNG encoding and Lanczos resizing use the CPU.
- **L20 FP8 exchange**. The quantized Ulysses path moves Q, K, V, and attention output through page-locked shared memory. `H3_OPT_QUANT_ULYSSES=0` selects the NCCL exchange.
