# MiniMax-H3-Image-Accel

![MiniMax H3 Image Accel. Native 4K generation and editing on NVIDIA L20 and RTX 5090.](docs/assets/social-preview.png)

Run **MiniMax-H3-Image on a single L20 or RTX 5090**, or spread inference across multiple GPUs. The SGLang runtime supports native 4K generation and reference-image editing. Start with BF16 and all six native denoising passes. Opt into FP8 and shorter schedules when you want to trade reference fidelity for lower latency.

[Quickstart](#quickstart) · [Runtime guide](docs/runtime.md) · [Benchmark your GPU](docs/benchmarking.md) · [Releases](https://github.com/devin-lai/MiniMax-H3-Image-Accel/releases)

## Choose a profile

| Hardware | Start here | What it does |
| --- | --- | --- |
| **1 × NVIDIA L20** | `single` | Keeps most of the encoder and the VAE in VRAM. Streams the DiT and checks CUDA graph output against eager decoding before replay. |
| **1 × RTX 5090** | `single` | Keeps a subset of BF16 DiT blocks in VRAM and prefetches the remaining layers from host RAM. |
| **4 × L20 or RTX 5090** | `auto` | Uses BF16 tensor parallelism (TP4) and dense FlashAttention. On L20, eligible host reductions are checked against NCCL before use. |
| **When latency or throughput comes first** | `single-fast`, `fast`, `latency`, `throughput` | Opt-in FP8 or MXFP8 with SageAttention2. These profiles can change the output for the same seed. |

`accurate` is an alias for `auto`. If TP4 cannot fit, `auto` can fall back to an idle L20. Use `single` explicitly on one RTX 5090. [Full profile matrix](docs/runtime.md#gpu-profiles)

## Quickstart

You’ll need a **Linux x86-64 host**, a CUDA 13.0-compatible NVIDIA driver, a C++20 compiler, and storage for **126.08 GiB of model files** plus the environment and caches. Single-GPU profiles also use host RAM for offload. The checkpoints come from Hugging Face under their own territory and use restrictions. [MiniMax-H3-Image license](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) · [Install requirements](docs/runtime.md#requirements)

```bash
git clone https://github.com/devin-lai/MiniMax-H3-Image-Accel.git
cd MiniMax-H3-Image-Accel
bin/install.sh
bin/download.sh
```

Start the profile for your hardware. Run one of these commands.

```bash
# Four GPUs, with a single-L20 fallback
bin/start_profile.sh auto

# One L20 or RTX 5090
H3_GPUS=0 bin/start_profile.sh single
```

When `bin/h3ctl status` reports a healthy server, generate a seeded image.

```bash
.venv/bin/python bin/generate.py \
  "An editorial photograph of a cobalt perfume bottle on pale stone" \
  --size 2048x2048 --seed 731 --profile six --quality exact \
  --output generated/bottle.png
```

Use `--size 4096x4096` for native 4K or pass `--image` for reference editing. The CLI saves a PNG and a JSON record of the request. [Edit example and HTTP API](docs/runtime.md#generate-and-edit)

## Benchmark your GPU

Compare two configurations with the same prompt, seed, resolution, GPU layout, and quality preset. The benchmark tools record warmed HTTP latency, concurrent images per minute, and per-image hashes. The accuracy audit reports PNG byte equality and can require identical decoded pixels or enforce PSNR and SSIM thresholds per image. [Run a matched benchmark](docs/benchmarking.md)

## Explore the source

| Path | Contents |
| --- | --- |
| [`bin/`](bin/) | Install, serve, generate, benchmark, and compare |
| [`configs/`](configs/) | Request profiles, example payloads, and pinned checkpoint hashes |
| [`tests/`](tests/) | Profile selection, image comparison, and CUDA correctness tests |
| [`vendor/`](vendor/) | H3 inference patch for pinned SGLang, SageAttention sources, and optional SageAttention3 build headers |

Reproductions on different GPU layouts and focused fixes are welcome. [Contributing guide](CONTRIBUTING.md) · [GitHub Issues](https://github.com/devin-lai/MiniMax-H3-Image-Accel/issues)

## License

Project code is available under [Apache 2.0](LICENSE). Vendored components retain their licenses, and the MiniMax checkpoints are governed by their own agreements. [Third-party notices and model licenses](THIRD_PARTY_NOTICES.md)
