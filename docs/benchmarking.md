# Benchmarking and accuracy

[Project README](../README.md) · [Published benchmark data](benchmarks/README.md) · [Runtime guide](runtime.md)

## Run a matched comparison

Measure warmed serial latency with [`benchmark.py`](../bin/benchmark.py). Use the same physical GPUs, prompt, seed, input image, resolution, request profile, quality preset, and PNG compression on both sides. Choose unique labels because results are appended to `runs/<label>/results.jsonl`.

The example below isolates the transport on **four L20s**. Run each benchmark after `bin/h3ctl status` reports a healthy server. Adjust `H3_GPUS` to your machine and use the same selection for both runs.

```bash
# Reference
H3_GPUS=0,1,2,3 H3_NO_CONDITIONING_CACHE=1 H3_OPT_HOST_TP=0 \
  H3_OPT_HOST_ULYSSES=0 bin/start_profile.sh accurate

.venv/bin/python bin/benchmark.py --label bf16-nccl \
  --quality exact --repeat 3 --warmup-matched

bin/h3ctl stop

# Same layout with calibrated host transport
H3_GPUS=0,1,2,3 H3_NO_CONDITIONING_CACHE=1 bin/start_profile.sh accurate

.venv/bin/python bin/benchmark.py --label bf16-host \
  --quality exact --repeat 3 --warmup-matched
```

The default suite covers 2K product, 2K portrait, 2K reference editing, and native 4K still life. The edit case uses the bundled geometric input in a fresh checkout. Use `--cases smoke` for a 1K check or `--seed-offset` to collect another matched set. The benchmark CLI defaults to upstream `lossless`, so specify `--quality exact` when comparing the native math.

For a single-RTX-5090 BF16 comparison, use `single-reference` and `single` with the same card and cache setting. To isolate CUDA replay on a single L20, compare `single` with `H3_OPT_EXACT_VAE_GRAPH=0` and with replay enabled, using `exact` on both sides. This measures the decoder change with the same offload policy.

## Check output equality

```bash
.venv/bin/python bin/validate_accuracy.py --runs runs \
  --reference bf16-nccl --candidate bf16-host --pixel-identical
```

The audit verifies matching case metadata and validates each file against its recorded hash. It reports whether the PNG hashes match and, with `--pixel-identical`, fails on any decoded pixel difference. For an approximate profile, omit `--pixel-identical` to apply the per-case gates of RGB PSNR ≥ 40 dB and luma SSIM ≥ 0.99. These scores quantify closeness to the chosen reference. Review full frames and matched crops to assess prompt adherence, typography, and texture.

Use TP1 versus TP1 and TP4 versus TP4 when checking equality. The order of BF16 operations can differ across topologies. The `exact` and `lossless` presets also use different arithmetic paths.

## Build an image stack

After checking the two matched runs above, export their full frames and 1:1 crops.

```bash
.venv/bin/python bin/make_comparisons.py --runs runs \
  --variant 'bf16-nccl=BF16 NCCL' \
  --variant 'bf16-host=BF16 verified host transport' \
  --output runs/bf16-comparison
```

Open `runs/bf16-comparison/side-by-side.png` for the full-frame comparison and `detail-crops.png` for matched source-pixel regions. The same folder contains `pixel-differences.csv`.

## Measure throughput

Repeat a URL for concurrent clients on the same endpoint. Use different URLs to exercise separate instances.

```bash
.venv/bin/python bin/benchmark_throughput.py --label two-clients \
  --url http://127.0.0.1:30000 --url http://127.0.0.1:30000 \
  --case product --quality exact --requests-per-client 8
```

Serial latency includes server PNG/base64 encoding and receipt of the full HTTP response. Model startup, warmups, and client decode and writes are outside that interval. Throughput is completed images over concurrent wall time, with client PNG validation included. Record the GPU count, precision, attention backend, pass count, quality, cache policy, trial count, and median or p95 with any timing you report.

## Tools and records

- [`summarize_benchmarks.py`](../bin/summarize_benchmarks.py) summarizes trial latency and PNG hash consistency.
- [`compare_quality.py`](../bin/compare_quality.py) calculates PSNR and SSIM. [`make_comparisons.py`](../bin/make_comparisons.py) builds visual comparisons from local runs. [`compare_run_stack.py`](../bin/compare_run_stack.py) verifies matched metadata before building a stack.
- The `export_*` scripts and [`plot_l20.py`](../bin/plot_l20.py) rebuild historical figures from the original study records. Plotting uses NumPy, Pillow, and Matplotlib.

Generated PNGs and benchmark records are written under the ignored `runs/` directory. MiniMax’s model agreements govern use and distribution of outputs and results. [Model agreements](../THIRD_PARTY_NOTICES.md#model-files-and-results)
