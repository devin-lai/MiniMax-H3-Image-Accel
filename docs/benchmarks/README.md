# Benchmark figures and data

[Headline speedups](../../README.md#benchmark-results) · [Fidelity charts by optimization](fidelity.md) · [Run the benchmarks](../benchmarking.md)

## What the reference measures

The BF16 controls run the pinned official MiniMax-H3-Image checkpoint with six native passes in the project's [pinned SGLang runtime](../../provenance/sources.json). The study did not benchmark a standalone, unmodified MiniMax/Diffusers installation. The headline chart compares the builds below within the same GPU count and preset. The upstream `lossless` preset enables fusion and tile batching and can change the seeded output relative to `exact`; both return lossless PNGs.

| Hardware | Reference for the headline speedup | What changed |
| --- | --- | --- |
| 1 L20, `exact` | Streamed BF16/FlashAttention (`single-exact-reference`) | Encoder and VAE residency, then verified VAE graph replay |
| 4 L20s, `lossless`, BF16 | BF16/FlashAttention TP4 with NCCL on spread PCIe placement (`accuracy-reference-final`) | Verified BF16 host transport at the same placement |
| 4 L20s, `lossless`, FP8 comparison | Earlier BF16/FlashAttention TP4 on the original PCIe placement (`l20-bf16`) | FP8/SageAttention2 with Ulysses4 and quantized exchange; the comparison includes the layout change |
| 1 RTX 5090, `lossless` | Streamed BF16/FlashAttention (`single-bf16-stream` on the primary card) | BF16 DiT residency, or the opt-in MXFP8 DiT/encoder and SageAttention2 build |
| 4 RTX 5090s, `lossless` | Tuned BF16 (`bf16-tuned`) | MXFP8/SageAttention2, TP2 + Ulysses2, resident components and GPU output packing |

## Latency

| Workload | Figures | Data |
| --- | --- | --- |
| L20 serial latency | [One L20 with native exact quality](single-exact.png) · [Single-card experiments](single-l20-tradeoffs.png) · [Four L20s with BF16](l20-accurate.png) · [FP8 and shorter schedules](l20-fast.png) | [Latency CSV](latency.csv) |
| RTX 5090 serial latency | [One card](single-rtx5090.png) · [Four cards](rtx5090.png) | [Primary single-card CSV](single-rtx5090-latency.csv) · [Four-card and single-card follow-up rows](latency.csv) |

The latency tables report warmed serial HTTP medians and observed min–max ranges in seconds. The timer includes server PNG/base64 encoding and receipt of the full localhost response. Model loading, warmups, cold compilation, and client decoding and writes fall outside that interval. The single-L20 `exact` and single-RTX-5090 latency runs disabled conditioning-cache hits.

Compare within the same GPU count, workload, pass schedule, and request preset to isolate a runtime change. On four L20s the move from the original BF16 placement to the verified host build cut latency by **29.1–31.3%** across the four workloads; changing transport alone at the same placement cut it by **5.1–5.9%**. Within four-GPU, six-pass FP8/SageAttention2 builds, the earlier NCCL build took **3.45 s / 14.72 s** (2K product / native 4K), versus **2.29 s / 10.14 s** with quantized exchange.

The four-RTX-5090 chart includes both `exact` and `lossless` controls, each with one timed trial per case. The optimized builds use three trials; the six-pass 2K product result ranged from 2.39 to 9.65 s. The single-L20 exploratory rows generally use one trial. The eight-GPU FP8 rows use twice the GPUs of the four-L20 BF16 reference.

## Throughput

All rows below use six passes. Rates are completed images per minute measured over wall time after warmup.

| Hardware and workload | Reference → optimized | Images/min | Gain |
| --- | --- | ---: | ---: |
| 1 L20, 2K, `exact` | Streamed BF16 → residency and replay | 3.62 → 5.12 | +41.3% |
| 1 L20, 2K, `lossless` | Streamed BF16 → residency | 4.61 → 5.22 | +13.1% |
| 8 L20s, 2K | FP8/NCCL → FP8 with quantized exchange | 42.10 → 58.20 | +38.2% |
| 8 L20s, native 4K | FP8/NCCL → FP8 with quantized exchange | 8.91 → 11.06 | +24.1% |
| 1 RTX 5090, 2K | Streamed BF16 → MXFP8/SageAttention2 | 8.72 → 18.17 | +108.3% |
| 1 RTX 5090, 2K, repeated prompt | Streamed BF16 → BF16 residency and conditioning cache hits | 8.59 → 9.98 | +16.1% |

The one-L20 runs use eight requests from two queued clients on one endpoint. The eight-L20 runs use four clients and four two-GPU replicas. The single-RTX-5090 runs use six requests from one client. The cache-hit comparison uses its own same-card reference and repeats the prompt with different seeds.

[L20 chart](throughput.png) · [L20 CSV](throughput.csv) · [Single RTX 5090 primary CSV](single-rtx5090-throughput.csv) · [Single RTX 5090 follow-up CSV](single-rtx5090-follow-up-throughput.csv)

## BF16 output checks

| Hardware | Matched comparison | Request quality | Byte-identical PNG pairs |
| --- | --- | --- | ---: |
| One L20 | Streamed BF16 and BF16 residency with VAE replay | `exact` | 8 / 8 |
| One L20 | Streamed BF16 and BF16 residency | `lossless` | 12 / 12 |
| Four L20s | BF16 NCCL and verified BF16 host transport on the same PCIe placement | `lossless` | 12 / 12 |
| One RTX 5090 | Streamed BF16 and BF16 with resident DiT blocks | `lossless` | 12 / 12 |

Each count covers the audited prompt and seed pairs. The [fidelity charts](fidelity.md) include approximate profiles and intermediate experiments. Run the [output equality check](../benchmarking.md#check-output-equality) and [image-stack exporter](../benchmarking.md#build-an-image-stack) to inspect your own runs.

## Data sources and chart regeneration

The benchmark records come from the `minimax-h3-image-opt` study at commit `331671cf870ad8e86bb8b97b7bb1a70b5a383a81`. The L20 and four-RTX-5090 timing series are from `comparisons/overview/plotted-latency.csv`, with the one-L20 graph ablation from `provenance/single-l20/benchmarks.csv`. The one-RTX-5090 timings and follow-ups are from `provenance/single5090/`. Each [fidelity row](fidelity.csv) names its original CSV and both run IDs. The four-RTX-5090 [PNG hash data](quad-rtx5090-png-hashes.csv) comes from `comparisons/rtx5090/png-hashes.csv`, plus the headline BF16 and MXFP8 pair matched directly from the request records in `provenance/results/`.

With Matplotlib installed, regenerate the headline speedup and fidelity charts from the bundled CSVs:

```bash
python3 bin/export_benchmark_report.py
```
