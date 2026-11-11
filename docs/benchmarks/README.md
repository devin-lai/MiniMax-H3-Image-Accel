# Benchmark figures and data

These wall-clock measurements come from the project's HTTP benchmark harness running the pinned checkpoint in SGLang. The figures retain the original timing scales, precision settings, GPU counts, and sample counts. [Run the benchmarks](../benchmarking.md)

| Workload | Figures | Data |
| --- | --- | --- |
| L20 serial latency | [One L20 with native exact quality](single-exact.png) · [Four L20s with BF16](l20-accurate.png) · [FP8 and shorter schedules](l20-fast.png) | [Latency CSV](latency.csv) |
| RTX 5090 serial latency | [One card](single-rtx5090.png) · [Four cards](rtx5090.png) | [Single-card CSV](single-rtx5090-latency.csv) · [Four-card rows in the latency CSV](latency.csv) |
| Completed requests per minute | [One and eight L20s](throughput.png) | [L20 CSV](throughput.csv) · [Single RTX 5090 CSV](single-rtx5090-throughput.csv) |

The latency tables report warmed serial HTTP medians and observed min–max ranges in seconds. The timer includes server PNG/base64 encoding and receipt of the full localhost response. Model loading, warmups, cold compilation, and client decoding and writes fall outside that interval. The single-L20 `exact` and single-RTX-5090 latency runs disabled conditioning-cache hits.

Compare within the same GPU count, workload, pass schedule, and request preset to isolate a runtime change. The four-L20 BF16 chart includes an earlier GPU placement and a matched pair on the spread placement. The four-RTX-5090 chart includes both `exact` and upstream `lossless` rows, with trial counts printed on each bar.

The throughput runs measure completed images over wall-clock time after warmup. The one-L20 measurements use two queued clients on one endpoint. The eight-L20 measurements use four clients and four two-GPU replicas. The single-RTX-5090 throughput series uses six requests from one client.

## BF16 output checks

| Hardware | Matched comparison | Request quality | Byte-identical PNG pairs |
| --- | --- | --- | ---: |
| One L20 | Streamed BF16 and BF16 residency with VAE replay | `exact` | 8 / 8 |
| Four L20s | BF16 NCCL and verified BF16 host transport on the same PCIe placement | `lossless` | 12 / 12 |

Each count covers the audited prompt and seed pairs. Run the [output equality check](../benchmarking.md#check-output-equality) and [image-stack exporter](../benchmarking.md#build-an-image-stack) to inspect your own runs.
