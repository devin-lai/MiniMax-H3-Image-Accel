"""Export the GitHub comparison guide's charts, tables and complete run index.

Uses committed timing CSVs and request records; no GPU or model weights needed.
Run: python3 bin/export_benchmark_overview.py
"""

import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "comparisons/overview"
CASES = ("product", "portrait", "edit", "native-4k")
TITLES = ("2K product", "2K portrait", "2K edit", "Native 4K")
SOURCES = {
    "single": "provenance/single-l20/benchmarks.csv",
    "accurate": "provenance/accuracy/benchmarks.csv",
    "l20": "provenance/l20/benchmarks.csv",
    "rtx5090": "provenance/benchmarks.csv",
}
COLORS = {"reference": "#758395", "native": "#25816c",
          "earlier": "#3979b9", "approximate": "#c75d2c"}


def build(source, run, name, gpus, quality="lossless", passes="6", role="approximate"):
    return dict(source=source, run=run, name=name, gpus=gpus, quality=quality,
                passes=passes, role=role)


GROUPS = {
    "single-exact": (
        "One L20: native exact quality",
        [build("single", "single-exact-reference", "Native BF16 reference", 1, "exact", role="reference"),
         build("single", "single-exact-optimized", "single: residency + verified graphs", 1, "exact", role="native")],
        "Same physical GPU; 3 warmed trials/case; conditioning cache hits disabled. Identical PNGs on the tested suite.",
    ),
    "l20-accurate": (
        "Four L20s: BF16 with the reference TP4 arithmetic",
        [build("l20", "l20-bf16", "Earlier native BF16 / original placement", 4, role="reference"),
         build("accurate", "accuracy-reference-final", "Fresh BF16 + NCCL / spread placement", 4, role="earlier"),
         build("accurate", "accuracy-auto-final", "accurate: spread + verified BF16 transport", 4, role="native")],
        "All rows: upstream lossless quality, 6 passes, 3 trials/case. Placement and transport gains are shown separately.",
    ),
    "l20-fast": (
        "L20: approximate precision, exchange and pass schedules",
        [build("l20", "l20-bf16", "Native BF16 / TP4", 4, role="reference"),
         build("l20", "fp8-sage-tp1sp4", "Earlier FP8 + Sage2 / NCCL U4", 4, role="earlier"),
         build("l20", "l20q-sp4", "fast: FP8 + quantized U4", 4),
         build("l20", "l20q-sp4-four", "fast + four merged passes", 4, passes="4 merged"),
         build("l20", "l20q-sp4-three", "fast + three merged passes", 4, passes="3 merged"),
         build("l20", "l20q-sp8", "latency: FP8 + quantized U8", 8),
         build("l20", "l20q-sp8-four", "latency + four merged passes", 8, passes="4 merged"),
         build("l20", "l20q-sp8-three", "latency + three merged passes", 8, passes="3 merged")],
        "Quality=lossless. GPU counts and passes vary as labeled; 2-3 trials/case. Approximate rows change images.",
    ),
    "rtx5090": (
        "Four RTX 5090s: recorded native and optimized builds",
        [build("rtx5090", "bf16-exact", "Native BF16 / exact", 4, "exact", role="reference"),
         build("rtx5090", "bf16-tuned", "Tuned BF16 / upstream lossless", 4, role="earlier"),
         build("rtx5090", "mxfp8-fa4", "MXFP8 + FlashAttention", 4),
         build("rtx5090", "gpu-output-six-v2", "fast: MXFP8 + Sage2 + GPU output", 4),
         build("rtx5090", "gpu-output-balanced-v2", "fast + balanced", 4, passes="4 merged"),
         build("rtx5090", "gpu-output-extreme-v2", "fast + extreme", 4, passes="3 merged")],
        "Exact/lossless quality differ as labeled. First 3 rows: 1 trial/case; final builds: 3. Pixel-quality scores unavailable.",
    ),
    "single-tradeoffs": (
        "One L20: upstream lossless quality and approximate experiments",
        [build("single", "single-final-baseline", "Native BF16 reference", 1, role="reference"),
         build("single", "single-lossless-optimized", "single: BF16 residency", 1, role="native"),
         build("single", "single-int8-convrot", "Full DiT ConvRot INT8", 1),
         build("single", "single-int8-protected", "Selective DiT INT8 / protected blocks", 1),
         build("single", "single-int8-mlp12", "INT8 only in last 12 MLPs", 1),
         build("single", "single-fp8-fa", "DiT FP8 + FlashAttention", 1),
         build("single", "single-fp8-sage6", "DiT FP8 + Sage2", 1),
         build("single", "single-bf16-sage-fp16", "BF16 + Sage2 FP16 PV", 1),
         build("single", "single-bf16-compiled", "BF16 regional compile / warmed", 1),
         build("single", "single-bf16-five-merge", "BF16 / five merged passes", 1, passes="5 merged"),
         build("single", "single-bf16-tea015", "BF16 / TeaCache 0.15", 1, passes="6 + reuse"),
         build("single", "single-bf16-early-reuse", "BF16 / early residual reuse", 1, passes="6 + reuse")],
        "Reference/BF16 single: 3 trials/case; experiments: 1, compile: 1-2 warmed. Missing bars mean not measured.",
    ),
}


def read_csv(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load():
    return {source: {(row["label"], row["case"]): row for row in read_csv(path)}
            for source, path in SOURCES.items()}


def timing(data, spec, case):
    row = data[spec["source"]].get((spec["run"], case))
    if row is None:
        return None
    row = dict(row)
    # This raw summary includes one cold compilation call; keep it in the
    # catalog, but use the manifest's explicit exclusion for warmed charts.
    if spec["run"] == "single-bf16-compiled":
        manifest = json.loads((ROOT / "configs/single-l20-comparisons.json").read_text())
        variant = next(v for v in manifest["variants"] if v["run"] == spec["run"])
        excluded = variant.get("exclude_timing_trials", {}).get(case, [])
        records = ROOT / "provenance/single-l20/records" / spec["run"] / "results.jsonl"
        values = [r["wall_seconds"] for line in records.read_text().splitlines()
                  if (r := json.loads(line))["case"] == case and r["trial"] not in excluded]
        assert values
        row.update(median_seconds=statistics.median(values), trials=len(values),
                   min_seconds=min(values), max_seconds=max(values))
    return row


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latency_chart(plt, data, key, title, builds, caption):
    cases = CASES if key == "single-exact" else ("product", "native-4k")
    if len(cases) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), facecolor="white")
        fig.subplots_adjust(left=.27, right=.96, top=.85, bottom=.18, hspace=.65, wspace=1.15)
        axes = list(axes.flat)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5.7, 2.5 + .49 * len(builds))),
                                 facecolor="white", sharey=True)
        fig.subplots_adjust(left=.30, right=.97, top=.82, bottom=.23, wspace=.14)
    for ax, case in zip(axes, cases):
        maximum = max(float(timing(data, b, case)["median_seconds"])
                      for b in builds if timing(data, b, case))
        for y, spec in enumerate(builds):
            row = timing(data, spec, case)
            if row is None:
                ax.text(maximum * .025, y, "not measured", va="center", color="#697586", fontsize=9)
                continue
            value = float(row["median_seconds"])
            ax.barh(y, value, height=.57, color=COLORS[spec["role"]])
            ax.text(value + maximum * .025, y, f"{value:.2f} s  (n={row['trials']})",
                    va="center", fontsize=8.5, color="#223046")
        labels = [f"{b['name']}\n{b['gpus']} GPU{'s' if b['gpus'] > 1 else ''} / {b['passes']} passes / {b['quality']}"
                  for b in builds]
        ax.set_yticks(range(len(builds)), labels, fontsize=8.7)
        ax.set_ylim(len(builds) - .5, -.5)
        ax.set_xlim(0, maximum * 1.40)
        ax.set_title(TITLES[CASES.index(case)], loc="left", fontsize=11, weight="bold", pad=10)
        ax.set_xlabel("Median HTTP seconds · lower is better", fontsize=9, color="#526176")
        ax.xaxis.grid(True, color="#e7ebf0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", labelsize=8, colors="#526176")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, x=.035, y=.965, ha="left", fontsize=16, weight="bold", color="#172b45")
    fig.text(.035, .085, caption, fontsize=9, color="#526176")
    fig.text(.035, .045, "Pinned official checkpoint in SGLang; unmodified official pipeline not benchmarked. Raw values: plotted-latency.csv.",
             fontsize=8.5, color="#526176")
    save(fig, key)
    plt.close(fig)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=170, facecolor="white")
    fig.savefig(OUTPUT / f"{name}.svg", facecolor="white", metadata={"Date": None})


def throughput_chart(plt):
    summary = json.loads((ROOT / "provenance/single-l20/summary.json").read_text())["throughput"]
    multi = {r["label"]: r for r in read_csv("provenance/l20/throughput.csv")}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), facecolor="white")
    fig.subplots_adjust(left=.07, right=.97, top=.76, bottom=.25, wspace=.40)
    rows = []
    for index, (ax, title) in enumerate(zip(axes, ("One L20 · 2K", "Eight L20s · 2K", "Eight L20s · 4K"))):
        groups = ("exact", "lossless") if index == 0 else ("six", "four", "three")
        maximum = 0
        for y, group in enumerate(groups):
            for side, offset, color in (("reference", -.19, COLORS["reference"]),
                                        ("optimized", .19, COLORS["native"])):
                if index == 0:
                    value = summary[group][side]
                    request_count, clients, gpus = 8, 2, 1
                else:
                    label = f"l20{'q' if side == 'optimized' else ''}-dp4-{group}-{'2k' if index == 1 else '4k'}"
                    record = multi[label]
                    value = float(record["images_per_minute"])
                    request_count, clients, gpus = int(record["images"]), int(record["clients"]), 8
                maximum = max(maximum, value)
                ax.barh(y + offset, value, height=.33, color=color)
                ax.text(value, y + offset, f"  {value:.2f}", va="center", fontsize=9, color="#223046")
                rows.append(dict(panel=title, setting=group, build=side, gpus=gpus,
                                 images=request_count, clients=clients, images_per_minute=value))
        ax.set_yticks(range(len(groups)), groups, fontsize=10)
        ax.set_ylim(len(groups) - .5, -.5)
        ax.set_xlim(0, maximum * 1.28)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        ax.set_xlabel("Images/minute · higher is better", fontsize=9, color="#526176")
        ax.xaxis.grid(True, color="#e7ebf0", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Measured throughput: compare bars within each panel", x=.035, y=.96,
                 ha="left", fontsize=16, weight="bold", color="#172b45")
    fig.text(.035, .865, "1 GPU: native BF16 reference vs residency/graphs.   8 GPUs: earlier FP8/NCCL vs quantized exchange, four 2-GPU replicas.",
             fontsize=10, color="#526176")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS["reference"], label="Panel reference"),
                        Patch(color=COLORS["native"], label="Panel optimized build")],
               loc="lower center", bbox_to_anchor=(.5, .10), ncol=2, frameon=False, fontsize=9)
    fig.text(.035, .045, "1 GPU: 8 images / 2 clients / 6 passes. 8 GPUs: 4 clients; six/four/three passes; 12-24 images. Warmups excluded.",
             fontsize=9, color="#526176")
    save(fig, "throughput")
    plt.close(fig)
    csv_file(OUTPUT / "plotted-throughput.csv", rows)


def tables(data):
    path = ROOT / "comparisons/README.md"
    text = path.read_text()
    plotted = []
    for key, (_, builds, _) in GROUPS.items():
        lines = ["| Build | GPUs | Quality | Passes | 2K product | 2K portrait | 2K edit | Native 4K | Trials/case |",
                 "|---|---:|---|---|---:|---:|---:|---:|---|"]
        for spec in builds:
            cells, counts = [], set()
            for case in CASES:
                row = timing(data, spec, case)
                cells.append(f"{float(row['median_seconds']):.2f}" if row else "—")
                if row:
                    counts.add(int(row["trials"]))
                    plotted.append(dict(group=key, build=spec["name"], run=spec["run"],
                                        gpus=spec["gpus"], quality=spec["quality"], passes=spec["passes"],
                                        case=case, trials=row["trials"], median_seconds=row["median_seconds"],
                                        min_seconds=row["min_seconds"], max_seconds=row["max_seconds"],
                                        source_csv=SOURCES[spec["source"]]))
            sample = str(min(counts)) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
            lines.append(f"| {spec['name']} | {spec['gpus']} | {spec['quality']} | {spec['passes']} | "
                         + " | ".join(cells) + f" | {sample} |")
        start, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
        assert text.count(start) == text.count(end) == 1, key
        before, rest = text.split(start)
        _, after = rest.split(end)
        text = before + start + "\n\n" + "\n".join(lines) + "\n\n" + end + after
    path.write_text(text)
    csv_file(OUTPUT / "plotted-latency.csv", plotted)


def run_metadata(source, label, row):
    if source == "single":
        gpus = 1
        paths = [ROOT / "provenance/single-l20/records" / label / "results.jsonl"]
    elif source == "accurate":
        gpus = 4
        paths = [ROOT / "provenance/accuracy/runs" / label / "results.jsonl"]
    elif source == "rtx5090":
        gpus = 4
        paths = [ROOT / "provenance/results" / folder / label / "results.jsonl"
                 for folder in ("runs", "portable")]
    else:
        match = re.search(r"tp(\d+)(?:sp(\d+))?", label)
        if match:
            gpus = int(match[1]) * int(match[2] or 1)
        elif label.startswith("l20q-sp"):
            gpus = int(label.split("-sp")[1].split("-")[0])
        elif label == "l20-dp4-smoke":
            gpus = 8
        elif label == "l20-bf16":
            gpus = 4
        else:
            raise ValueError(f"GPU count not declared for {label}")
        paths = []
    quality = "lossless" if source in ("l20", "accurate") else "not recorded"
    for path in paths:
        if path.is_file():
            quality = json.loads(path.read_text().splitlines()[0])["quality"]
            break
    opt = json.loads(row["h3_opt"]) or {}
    passes = len(opt["steps"]) if opt.get("steps") else 6
    suffix = " merged" if opt.get("merge_heads") else ""
    if opt.get("tea_threshold") or opt.get("reuse_steps"):
        suffix += " + reuse"
    return dict(gpus=gpus, quality=quality, passes=f"{passes}{suffix}")


def catalog(data):
    records = []
    run_count = sum(len({label for label, _ in lookup}) for lookup in data.values())
    lines = ["# Every recorded build/run", "",
             "[Comparison guide](../README.md) · [Machine-readable index](build-catalog.csv)", "",
             f"This index preserves all {run_count} run IDs from the four source summaries, including reference, seed-validation and intermediate runs. "
             "Values are recorded median HTTP seconds; a dash means unmeasured. Trial counts vary by case. "
             "The raw single-GPU compiled product median includes its cold call; use the warmed value in the guide for that comparison. "
             "GPU counts, quality levels and references are defined in the guide; these raw rows are not one hardware-normalized leaderboard.", ""]
    lines += ["Rejected and preliminary experiments are included. A row here does not assert reference-pixel equality or mark a build as recommended; "
              "use the guide's named profiles and quality evidence for those conclusions.", ""]
    names = {"single": "Single-L20 series", "accurate": "Four-L20 accuracy series",
             "l20": "Earlier multi-L20 series", "rtx5090": "Four-RTX-5090 series"}
    for source, lookup in data.items():
        lines += [f"## {names[source]}", "",
                  f"[Source CSV](../../{SOURCES[source]})", "",
                  "| Run ID | GPUs | Quality | Passes | 2K product | 2K portrait | 2K edit | Native 4K | Other cases | Trials/case |",
                  "|---|---:|---|---|---:|---:|---:|---:|---|---|"]
        for label in sorted({label for label, _ in lookup}):
            cells = []
            run = [row for (name, _), row in lookup.items() if name == label]
            counts = {int(row["trials"]) for row in run}
            meta = run_metadata(source, label, run[0])
            for row in run:
                records.append(dict(series=source, source_csv=SOURCES[source], **meta, **row))
            for case in CASES:
                row = lookup.get((label, case))
                cells.append(f"{float(row['median_seconds']):.2f}" if row else "—")
            sample = str(min(counts)) if len(counts) == 1 else f"{min(counts)}–{max(counts)}"
            other = "; ".join(f"{row['case']} ({row['size']}): {float(row['median_seconds']):.2f} s"
                              for row in run if row["case"] not in CASES) or "—"
            lines.append(f"| `{label}` | {meta['gpus']} | {meta['quality']} | {meta['passes']} | "
                         + " | ".join(cells) + f" | {other} | {sample} |")
        lines.append("")
    (OUTPUT / "build-catalog.md").write_text("\n".join(lines))
    assert len(records) == sum(map(len, data.values()))
    csv_file(OUTPUT / "build-catalog.csv", records)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                                "svg.hashsalt": "h3-benchmark-overview", "axes.unicode_minus": False})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load()
    for key, (title, builds, caption) in GROUPS.items():
        latency_chart(plt, data, key, title, builds, caption)
    throughput_chart(plt)
    tables(data)
    catalog(data)
    print("Exported 6 performance charts (PNG/SVG), guide tables and the complete run catalog.")


if __name__ == "__main__":
    main()
