"""Audit and plot the recorded single-card RTX 5090 experiments.

Uses the same strict request/hash checks and full-resolution RGB PSNR / 8x8
luma SSIM as validate_accuracy.py. Missing main cases fail; extra seed runs
are compared with the matching reference seed, never with the first seed.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

from validate_accuracy import audit, pixels, records

CASES = ("product", "portrait", "edit", "native-4k")
VARIANTS = {
    "single-bf16-stream": "BF16 streaming reference",
    "single-bf16-tuned": "BF16 resident subset",
    "single-int8-mlp": "INT8 MLP / BF16 attention + encoder",
    "single-bf16-compiled": "BF16 / compiled, eager precision",
    "single-mxfp8-dit": "MXFP8 DiT / BF16 encoder / FA",
    "single-mxfp8-sage": "MXFP8 DiT + encoder / Sage2",
    "single-mxfp8-sage-four": "MXFP8 + Sage2 / four passes",
    "single-mxfp8-sage-tea": "MXFP8 + Sage2 / Tea 0.15",
    "single-mxfp8-sage-three": "MXFP8 + Sage2 / three passes",
    "single-mxfp8-compiled": "MXFP8 + Sage2 / compiled",
    "single-bf16-cache": "BF16 resident subset / conditioning hits",
}
TITLES = ("2K product", "2K portrait", "2K reference edit", "Native 4K")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    latency, quality, memory, throughput = [], [], [], []
    reference = "single-bf16-stream"
    available = [name for name in VARIANTS if (runs / name / "results.jsonl").exists()]
    if reference not in available:
        raise ValueError("Missing single-card BF16 reference")
    for name in available:
        directory = runs / name
        items = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        first = records(directory)
        for case in CASES:
            pixels(directory, first[case])
            group = [row for row in items if row["case"] == case]
            trials = [row["trial"] for row in group]
            if len(trials) != len(set(trials)) or sorted(trials) != list(range(1, len(trials) + 1)):
                raise ValueError(f"Duplicate or missing trials: {name}/{case}")
            seconds = [row["wall_seconds"] for row in group]
            latency.append(dict(label=name, case=case, trials=len(group),
                                median_seconds=statistics.median(seconds),
                                min_seconds=min(seconds), max_seconds=max(seconds),
                                repeated_png_identical=len({row["sha256"] for row in group}) == 1))
        if name != reference:
            quality.extend(audit(runs / reference, directory, CASES))
        for suffix in ("-seed-1", "-seed-17"):
            if not (runs / (name + suffix) / "results.jsonl").exists():
                continue
            if name != reference:
                quality.extend(audit(runs / (reference + suffix), runs / (name + suffix), CASES))
        perf = [json.loads(path.read_text()) for path in directory.glob("*.perf.json")]
        if perf:
            checkpoints = [p["memory_checkpoints"]["after_forward"] for p in perf]
            memory.append(dict(label=name,
                               peak_torch_allocated_gib=max(p["peak_allocated_mb"] for p in checkpoints) / 1024,
                               peak_torch_reserved_gib=max(p["peak_reserved_mb"] for p in checkpoints) / 1024,
                               peak_host_anon_gib=max(p["peak_host_anon_mb"] for p in checkpoints) / 1024))
        throughput_path = runs / "throughput" / f"{name}.json"
        if throughput_path.exists():
            data = json.loads(throughput_path.read_text())
            throughput.append({key: data[key] for key in ("label", "case", "clients", "images", "wall_seconds",
                                                          "images_per_minute", "median_request_seconds", "p95_request_seconds")})
    write_csv(output / "benchmarks-main.csv", latency)
    write_csv(output / "quality.csv", quality)
    write_csv(output / "memory.csv", memory)
    write_csv(output / "throughput.csv", throughput)

    screened, parity = [], []
    for directory in sorted(runs.glob("single-*-screen")):
        if not (directory / "results.jsonl").exists():
            continue
        first = records(directory)
        cases = [case for case in CASES if case in first]
        items = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        for row in audit(runs / reference, directory, cases):
            seconds = [item["wall_seconds"] for item in items if item["case"] == row["case"]]
            screened.append({**row, "trials": len(seconds), "median_seconds": statistics.median(seconds),
                             "min_seconds": min(seconds), "max_seconds": max(seconds)})
        parity_reference = {
            "single-mxfp8-compiled-screen": "single-mxfp8-sage",
            "single-mxfp8-dit-compiled-screen": "single-mxfp8-dit-screen",
        }.get(directory.name)
        if parity_reference and (runs / parity_reference / "results.jsonl").exists():
            parity.extend(audit(runs / parity_reference, directory, cases, identical=True))
    write_csv(output / "screening.csv", screened)
    write_csv(output / "compiled-parity.csv", parity)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#607787", "#247a63", "#6a9361", "#5b9c91", "#4286b4", "#d69b35", "#bf7142", "#9c7cb5", "#b65a60", "#5466b7", "#247a63"]
    color = dict(zip(VARIANTS, colors))
    fig, axes = plt.subplots(2, 2, figsize=(17, max(8, len(available) * 1.05)), constrained_layout=True)
    for case, title, ax in zip(CASES, TITLES, axes.flat):
        rows = [next(row for row in latency if row["label"] == name and row["case"] == case) for name in available]
        x = [row["median_seconds"] for row in rows]
        error = [[v - row["min_seconds"] for v, row in zip(x, rows)],
                 [row["max_seconds"] - v for v, row in zip(x, rows)]]
        bars = ax.barh(np.arange(len(rows)), x, xerr=error, color=[color[name] for name in available], capsize=3)
        ax.set_yticks(np.arange(len(rows)), [VARIANTS[name] for name in available], fontsize=9)
        ax.invert_yaxis()
        limit = max(row["max_seconds"] for row in rows)
        ax.set_xlim(0, limit * 1.19)
        ax.set_xlabel("Seconds per complete HTTP response; lower is faster")
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        for bar, row, value in zip(bars, rows, x):
            ax.text(row["max_seconds"] + limit * .02, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va="center", fontsize=9)
    cache_policy = (
        "Cache disabled except the explicitly labeled conditioning-hit row"
        if "single-bf16-cache" in available else "Conditioning cache disabled"
    )
    fig.suptitle("One RTX 5090 • matched prompts and seeds • warmed, serial, lossless PNG\n"
                 f"Medians with measured min–max ranges; {cache_policy}", fontsize=15, weight="bold")
    fig.savefig(output / "latency.png", dpi=160)
    plt.close(fig)

    candidates = [name for name in available if name != reference]
    if candidates:
        grid = np.array([[next(row["ssim"] for row in quality if row["candidate"] == name and row["case"] == case)
                          for case in CASES] for name in candidates])
        fig, ax = plt.subplots(figsize=(11, max(3.4, len(candidates) * .7)), constrained_layout=True)
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), TITLES)
        ax.set_yticks(range(len(candidates)), [VARIANTS[name] for name in candidates])
        for row in range(len(candidates)):
            for col in range(4):
                same = next(q["pixel_identical"] for q in quality
                            if q["candidate"] == candidates[row] and q["case"] == CASES[col])
                ax.text(col, row, f"{grid[row, col]:.4f}" + ("\nidentical pixels" if same else ""),
                        ha="center", va="center", color="#132126" if grid[row, col] > .55 else "white", fontsize=10)
        ax.set_title("Full-resolution luma SSIM against the single-card BF16 reference\nSame prompt, seed, canvas and edit reference; each case shown separately", loc="left", pad=15)
        fig.colorbar(im, ax=ax, label="SSIM (higher is closer)", fraction=.035)
        fig.savefig(output / "quality.png", dpi=160)
        plt.close(fig)
    print(f"Exported {len(latency)} latency rows and {len(quality)} matched-image audits to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/single5090"))
    parser.add_argument("--output", type=Path, default=Path("provenance/single5090"))
    args = parser.parse_args()
    export(args.runs, args.output)
