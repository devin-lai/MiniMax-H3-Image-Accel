"""Render the published speedup and fidelity charts from the bundled CSVs."""

import argparse
import csv
from collections import defaultdict
from math import isinf
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch


DATA = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
CASES = (
    ("product", "2K product", 731),
    ("portrait", "2K portrait", 1234),
    ("edit", "2K edit", 42),
    ("native-4k", "Native 4K", 802),
)
# Each pair uses six passes and the same GPU count and request preset.
PAIRS = (
    ("single-exact", "single-exact-reference", "single-exact-optimized",
     "1 × L20 · BF16 residency + replay", "vs streamed BF16 · exact", True),
    ("l20-accurate", "accuracy-reference-final", "accuracy-auto-final",
     "4 × L20 · BF16 verified transport", "vs BF16/NCCL on the same placement · lossless", True),
    ("single-rtx5090", "single-bf16-stream", "single-bf16-tuned",
     "1 × RTX 5090 · BF16 residency", "vs streamed BF16 · lossless", True),
    ("l20-fast", "l20-bf16", "l20q-sp4",
     "4 × L20 · FP8 + quantized exchange", "vs earlier BF16/TP4 · lossless", False),
    ("single-rtx5090", "single-bf16-stream", "single-mxfp8-sage",
     "1 × RTX 5090 · MXFP8 + SageAttention2", "vs streamed BF16 · lossless", False),
    ("rtx5090", "bf16-tuned", "gpu-output-six-v2",
     "4 × RTX 5090 · MXFP8 + SageAttention2", "vs tuned BF16 · lossless", False),
)
STACKS = (
    ("fidelity-single-l20.png", "One L20: fidelity by optimization", (
        ("l20-single-exact", "Exact preset · vs streamed BF16"),
        ("l20-single-lossless", "Upstream lossless preset · vs streamed BF16"),
        ("l20-quality-preset", "Preset change · vs streamed BF16 with exact"),
        ("l20-tp1-vs-tp4", "GPU partitioning · vs BF16 TP4 with lossless"),
    )),
    ("fidelity-multi-l20.png", "Multiple L20s: fidelity by optimization", (
        ("l20-bf16-transport", "BF16 · vs NCCL on the same spread PCIe placement"),
        ("l20-bf16-placement", "BF16 · vs NCCL on the original PCIe placement"),
        ("l20-bf16-layout", "BF16 · vs TP4 in the layout screen"),
        ("l20-multi-fp8", "FP8 + SageAttention2 · vs earlier six-pass BF16 TP4"),
    )),
    ("fidelity-single-rtx5090.png", "One RTX 5090: fidelity by optimization", (
        ("rtx5090-single", "Main series · vs streamed BF16 on the primary card"),
        ("rtx5090-follow-up", "Follow-up · vs streamed BF16 on the follow-up card"),
        ("rtx5090-compiled-fp8", "Compile screen · vs eager MXFP8 DiT on the follow-up card"),
    )),
)
WIDTH = 12
LEFT = 0.38
GRID = 4.66
COL = (WIDTH - GRID - LEFT) / len(CASES)
INK = "#172d40"
MUTED = "#4f6476"
TEAL = "#247d6c"
ORANGE = "#ba592e"


def read_csv(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def label(ax, x, y, content, size=11, color=INK, bold=False, align="left"):
    ax.text(x, y, content, fontsize=size, color=color, ha=align, va="center",
            weight="bold" if bold else "normal")


def box(ax, x, y, width, height, color):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0,rounding_size=0.035",
                                facecolor=color, linewidth=0))


def canvas(height, title, subtitle):
    fig = plt.figure(figsize=(WIDTH, height), facecolor="#fafcfe")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, WIDTH), ylim=(height, 0))
    ax.set_axis_off()
    label(ax, LEFT, 0.39, title, 22, bold=True)
    label(ax, LEFT, 0.78, subtitle, 11, MUTED)
    for j, (_, name, _) in enumerate(CASES):
        label(ax, GRID + (j + 0.5) * COL, 1.26, name, 11, bold=True, align="center")
    ax.hlines(1.45, LEFT, WIDTH - LEFT, color="#d6e1ea", linewidth=0.8)
    return fig, ax


def save(fig, output):
    fig.savefig(output, dpi=150)
    plt.close(fig)


def speedup_chart(output):
    rows = read_csv("latency.csv") + [
        dict(row, group="single-rtx5090", run=row["label"])
        for row in read_csv("single-rtx5090-latency.csv")
    ]
    seconds = {(row["group"], row["run"], row["case"]): float(row["median_seconds"])
               for row in rows}
    height = 1.88 + 0.55 * len(PAIRS)
    fig, ax = canvas(height, "Six-pass speedup vs. BF16",
                     "Warmed HTTP latency · same GPU count and request preset within each pair")
    for i, (group, reference, candidate, name, detail, exact) in enumerate(PAIRS):
        y = 1.55 + 0.55 * i
        label(ax, LEFT, y + 0.13, name, bold=True)
        label(ax, LEFT, y + 0.34, detail, 9.3, MUTED)
        color = TEAL if exact else ORANGE
        for j, (case, _, _) in enumerate(CASES):
            ratio = seconds[group, reference, case] / seconds[group, candidate, case]
            x = GRID + j * COL + 0.13
            width = COL - 0.26
            label(ax, x + width / 2, y + 0.12, f"{ratio:.2f}×", 12, color, True, "center")
            box(ax, x, y + 0.26, width, 0.09, "#e9eff4")
            box(ax, x, y + 0.26, width * ratio / 3.5, 0.09, color)
            ax.vlines(x + width / 3.5, y + 0.24, y + 0.37, color=INK, linewidth=0.8)
    label(ax, LEFT, height - 0.16, "■  Audited PNGs matched", 9.5, TEAL, True)
    label(ax, 2.48, height - 0.16, "■  Approximate build", 9.5, ORANGE, True)
    label(ax, GRID, height - 0.16, "Tick = 1× (reference)  ·  Higher is faster", 9.5, MUTED)
    save(fig, output)


def fidelity_cell(ax, x, y, row):
    if row is None:
        fill, first, second = "#edf1f5", "—", ""
    else:
        ssim = float(row["ssim"])
        psnr = float(row["psnr_db"])
        if ssim == 1 and isinf(psnr):
            fill, first, second = "#d5eee5", "Identical", "1.0000  /  ∞ dB"
        else:
            # Keep the scale fixed across panels so color is comparable.
            strength = min(1, max(0, (1 - ssim) / 0.42)) ** 0.65
            fill = tuple(a + (b - a) * strength
                         for a, b in zip(to_rgb("#fff2dc"), to_rgb("#efa19a")))
            first, second = f"{ssim:.4f}", f"{psnr:.1f} dB"
    box(ax, x + 0.04, y, COL - 0.08, 0.42, fill)
    label(ax, x + COL / 2, y + (0.13 if second else 0.21), first, 11, bold=True, align="center")
    if second:
        label(ax, x + COL / 2, y + 0.30, second, 9, MUTED, align="center")


def stack_chart(output, title, subtitle, sections, draw_cell, legend):
    height = 1.91 + sum(0.43 + 0.49 * len(variants) for _, variants in sections)
    fig, ax = canvas(height, title, subtitle)
    y = 1.48
    for heading, variants in sections:
        label(ax, LEFT, y + 0.17, heading, 11, TEAL, bold=True)
        y += 0.35
        for name, cases in variants.items():
            label(ax, LEFT, y + 0.21, name, 10.6)
            for j, (case, _, _) in enumerate(CASES):
                draw_cell(ax, GRID + j * COL, y, cases.get(case))
            y += 0.49
        y += 0.08
    label(ax, LEFT, height - 0.17, legend, 9.5, MUTED)
    save(fig, output)


def fidelity_charts(output_dir):
    main_seeds = {case: seed for case, _, seed in CASES}
    panels = defaultdict(dict)
    for row in read_csv("fidelity.csv"):
        case = row["case"]
        if int(row["seed"]) == main_seeds[case]:
            panels[row["panel"]].setdefault(row["optimization"], {})[case] = row

    for filename, title, sections in STACKS:
        stack_chart(output_dir / filename, title,
                    "Full resolution, same prompt and seed · luma SSIM above RGB PSNR",
                    [(heading, panels[key]) for key, heading in sections], fidelity_cell,
                    "Green = identical pixels   ·   Darker orange = lower SSIM   ·   — = not measured")


def hash_cell(ax, x, y, row):
    status = row["png_bytes"] if row else None
    caption, fill = {None: ("—", "#edf1f5"), "same": ("Same bytes", "#d5eee5"),
                     "different": ("Different", "#f8dfa7")}[status]
    box(ax, x + 0.04, y, COL - 0.08, 0.42, fill)
    label(ax, x + COL / 2, y + 0.21, caption, 11, bold=True, align="center")


def hash_chart(output):
    panels = defaultdict(dict)
    for row in read_csv("quad-rtx5090-png-hashes.csv"):
        ref = row["reference"]
        panel = "cpu-output" if ref.startswith("runs/final-") else ref
        panels[panel].setdefault(row["scheme"].removesuffix(" *"), {})[row["case"]] = row
    stack_chart(output, "Four RTX 5090s: recorded PNG matches",
                "Matched first-trial requests · hashes show byte equality only",
                [
                    ("Headline fast build · vs tuned six-pass BF16", panels["runs/bf16-tuned"]),
                    ("Earlier experiments · vs six-pass MXFP8 + SageAttention2, TP2 + U2",
                     panels["runs/r2-warmed-base"]),
                    ("Prefix KV reuse · vs full encoder prefill", panels["runs/r2-prefix-full"]),
                    ("GPU output packing · vs CPU packing with the same schedule", panels["cpu-output"]),
                ], hash_cell,
                "Green = same PNG bytes   ·   Gold = different PNG bytes   ·   — = not recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA, help="directory for the PNG charts")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    speedup_chart(output / "speedup.png")
    fidelity_charts(output)
    hash_chart(output / "quad-rtx5090-png-hashes.png")


if __name__ == "__main__":
    main()
