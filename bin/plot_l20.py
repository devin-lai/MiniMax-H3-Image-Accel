"""Draw the L20 charts from the CSVs in provenance/l20.

  python bin/plot_l20.py                      # needs matplotlib
  python bin/plot_l20.py --data provenance/l20 --output provenance/l20
"""
import argparse
import csv
from pathlib import Path

INK, MUTED, GRID, SURFACE = "#1f2328", "#59636e", "#e4e6ea", "#fcfcfb"
GRAY, BLUE, ORANGE, AQUA = "#9aa1ab", "#2a78d6", "#eb6834", "#1baf7a"


def read(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for label in ax.get_yticklabels():
        label.set_color(INK)


def legend(fig, plt, entries):
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in entries]
    fig.legend(handles, [name for name, _ in entries], loc="lower center", ncol=len(entries),
               frameon=False, fontsize=9, labelcolor=INK, handlelength=1, handleheight=1)


def latency(plt, data, output):
    median = {(r["label"], r["case"]): float(r["median_seconds"]) for r in read(data / "benchmarks.csv")}
    rows = [
        ("BF16, FlashAttention, TP4", "l20-bf16", GRAY),
        ("FP8, Ulysses4, NCCL exchange", "fp8-sage-tp1sp4", BLUE),
        ("FP8, Ulysses4, quantized exchange", "l20q-sp4", ORANGE),
        ("FP8, Ulysses8, quantized exchange", "l20q-sp8", ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=SURFACE, sharey=True)
    for ax, (case, title) in zip(axes, (("product", "2048 × 2048"), ("native-4k", "4096 × 4096"))):
        values = [median[(label, case)] for _, label, _ in rows]
        bars = ax.barh(range(len(rows)), values, height=0.5, color=[c for *_, c in rows])
        ax.set_yticks(range(len(rows)), [name for name, *_ in rows])
        if ax is axes[0]:
            ax.invert_yaxis()  # the y axis is shared
        for bar, value in zip(bars, values):
            note = f"{value:.2f} s" if value == values[0] else f"{value:.2f} s  ({values[0] / value:.1f}×)"
            ax.text(value, bar.get_y() + bar.get_height() / 2, "  " + note, va="center", fontsize=9, color=INK)
        ax.set_xlim(0, values[0] * 1.28)
        style(ax, title, "median request seconds, six passes")
    fig.suptitle("Six-pass request latency on NVIDIA L20", x=0.01, ha="left", fontsize=13, color=INK)
    legend(fig, plt, [("BF16 reference", GRAY), ("previous FP8 runtime", BLUE), ("this work", ORANGE)])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(output / "latency.png", dpi=160)


def throughput(plt, data, output):
    rate = {r["label"]: float(r["images_per_minute"]) for r in read(data / "throughput.csv")}
    passes = (("six", "Six passes"), ("four", "Four merged passes"), ("three", "Three merged passes"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE, sharey=True)
    for ax, (suffix, title) in zip(axes, (("2k", "2048 × 2048"), ("4k", "4096 × 4096"))):
        top = 0
        for index, (key, _) in enumerate(passes):
            old, new = rate[f"l20-dp4-{key}-{suffix}"], rate[f"l20q-dp4-{key}-{suffix}"]
            top = max(top, new)
            ax.barh(index - 0.19, old, height=0.34, color=BLUE)
            ax.barh(index + 0.19, new, height=0.34, color=ORANGE)
            ax.text(old, index - 0.19, f"  {old:.1f}", va="center", fontsize=9, color=INK)
            ax.text(new, index + 0.19, f"  {new:.1f}  (+{(new / old - 1) * 100:.0f}%)", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(passes)), [name for _, name in passes])
        if ax is axes[0]:
            ax.invert_yaxis()  # the y axis is shared
        ax.set_xlim(0, top * 1.25)
        style(ax, title, "images per minute, eight GPUs")
    fig.suptitle("Node throughput, four two-GPU replicas", x=0.01, ha="left", fontsize=13, color=INK)
    legend(fig, plt, [("NCCL exchange", BLUE), ("quantized host exchange", ORANGE)])
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(output / "throughput.png", dpi=160)


def steps(plt, data, output):
    rows = read(data / "step-profile.csv")
    parts = (("gemm_ms", "FP8 linear layers", BLUE), ("attention_kernel_ms", "attention kernel", AQUA),
             ("exchange_ms", "Ulysses exchange", ORANGE), ("rest", "quantize for exchange and other", GRAY))
    names = {"fp8-sage-tp1sp4": "Ulysses4, NCCL", "l20q-sp4": "Ulysses4, quantized", "l20q-sp8": "Ulysses8, quantized"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=SURFACE, sharey=True)
    for ax, size in zip(axes, ("2048x2048", "4096x4096")):
        group = [r for r in rows if r["size"] == size]
        for index, row in enumerate(group):
            row["rest"] = float(row["exchange_quantization_ms"]) + float(row["other_ms"])
            left = 0.0
            for key, _, color in parts:
                width = float(row[key])
                ax.barh(index, width, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=1.5)
                if width > 0.09 * float(group[0]["step_ms"]):
                    ax.text(left + width / 2, index, f"{width:.0f}", ha="center", va="center", fontsize=8.5, color="white")
                left += width
            ax.text(left, index, f"  {float(row['step_ms']):.0f} ms", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(group)), [names[r["profile"].split(" ")[0]] for r in group])
        if ax is axes[0]:
            ax.invert_yaxis()  # the y axis is shared
        ax.set_xlim(0, float(group[0]["step_ms"]) * 1.15)
        style(ax, size.replace("x", " × "), "milliseconds per denoising pass (profiled)")
    fig.suptitle("Where one denoising pass spends its time", x=0.01, ha="left", fontsize=13, color=INK)
    legend(fig, plt, [(name, color) for _, name, color in parts])
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(output / "step-breakdown.png", dpi=160)


def main():
    root = Path(__file__).resolve().parents[1] / "provenance" / "l20"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for chart in (latency, throughput, steps):
        chart(plt, args.data, args.output)
    print(f"Wrote latency.png, throughput.png and step-breakdown.png to {args.output}")


if __name__ == "__main__":
    main()
