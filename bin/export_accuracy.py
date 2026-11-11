"""Export matched L20 accuracy charts from the recorded benchmark PNGs.

The default command redraws the numeric chart from comparisons/l20/accuracy.csv.
Use --measure to recompute that CSV and --image-sheets to redraw the visual
comparisons. Both options need the original first-trial PNGs and results.jsonl
files under runs/l20/images (which are intentionally not tracked by Git).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "l20-bf16"
CASES = (
    ("product", "2K product", 731, (0.50, 0.55)),
    ("portrait", "2K portrait", 1234, (0.50, 0.46)),
    ("edit", "2K edit", 42, (0.50, 0.50)),
    ("native-4k", "4K still life", 802, (0.66, 0.62)),
)
# IDs come directly from runs/l20/images and provenance/l20/benchmarks.csv.
SCHEMES = (
    ("fp8-sage-tp2", "NCCL · TP2 · 6", "NCCL TP2\n2 GPUs · 6 passes"),
    ("fp8-sage-tp1sp2", "NCCL · U2 · 6", "NCCL Ulysses2\n2 GPUs · 6 passes"),
    ("fp8-sage-tp2sp2", "NCCL · TP2+U2 · 6", "NCCL TP2 + U2\n4 GPUs · 6 passes"),
    ("fp8-sage-tp1sp4", "NCCL · U4 · 6", "NCCL Ulysses4\n4 GPUs · 6 passes"),
    ("l20-dp4-smoke", "NCCL · 4 replicas · 6", "NCCL 4 replicas\n2 GPUs each · 6 passes"),
    ("l20q-sp4", "Quantized · U4 · 6", "Quantized Ulysses4\n4 GPUs · 6 passes"),
    ("l20q-sp8", "Quantized · U8 · 6", "Quantized Ulysses8\n8 GPUs · 6 passes"),
    ("fp8-sage-tp1sp4-four", "NCCL · U4 · 4 merged", "NCCL Ulysses4\n4 GPUs · 4 merged"),
    ("fp8-sage-tp1sp4-three", "NCCL · U4 · 3 merged", "NCCL Ulysses4\n4 GPUs · 3 merged"),
    ("fp8-sage-tp2sp2-four", "NCCL · TP2+U2 · 4 merged", "NCCL TP2 + U2\n4 GPUs · 4 merged"),
    ("fp8-sage-tp2sp2-three", "NCCL · TP2+U2 · 3 merged", "NCCL TP2 + U2\n4 GPUs · 3 merged"),
    ("fp8-sage-tp2sp2-tea", "NCCL · TP2+U2 · Tea 0.4", "NCCL TP2 + U2\n4 GPUs · Tea 0.4"),
    ("l20q-sp4-four", "Quantized · U4 · 4 merged", "Quantized Ulysses4\n4 GPUs · 4 merged"),
    ("l20q-sp4-three", "Quantized · U4 · 3 merged", "Quantized Ulysses4\n4 GPUs · 3 merged"),
    ("l20q-sp8-four", "Quantized · U8 · 4 merged", "Quantized Ulysses8\n8 GPUs · 4 merged"),
    ("l20q-sp8-three", "Quantized · U8 · 3 merged", "Quantized Ulysses8\n8 GPUs · 3 merged"),
)
LABELS = {key: (row, column) for key, row, column in SCHEMES}
LABELS[REFERENCE] = ("BF16 reference", "BF16 reference\nTP4 · 6 passes")
GROUPS = {
    "precision-and-layout": ("Six-pass precision and layouts", (
        REFERENCE, "fp8-sage-tp2", "fp8-sage-tp1sp2", "fp8-sage-tp2sp2",
        "fp8-sage-tp1sp4", "l20-dp4-smoke", "l20q-sp4", "l20q-sp8")),
    "nccl-pass-and-cache": ("NCCL: pass schedules and residual reuse", (
        REFERENCE, "fp8-sage-tp1sp4", "fp8-sage-tp1sp4-four",
        "fp8-sage-tp1sp4-three", "fp8-sage-tp2sp2", "fp8-sage-tp2sp2-four",
        "fp8-sage-tp2sp2-three", "fp8-sage-tp2sp2-tea")),
    "quantized-pass-schedules": ("Quantized exchange: pass schedules", (
        REFERENCE, "l20q-sp4", "l20q-sp4-four", "l20q-sp4-three",
        "l20q-sp8", "l20q-sp8-four", "l20q-sp8-three")),
}
MATCH_KEYS = ("case", "size", "seed", "task", "prompt", "reference_sha256", "quality", "output_compression")
Image.MAX_IMAGE_PIXELS = None


def matched(reference, candidate):
    for key in MATCH_KEYS:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Mismatched {key} for {candidate.get('label')}/{candidate.get('case')}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(runs):
    """Require matching records, image dimensions and original PNG hashes."""
    found = {}
    for label in (REFERENCE, *(item[0] for item in SCHEMES)):
        directory = runs / label
        record_file = directory / "results.jsonl"
        if not record_file.is_file():
            raise FileNotFoundError(record_file)
        records = {}
        for line in record_file.read_text().splitlines():
            record = json.loads(line)
            if record.get("trial") == 1:
                if record["case"] in records:
                    raise ValueError(f"Duplicate trial 1 in {record_file}")
                records[record["case"]] = record
        for case, _, _, _ in CASES:
            path = directory / f"{case}.png"
            record = records.get(case)
            if record is None and not path.exists():
                continue  # That case was not collected for this profile.
            if record is None or not path.is_file():
                raise ValueError(f"Missing image or trial 1 metadata: {label}/{case}")
            if record.get("label") != label or sha256(path) != record.get("sha256"):
                raise ValueError(f"PNG hash or label does not match: {label}/{case}")
            with Image.open(path) as image:
                expected_size = tuple(map(int, record["size"].split("x")))
                if image.size != expected_size:
                    raise ValueError(f"Image size does not match metadata: {label}/{case}")
            found[(label, case)] = (path, record)
    for case, _, _, _ in CASES:
        if (REFERENCE, case) not in found:
            raise ValueError(f"Missing reference for {case}")
        ref = found[(REFERENCE, case)][1]
        for label, _, _ in SCHEMES:
            if (label, case) in found:
                matched(ref, found[(label, case)][1])
    return found


def box(image, size=8):
    total = np.cumsum(np.cumsum(np.pad(image, ((1, 0), (1, 0))), axis=0), axis=1)
    return (total[size:, size:] - total[:-size, size:] - total[size:, :-size]
            + total[:-size, :-size]) / size**2


def compare_arrays(reference, candidate, tile_rows=256):
    """RGB PSNR/MAE and 8x8 box-window luma SSIM, evaluated in strips.

    Strips overlap by seven rows so every valid SSIM window is counted once.
    This matches bin/compare_quality.py while keeping 4K memory bounded.
    """
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Expected equally sized RGB images")
    height, width, _ = reference.shape
    if min(height, width) < 8 or tile_rows < 1:
        raise ValueError("Images must be at least 8x8 and tile_rows positive")
    abs_sum = squared_sum = 0.0
    for start in range(0, height, tile_rows):
        a = reference[start:start + tile_rows].astype(np.float64)
        b = candidate[start:start + tile_rows].astype(np.float64)
        diff = a - b
        abs_sum += float(np.abs(diff).sum())
        squared_sum += float(np.square(diff).sum())
    count = height * width * 3
    mse = squared_sum / count
    psnr = float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)
    luma = np.array([0.299, 0.587, 0.114])
    ssim_sum = 0.0
    windows = 0
    for start in range(0, height - 7, tile_rows):
        stop = min(start + tile_rows, height - 7)
        x = reference[start:stop + 7].astype(np.float64) @ luma
        y = candidate[start:stop + 7].astype(np.float64) @ luma
        mx, my = box(x), box(y)
        vx, vy = box(x * x) - mx * mx, box(y * y) - my * my
        covariance = box(x * y) - mx * my
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        scores = ((2 * mx * my + c1) * (2 * covariance + c2)
                  / ((mx * mx + my * my + c1) * (vx + vy + c2)))
        ssim_sum += float(scores.sum())
        windows += scores.size
    return psnr, ssim_sum / windows, abs_sum / count


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def measure(found, output):
    rows = []
    for case, _, _, _ in CASES:
        reference = read_image(found[(REFERENCE, case)][0])
        for label, _, _ in SCHEMES:
            if (label, case) not in found:
                continue
            path, record = found[(label, case)]
            psnr, ssim, mae = compare_arrays(reference, read_image(path))
            row = {"reference": REFERENCE, "candidate": label, "case": case,
                   "size": record["size"], "seed": record["seed"],
                   "psnr_db": round(psnr, 2), "ssim": round(ssim, 4),
                   "mean_abs_diff": round(mae, 3),
                   "reference_png_sha256": found[(REFERENCE, case)][1]["sha256"],
                   "candidate_png_sha256": record["sha256"]}
            rows.append(row)
            print(f"{case:10} {label:28} PSNR {psnr:.2f}  SSIM {ssim:.4f}", flush=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_metrics(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    known_labels = {item[0] for item in SCHEMES}
    known_cases = {item[0] for item in CASES}
    for row in rows:
        key = (row["candidate"], row["case"])
        if (row["reference"] != REFERENCE or key in seen or key[0] not in known_labels
                or key[1] not in known_cases):
            raise ValueError(f"Invalid or duplicate accuracy row: {key}")
        seen.add(key)
    return rows


def validate_metrics_for_images(rows, found):
    """Do not attach scores from a different set of PNGs to the image sheets."""
    expected = {(label, case) for label, _, _ in SCHEMES for case, *_ in CASES
                if (label, case) in found}
    actual = {(row["candidate"], row["case"]) for row in rows}
    if actual != expected:
        raise ValueError("Accuracy CSV coverage differs from the source images; rerun with --measure")
    for row in rows:
        label, case = row["candidate"], row["case"]
        if (row.get("reference_png_sha256") != found[(REFERENCE, case)][1]["sha256"]
                or row.get("candidate_png_sha256") != found[(label, case)][1]["sha256"]):
            raise ValueError(f"Accuracy CSV hashes do not match {label}/{case}; rerun with --measure")


def plot_metrics(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    values = {(r["candidate"], r["case"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), facecolor="#fbfcfe")
    fig.subplots_adjust(left=0.275, right=0.96, top=0.78, bottom=0.14, wspace=0.16)
    fig.text(.025, .955, "L20 accuracy against the six-pass BF16 reference", fontsize=20,
             weight="bold", color="#182433")
    fig.text(.025, .917, "Same prompt, seed, resolution and edit input · first-trial full-resolution PNGs",
             fontsize=11, color="#516170")
    fig.text(.025, .889, "Higher values mean closer pixels; they are not a perceptual-quality guarantee. Gray = not measured.",
             fontsize=10, color="#516170")
    cases = [case for case, *_ in CASES]
    for ax, (key, title, low, high, fmt) in zip(axes, (
            ("psnr_db", "RGB PSNR (dB)", 15, 34, ".2f"),
            ("ssim", "Luma SSIM", .55, 1, ".3f"))):
        grid = np.full((len(SCHEMES), len(CASES)), np.nan)
        for i, (label, _, _) in enumerate(SCHEMES):
            for j, case in enumerate(cases):
                if (label, case) in values:
                    grid[i, j] = float(values[(label, case)][key])
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#e8ebef")
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=Normalize(low, high), aspect="auto")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#182433", pad=12)
        ax.set_xticks(range(len(CASES)), [name for _, name, *_ in CASES], rotation=28, ha="left")
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#263646")
        ax.set_yticks(range(len(SCHEMES)), [name for _, name, _ in SCHEMES] if ax is axes[0] else [])
        for y in range(len(SCHEMES)):
            for x in range(len(CASES)):
                value = grid[y, x]
                text = "—" if np.isnan(value) else format(value, fmt)
                color = "#ffffff" if not np.isnan(value) and (value - low) / (high - low) > .54 else "#1c3042"
                ax.text(x, y, text, ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(np.arange(-.5, len(CASES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(SCHEMES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.035, pad=.035, aspect=30).ax.tick_params(labelsize=8)
    fig.text(.025, .055, "U = Ulysses degree; TP = tensor parallel degree. Four- and three-pass schedules merge heads.",
             fontsize=10, color="#516170")
    fig.text(.025, .03, "Each entry is one matched image, not a statistical estimate. See accuracy.csv for exact values.",
             fontsize=10, color="#516170")
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else
             ["DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_sheet(group, title, found, rows, output, detail=False):
    metrics = {(r["candidate"], r["case"]): r for r in rows}
    tile, gap, row_label, margin = 256, 12, 135, 26
    col = tile + 12
    header, row_height = 170, tile + 57
    width = 2 * margin + row_label + len(group) * col + (len(group) - 1) * gap
    height = header + len(CASES) * row_height + 60
    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill="#192b3b", font=font(26, True))
    subtitle = ("Matched detail crops: 25% of 2K canvas, 40% of 4K canvas, resized for display" if detail else
                "Matched full frames, resized for display")
    draw.text((margin, 57), subtitle + "  ·  scores use full-resolution images against BF16",
              fill="#536677", font=font(16))
    x0 = margin + row_label
    for i, label in enumerate(group):
        x = x0 + i * (col + gap)
        draw.multiline_text((x + 6, 105), LABELS[label][1], fill="#20394d", font=font(17, True), spacing=4)
    for index, (case, case_title, seed, anchor) in enumerate(CASES):
        y = header + index * row_height
        draw.text((margin, y + 10), case_title, fill="#20394d", font=font(17, True))
        draw.text((margin, y + 35), f"seed {seed}", fill="#657786", font=font(14))
        for i, label in enumerate(group):
            x = x0 + i * (col + gap)
            draw.rounded_rectangle((x, y, x + col, y + row_height - 10), radius=5, fill="white", outline="#d8dfe6")
            item = found.get((label, case))
            if item is None:
                draw.rectangle((x + 6, y + 6, x + 6 + tile, y + 6 + tile), fill="#e8edf1")
                draw.text((x + 45, y + tile // 2), "NOT MEASURED", fill="#6c7d8c", font=font(17, True))
                continue
            with Image.open(item[0]) as original:
                image = original.convert("RGB")
                if detail:
                    crop_size = round(image.width * (0.40 if case == "native-4k" else 0.25))
                    left = min(max(0, round(image.width * anchor[0] - crop_size / 2)), image.width - crop_size)
                    top = min(max(0, round(image.height * anchor[1] - crop_size / 2)), image.height - crop_size)
                    image = image.crop((left, top, left + crop_size, top + crop_size))
                image = ImageOps.fit(image, (tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(image, (x + 6, y + 6))
            if label == REFERENCE:
                score = "Reference image"
            else:
                row = metrics[(label, case)]
                score = f"SSIM {float(row['ssim']):.3f}  |  PSNR {float(row['psnr_db']):.2f} dB"
            draw.text((x + 8, y + tile + 16), score, fill="#344c60", font=font(14, True))
    draw.text((margin, height - 34), "Missing cases were not measured. FP8, attention and reduced passes can change pixels.",
              fill="#657786", font=font(15))
    canvas.save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/l20/images")
    parser.add_argument("--output", type=Path, default=ROOT / "comparisons/l20")
    parser.add_argument("--measure", action="store_true", help="recompute CSV from original PNGs")
    parser.add_argument("--image-sheets", action="store_true", help="redraw full-frame and crop comparisons")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    found = inventory(args.runs) if args.measure or args.image_sheets else None
    csv_path = args.output / "accuracy.csv"
    rows = measure(found, csv_path) if args.measure else load_metrics(csv_path)
    if args.image_sheets:
        validate_metrics_for_images(rows, found)
    plot_metrics(rows, args.output / "accuracy-metrics.png")
    if args.image_sheets:
        for name, (title, labels) in GROUPS.items():
            for detail in (False, True):
                suffix = "details" if detail else "frames"
                image_sheet(labels, title, found, rows, args.output / f"{name}-{suffix}.png", detail)
    print(f"Exported charts to {args.output}")


if __name__ == "__main__":
    main()
