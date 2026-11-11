"""PSNR and SSIM between matched benchmark images of two or more runs.

  compare_quality.py --runs runs --reference l20-bf16 --candidate fp8-sage-tp1sp4 \
      --candidate my-run --output quality.csv

Each run directory holds first-trial PNGs named after the benchmark case
(product.png, portrait.png, edit.png, native-4k.png). SSIM is the standard
single-scale luma index with an 8x8 box window, computed with integral
images so 4K frames need only NumPy and Pillow.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def box(x, size):
    total = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), axis=0), axis=1)
    return (total[size:, size:] - total[:-size, size:] - total[size:, :-size]
            + total[:-size, :-size]) / size**2


def ssim(a, b, size=8):
    # Avoid a BLAS dispatch for a three-channel weighted sum. Some macOS
    # Accelerate/NumPy combinations report spurious FP exceptions here.
    def luma(rgb):
        return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    x, y = luma(a), luma(b)
    mx, my = box(x, size), box(y, size)
    vx, vy = box(x * x, size) - mx * mx, box(y * y, size) - my * my
    cov = box(x * y, size) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean(((2 * mx * my + c1) * (2 * cov + c2))
                         / ((mx * mx + my * my + c1) * (vx + vy + c2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--cases", nargs="+",
                        default=["product", "portrait", "edit", "native-4k"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        reference = args.runs / args.reference / f"{case}.png"
        if not reference.exists():
            continue
        target = load(reference)
        for candidate in args.candidate:
            path = args.runs / candidate / f"{case}.png"
            if not path.exists():
                continue
            image = load(path)
            rows.append({"reference": args.reference, "candidate": candidate, "case": case,
                         "psnr_db": round(psnr(image, target), 2),
                         "ssim": round(ssim(image, target), 4),
                         "mean_abs_diff": round(float(np.mean(np.abs(image - target))), 3)})
            print(rows[-1], flush=True)
    if args.output and rows:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
