"""Fail a matched-image accuracy audit if any case misses the requested limits.

Checks recorded PNG hashes and request metadata before measuring RGB PSNR and
8x8 luma SSIM. Missing cases, duplicate trials and mismatched requests fail.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compare_quality import psnr, ssim

MATCH_KEYS = ("case", "size", "seed", "task", "prompt", "reference_sha256",
              "quality", "output_compression")


def records(directory):
    found = {}
    for line in (directory / "results.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item.get("trial") != 1:
            continue
        case = item["case"]
        if case in found:
            raise ValueError(f"Duplicate trial 1: {directory}/{case}")
        found[case] = item
    return found


def pixels(directory, item):
    path = directory / (item["case"] + ".png")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Image hash mismatch: {path}")
    with Image.open(path) as image:
        if image.size != tuple(map(int, item["size"].split("x"))):
            raise ValueError(f"Image size mismatch: {path}")
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def audit(reference, candidate, cases, min_psnr=40.0, min_ssim=0.99, identical=False):
    refs, candidates = records(reference), records(candidate)
    rows = []
    for case in cases:
        a, b = refs[case], candidates[case]  # missing cases must fail
        for item in (a, b):
            for key in ("case", "size", "seed", "task", "prompt", "quality"):
                if key not in item:
                    raise ValueError(f"Missing {key}: {case}")
        for key in MATCH_KEYS:
            if a.get(key) != b.get(key):
                raise ValueError(f"Mismatched {key}: {case}")
        pa, pb = pixels(reference, a), pixels(candidate, b)
        same = bool(np.array_equal(pa, pb))
        p, s = (float("inf"), 1.0) if same else (float(psnr(pa, pb)), ssim(pa, pb))
        rows.append({
            "case": case, "reference": reference.name, "candidate": candidate.name,
            "psnr_db": p, "ssim": s, "pixel_identical": same,
            "png_identical": a["sha256"] == b["sha256"],
            "passed": same if identical else p >= min_psnr and s >= min_ssim,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cases", nargs="+", default=["product", "portrait", "edit", "native-4k"])
    parser.add_argument("--min-psnr", type=float, default=40.0)
    parser.add_argument("--min-ssim", type=float, default=0.99)
    parser.add_argument("--pixel-identical", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = audit(args.runs / args.reference, args.runs / args.candidate, args.cases,
                 args.min_psnr, args.min_ssim, args.pixel_identical)
    for row in rows:
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
