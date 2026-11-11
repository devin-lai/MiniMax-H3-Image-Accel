"""Export matched full frames, 1:1 crops and metrics for a run manifest.

Every PNG is checked against its benchmark record before it is displayed.
Missing cases remain labeled gaps. Metrics use full-resolution RGB images,
even when the sheet displays a crop or a resized full frame.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from export_accuracy import MATCH_KEYS, compare_arrays, font, sha256


CASES = ("product", "portrait", "edit", "native-4k")
TITLES = {"product": "2K product", "portrait": "2K portrait",
          "edit": "2K reference edit", "native-4k": "Native 4K"}
CROPS = (("product", "Bottle label", .50, .55),
         ("portrait", "Face / skin", .62, .34),
         ("edit", "Face / identity", .73, .37),
         ("edit", "Carved lettering", .50, .72),
         ("native-4k", "Notecard / paper", .64, .64))


def inventory(runs, variants, reference, cases=CASES, allow_quality_difference=False):
    found = {}
    for variant in variants:
        name = variant["run"]
        directory = runs / name
        records, timings = {}, {}
        excluded = variant.get("exclude_timing_trials", {})
        for line in (directory / "results.jsonl").read_text().splitlines():
            item = json.loads(line)
            case = item["case"]
            if item.get("label") != name:
                raise ValueError(f"Mismatched run label: {directory}")
            required = ("case", "size", "seed", "task", "prompt", "quality", "output_compression", "sha256")
            if any(key not in item for key in required) or (
                item["task"] == "edit" and "reference_sha256" not in item
            ):
                raise ValueError(f"Incomplete request metadata: {directory}/{case}")
            if allow_quality_difference and item["quality"] != variant.get("quality"):
                raise ValueError(f"Quality comparison requires a declared quality: {directory}/{case}")
            if item["trial"] not in excluded.get(case, []):
                timings.setdefault(case, []).append(item["wall_seconds"])
            if item["trial"] == 1:
                if case in records:
                    raise ValueError(f"Duplicate trial 1: {directory}/{case}")
                records[case] = item
        for case in cases:
            path = directory / f"{case}.png"
            record = records.get(case)
            if record is None and not path.exists():
                continue
            if record is None or not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Missing image/record or invalid PNG hash: {path}")
            with Image.open(path) as image:
                if image.size != tuple(map(int, record["size"].split("x"))):
                    raise ValueError(f"Mismatched image dimensions: {path}")
            seconds = statistics.median(timings[case]) if timings.get(case) else None
            found[name, case] = (path, record, seconds)
    keys = tuple(key for key in MATCH_KEYS if key != "quality" or not allow_quality_difference)
    for case in cases:
        ref = found[reference, case][1]
        for variant in variants:
            item = found.get((variant["run"], case))
            if item and any(ref.get(key) != item[1].get(key) for key in keys):
                raise ValueError(f"Mismatched benchmark request: {variant['run']}/{case}")
    return found


def measure(found, variants, reference, cases=CASES):
    rows = []
    for case in cases:
        ref_path, ref_record, _ = found[reference, case]
        with Image.open(ref_path) as image:
            ref = np.asarray(image.convert("RGB"))
        for variant in variants:
            name = variant["run"]
            if name == reference or (name, case) not in found:
                continue
            path, record, seconds = found[name, case]
            with Image.open(path) as image:
                candidate = np.asarray(image.convert("RGB"))
            psnr, ssim, mae = compare_arrays(ref, candidate)
            rows.append(dict(reference=reference, candidate=name, case=case,
                             size=record["size"], seed=record["seed"],
                             psnr_db=psnr, ssim=ssim, mean_abs_diff=mae,
                             pixel_identical=bool(np.array_equal(ref, candidate)),
                             median_seconds=seconds,
                             reference_png_sha256=ref_record["sha256"],
                             candidate_png_sha256=record["sha256"]))
            print(f"{name}/{case}: PSNR {psnr:.3f}, SSIM {ssim:.6f}", flush=True)
    return rows


def wrapped(draw, value, width, face):
    lines = []
    for paragraph in value.splitlines():
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if line and draw.textlength(trial, font=face) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def sheet(group, reference, variants, found, metrics, output, tile=420, detail=False, cases=CASES):
    columns = [variants[name] for name in [reference, *group["runs"]]]
    margin, gap, heading, row_height = 24, 14, 204, tile + 105
    width = margin * 2 + len(columns) * (tile + gap) - gap
    cells = tuple(row for row in CROPS if row[0] in cases) if detail else tuple(
        (case, TITLES[case], None, None) for case in cases)
    height = heading + row_height * len(cells) + 110
    canvas = Image.new("RGB", (width, height), "#f4f6f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), group["title"], fill="#192b3b", font=font(28, True))
    mode = f"{tile} × {tile} source-pixel crops (1:1)" if detail else "Full frames resized for display"
    draw.text((margin, 60), mode + " · same prompt, seed, resolution and reference input",
              fill="#536677", font=font(17))
    hardware = group.get("hardware_note", "All runs use one L20.")
    draw.text((margin, 88), hardware + " Scores use full-resolution images; gray = not measured.",
              fill="#536677", font=font(17))
    for index, variant in enumerate(columns):
        x = margin + index * (tile + gap)
        text = wrapped(draw, variant["label"], tile, font(19, True))
        draw.multiline_text((x, 128), text, font=font(19, True), fill="#20394d", spacing=5)
    for row, (case, title, cx, cy) in enumerate(cells):
        y = heading + row * row_height
        ref_record = found[reference, case][1]
        draw.text((margin, y), f"{TITLES[case]} / {title}" if detail else title,
                  fill="#20394d", font=font(19, True))
        for index, variant in enumerate(columns):
            name = variant["run"]
            x, top = margin + index * (tile + gap), y + 31
            draw.rectangle((x, top, x + tile, top + tile), fill="#e2e7ed")
            item = found.get((name, case))
            if item is None:
                draw.text((x + 90, top + tile // 2), "NOT MEASURED", fill="#657786", font=font(19, True))
                continue
            path, record, seconds = item
            with Image.open(path) as original:
                image = original.convert("RGB")
                if detail:
                    left = min(max(0, round(image.width * cx - tile / 2)), image.width - tile)
                    upper = min(max(0, round(image.height * cy - tile / 2)), image.height - tile)
                    image = image.crop((left, upper, left + tile, upper + tile))
                else:
                    image = ImageOps.contain(image, (tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(image, (x, top))
            elapsed = f"{seconds:.2f}s median HTTP" if seconds is not None else "Timing unavailable"
            score = "Reference" if name == reference else (
                "IDENTICAL PIXELS" if metrics[name, case]["pixel_identical"] else
                f"SSIM {metrics[name, case]['ssim']:.4f} · PSNR {metrics[name, case]['psnr_db']:.2f} dB")
            draw.text((x, top + tile + 9), score, fill="#20394d", font=font(16, True))
            draw.text((x, top + tile + 34), elapsed, fill="#657786", font=font(15))
        draw.text((width - 280, y + 4), f"{ref_record['size']} · seed {ref_record['seed']}",
                  fill="#657786", font=font(15))
    draw.text((margin, height - 92), "Higher PSNR/SSIM means closer pixels, not a general perceptual-quality score.",
              fill="#536677", font=font(17))
    note = group.get("timing_note", "Warmups excluded; see manifest for cold compilation exclusions and the raw records for trial counts.")
    draw.multiline_text((margin, height - 65), wrapped(draw, note, width - 2 * margin, font(15)),
                        fill="#536677", font=font(15), spacing=4)
    canvas.save(output, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    variants = {item["run"]: item for item in config["variants"]}
    reference = config["reference"]
    cases = tuple(config.get("cases", CASES))
    if not cases or any(case not in CASES for case in cases):
        raise ValueError("Manifest cases must name measured benchmark cases")
    found = inventory(args.runs, list(variants.values()), reference, cases,
                      config.get("allow_quality_difference", False))
    rows = measure(found, list(variants.values()), reference, cases)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {(row["candidate"], row["case"]): row for row in rows}
    for group in config["groups"]:
        for detail in (False, True):
            suffix = "details" if detail else "frames"
            sheet(group, reference, variants, found, metrics,
                  args.output / f"{group['name']}-{suffix}.png", detail=detail, cases=cases)


if __name__ == "__main__":
    main()
