"""Make aligned contact sheets and pixel statistics from completed benchmark runs."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, ImageStat

TITLES = {
    "product": "Product photograph / typography",
    "portrait": "Portrait / skin and fabric",
    "edit": "Reference edit / identity and lettering",
    "native-4k": "Native 4K still life / fine lines",
    "smoke": "Smoke test",
}
CROP_ANCHORS = {"product": (0.50, 0.55), "portrait": (0.50, 0.46),
                "edit": (0.50, 0.50), "native-4k": (0.60, 0.65)}
BACKGROUND, CARD, INK, MUTED = "#101419", "#202830", "#f4f6f8", "#acb9c6"


def font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def image(path):
    with Image.open(path) as src:
        return src.convert("RGB")


def pixel_stats(a, b):
    if a.size != b.size:
        raise ValueError("Image sizes differ")
    d = ImageChops.difference(a, b)
    histogram = d.histogram()
    mse = sum((i % 256)**2 * n for i, n in enumerate(histogram)) / (a.width * a.height * 3)
    return {
        "pixel_identical": mse == 0,
        "rgb_mae_0_to_255": round(sum(ImageStat.Stat(d).mean) / 3, 5),
        "rgb_psnr_db": "" if mse == 0 else round(10 * math.log10(255**2 / mse), 4),
    }


def sheet(runs, cases, out, details=False):
    margin, col, gap, tile, label, head = 26, 442, 12, 420, 38, 100
    row_h = tile + 75
    canvas = Image.new("RGB", (2 * margin + len(runs)*col + (len(runs)-1)*gap,
                                head + len(cases)*row_h + 22), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "MiniMax H3 Image  /  " + ("1:1 detail crops" if details else "Matched seed comparisons"),
              fill=INK, font=font(27, True))
    draw.text((margin, 57), "Same prompt, seed and canvas. Approximate passes and precision are labeled below.",
              fill=MUTED, font=font(16))
    for row, (name, base_record) in enumerate(cases):
        y = head + row * row_h
        draw.text((margin, y), f"{TITLES.get(name, name)}    {base_record['size']}    seed {base_record['seed']}",
                  fill=INK, font=font(18, True))
        for i, (directory, title, records) in enumerate(runs):
            x, top = margin + i*(col+gap), y + label
            draw.rounded_rectangle((x, y + 28, x+col, y+row_h-8), radius=8, fill=CARD)
            record = records.get(name)
            if record is None:
                draw.text((x+12, top+10), f"{title}: unavailable", fill=MUTED, font=font(17))
                continue
            im = image(directory / (name + ".png"))
            if details:
                anchor = CROP_ANCHORS.get(name, (0.5, 0.5))
                left = min(max(0, int(im.width*anchor[0] - tile/2)), im.width-tile)
                upper = min(max(0, int(im.height*anchor[1] - tile/2)), im.height-tile)
                thumb = im.crop((left, upper, left+tile, upper+tile))
            else:
                thumb = ImageOps.contain(im, (tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x+(col-thumb.width)//2, top+(tile-thumb.height)//2))
            draw.text((x+11, top+tile+7), f"{title}   ·   {record['median_seconds']:.2f}s",
                      fill=INK, font=font(16, True))
    canvas.save(out, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True, help="run-directory=display name; native first")
    p.add_argument("--cases", nargs="+", choices=TITLES, default=tuple(TITLES))
    args = p.parse_args()
    runs = []
    for pair in args.variant:
        name, title = pair.split("=", 1)
        directory = args.runs / name
        records, timings = {}, {}
        for line in (directory / "results.jsonl").read_text().splitlines():
            item = json.loads(line)
            timings.setdefault(item["case"], []).append(item["wall_seconds"])
            if item["trial"] == 1:
                records[item["case"]] = item
        for case, item in records.items():
            item["median_seconds"] = statistics.median(timings[case])
        runs.append((directory, title, records))
    if len(runs) < 2:
        raise ValueError("At least two completed profiles are needed for a comparison")
    args.output.mkdir(parents=True, exist_ok=True)
    base, _, records = runs[0]
    cases = [(k, records[k]) for k in args.cases if k in records and any(k in r for _, _, r in runs[1:])]
    if not cases:
        raise ValueError("No matching cases found")
    for details, name in [(False, "side-by-side.png"), (True, "detail-crops.png")]:
        sheet(runs, cases, args.output / name, details)
    rows = []
    for name, ref in cases:
        a = image(base / (name + ".png"))
        for d, title, r in runs[1:]:
            if name not in r:
                continue
            assert all(ref.get(k) == r[name].get(k) for k in ("seed", "size", "prompt", "reference_sha256"))
            b = image(d / (name + ".png"))
            rows.append({"case": name, "variant": title, "seed": ref["seed"], "size": ref["size"],
                         **pixel_stats(a, b)})
    with (args.output / "pixel-differences.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
