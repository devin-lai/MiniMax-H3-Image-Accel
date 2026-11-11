"""Chart byte-level PNG comparisons from the recorded RTX 5090 results.

The original 5090 images are not in this repository. Equal SHA-256 hashes
establish matching PNG bytes; unequal hashes do not quantify pixel changes.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = (("product", "2K product"), ("portrait", "2K portrait"),
         ("edit", "2K edit"), ("native-4k", "4K still life"))
BASE = "runs/r2-warmed-base"
# (candidate path, reference path, display label). Paths are relative to
# provenance/results. Each row states its reference when it differs from BASE.
SCHEMES = (
    ("runs/bf16-tuned", BASE, "BF16 tuned"),
    ("runs/mxfp8-fa4", BASE, "MXFP8 + FlashAttention"),
    ("runs/sage2", BASE, "SageAttention2 (earlier layout)"),
    ("runs/mxfp8-sage2-tp2sp2", BASE, "MXFP8 + Sage2 · TP2 + U2"),
    ("runs/r2-resident", BASE, "Resident components"),
    ("runs/r2-four-merged", BASE, "Four merged passes"),
    ("runs/r2-three", BASE, "Three original heads"),
    ("runs/r2-three-merged", BASE, "Three merged passes"),
    ("runs/r2-tea015-all", BASE, "TeaCache · threshold 0.15"),
    ("runs/r2-tea04", BASE, "TeaCache · threshold 0.4"),
    ("runs/r2-tea3", BASE, "TeaCache · aggressive reuse"),
    ("runs/r2-sage3", BASE, "SageAttention3"),
    ("runs/r2-compiled", BASE, "Regional torch.compile"),
    ("runs/r2-graph", BASE, "Breakable CUDA graphs"),
    ("runs/r2-prefix-kv", "runs/r2-prefix-full", "Prefix KV vs full prefill *"),
    ("portable/gpu-output-six-v2", "runs/final-six", "GPU output · six vs CPU output *"),
    ("portable/gpu-output-balanced-v2", "runs/final-balanced", "GPU output · four vs CPU output *"),
    ("portable/gpu-output-extreme-v2", "runs/final-extreme", "GPU output · three vs CPU output *"),
)
MATCH_KEYS = ("case", "size", "seed", "task", "prompt", "reference_sha256", "quality", "output_compression")


def first_trials(root, label):
    records = {}
    path = root / label / "results.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("trial") == 1:
            if row["case"] in records:
                raise ValueError(f"Duplicate trial 1 in {path}")
            records[row["case"]] = row
    return records


def comparison_rows(root):
    records = {name: first_trials(root, name) for pair in SCHEMES for name in pair[:2]}
    rows = []
    for candidate, reference, label in SCHEMES:
        for case, _ in CASES:
            test = records[candidate].get(case)
            ref = records[reference].get(case)
            if test is None:
                continue
            if ref is None:
                raise ValueError(f"No reference for {candidate}/{case}")
            for key in MATCH_KEYS:
                if ref.get(key) != test.get(key):
                    raise ValueError(f"Mismatched {key} for {candidate}/{case}")
            rows.append({"candidate": candidate, "reference": reference, "scheme": label,
                         "case": case, "size": test["size"], "seed": test["seed"],
                         "reference_sha256": ref["sha256"], "candidate_sha256": test["sha256"],
                         "png_bytes": "same" if ref["sha256"] == test["sha256"] else "different"})
    return rows


def draw(rows, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    values = {(r["candidate"], r["case"]): r for r in rows}
    grid = np.full((len(SCHEMES), len(CASES)), -1, dtype=int)
    for i, (candidate, _, _) in enumerate(SCHEMES):
        for j, (case, _) in enumerate(CASES):
            if (candidate, case) in values:
                grid[i, j] = int(values[(candidate, case)]["png_bytes"] == "same")
    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#fbfcfe")
    fig.subplots_adjust(left=.385, right=.96, top=.82, bottom=.15)
    fig.text(.025, .955, "RTX 5090: recorded PNG fingerprint comparisons", fontsize=19, weight="bold", color="#182433")
    fig.text(.025, .916, "Matched first-trial requests. Images are unavailable here, so PSNR and SSIM cannot be computed.",
             fontsize=10, color="#516170")
    fig.text(.025, .889, "A different hash only means PNG bytes differ; it does not measure pixel or perceptual change.",
             fontsize=10, color="#516170")
    ax.imshow(grid, cmap=ListedColormap(["#e8ebef", "#f6d99e", "#a9d7c2"]), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CASES)), [title for _, title in CASES], rotation=25, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(SCHEMES)), [label for _, _, label in SCHEMES])
    ax.tick_params(axis="both", length=0, labelsize=10, colors="#263646")
    for y in range(len(SCHEMES)):
        for x in range(len(CASES)):
            ax.text(x, y, {1: "same", 0: "different", -1: "—"}[grid[y, x]],
                    ha="center", va="center", color="#243746", fontsize=9)
    ax.set_xticks(np.arange(-.5, len(CASES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(SCHEMES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(handles=[Patch(color="#a9d7c2", label="Same PNG bytes"),
                        Patch(color="#f6d99e", label="Different PNG bytes"),
                        Patch(color="#e8ebef", label="Not recorded")],
               loc="lower center", bbox_to_anchor=(.58, .09), ncol=3, frameon=False, fontsize=10)
    fig.text(.025, .055, "Reference: warmed six-pass MXFP8 + Sage2, TP2 + U2. * uses its named matched reference.",
             fontsize=9, color="#516170")
    fig.text(.025, .03, "Hashes and reference identifiers are in png-hashes.csv; see the README for the evidence limits.",
             fontsize=9, color="#516170")
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "provenance/results")
    parser.add_argument("--output", type=Path, default=ROOT / "comparisons/rtx5090")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(args.records)
    with (args.output / "png-hashes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(rows, args.output / "png-hashes.png")
    print(f"Exported {len(rows)} matched PNG fingerprint comparisons to {args.output}")


if __name__ == "__main__":
    main()
