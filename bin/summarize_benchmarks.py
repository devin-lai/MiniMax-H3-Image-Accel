"""Summarize all recorded trials; keep cold/single-trial runs identifiable."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for directory in args.runs:
        for path in sorted(directory.glob("*/results.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if "trial" in row:
                    groups.setdefault((row["label"], row["case"]), []).append(row)
    rows = []
    for (label, case), group in groups.items():
        sample = group[0]
        values = [r["wall_seconds"] for r in group]
        rows.append(dict(label=label, case=case, size=sample["size"], seed=sample["seed"],
                         trials=len(group), median_seconds=round(statistics.median(values), 4),
                         min_seconds=min(values), max_seconds=max(values),
                         repeated_png_identical=len({r["sha256"] for r in group}) == 1,
                         h3_opt=json.dumps(sample.get("h3_opt"), sort_keys=True)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} benchmark groups to {args.output}")


if __name__ == "__main__":
    main()
