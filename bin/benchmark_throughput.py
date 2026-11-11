"""Measure sustained H3 image throughput across one or more local endpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
from threading import Barrier
import time
from types import SimpleNamespace

from benchmark import CASES, ROOT, make_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", choices=CASES, default="product")
    parser.add_argument("--url", action="append", help="Repeat a URL for concurrent clients")
    parser.add_argument("--requests-per-client", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/throughput")
    parser.add_argument("--opt", type=json.loads)
    parser.add_argument("--quality", choices=["exact", "lossless"], default="lossless")
    parser.add_argument("--compression", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.requests_per_client < 1:
        parser.error("--requests-per-client must be positive")

    urls = args.url or ["http://127.0.0.1:30000"]
    size, seed, prompt, edit = CASES[args.case]
    if isinstance(prompt, Path):
        prompt = prompt.read_text().strip()
    request_args = [
        SimpleNamespace(url=url, quality=args.quality, compression=args.compression,
                        timeout=args.timeout, stage_profile=False, attention=None, opt=args.opt)
        for url in urls
    ]
    start = []
    gate = Barrier(len(urls) + 1, action=lambda: start.append(time.perf_counter()))

    def run(client, request):
        gate.wait()
        rows = []
        for i in range(args.requests_per_client):
            _, record = make_request(
                request, size, seed + client * args.requests_per_client + i,
                prompt, edit, args.case, args.output,
            )
            rows.append({"client": client, "url": request.url, **record})
        return rows

    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        warmups = [
            pool.submit(make_request, request, size, seed - i - 1, prompt, edit,
                        args.case, args.output)
            for i, request in enumerate(request_args)
        ]
        for warmup in warmups:
            warmup.result()
        futures = [pool.submit(run, i, request) for i, request in enumerate(request_args)]
        gate.wait()
        rows = [row for future in futures for row in future.result()]
    elapsed = time.perf_counter() - start[0]
    latencies = sorted(row["wall_seconds"] for row in rows)
    summary = {
        "label": args.label, "case": args.case, "size": size, "prompt": prompt,
        "quality": args.quality,
        "clients": len(urls), "endpoints": len(set(urls)), "images": len(rows),
        "wall_seconds": round(elapsed, 4),
        "images_per_minute": round(60 * len(rows) / elapsed, 4),
        "median_request_seconds": round(statistics.median(latencies), 4),
        "p95_request_seconds": latencies[math.ceil(len(latencies) * 0.95) - 1],
        "requests": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("requests", "prompt")}))


if __name__ == "__main__":
    main()
