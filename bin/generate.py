"""Generate with an explicit, reviewable H3 image optimization profile."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import benchmark


def main():
    profiles = json.loads((benchmark.ROOT / "configs/request-profiles.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--profile", choices=profiles, default="six")
    parser.add_argument("--size", default="2048x2048")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--quality", choices=["exact", "lossless"], default="exact",
                        help="exact preserves the native math; lossless enables upstream quality-gated fusions")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prefix-kv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or benchmark.ROOT / "generated" / f"{int(time.time())}-{args.seed}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    opt = dict(profiles[args.profile])
    opt["prefix_kv"] = args.prefix_kv
    request_args = SimpleNamespace(url=args.url, quality=args.quality, compression=1,
                                   timeout=900, stage_profile=False, attention=None, opt=opt)
    png, record = benchmark.make_request(
        request_args, args.size, args.seed, args.prompt, bool(args.image),
        "generation", output.parent, reference_image=args.image,
    )
    output.write_bytes(png)
    record.update(prompt=args.prompt, profile=args.profile, output=str(output))
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
