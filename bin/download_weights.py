"""Download only the pinned native H3 image checkpoint files from Hugging Face."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("H3_ROOT", Path(__file__).resolve().parents[1]))
REVISIONS = {
    "MiniMax-H3-Image": "6534ea1c979a5f8cb00c2bd1025d5ca4ee8d8901",
    "MiniMax-H3": "0919fb684102ef115126612f3fbb84d1027e7317",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path,
                        default=Path(os.environ.get("H3_MODELS_DIR", ROOT / "models")))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/weights-manifest.json").read_text())
    if args.list:
        print(json.dumps({"revisions": REVISIONS, "files": len(manifest),
                          "total_gib": round(sum(row[2] for row in manifest) / 2**30, 2)}, indent=2))
        return
    args.models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / "cache/hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # local_dir keeps each shard once. No overlapping diffusion-format weights.
    from disk_guard import check
    needed = sum(size for model, rel, size, _ in manifest
                 if not (args.models_dir / model / rel).exists())
    check(args.models_dir, needed + (2 * 1024**3 if needed else 0))
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
    for index, (model, rel, size, expected) in enumerate(manifest, 1):
        path = args.models_dir / model / rel
        if not path.exists() or path.stat().st_size != size:
            if args.verify_only:
                raise RuntimeError(f"Missing or truncated checkpoint file: {path}")
            check(args.models_dir)
            for attempt in range(5):
                try:
                    downloaded = hf_hub_download(
                        repo_id=f"MiniMaxAI/{model}", filename=rel,
                        revision=REVISIONS[model], local_dir=args.models_dir / model,
                    )
                    if Path(downloaded).resolve() != path.resolve():
                        raise RuntimeError("Unexpected download destination")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(min(10, 2**attempt))
        if path.stat().st_size != size or sha256(path) != expected:
            raise RuntimeError(f"SHA256 verification failed: {path}")
        print(f"[{index}/{len(manifest)}] verified {model}/{rel}", flush=True)
        check(args.models_dir)
    ready = {"revisions": REVISIONS, "verified_files": len(manifest),
             "manifest_sha256": sha256(ROOT / "configs/weights-manifest.json")}
    (args.models_dir / "verified.json").write_text(json.dumps(ready, indent=2) + "\n")
    print("All official model files verified.")


if __name__ == "__main__":
    main()
