"""Prepare the pinned SGLang runtime and apply the H3 inference changes."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def prepare(root: Path) -> None:
    target = root / "vendor" / "sglang"
    entry = Path("python/sglang/multimodal_gen/runtime/models/vaes/h3_exact_graph.py")
    if target.exists() and not target.is_symlink():
        if not (target / entry).is_file():
            raise RuntimeError(f"{target} exists without the H3 inference changes.")
        print(f"Using the existing SGLang runtime at {target}")
        return

    source = json.loads((root / "provenance" / "sources.json").read_text())["sglang"]
    revision = source["commit"]
    patch = root / "vendor" / "sglang-h3.patch"
    patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    cache = root / "cache" / "sglang-source"
    prepared = cache / f"{revision}-{patch_hash}"
    cache.mkdir(parents=True, exist_ok=True)

    with (cache / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_symlink() and target.resolve() == prepared and (prepared / entry).is_file():
            print(f"Using cached SGLang runtime at {prepared}")
            return

        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=cache) as work:
            stage = Path(work)
            if not prepared.exists():
                repository = stage / "upstream.git"
                snapshot = stage / "snapshot"
                snapshot.mkdir()
                subprocess.run(["git", "init", "--quiet", "--bare", repository], check=True)
                git = ["git", "-C", repository]
                subprocess.run(
                    [*git, "fetch", "--quiet", "--depth=1", "--no-tags", source["url"], revision],
                    check=True,
                )
                fetched = subprocess.check_output(
                    [*git, "rev-parse", "FETCH_HEAD"], text=True
                ).strip()
                if fetched != revision:
                    raise RuntimeError(f"Expected SGLang {revision}, fetched {fetched}.")

                archive = stage / "sglang.tar"
                with archive.open("wb") as stream:
                    subprocess.run(
                        [*git, "archive", revision, "LICENSE", "python"],
                        stdout=stream, check=True,
                    )
                subprocess.run(["tar", "-xf", archive, "-C", snapshot], check=True)
                subprocess.run(
                    ["git", f"--git-dir={repository}", f"--work-tree={snapshot}",
                     "apply", "--whitespace=nowarn", patch],
                    cwd=snapshot, check=True,
                )
                snapshot.rename(prepared)

            if not (prepared / entry).is_file():
                raise RuntimeError(f"Incomplete SGLang runtime at {prepared}.")
            link = stage / "sglang"
            link.symlink_to(os.path.relpath(prepared, target.parent), target_is_directory=True)
            link.replace(target)

    print(f"Prepared SGLang {revision} with the H3 inference changes.")


if __name__ == "__main__":
    prepare(Path(os.environ.get("H3_ROOT", Path(__file__).resolve().parents[1])).resolve())
