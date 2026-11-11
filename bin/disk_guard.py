"""Run a child while keeping the containing filesystem below a hard limit."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def check(path, reserve=0, limit=0.78):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    usage = shutil.disk_usage(path)
    if (usage.used + reserve) / usage.total >= limit:
        raise RuntimeError(f"Disk guard: {path} is {usage.used / usage.total:.1%} used; "
                           f"{reserve / 2**30:.1f} GiB additional space exceeds {limit:.0%}.")
    return usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    check(args.path, int(args.reserve_gib * 2**30))
    if not command:
        return
    child = subprocess.Popen(command, start_new_session=True)
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while child.poll() is None:
            check(args.path)
            time.sleep(1)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
