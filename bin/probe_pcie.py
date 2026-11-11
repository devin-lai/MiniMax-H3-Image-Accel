"""Find which GPUs share a PCIe uplink, by timing concurrent host copies.

Containers often hide the PCIe tree, so `nvidia-smi topo` and sysfs cannot
tell whether two GPUs sit behind one switch port. Two GPUs that share an
uplink each get about half of their solo device-to-host bandwidth when they
copy at the same time. The host-staged Ulysses exchange is bandwidth bound,
so the launcher places a replica's ranks on different uplinks when it can.

Prints JSON: solo bandwidths (GB/s) and the uplink groups as lists of
PCI-ordered CUDA ordinals. Run it on idle GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time


def measure(devices, megabytes=32, rounds=12):
    import torch

    size = megabytes << 20
    buffers = {}
    for device in devices:
        gpu = torch.empty(size, dtype=torch.uint8, device=f"cuda:{device}").zero_()
        host = torch.empty(size, dtype=torch.uint8).pin_memory()
        host.copy_(gpu)
        buffers[device] = (gpu, host)
    barrier = threading.Barrier(len(devices))
    rates = {}

    def worker(device):
        gpu, host = buffers[device]
        barrier.wait()
        start = time.perf_counter()
        for _ in range(rounds):
            host.copy_(gpu, non_blocking=True)
            torch.cuda.synchronize(device)
        rates[device] = megabytes * rounds / 1024 / (time.perf_counter() - start)

    threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return rates


def probe():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch

    count = torch.cuda.device_count()
    solo = {d: measure([d])[d] for d in range(count)}
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a in range(count):
        for b in range(a + 1, count):
            if find(a) == find(b):
                continue
            pair = measure([a, b])
            if pair[a] < 0.7 * solo[a] and pair[b] < 0.7 * solo[b]:
                parent[find(b)] = find(a)
    groups = {}
    for device in range(count):
        groups.setdefault(find(device), []).append(device)
    return {
        "solo_gb_per_s": [round(solo[d], 1) for d in range(count)],
        "uplink_groups": sorted(groups.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="also write the JSON to this file")
    args = parser.parse_args()
    text = json.dumps(probe())
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
