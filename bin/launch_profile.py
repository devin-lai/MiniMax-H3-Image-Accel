"""Choose an H3 server profile for the visible CUDA devices."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

SINGLE_MODES = {
    "single", "single-reference", "single-fp8", "single-fast", "single-int8-mlp"
}


@dataclass
class GPU:
    index: int
    uuid: str
    name: str
    bus: str
    capability: str
    memory_mib: int
    free_mib: int
    cpus: str = ""
    numa_node: int | None = None
    uplink: int | None = None
    utilization: int = 0

    @property
    def family(self):
        if self.capability == "12.0" and "RTX 5090" in self.name:
            return "rtx5090"
        if self.capability == "8.9" and self.name == "NVIDIA L20":
            return "l20"
        return "cuda"


def inventory():
    fields = "index,uuid,name,pci.bus_id,compute_cap,memory.total,memory.free,utilization.gpu"
    rows = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = [
        GPU(int(index), uuid, name, bus, capability, int(total), int(free),
            utilization=int(utilization) if utilization.strip().isdigit() else 0)
        for index, uuid, name, bus, capability, total, free, utilization
        in csv.reader(rows.splitlines(), skipinitialspace=True)
    ]
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
    ).stdout
    affinity_column = -3 if "GPU NUMA ID" in topology else -2
    by_index = {gpu.index: gpu for gpu in gpus}
    for line in topology.splitlines():
        columns = line.split()
        if not columns or not columns[0].startswith("GPU"):
            continue
        index = columns[0][3:]
        if index.isdigit() and int(index) in by_index and len(columns) > len(gpus):
            cpus = columns[affinity_column]
            if cpus and all(c in "0123456789,-" for c in cpus):
                by_index[int(index)].cpus = cpus
            numa_node = columns[affinity_column + 1]
            if numa_node.isdigit():
                by_index[int(index)].numa_node = int(numa_node)
    # The server sets CUDA_DEVICE_ORDER=PCI_BUS_ID. Numeric selectors refer
    # to CUDA ordinals in that order, which need not match NVML indices.
    return sorted(gpus, key=lambda gpu: gpu.bus)


def assign_uplinks(gpus, root, env):
    """Label GPUs with their PCIe uplink group, probing once per GPU set.

    The probe (bin/probe_pcie.py) needs idle devices, so it only runs when
    every GPU is free; its result is cached under cache/pcie-uplinks.json.
    H3_PCIE_PROBE=0 skips it and keeps the NUMA-only placement.
    """
    if env.get("H3_PCIE_PROBE", "1") == "0" or len(gpus) < 3:
        return
    if not all(gpu.family == "l20" for gpu in gpus):
        return
    cache = Path(root) / "cache" / "pcie-uplinks.json"
    uuids = [gpu.uuid for gpu in gpus]
    groups = None
    try:
        saved = json.loads(cache.read_text())
        if saved.get("uuids") == uuids:
            groups = saved["uplink_groups"]
    except (OSError, ValueError, KeyError):
        pass
    if groups is None:
        python = Path(root) / ".venv" / "bin" / "python"
        if not python.exists() or any(g.free_mib < 0.95 * g.memory_mib for g in gpus):
            return
        probe_env = {k: v for k, v in env.items() if k != "CUDA_VISIBLE_DEVICES"}
        probe_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        try:
            result = json.loads(subprocess.check_output(
                [str(python), str(Path(root) / "bin" / "probe_pcie.py")],
                text=True, env=probe_env, timeout=300, stderr=subprocess.DEVNULL,
            ).strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return
        groups = result["uplink_groups"]
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"uuids": uuids, **result}) + "\n")
        except OSError:
            pass
    for label, group in enumerate(groups):
        for ordinal in group:
            if ordinal < len(gpus):
                gpus[ordinal].uplink = label


def select_devices(gpus, value):
    chosen = []
    for selector in value.split(","):
        selector = selector.strip()
        if selector.isdigit() and int(selector) < len(gpus):
            ordinal = int(selector)
        else:
            matches = [
                i for i, gpu in enumerate(gpus)
                if selector.startswith("GPU-") and gpu.uuid.startswith(selector)
            ]
            if len(matches) != 1:
                raise ValueError(f"CUDA device selector {selector!r} does not identify a GPU")
            ordinal = matches[0]
        if ordinal in chosen:
            raise ValueError(f"Duplicate CUDA device selector {selector!r}")
        chosen.append(ordinal)
    return chosen


def cpu_affinity(gpus):
    if not hasattr(os, "sched_getaffinity"):
        return ""
    cpus = {gpu.cpus for gpu in gpus}
    if len(cpus) != 1 or "" in cpus:
        return ""
    available = os.sched_getaffinity(0)
    chosen = set()
    affinity = cpus.pop()
    for interval in affinity.split(","):
        start, _, stop = interval.partition("-")
        chosen.update(range(int(start), int(stop or start) + 1))
    if chosen <= available:
        return affinity
    return ",".join(map(str, sorted(available & chosen)))


def parallel_layouts(family, mode, env):
    if mode in SINGLE_MODES:
        if int(env.get("H3_TP") or 1) != 1 or int(env.get("H3_ULYSSES") or 1) != 1:
            raise ValueError("Single-card profiles require TP=1 and Ulysses=1")
        return [(1, 1)]
    if mode in ("auto", "accurate", "native") or family == "cuda":
        defaults = [(4, 1)]
    elif family == "rtx5090":
        defaults = [(2, 2)]
    elif mode == "throughput":
        defaults = [(1, 2)]
    elif mode == "latency":
        defaults = [(1, 8), (1, 4), (1, 2)]
    else:
        defaults = [(1, 4)]
        if mode != "fp8-dit":
            defaults.append((1, 2))
    if env.get("H3_TP") or env.get("H3_ULYSSES"):
        tp, ulysses = defaults[0]
        tp = int(env.get("H3_TP") or tp)
        ulysses = int(env.get("H3_ULYSSES") or ulysses)
        if tp < 1 or ulysses < 1:
            raise ValueError("H3_TP and H3_ULYSSES must be positive integers")
        return [(tp, ulysses)]
    return defaults


def _uplink_queues(gpus, members):
    queues = {}
    for i in members:
        key = ("solo", i) if gpus[i].uplink is None else ("shared", gpus[i].uplink)
        queues.setdefault(key, []).append(i)
    return sorted(queues.values(), key=lambda queue: queue[0])


def _spread(gpus, members, count):
    """``count`` of ``members`` on as many PCIe uplinks as possible."""
    queues = _uplink_queues(gpus, members)
    chosen = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return [i for i in members if i in chosen], min(count, len(queues))


def _replicas(gpus, members, count):
    """Consecutive ``count``-GPU replicas, each spread over PCIe uplinks."""
    queues = _uplink_queues(gpus, members)
    chosen = []
    while sum(map(len, queues)) >= count:
        queues.sort(key=lambda queue: (-len(queue), queue[0] if queue else 0))
        replica = [queue.pop(0) for queue in queues[:count] if queue]
        while len(replica) < count:
            replica.append(max(queues, key=len).pop(0))
        chosen.extend(sorted(replica))
    return chosen


def pick_group(gpus, group, count, explicit, replicas):
    if explicit:
        matches = len(group) % count == 0 if replicas else len(group) == count
        return group if matches else []
    local = {}
    for i in group:
        local.setdefault(gpus[i].cpus, []).append(i)
    if replicas:
        return [i for socket in local.values() for i in _replicas(gpus, socket, count)]
    # One NUMA node unless spanning nodes reaches more independent uplinks:
    # the Ulysses exchange is bound by host-link bandwidth, not by NUMA.
    options = [_spread(gpus, socket, count) for socket in local.values()
               if len(socket) >= count]
    options.append(_spread(gpus, group, count))
    best = max(uplinks for _, uplinks in options)
    return next(chosen for chosen, uplinks in options
                if uplinks == best and len(chosen) == count)


def plan(gpus, mode, env):
    explicit = env.get("H3_GPUS")
    visible = (
        select_devices(gpus, env["CUDA_VISIBLE_DEVICES"])
        if "CUDA_VISIBLE_DEVICES" in env and not explicit else list(range(len(gpus)))
    )
    if explicit:
        visible = select_devices(gpus, explicit)
    else:
        visible = [
            i for i in visible
            if gpus[i].free_mib >= 0.85 * gpus[i].memory_mib
            and (mode not in SINGLE_MODES or gpus[i].utilization < 20)
        ]
    if mode in SINGLE_MODES:
        if any(int(env.get(key) or "1") != 1 for key in ("H3_TP", "H3_ULYSSES", "H3_DP")):
            raise ValueError("Single-card profiles require TP=Ulysses=DP=1")
        if explicit and len(visible) != 1:
            raise ValueError("Single-card profiles require exactly one H3_GPUS selector")
        if mode == "single" and not any(gpus[i].family == "rtx5090" for i in visible):
            return single_config(gpus, visible, env, explicit)
    for family in ("rtx5090", "l20", "cuda"):
        if mode in SINGLE_MODES and family != "rtx5090":
            continue
        if mode == "sage3" and family != "rtx5090":
            continue
        if mode in ("fp8-dit", "throughput", "latency") and family != "l20":
            continue
        if mode in ("compiled", "graph") and family == "cuda":
            continue
        native = mode in ("auto", "accurate", "native", "single", "single-reference") or family == "cuda"
        groups = {}
        for i in visible:
            gpu = gpus[i]
            major = gpu.capability.split(".")[0]
            if gpu.family == family and major.isdigit() and int(major) >= 8:
                groups.setdefault((gpu.name, gpu.capability), []).append(i)
        for group in groups.values():
            if explicit and len(group) != len(visible):
                continue
            for tp, ulysses in parallel_layouts(family, mode, env):
                count = tp * ulysses
                if len(group) < count:
                    continue
                chosen = pick_group(gpus, group, count, explicit, mode == "throughput")
                if chosen and (mode in SINGLE_MODES or not native or sum(gpus[i].memory_mib for i in chosen) >= 128000):
                    return profile_config(
                        family, mode, native, gpus, chosen, tp, ulysses, env
                    )
    if mode in ("auto", "accurate"):
        return single_config(gpus, visible, env, explicit)
    raise ValueError(
        f"No homogeneous CUDA GPU group can run profile {mode!r}. "
        "Check visible devices and the TP × Ulysses replica size."
    )


def single_config(gpus, visible, env, explicit):
    """Use spare VRAM for the encoder; hide streamed DiT copies under compute."""
    if any(int(env.get(key) or "1") != 1 for key in ("H3_TP", "H3_ULYSSES", "H3_DP")):
        raise ValueError("The single profile requires TP=Ulysses=DP=1")
    if explicit and len(visible) != 1:
        raise ValueError("The single profile requires exactly one H3_GPUS selector")
    candidates = [i for i in visible if gpus[i].family == "l20"
                  and gpus[i].memory_mib >= 46000
                  and gpus[i].free_mib >= 0.95 * gpus[i].memory_mib]
    if not candidates:
        raise ValueError("The single profile requires an idle 48 GB L20")
    ratio = float(env.get("H3_SINGLE_ENCODER_RESIDENT", "0.8"))
    if not 0 <= ratio < 1:
        raise ValueError("H3_SINGLE_ENCODER_RESIDENT must be a ratio in [0, 1)")
    config = profile_config("l20", "single", True, gpus, [candidates[0]], 1, 1, env)
    config["env"]["PYTORCH_ALLOC_CONF"] = (
        env.get("PYTORCH_ALLOC_CONF") or env.get("PYTORCH_CUDA_ALLOC_CONF")
        or "expandable_segments:True"
    )
    config["env"]["H3_OPT_EXACT_VAE_GRAPH"] = env.get("H3_OPT_EXACT_VAE_GRAPH", "1")
    config["env"]["H3_OPT_DEFAULT_QUALITY"] = env.get("H3_OPT_DEFAULT_QUALITY", "exact")
    config["flags"] = [
        "--encoder-parallel", "auto",
        "--component-residency", "all=layerwise-offload,video_vae=resident",
        "--layerwise-prefetch-size", "transformer=1,text_encoder=1",
        "--layerwise-resident-layers", f"text_encoder={ratio:g}",
        "--layerwise-residency-lifetime", "text_encoder=permanent",
    ]
    return config


def profile_config(family, mode, native, gpus, chosen, tp, ulysses, env):
    selected = [gpus[i] for i in chosen]
    quant = "native" if native else ("mxfp8-all" if family == "rtx5090" else "fp8-all")
    if mode == "fp8-dit":
        quant = mode
    attention = "fa" if native else ("sage_attn_3" if mode == "sage3" else "sage_attn")
    replicas = len(chosen) // (tp * ulysses)
    settings = {
        "H3_GPUS": ",".join(map(str, chosen)),
        "H3_TP": str(tp),
        "H3_ULYSSES": str(ulysses),
        "H3_DP": str(replicas),
        "H3_ATTENTION": env.get("H3_ATTENTION") or attention,
        "H3_CPUS": env["H3_CPUS"] if "H3_CPUS" in env else cpu_affinity(selected),
    }
    if family == "l20":
        # CUDA host allocations bypass small /dev/shm mounts for NCCL.
        settings["NCCL_CUMEM_HOST_ENABLE"] = env.get("NCCL_CUMEM_HOST_ENABLE", "1")
        if mode in ("auto", "accurate"):
            settings["H3_OPT_HOST_TP"] = env.get("H3_OPT_HOST_TP", "1")
            settings["H3_OPT_HOST_ULYSSES"] = env.get("H3_OPT_HOST_ULYSSES", "1")
            settings["H3_OPT_HOST_A2A_CHUNK_KB"] = env.get("H3_OPT_HOST_A2A_CHUNK_KB", "1536")
            settings["H3_OPT_HOST_A2A_MAX_CHUNKS"] = env.get("H3_OPT_HOST_A2A_MAX_CHUNKS", "16")
        if tp == 1 and not native:
            settings["H3_OPT_LAYERWISE_FP8_LOAD"] = env.get("H3_OPT_LAYERWISE_FP8_LOAD", "1")
        if not native:
            # cuBLASLt row-wise FP8 for the projections that narrow (out_proj, fc2).
            settings["H3_OPT_FP8_CUBLASLT"] = env.get("H3_OPT_FP8_CUBLASLT", "1")
        if ulysses > 1 and not native and settings["H3_ATTENTION"] == "sage_attn":
            # Quantize before the Ulysses exchange and move the bytes through
            # page-locked host memory. Each rank's shard must be 64-row aligned.
            settings["H3_OPT_QUANT_ULYSSES"] = env.get("H3_OPT_QUANT_ULYSSES", "1")
            if settings["H3_OPT_QUANT_ULYSSES"] == "1":
                settings["H3_OPT_SEQ_ALIGN"] = env.get("H3_OPT_SEQ_ALIGN", str(64 * ulysses))
    flags = []
    if family == "rtx5090" and mode in SINGLE_MODES:
        if selected[0].memory_mib < 32000:
            raise ValueError("Single-card RTX 5090 profiles require a 32 GB card")
        settings["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
        settings["NCCL_CUMEM_HOST_ENABLE"] = env.get("NCCL_CUMEM_HOST_ENABLE", "1")
        # Borrow compatible native CPU checkpoint tensors until the offload
        # manager consolidates them into pinned slabs, avoiding a full copy.
        settings["H3_OPT_ENCODER_MMAP"] = env.get("H3_OPT_ENCODER_MMAP", "1")
        settings["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        if mode in ("single", "single-reference", "single-int8-mlp"):
            resident_layers = 0 if mode == "single-reference" else int(
                env.get("H3_SINGLE_RESIDENT_LAYERS", "24")
            )
            if not 0 <= resident_layers <= 50:
                raise ValueError("H3_SINGLE_RESIDENT_LAYERS must be an integer from 0 to 50")
            settings["H3_ATTENTION"] = env.get("H3_ATTENTION") or "fa"
            flags.extend([
                "--component-residency", "all=resident,transformer=layerwise-offload,text_encoder=layerwise-offload",
                "--layerwise-prefetch-size", "transformer=2,text_encoder=2",
                "--layerwise-resident-layers", f"transformer={resident_layers}",
                "--layerwise-residency-lifetime", "transformer=permanent",
            ])
            if mode == "single-int8-mlp":
                quant = "int8-mlp"
                settings["H3_OPT_INT8_EXPERIMENTAL"] = "1"
                settings["H3_OPT_LAYERWISE_QUANT_LOAD"] = "1"
                settings["SGLANG_DIFFUSION_CONVROT_INT8_BACKEND"] = env.get(
                    "SGLANG_DIFFUSION_CONVROT_INT8_BACKEND", "jit"
                )
        else:
            quant = "mxfp8-dit" if mode == "single-fp8" else "mxfp8-all"
            settings["H3_ATTENTION"] = env.get("H3_ATTENTION") or (
                "fa" if mode == "single-fp8" else "sage_attn"
            )
            settings["H3_OPT_LAYERWISE_FP8_LOAD"] = "1"
            flags.extend([
                "--component-residency", "all=resident,text_encoder=layerwise-offload",
                "--layerwise-prefetch-size", "text_encoder=2",
            ])
    spans_nodes = len({gpu.numa_node for gpu in selected}) > 1
    if (
        (replicas > 1 or spans_nodes)
        and "H3_CPUS" not in env
        and all(gpu.numa_node is not None for gpu in selected)
    ):
        flags.extend(["--numa-node", *(str(gpu.numa_node) for gpu in selected)])
    if mode == "compiled":
        flags.extend(["--enable-torch-compile", "true", "--regional-compile", "true"])
    elif mode == "graph":
        settings["H3_OPT_GRAPH_EXPERIMENTAL"] = "1"
        flags.extend(["--enable-breakable-cuda-graph", "true",
                      "--warmup-mode", "server", "--warmup-steps", "6"])
    return {"hardware": family, "quantization": quant, "env": settings, "flags": flags}


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args and not args[0].startswith("-") else "auto"
    if mode not in {"auto", "accurate", "fast", "latency", "native", "throughput", "compiled", "graph",
                    "sage3", "fp8-dit", *SINGLE_MODES} or (
        args and args[0] in {"--help", "-h"}
    ):
        print("Usage: start_profile.sh [auto|accurate|fast|latency|native|throughput|compiled|graph|sage3|fp8-dit|single|single-reference|single-fp8|single-fast|single-int8-mlp]"
              " [--print-config | server flags]", file=sys.stderr)
        return 0 if args and args[0] in {"--help", "-h"} else 2
    try:
        gpus = inventory()
        if mode not in SINGLE_MODES:
            assign_uplinks(gpus, os.environ["H3_ROOT"], os.environ)
        config = plan(gpus, mode, os.environ)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"H3 profile selection failed: {exc}", file=sys.stderr)
        return 2
    if args == ["--print-config"]:
        print(json.dumps(config, indent=2))
        return 0
    root = Path(os.environ["H3_ROOT"])
    env = {**os.environ, **config["env"]}
    os.execve(root / "bin/h3ctl", [
        "h3ctl", "start", config["quantization"], *config["flags"], *args,
    ], env)


if __name__ == "__main__":
    sys.exit(main())
