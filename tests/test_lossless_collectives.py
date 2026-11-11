"""Real multi-process CUDA comparisons with NCCL (no model weights).

Run with four visible GPUs and the serving environment. Exercises overlapping
two-rank groups, byte-preserving BF16 all-to-all (including NaN bit patterns),
repeated buffer reuse, and two-rank summation against NCCL.
"""

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, port):
    from sglang.multimodal_gen.runtime.distributed.h3_lossless_collectives import (
        host_all_to_all,
        host_bf16_all_reduce,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    os.environ["H3_OPT_HOST_ULYSSES"] = "1"
    os.environ["H3_OPT_HOST_TP"] = "1"
    pairs = [dist.new_group(ranks) for ranks in ((0, 1), (2, 3), (0, 2), (1, 3))]
    mine = [pairs[0 if rank < 2 else 1], pairs[2 if rank % 2 == 0 else 3]]
    try:
        for group in (dist.group.WORLD, *mine, dist.group.WORLD, *mine):
            n = dist.get_world_size(group)
            for rows in (17, 4097, 137):
                # Compare raw bits so NaN payloads and signed zeros count.
                src = torch.randint(-32768, 32768, (n, rows, 128),
                                    device=rank, dtype=torch.int16).view(torch.bfloat16)
                expected, actual = torch.empty_like(src), torch.empty_like(src)
                dist.all_to_all_single(expected, src, group=group)
                assert host_all_to_all(src, actual, group)
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            for rows in ((16384, 32768) if n == 4 else (1024, 4096)):
                for _ in range(3):
                    src = torch.randn(rows, 256, device=rank, dtype=torch.bfloat16)
                    expected = src.clone()
                    dist.all_reduce(expected, group=group)
                    actual = host_bf16_all_reduce(src, group)
                    assert actual is not None
                    assert torch.equal(actual, expected)
        # Strided tensors retain NCCL.
        src = torch.ones(1024, 256, device=rank, dtype=torch.bfloat16)
        assert host_bf16_all_reduce(src[:, ::2], mine[0]) is None
        print(f"rank {rank}: NCCL byte equality and BF16 sums passed", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA GPUs")
class LosslessCollectiveTests(unittest.TestCase):
    def test_against_nccl(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(worker, args=(4, port), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
