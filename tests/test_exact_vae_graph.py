"""CUDA equality, output lifetime and invalidation for exact VAE replay."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/sglang/python"))
import torch
from sglang.multimodal_gen.runtime.models.vaes.h3_exact_graph import (
    H3ExactDecoderGraph,
    _same_bits,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(
            32, 32, bias=False, device="cuda", dtype=torch.bfloat16
        )

    def forward(self, hidden, rope):
        return hidden + self.proj(hidden) + rope[0]


class ByteCheckTests(unittest.TestCase):
    def test_signed_zero_is_not_byte_equal(self):
        self.assertFalse(_same_bits(torch.tensor([0.0]), torch.tensor([-0.0])))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ExactGraphTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (8, 9):
            self.skipTest("L20/Ada path")
        self.owner = torch.nn.Module().eval()
        self.owner.decode_fast_path_gate = SimpleNamespace(enabled=False)
        self.owner.layerwise_offload_managers = []
        self.blocks = torch.nn.ModuleList([Block(), Block()]).eval()
        self.cache = H3ExactDecoderGraph(self.owner)
        self.rope = (torch.randn(1, 1, 32, device="cuda", dtype=torch.bfloat16),)
        self.x = torch.randn(1, 64, 32, device="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def test_replays_keep_independent_outputs(self):
        self.cache.maybe_run(self.blocks, self.x, self.rope)
        first = self.cache.maybe_run(self.blocks, self.x + 1, self.rope)
        saved = first.clone()
        for index in range(2, 5):
            x = self.x + index
            expected = self.cache.eager(self.blocks, x, self.rope)
            actual = self.cache.maybe_run(self.blocks, x, self.rope)
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(first, saved))
        self.assertEqual(self.cache.captures, 1)
        self.assertEqual(self.cache.replays, 4)

    @torch.no_grad()
    def test_weight_replacement_does_not_replay_stale_addresses(self):
        self.cache.maybe_run(self.blocks, self.x, self.rope)
        self.blocks[0].proj.weight = torch.nn.Parameter(
            self.blocks[0].proj.weight.clone() * 0.5
        )
        expected = self.cache.eager(self.blocks, self.x, self.rope)
        actual = self.cache.maybe_run(self.blocks, self.x, self.rope)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(self.cache.captures, 2)
        expected = self.cache.eager(self.blocks, self.x * 2, self.rope)
        actual = self.cache.maybe_run(self.blocks, self.x * 2, self.rope)
        self.assertTrue(torch.equal(actual, expected))

    @torch.no_grad()
    def test_offload_and_fast_quality_bypass_capture(self):
        self.owner.decode_fast_path_gate.enabled = True
        self.assertIsNone(self.cache.maybe_run(self.blocks, self.x, self.rope))
        self.owner.decode_fast_path_gate.enabled = False
        self.owner.layerwise_offload_managers = [SimpleNamespace(enabled=True)]
        self.assertIsNone(self.cache.maybe_run(self.blocks, self.x, self.rope))
        self.assertEqual(self.cache.captures, 0)

    @torch.no_grad()
    def test_rope_tables_and_shapes_do_not_alias_a_capture(self):
        self.cache.maybe_run(self.blocks, self.x, self.rope)
        new_rope = (self.rope[0] + 0.1,)
        expected = self.cache.eager(self.blocks, self.x * 2, new_rope)
        actual = self.cache.maybe_run(self.blocks, self.x * 2, new_rope)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(self.cache.captures, 2)
        x = self.x[:, :32].clone(memory_format=torch.contiguous_format)
        self.cache.maybe_run(self.blocks, x, new_rope)
        expected = self.cache.eager(self.blocks, x * 2, new_rope)
        actual = self.cache.maybe_run(self.blocks, x * 2, new_rope)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(self.cache.captures, 3)
        self.assertLessEqual(len(self.cache.entries), 2)


if __name__ == "__main__":
    unittest.main()
