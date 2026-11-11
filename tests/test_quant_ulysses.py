"""Quantize-before-exchange Ulysses attention against SageAttention2 and exact attention.

Runs on one SM89 GPU without model weights. The four ranks are simulated by
slicing one sequence into shards and concatenating what each rank would receive.
"""
import unittest

import torch
import torch.nn.functional as F

try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3_qulysses as Q
    READY = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 9)
except Exception:  # SageAttention2 is built per GPU family
    READY = False

W, R, H, C = 4, 1088, 56, 128  # ranks, rows per rank, heads, head width
T, HD = W * R, H // W
USED = T - 150  # trailing padding rows


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@unittest.skipUnless(READY, "needs an SM89 GPU with SageAttention2")
class QuantUlyssesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = "cuda"
        qkv = torch.randn(T, 3 * H * C, device=dev, dtype=torch.bfloat16)
        q, k, v = (t.view(T, H, C) for t in qkv.split(H * C, dim=-1))
        cls.q = q * 3
        cls.k = k + 0.7 * torch.randn(1, H, C, device=dev, dtype=torch.bfloat16)
        cls.v = v * (1 + torch.rand(1, H, C, device=dev, dtype=torch.bfloat16) * 4)
        cls.shards = [
            (cls.q[r * R:(r + 1) * R], cls.k[r * R:(r + 1) * R], cls.v[r * R:(r + 1) * R],
             min(max(USED - r * R, 0), R))
            for r in range(W)
        ]
        cls.stats = torch.stack([Q.local_stats(k, v, live) for _, k, v, live in cls.shards])
        value_range = cls.stats[:, 1].amax(0)
        parts = [Q.quantize_shard(q, k, v, cls.stats, live, USED) for q, k, v, live in cls.shards]
        cls.outs = []
        for d in range(W):
            cat = lambda i: torch.cat([p[i][d] for p in parts], dim=0)
            v8 = torch.cat([p[2][d].view(torch.uint8) for p in parts], dim=0).view(torch.float8_e4m3fn)
            cls.outs.append(Q.sage_attend(
                cat(0), cat(1), v8, cat(3), cat(4), value_range[d * HD:(d + 1) * HD], USED, C ** -0.5))

    def test_statistics_match_reference(self):
        expected = torch.stack([
            torch.stack([k[:live].float().sum(0), v[:live].float().abs().amax(0)])
            for _, k, v, live in self.shards
        ])
        torch.testing.assert_close(self.stats, expected, rtol=2e-3, atol=2e-2)

    def test_attention_error_matches_sage(self):
        q, k, v = (t[:USED] for t in (self.q, self.k, self.v))
        mine = torch.stack(self.outs, 1).view(USED, H, C).float()
        sage = sageattn_qk_int8_pv_fp8_cuda(
            q[None], k[None], v[None], tensor_layout="NHD", pv_accum_dtype="fp32+fp16")[0].float()
        exact = F.scaled_dot_product_attention(
            *(t.float().transpose(0, 1)[None] for t in (q, k, v)))[0].transpose(0, 1)
        self.assertLess(rel(mine, exact), 1.05 * rel(sage, exact))
        self.assertLess(rel(mine, exact), 0.05)

    def test_power_of_two_row_merge(self):
        quant = [Q.row_quantize(o, T) for o in self.outs]
        full = torch.stack(self.outs, 1).view(USED, H * C).float()
        for r in range(W):
            staged = torch.stack([c.view(torch.uint8)[r * R:(r + 1) * R] for c, _ in quant])
            scales = torch.stack([s[r * R:(r + 1) * R] for _, s in quant]).contiguous()
            merged, common = Q.row_merge(staged.contiguous().view(torch.float8_e4m3fn), scales)
            live = min(max(USED - r * R, 0), R)
            expected = full[r * R:r * R + live]
            ref_q, ref_s = sglang_per_token_quant_fp8(expected.to(torch.bfloat16).contiguous())
            error = rel(merged.float()[:live] * common[:live], expected)
            self.assertLess(error, 1.1 * rel(ref_q.float() * ref_s, expected))
            self.assertTrue(bool((merged.view(torch.uint8)[live:] == 0).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
