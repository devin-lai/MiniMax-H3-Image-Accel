"""Correctness checks for the experimental H3 optimizer (no model weights)."""
import contextlib
import unittest
from unittest.mock import patch
import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_opt import H3ImageOptState, validate_h3_opt, fuse_velocity_heads
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import denoise_loop


class PolicyTests(unittest.TestCase):
    def test_merged_heads_equal_shared_hidden_euler_updates(self):
        hidden = torch.randn(3, 5)
        weight, bias = torch.randn(2, 4, 5), torch.randn(2, 4)
        deltas = (.1, .3)
        w, b = fuse_velocity_heads(weight, bias, deltas)
        expected = sum(dt * torch.nn.functional.linear(hidden, wi, bi) for dt, wi, bi in zip(deltas, weight, bias))
        actual = sum(deltas) * torch.nn.functional.linear(hidden, w, b)
        torch.testing.assert_close(actual, expected)

    def test_invalid_head_schedules(self):
        for value in ({"steps": [1, 3, 5]}, {"steps": [0, 5, 2]}, {"steps": [0, True, 5]},
                      {"tea_threshold": float("nan")}, {"reuse_steps": [5]}, {"unknown": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_h3_opt(value)

    def test_cache_survives_inplace_blocks_and_forces_final_pass(self):
        state = H3ImageOptState(
            {"tea_threshold": 100.0, "tea_warmup": 1}, list(range(6)), 0, 4
        )
        hidden = torch.ones(4, 8)
        full, _ = state.begin(hidden, 0)
        self.assertTrue(full)
        hidden.add_(2)
        state.finish(hidden)
        full, reused = state.begin(torch.full_like(hidden, 5), 1)
        self.assertFalse(full)
        torch.testing.assert_close(reused, torch.full_like(hidden, 7))
        full, _ = state.begin(torch.ones_like(hidden), 5)
        self.assertTrue(full)
        state.release()
        self.assertIsNone(state.residual)

    def test_sparse_schedule_uses_original_heads_and_sigma_distances(self):
        count, width = 4, 96
        positive = type("Branch", (), {
            "packed": {"update_mask": torch.ones(count, dtype=torch.bool), "text_pos": torch.arange(2)},
            "text_embeddings": torch.zeros(2, 5120), "token_tags": torch.zeros(8), "attn_metadata": None,
        })()

        class FakeBranch:
            def __init__(self, **kwargs):
                self.img_pos_dev = torch.arange(count)
                self.local_row_slice = slice(0, 8)
            def prepare_timestep_plan(self, **kwargs):
                return [(torch.tensor([t]),) for t in kwargs["video_timesteps"]]
            def forward_kwargs(self, **kwargs):
                return {"x": torch.zeros(1, 8, width)}

        class FakeModel:
            adaln_cache = None
            def prepare_adaln_plans(self, plan): return None

        seen = []
        def run(model, kwargs, step):
            seen.append(step)
            return torch.full((count, width), float(step + 1)), torch.empty(0)
        grid = [1, .96, .88, .75, .53, .23, 0]
        with patch.object(denoise_loop, "MiniMaxH3DenoiseBranch", FakeBranch), patch.object(
            denoise_loop, "set_forward_context", lambda **kw: contextlib.nullcontext()
        ):
            out = denoise_loop.minimax_h3_image_denoise_loop(
                model=FakeModel(), model_forward=run, positive=positive,
                initial_video_rows=torch.zeros(count, width), keyframe_cond_rows=None,
                sigmas=grid, device=torch.device("cpu"), opt_config={"steps": [0, 2, 5]},
            )
        self.assertEqual(seen, [0, 2, 5])
        expected = (grid[0] - grid[2]) + 3 * (grid[2] - grid[5]) + 6 * grid[5]
        torch.testing.assert_close(out, torch.full_like(out, expected))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PrefixKVTests(unittest.TestCase):
    def test_gpu_pixel_packing_matches_native_cpu_quantization(self):
        from sglang.multimodal_gen.runtime.models.vaes.h3_image_output import h3_image_uint8_host
        for dtype in (torch.float32, torch.bfloat16):
            values = (torch.randn(2, 3, 31, 37, device="cuda") * .7 + .5).to(dtype)
            values = values.transpose(2, 3)
            expected = (values.float().cpu() * 255).clamp(0, 255).to(torch.uint8)
            actual = h3_image_uint8_host(values)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(actual.permute(0, 2, 3, 1).is_contiguous())

    def test_native_qwen_prefix_continuation_matches_full_prefill(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLTextModel
        from sglang.multimodal_gen.runtime.models.encoders.h3_prefix_cache import H3PrefixKVCache
        from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
        from sglang.multimodal_gen.runtime.server_args import set_global_server_args
        from types import SimpleNamespace
        set_global_server_args(SimpleNamespace(attention_backend="torch_sdpa", enable_attention_backend_autotune=False))
        cfg = Qwen3VLTextConfig(
            vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            rope_scaling={"rope_type": "default", "mrope_section": [8, 12, 12], "mrope_interleaved": True},
        )
        # FP32 validates causal alignment independently of BF16 kernel rounding.
        language = Qwen3VLTextModel(cfg).cuda().float().eval()
        with torch.no_grad():
            for name, parameter in language.named_parameters():
                parameter.fill_(1) if "norm" in name and parameter.ndim == 1 else parameter.uniform_(-.04, .04)
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.language_model = language
            def forward(self, **kwargs): return self.language_model(**kwargs)
        model = Model()
        gen = torch.Generator().manual_seed(419)
        old = torch.randint(0, 255, (1, 192), generator=gen)
        new = torch.cat((old[:, :128], torch.randint(0, 255, (1, 48), generator=gen)), dim=1)
        cache = H3PrefixKVCache(min_prefix_tokens=32, max_bytes=8 * 1024**2)
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            cache.encode(model, old, old.cuda())
            actual = cache.encode(model, new, new.cuda())
            expected = model(input_ids=new.cuda(), use_cache=False, return_dict=True).last_hidden_state
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        self.assertLess(float((actual.float() - expected.float()).square().mean().sqrt()), 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
