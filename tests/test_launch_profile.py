"""CPU-only checks for selecting a launch profile on both GPU families."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from launch_profile import GPU, plan


def devices(name, capability, count, memory):
    return [
        GPU(i, f"GPU-{i}", name, str(i), capability, memory, memory,
            "0-7" if i < 4 else "8-15", i // 4)
        for i in range(count)
    ]


class LaunchProfileTests(unittest.TestCase):
    def test_single_uses_one_l20_and_native_math(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        config = plan(gpus, "single", {"H3_GPUS": "5"})
        self.assertEqual(config["quantization"], "native")
        self.assertEqual(config["env"]["H3_GPUS"], "5")
        for key in ("H3_TP", "H3_ULYSSES", "H3_DP"):
            self.assertEqual(config["env"][key], "1")
        self.assertEqual(config["env"]["H3_ATTENTION"], "fa")
        self.assertNotIn("H3_OPT_QUANT_ULYSSES", config["env"])
        self.assertIn("text_encoder=permanent", config["flags"])
        self.assertIn("all=layerwise-offload,video_vae=resident", config["flags"])
        self.assertEqual(config["env"]["H3_OPT_EXACT_VAE_GRAPH"], "1")
        self.assertEqual(config["env"]["H3_OPT_DEFAULT_QUALITY"], "exact")

    def test_auto_falls_back_to_single_on_one_visible_l20(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        config = plan(gpus, "auto", {"CUDA_VISIBLE_DEVICES": "GPU-5"})
        self.assertEqual(config["env"]["H3_GPUS"], "5")
        self.assertEqual(config["env"]["H3_TP"], "1")
        self.assertIn("text_encoder=permanent", config["flags"])

    def test_single_rejects_multiple_devices_or_parallel_ranks(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        for env in ({"H3_GPUS": "4,5"}, {"H3_TP": "2"},
                    {"H3_ULYSSES": "2"}, {"H3_DP": "2"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                plan(gpus, "single", env)

    def test_single_requires_free_memory_and_supports_a_smaller_cache(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        for gpu in gpus:
            gpu.free_mib = 42000
        with self.assertRaisesRegex(ValueError, "idle 48 GB L20"):
            plan(gpus, "single", {})
        gpus[3].free_mib = 49000
        config = plan(gpus, "single", {"H3_SINGLE_ENCODER_RESIDENT": "0.6"})
        self.assertEqual(config["env"]["H3_GPUS"], "3")
        self.assertIn("text_encoder=0.6", config["flags"])
        with self.assertRaisesRegex(ValueError, "ratio"):
            plan(gpus, "single", {"H3_SINGLE_ENCODER_RESIDENT": "1"})

    def test_single_card_profiles_select_exactly_one_5090(self):
        gpus = devices("NVIDIA GeForce RTX 5090", "12.0", 8, 32607)
        gpus[0].utilization = 100
        gpus[1].free_mib = 4000
        for mode in ("single", "single-reference", "single-fp8", "single-fast", "single-int8-mlp"):
            with self.subTest(mode=mode):
                config = plan(gpus, mode, {})
                self.assertEqual(config["env"]["H3_GPUS"], "2")
                self.assertEqual(config["env"]["H3_TP"], "1")
                self.assertEqual(config["env"]["H3_ULYSSES"], "1")
                self.assertEqual(config["env"]["H3_DP"], "1")

    def test_single_preserves_bf16_and_allows_an_explicit_idle_card(self):
        config = plan(devices("NVIDIA GeForce RTX 5090", "12.0", 8, 32607),
                      "single", {"H3_GPUS": "5", "H3_SINGLE_RESIDENT_LAYERS": "24"})
        self.assertEqual(config["quantization"], "native")
        self.assertEqual(config["env"]["H3_ATTENTION"], "fa")
        self.assertEqual(config["env"]["H3_GPUS"], "5")
        self.assertIn("transformer=24", config["flags"])
        self.assertIn("transformer=permanent", config["flags"])

    def test_single_fp8_keeps_the_encoder_and_attention_in_bf16(self):
        config = plan(devices("NVIDIA GeForce RTX 5090", "12.0", 1, 32607), "single-fp8", {})
        self.assertEqual(config["quantization"], "mxfp8-dit")
        self.assertEqual(config["env"]["H3_ATTENTION"], "fa")
        self.assertEqual(config["env"]["H3_OPT_LAYERWISE_FP8_LOAD"], "1")
        self.assertIn("all=resident,text_encoder=layerwise-offload", config["flags"])

    def test_single_fast_is_an_explicit_precision_tradeoff(self):
        config = plan(devices("NVIDIA GeForce RTX 5090", "12.0", 1, 32607), "single-fast", {})
        self.assertEqual(config["quantization"], "mxfp8-all")
        self.assertEqual(config["env"]["H3_ATTENTION"], "sage_attn")

    def test_single_int8_only_quantizes_the_dit_mlp(self):
        config = plan(devices("NVIDIA GeForce RTX 5090", "12.0", 1, 32607), "single-int8-mlp", {})
        self.assertEqual(config["quantization"], "int8-mlp")
        self.assertEqual(config["env"]["H3_ATTENTION"], "fa")
        self.assertEqual(config["env"]["H3_OPT_INT8_EXPERIMENTAL"], "1")
        self.assertEqual(config["env"]["SGLANG_DIFFUSION_CONVROT_INT8_BACKEND"], "jit")
        self.assertIn("transformer=24", config["flags"])

    def test_single_rejects_multiple_cards_or_parallel_ranks(self):
        gpus = devices("NVIDIA GeForce RTX 5090", "12.0", 8, 32607)
        for env in ({"H3_GPUS": "4,5"}, {"H3_TP": "2"}, {"H3_ULYSSES": "2"}, {"H3_DP": "2"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                plan(gpus, "single", env)

    def test_single_rejects_untested_hardware(self):
        with self.assertRaises(ValueError):
            plan(devices("NVIDIA A100", "8.0", 1, 81920), "single", {})

    def test_rtx_specific_single_profiles_reject_l20(self):
        for mode in ("single-reference", "single-fp8", "single-fast", "single-int8-mlp"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plan(devices("NVIDIA L20", "8.9", 1, 49140), mode, {})

    def test_single_honors_hardware_selection_in_a_mixed_inventory(self):
        gpus = [*devices("NVIDIA L20", "8.9", 1, 49140),
                *devices("NVIDIA GeForce RTX 5090", "12.0", 1, 32607)]
        l20 = plan(gpus, "single", {"H3_GPUS": "0"})
        rtx = plan(gpus, "single", {"H3_GPUS": "1"})
        self.assertEqual(l20["hardware"], "l20")
        self.assertEqual(l20["env"]["H3_OPT_EXACT_VAE_GRAPH"], "1")
        self.assertNotIn("H3_OPT_ENCODER_MMAP", l20["env"])
        self.assertEqual(rtx["hardware"], "rtx5090")
        self.assertEqual(rtx["env"]["H3_OPT_ENCODER_MMAP"], "1")
        self.assertNotIn("H3_OPT_EXACT_VAE_GRAPH", rtx["env"])

    def test_5090_fast_retains_mxfp8_and_sage2(self):
        config = plan(devices("NVIDIA GeForce RTX 5090", "12.0", 8, 32607), "fast", {})
        self.assertEqual(config["quantization"], "mxfp8-all")
        self.assertEqual(config["env"]["H3_ATTENTION"], "sage_attn")
        self.assertEqual(config["env"]["H3_GPUS"], "0,1,2,3")

    def test_l20_uses_fp8_and_prefers_one_numa_group(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        gpus[0].free_mib = 5000
        config = plan(gpus, "fast", {})
        self.assertEqual(config["quantization"], "fp8-all")
        self.assertEqual(config["env"]["H3_GPUS"], "4,5,6,7")
        self.assertEqual(config["env"]["H3_TP"], "1")
        self.assertEqual(config["env"]["H3_ULYSSES"], "4")
        self.assertEqual(config["env"]["H3_OPT_LAYERWISE_FP8_LOAD"], "1")
        self.assertEqual(config["env"]["H3_OPT_QUANT_ULYSSES"], "1")
        self.assertEqual(config["env"]["H3_OPT_SEQ_ALIGN"], "256")
        self.assertEqual(config["flags"], [])

    def test_visible_devices_are_preserved(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        config = plan(gpus, "native", {"CUDA_VISIBLE_DEVICES": "GPU-5,GPU-4,GPU-6,GPU-7"})
        self.assertEqual(config["quantization"], "native")
        self.assertEqual(config["env"]["H3_GPUS"], "5,4,6,7")

    def test_l20_rejects_blackwell_attention(self):
        gpus = devices("NVIDIA L20", "8.9", 4, 49140)
        with self.assertRaisesRegex(ValueError, "No homogeneous CUDA GPU group"):
            plan(gpus, "sage3", {})

    def test_explicit_replica_size_matches_parallelism(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        with self.assertRaisesRegex(ValueError, "TP × Ulysses"):
            plan(gpus, "fast", {"H3_GPUS": "4,5,6"})
        config = plan(gpus, "fast", {"H3_GPUS": "4,5"})
        self.assertEqual(config["env"]["H3_GPUS"], "4,5")
        self.assertEqual(config["env"]["H3_ULYSSES"], "2")

    def test_throughput_groups_pairs_by_numa(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        gpus[0].free_mib = 5000
        config = plan(gpus, "throughput", {})
        self.assertEqual(config["env"]["H3_GPUS"], "1,2,4,5,6,7")
        self.assertEqual(config["env"]["H3_DP"], "3")
        self.assertEqual(config["flags"], ["--numa-node", "0", "0", "1", "1", "1", "1"])

    def test_l20_spreads_ranks_over_pcie_uplinks(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        for gpu in gpus:
            gpu.uplink = gpu.index // 2
        config = plan(gpus, "fast", {})
        self.assertEqual(config["env"]["H3_GPUS"], "0,2,4,6")
        self.assertEqual(config["flags"], ["--numa-node", "0", "0", "1", "1"])
        config = plan(gpus, "throughput", {})
        self.assertEqual(config["env"]["H3_GPUS"], "0,2,1,3,4,6,5,7")
        self.assertEqual(config["env"]["H3_OPT_SEQ_ALIGN"], "128")
        config = plan(gpus, "latency", {})
        self.assertEqual(config["env"]["H3_GPUS"], "0,1,2,3,4,5,6,7")
        self.assertEqual(config["env"]["H3_ULYSSES"], "8")
        self.assertEqual(config["env"]["H3_OPT_SEQ_ALIGN"], "512")

    def test_native_and_5090_keep_the_stock_exchange(self):
        gpus = devices("NVIDIA L20", "8.9", 8, 49140)
        self.assertNotIn("H3_OPT_QUANT_ULYSSES", plan(gpus, "native", {})["env"])
        config = plan(devices("NVIDIA GeForce RTX 5090", "12.0", 8, 32607), "auto", {})
        self.assertNotIn("H3_OPT_QUANT_ULYSSES", config["env"])

    def test_accuracy_is_the_default_on_both_families(self):
        for name, capability, memory in (("NVIDIA L20", "8.9", 49140),
                                          ("NVIDIA GeForce RTX 5090", "12.0", 32607)):
            for mode in ("auto", "accurate"):
                with self.subTest(gpu=name, mode=mode):
                    config = plan(devices(name, capability, 8, memory), mode, {})
                    self.assertEqual(config["quantization"], "native")
                    self.assertEqual(config["env"]["H3_ATTENTION"], "fa")
                    self.assertEqual(config["env"]["H3_TP"], "4")
                    self.assertEqual(config["env"]["H3_ULYSSES"], "1")
                    self.assertNotIn("H3_OPT_QUANT_ULYSSES", config["env"])
                    if name == "NVIDIA L20":
                        self.assertEqual(config["env"]["H3_OPT_HOST_TP"], "1")
                        self.assertEqual(config["env"]["H3_OPT_HOST_A2A_MAX_CHUNKS"], "16")

    def test_host_transport_can_be_disabled(self):
        config = plan(devices("NVIDIA L20", "8.9", 8, 49140), "accurate",
                      {"H3_OPT_HOST_TP": "0"})
        self.assertEqual(config["env"]["H3_OPT_HOST_TP"], "0")


if __name__ == "__main__":
    unittest.main()
