"""CPU-only checks for the exported comparison data and matching rules."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import compare_quality
import export_accuracy
import export_5090_hashes


class AccuracyExportTests(unittest.TestCase):
    def test_tiled_metrics_match_full_image_implementation(self):
        rng = np.random.default_rng(71)
        reference = rng.integers(0, 256, size=(83, 67, 3), dtype=np.uint8)
        candidate = rng.integers(0, 256, size=reference.shape, dtype=np.uint8)
        psnr, ssim, mae = export_accuracy.compare_arrays(reference, candidate, tile_rows=13)
        a, b = reference.astype(np.float64), candidate.astype(np.float64)
        self.assertAlmostEqual(psnr, compare_quality.psnr(a, b), places=10)
        self.assertAlmostEqual(ssim, compare_quality.ssim(a, b), places=10)
        self.assertAlmostEqual(mae, np.mean(np.abs(a - b)), places=10)

    def test_identical_images_and_invalid_sizes(self):
        image = np.full((16, 17, 3), 128, dtype=np.uint8)
        psnr, ssim, mae = export_accuracy.compare_arrays(image, image, tile_rows=3)
        self.assertEqual(psnr, float("inf"))
        self.assertAlmostEqual(ssim, 1.0)
        self.assertEqual(mae, 0.0)
        with self.assertRaises(ValueError):
            export_accuracy.compare_arrays(image, image[:7])

    def test_rejects_unmatched_reference_or_prompt(self):
        reference = {"label": "base", "case": "edit", "size": "2x2", "seed": 1,
                     "task": "edit", "prompt": "edit", "reference_sha256": "first"}
        candidate = dict(reference, label="candidate")
        export_accuracy.matched(reference, candidate)
        for field in ("seed", "prompt", "reference_sha256"):
            with self.subTest(field=field):
                changed = dict(candidate, **{field: "different"})
                with self.assertRaisesRegex(ValueError, field):
                    export_accuracy.matched(reference, changed)

    def test_image_sheet_groups_cover_every_l20_scheme(self):
        covered = {item for _, labels in export_accuracy.GROUPS.values() for item in labels}
        self.assertEqual(covered, {export_accuracy.REFERENCE, *(row[0] for row in export_accuracy.SCHEMES)})

    def test_scores_must_belong_to_the_pngs_being_charted(self):
        label, case = "l20q-sp4", "product"
        found = {(export_accuracy.REFERENCE, case): (None, {"sha256": "ref"}),
                 (label, case): (None, {"sha256": "candidate"})}
        row = {"candidate": label, "case": case, "reference_png_sha256": "ref",
               "candidate_png_sha256": "candidate"}
        export_accuracy.validate_metrics_for_images([row], found)
        with self.assertRaisesRegex(ValueError, "hashes"):
            export_accuracy.validate_metrics_for_images([dict(row, candidate_png_sha256="stale")], found)
        with self.assertRaisesRegex(ValueError, "coverage"):
            export_accuracy.validate_metrics_for_images([], found)

    def test_5090_hashes_use_the_named_reference(self):
        labels = {label for candidate, reference, _ in export_5090_hashes.SCHEMES
                  for label in (candidate, reference)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in labels:
                path = root / label / "results.jsonl"
                path.parent.mkdir(parents=True)
                sha = "b" * 64 if label == "runs/r2-prefix-kv" else "a" * 64
                records = [
                    {"case": case, "trial": 1, "size": "2048x2048", "seed": 731,
                     "prompt": "A blue circle", "quality": "exact", "sha256": sha}
                    for case, _ in export_5090_hashes.CASES
                ]
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
            rows = export_5090_hashes.comparison_rows(root)
        prefix = next(r for r in rows if r["candidate"] == "runs/r2-prefix-kv")
        self.assertEqual(prefix["reference"], "runs/r2-prefix-full")
        self.assertEqual(prefix["png_bytes"], "different")
        gpu_six = [r for r in rows if r["candidate"] == "portable/gpu-output-six-v2"]
        self.assertEqual(len(gpu_six), 4)
        self.assertTrue(all(r["reference"] == "runs/final-six" and r["png_bytes"] == "same" for r in gpu_six))


if __name__ == "__main__":
    unittest.main()
