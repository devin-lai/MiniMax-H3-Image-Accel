"""Verify that comparison stacks cannot silently pair different requests."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from compare_run_stack import CASES, inventory


class ComparisonInventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.variants = [{"run": name} for name in ("reference", "candidate")]
        self.data = {}
        for name in ("reference", "candidate"):
            directory = self.root / name
            directory.mkdir()
            rows = []
            for case in CASES if name == "reference" else ("product",):
                path = directory / f"{case}.png"
                Image.new("RGB", (32, 32), "navy").save(path)
                row = dict(label=name, case=case, trial=1, size="32x32", seed=71,
                           task="text-to-image", prompt="Matched fixture",
                           quality="lossless", output_compression=1,
                           wall_seconds=1.0, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                rows.append(row)
            self.data[name] = rows
            self.write(name)

    def write(self, name):
        (self.root / name / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.data[name]))

    def test_unmeasured_candidate_cases_remain_gaps(self):
        found = inventory(self.root, self.variants, "reference")
        self.assertIn(("candidate", "product"), found)
        self.assertNotIn(("candidate", "native-4k"), found)

    def test_rejects_a_different_seed(self):
        self.data["candidate"][0]["seed"] = 72
        self.write("candidate")
        with self.assertRaisesRegex(ValueError, "Mismatched benchmark request"):
            inventory(self.root, self.variants, "reference")

    def test_rejects_modified_pixels(self):
        Image.new("RGB", (32, 32), "red").save(self.root / "candidate/product.png")
        with self.assertRaisesRegex(ValueError, "invalid PNG hash"):
            inventory(self.root, self.variants, "reference")

    def test_rejects_duplicate_first_trials(self):
        self.data["candidate"].append(self.data["candidate"][0].copy())
        self.write("candidate")
        with self.assertRaisesRegex(ValueError, "Duplicate trial 1"):
            inventory(self.root, self.variants, "reference")

    def test_rejects_missing_request_metadata(self):
        for name in self.data:
            del self.data[name][0]["prompt"]
            self.write(name)
        with self.assertRaisesRegex(ValueError, "Incomplete request metadata"):
            inventory(self.root, self.variants, "reference")

    def test_quality_variants_require_an_explicit_declaration(self):
        for row in self.data["reference"]:
            row["quality"] = "exact"
        self.write("reference")
        with self.assertRaisesRegex(ValueError, "Mismatched benchmark request"):
            inventory(self.root, self.variants, "reference", cases=("product",))
        with self.assertRaisesRegex(ValueError, "declared quality"):
            inventory(self.root, self.variants, "reference", cases=("product",),
                      allow_quality_difference=True)
        self.variants[0]["quality"] = "exact"
        self.variants[1]["quality"] = "lossless"
        found = inventory(self.root, self.variants, "reference", cases=("product",),
                          allow_quality_difference=True)
        self.assertEqual(len(found), 2)


if __name__ == "__main__":
    unittest.main()
