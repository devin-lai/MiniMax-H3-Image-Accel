"""Accuracy gates reject altered images, missing evidence and unmatched inputs."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from validate_accuracy import audit


class AccuracyGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_image(self, label, value, seed=42):
        directory = self.root / label
        directory.mkdir(exist_ok=True)
        path = directory / "sample.png"
        Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(path)
        record = dict(case="sample", trial=1, size="16x16", seed=seed,
                      prompt="fixture", task="text-to-image", quality="lossless",
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (directory / "results.jsonl").write_text(json.dumps(record) + "\n")
        return directory

    def test_identical_and_degraded_images(self):
        ref = self.run_image("ref", 100)
        same = self.run_image("same", 100)
        changed = self.run_image("changed", 160)
        self.assertTrue(audit(ref, same, ["sample"], identical=True)[0]["passed"])
        result = audit(ref, changed, ["sample"])[0]
        self.assertFalse(result["passed"])
        self.assertLess(result["psnr_db"], 40)
        self.assertFalse(result["pixel_identical"])

    def test_missing_or_different_seed_fails(self):
        ref = self.run_image("ref", 100)
        candidate = self.run_image("candidate", 100, seed=43)
        with self.assertRaisesRegex(ValueError, "Mismatched seed"):
            audit(ref, candidate, ["sample"])
        with self.assertRaises(KeyError):
            audit(ref, candidate, ["unmeasured"])

    def test_modified_png_fails(self):
        ref = self.run_image("ref", 100)
        candidate = self.run_image("candidate", 100)
        Image.new("RGB", (16, 16), "white").save(candidate / "sample.png")
        with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
            audit(ref, candidate, ["sample"])


if __name__ == "__main__":
    unittest.main()
