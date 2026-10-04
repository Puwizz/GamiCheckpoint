"""Regression tests for the step-only classifier contract."""

import sys
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step_config import ORIGAMI_STEPS, STEP_CLASSES, STEP_NAMES  # noqa: E402


class Step6ContractTests(unittest.TestCase):
    checkpoint_path = ROOT / "checkpoints" / "origami_step6_best.pth"
    test_data_path = ROOT / "data" / "step6" / "processed" / "test" / "Chk1"

    def test_canonical_classes_and_names(self):
        self.assertEqual(
            STEP_CLASSES,
            ("Chk0_NonOrigami", "Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        )
        self.assertEqual(ORIGAMI_STEPS, ("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"))
        self.assertEqual(STEP_NAMES["Chk0_NonOrigami"], "Non-Origami / Background")
        self.assertEqual(STEP_NAMES["Chk4"], "Neck & Tail Setup")
        self.assertEqual(STEP_NAMES["Chk5"], "Head Fold")

    @unittest.skipUnless(checkpoint_path.is_file(), "local step6 checkpoint is not available")
    def test_checkpoint_is_step_only(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint.get("task"), "step-classification")
        self.assertEqual(checkpoint.get("classes"), list(STEP_CLASSES))

    @unittest.skipUnless(
        checkpoint_path.is_file() and test_data_path.is_dir(),
        "local checkpoint or bootstrap test data is not available",
    )
    def test_api_returns_step_without_fold_status(self):
        from app.app import app

        image_path = next(self.test_data_path.glob("*"))
        with TestClient(app) as client:
            response = client.post(
                "/api/predict",
                files={"file": (image_path.name, image_path.read_bytes(), "image/jpeg")},
            )
            model_info = client.get("/api/model-info").json()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["step"], STEP_CLASSES)
        self.assertIn("recognized", payload)
        self.assertIn("top_gap", payload)
        self.assertIn("recognition_thresholds", payload)
        self.assertNotIn("status", payload)
        self.assertNotIn("same_step_probabilities", payload)
        self.assertEqual(model_info["task"], "step-classification")
        self.assertEqual(model_info["classes"], list(STEP_CLASSES))
        self.assertIn(model_info["model_path"], ("origami_step6_best.pth", "origami_step6_final.pth"))


if __name__ == "__main__":
    unittest.main()
