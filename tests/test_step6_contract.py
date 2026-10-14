"""Regression tests for the step-only classifier contract."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step_config import ORIGAMI_STEPS, STEP_CLASSES, STEP_NAMES  # noqa: E402


class Step6ContractTests(unittest.TestCase):
    checkpoint_path = ROOT / "checkpoints" / "origami_step6_best.pth"

    def test_canonical_classes_and_names(self):
        self.assertEqual(
            STEP_CLASSES,
            ("Chk0_NonOrigami", "Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        )
        self.assertEqual(ORIGAMI_STEPS, ("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"))
        self.assertEqual(STEP_NAMES["Chk0_NonOrigami"], "Non-Origami / Background")
        self.assertEqual(STEP_NAMES["Chk4"], "Neck & Tail Setup")
        self.assertEqual(STEP_NAMES["Chk5"], "Head Fold")

    def test_checkpoint_schema_contract(self):
        if self.checkpoint_path.is_file():
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        else:
            checkpoint = {
                "task": "step-classification",
                "classes": list(STEP_CLASSES),
                "model_name": "mobilenet_v2",
            }
        self.assertEqual(checkpoint.get("task"), "step-classification")
        self.assertEqual(checkpoint.get("classes"), list(STEP_CLASSES))

    def test_api_returns_step_without_fold_status(self):
        import app.app as app_module

        dummy_jpeg = cv2.imencode(".jpg", np.full((100, 100, 3), 150, dtype=np.uint8))[1].tobytes()

        mock_verifier = MagicMock()
        mock_verifier.classes = list(STEP_CLASSES)
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "frame_ready": True,
            "recognized": True,
            "prediction": "Chk1",
            "step": "Chk1",
            "step_number": 1,
            "step_name": STEP_NAMES["Chk1"],
            "confidence": 95.0,
            "top_gap": 50.0,
            "recognition_thresholds": {"min_confidence": 55.0, "min_top_gap": 15.0},
            "roi": {"applied": True, "method": "white_paper", "bbox": [10, 10, 90, 90]},
            "top_predictions": [{"step": "Chk1", "confidence": 95.0}],
            "metrics": {},
        }
        mock_verifier.verify_bytes.return_value = mock_result

        with patch.object(app_module.state, "get_verifier", return_value=mock_verifier):
            with TestClient(app_module.app) as client:
                response = client.post(
                    "/api/predict?figure=crane",
                    files={"file": ("frame.jpg", dummy_jpeg, "image/jpeg")},
                )
                model_info = client.get("/api/model-info?figure=crane").json()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["step"], STEP_CLASSES)
        self.assertIn("recognized", payload)
        self.assertIn("top_gap", payload)
        self.assertIn("recognition_thresholds", payload)
        self.assertNotIn("status", payload)
        self.assertNotIn("same_step_probabilities", payload)

        self.assertEqual(model_info["task"], "step-classification")
        self.assertEqual(model_info["figure"], "crane")
        self.assertEqual(model_info["classes"], list(STEP_CLASSES))
        self.assertIn("model_loaded", model_info)
        self.assertIn("available_figures", model_info)


if __name__ == "__main__":
    unittest.main()
