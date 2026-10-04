"""Load a candidate checkpoint through the real API and predict one image."""

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step_config import STEP_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test a GamiSenpai checkpoint")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-class", choices=STEP_CLASSES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    image_path = args.image.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.environ["GAMISENPAI_MODEL_PATH"] = str(model_path)
    from fastapi.testclient import TestClient
    from app.app import app

    content_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"

    with TestClient(app) as client:
        model_info_res = client.get("/api/model-info")
        model_info_res.raise_for_status()
        model_info = model_info_res.json()

        if not model_info.get("model_loaded"):
            raise RuntimeError(
                f"API reported model failed to load from {model_path}: {model_info}"
            )

        with image_path.open("rb") as image_file:
            response = client.post(
                "/api/predict",
                files={"file": (image_path.name, image_file, content_type)},
            )
        response.raise_for_status()
        prediction = response.json()

    result = {
        "model": str(model_path),
        "image": str(image_path),
        "expected_class": args.expected_class,
        "model_info": model_info,
        "prediction": prediction,
    }

    if "error" in prediction:
        raise RuntimeError(f"API returned error: {prediction['error']}")

    if not prediction.get("frame_ready", True):
        reason = prediction.get("reason", "unknown")
        metrics = prediction.get("metrics", {})
        raise RuntimeError(
            f"API rejected frame as not ready for inference: reason='{reason}', metrics={metrics}"
        )

    actual_pred = prediction.get("prediction")
    if not actual_pred:
        raise RuntimeError(f"API did not return a valid prediction field: {prediction}")

    if args.expected_class and actual_pred != args.expected_class:
        conf = prediction.get("confidence", 0.0)
        raise AssertionError(
            f"Expected class '{args.expected_class}', but API predicted '{actual_pred}' "
            f"with {conf:.1f}% confidence."
        )

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Smoke-test report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
