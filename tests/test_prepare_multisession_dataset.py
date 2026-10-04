import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from prepare_dataset import (  # noqa: E402
    DatasetValidationError,
    GENERATED_MARKER,
    audit_sessions,
    find_step_directory,
    load_config,
    prepare_output,
)


class MultiSessionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        self.sessions = [f"session_{index:03d}" for index in range(1, 6)]
        image_index = 0
        for session_index, session in enumerate(self.sessions):
            for step in range(1, 7):
                folder_name = "Step1_noon" if session_index == 0 and step == 1 else f"Step{step}"
                step_dir = self.source / session / folder_name
                step_dir.mkdir(parents=True)
                image_index += 1
                color = (
                    (image_index * 37) % 256,
                    (image_index * 67) % 256,
                    (image_index * 97) % 256,
                )
                Image.new("RGB", (240, 240), color=color).save(step_dir / f"image_{image_index}.png")

        self.config_path = self.root / "split.json"
        self.config = {
            "schema_version": 1,
            "source_dir": "raw",
            "min_images_per_step": 1,
            "min_image_width": 224,
            "min_image_height": 224,
            "splits": {
                "train": self.sessions[:3],
                "val": [self.sessions[3]],
                "test": [self.sessions[4]],
            },
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def load_and_audit(self):
        config, source_dir, project_root = load_config(self.config_path, self.root)
        records, warnings = audit_sessions(config, source_dir)
        return config, source_dir, project_root, records, warnings

    def test_builds_train_val_test_from_complete_sessions(self):
        _, source_dir, project_root, records, warnings = self.load_and_audit()
        self.assertEqual(warnings, [])
        self.assertEqual(len(records), 30)

        output = self.root / "processed"
        manifest, summary = prepare_output(
            records,
            output,
            source_dir,
            project_root,
            self.config_path,
        )

        self.assertTrue((output / GENERATED_MARKER).is_file())
        self.assertTrue(manifest.is_file())
        self.assertTrue(summary.is_file())
        self.assertEqual(len(list((output / "train").glob("Chk*/*.png"))), 18)
        self.assertEqual(len(list((output / "val").glob("Chk*/*.png"))), 6)
        self.assertEqual(len(list((output / "test").glob("Chk*/*.png"))), 6)
        summary_data = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(summary_data["counts"]["train"]["Chk1"], 3)
        self.assertEqual(summary_data["counts"]["val"]["Chk6"], 1)
        self.assertEqual(summary_data["counts"]["test"]["Chk6"], 1)

    def test_rejects_exact_duplicate_across_splits(self):
        train_image = next((self.source / self.sessions[0] / "Step2").glob("*.png"))
        test_image = next((self.source / self.sessions[4] / "Step2").glob("*.png"))
        shutil.copy2(train_image, test_image)

        config, source_dir, _ = load_config(self.config_path, self.root)
        with self.assertRaises(DatasetValidationError) as context:
            audit_sessions(config, source_dir)
        self.assertTrue(
            any("crosses splits" in message for message in context.exception.errors),
            context.exception.errors,
        )

    def test_resolves_single_legacy_step_alias(self):
        directory, error = find_step_directory(self.source / self.sessions[0], "Step1")
        self.assertIsNone(error)
        self.assertEqual(directory.name, "Step1_noon")

    def test_refuses_to_overwrite_unmarked_directory(self):
        _, source_dir, project_root, records, _ = self.load_and_audit()
        output = self.root / "manual_files"
        output.mkdir()
        (output / "keep.txt").write_text("do not delete", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unmarked"):
            prepare_output(
                records,
                output,
                source_dir,
                project_root,
                self.config_path,
                overwrite=True,
            )
        self.assertTrue((output / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()
