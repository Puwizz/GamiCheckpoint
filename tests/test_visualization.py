"""
Unit tests for src/visualization.py (CORE-06)
"""

import sys
from pathlib import Path
import tempfile
import torch
import pytest

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from visualization import denormalize_image, save_train_samples_grid


class MockDataset:
    def __init__(self, classes, samples_per_class_map):
        self.classes = list(classes)
        self.samples = []
        for cls_name, count in samples_per_class_map.items():
            cls_idx = self.classes.index(cls_name)
            for _ in range(count):
                self.samples.append((f"/fake/{cls_name}.jpg", cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return a normalized dummy tensor (3, 32, 32)
        return torch.zeros((3, 32, 32), dtype=torch.float32), self.samples[idx][1]


def test_denormalize_image():
    tensor = torch.zeros((3, 16, 16), dtype=torch.float32)
    img = denormalize_image(tensor)
    assert img.shape == (16, 16, 3)
    assert img.dtype.name == "uint8"


def test_save_train_samples_grid_samples_per_class_1():
    """CORE-06: Verifies that samples_per_class=1 does not raise ZeroDivisionError."""
    classes = ["Chk1", "Chk2"]
    dataset = MockDataset(classes, {"Chk1": 5, "Chk2": 3})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "grid_1.png"
        save_train_samples_grid(dataset, out_path, classes=classes, samples_per_class=1)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_save_train_samples_grid_fewer_samples_than_requested():
    """Verify behavior when some classes have fewer samples than samples_per_class."""
    classes = ["Chk1", "Chk2", "Chk3"]
    dataset = MockDataset(classes, {"Chk1": 1, "Chk2": 4, "Chk3": 0})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "grid_fewer.png"
        save_train_samples_grid(dataset, out_path, classes=classes, samples_per_class=3)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_save_train_samples_grid_single_class_single_sample():
    """Verify single class and single sample grid layout."""
    classes = ["Chk1"]
    dataset = MockDataset(classes, {"Chk1": 2})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "grid_single.png"
        save_train_samples_grid(dataset, out_path, classes=classes, samples_per_class=1)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
