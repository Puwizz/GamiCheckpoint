import sys
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
import torch

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from visualization import denormalize_image, save_train_samples_grid
from scripts.dataset.visualize_labels import visualize_image_labels, main as visualize_labels_main


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
        return torch.zeros((3, 32, 32), dtype=torch.float32), self.samples[idx][1]


def test_denormalize_image():
    tensor = torch.zeros((3, 16, 16), dtype=torch.float32)
    img = denormalize_image(tensor)
    assert img.shape == (16, 16, 3)
    assert img.dtype.name == "uint8"


def test_save_train_samples_grid_samples_per_class_1():
    classes = ["Chk1", "Chk2"]
    dataset = MockDataset(classes, {"Chk1": 5, "Chk2": 3})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "grid_1.png"
        save_train_samples_grid(dataset, out_path, classes=classes, samples_per_class=1)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_save_train_samples_grid_fewer_samples_than_requested():
    classes = ["Chk1", "Chk2", "Chk3"]
    dataset = MockDataset(classes, {"Chk1": 1, "Chk2": 4, "Chk3": 0})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "grid_fewer.png"
        save_train_samples_grid(dataset, out_path, classes=classes, samples_per_class=3)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_save_train_samples_grid_single_class_single_sample():
    classes = ["Chk1"]
    dataset = MockDataset(classes, {"Chk1": 2})

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "grid_single.png"
        save_train_samples_grid(dataset, out_path, classes=classes, samples_per_class=1)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_visualize_image_labels_empty_lines():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated, cls_name = visualize_image_labels(img, [])
    assert cls_name == "Background"
    assert annotated.shape == (100, 100, 3)


def test_visualize_image_labels_bbox():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lines = ["0 0.5 0.5 0.2 0.2\n"]
    annotated, cls_name = visualize_image_labels(img, lines)
    assert cls_name == "Chk1"
    assert not np.array_equal(annotated, np.zeros((100, 100, 3), dtype=np.uint8))


def test_visualize_image_labels_polygon():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lines = ["1 0.1 0.1 0.2 0.2 0.3 0.1\n"]
    annotated, cls_name = visualize_image_labels(img, lines)
    assert cls_name == "Chk2"
    assert not np.array_equal(annotated, np.zeros((100, 100, 3), dtype=np.uint8))


def test_visualize_labels_main_empty_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lbl_dir = Path(tmp_dir) / "labels"
        img_dir = Path(tmp_dir) / "images"
        out_dir = Path(tmp_dir) / "plots"
        lbl_dir.mkdir()
        img_dir.mkdir()
        res = visualize_labels_main(lbl_dir=lbl_dir, img_dir=img_dir, out_dir=out_dir)
        assert res is None


def test_visualize_labels_main_with_data():
    with tempfile.TemporaryDirectory() as tmp_dir:
        lbl_dir = Path(tmp_dir) / "labels"
        img_dir = Path(tmp_dir) / "images"
        out_dir = Path(tmp_dir) / "plots"
        lbl_dir.mkdir()
        img_dir.mkdir()

        dummy_img = np.full((50, 50, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(img_dir / "sample1.jpg"), dummy_img)
        (lbl_dir / "sample1.txt").write_text("0 0.5 0.5 0.4 0.4\n")

        res = visualize_labels_main(lbl_dir=lbl_dir, img_dir=img_dir, out_dir=out_dir)
        assert res is not None
        assert res.exists()
