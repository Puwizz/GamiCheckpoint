import pytest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from scripts.dataset.auto_label_dataset import CLASS_MAP, convert_split

def test_class_map():
    assert len(CLASS_MAP) == 6
    assert CLASS_MAP["Chk1"] == 0
    assert CLASS_MAP["Chk6"] == 5

def test_convert_split_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_dir = tmp_path / "source"
        out_dir = tmp_path / "target"
        
        # Create dummy class directory and image
        chk1_dir = src_dir / "Chk1"
        chk1_dir.mkdir(parents=True)
        dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(chk1_dir / "sample.jpg"), dummy_img)
        
        convert_split("train", src_dir, out_dir, copy_images=True)
        
        # Verify output structure
        assert (out_dir / "images" / "train" / "sample.jpg").exists()
        label_file = out_dir / "labels" / "train" / "sample.txt"
        assert label_file.exists()
        
        content = label_file.read_text().strip().split()
        assert content[0] == "0" # Class Chk1 -> 0
        assert len(content) == 5
        cx, cy, nw, nh = map(float, content[1:])
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 < nw <= 1.0
        assert 0.0 < nh <= 1.0
