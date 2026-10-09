"""
Unit tests for src/utils.py (CORE-14)
"""

import os
from pathlib import Path
import pytest
from src.utils import split_dataset, count_dataset, safe_rmtree


def test_split_dataset_single_image_class(tmp_path):
    """CORE-14: Verify that a class with only 1 image allocates at least 1 image to train."""
    src_dir = tmp_path / "raw"
    cls_a = src_dir / "class_single"
    cls_b = src_dir / "class_multiple"
    cls_a.mkdir(parents=True)
    cls_b.mkdir(parents=True)

    # 1 image in class_single
    (cls_a / "img1.jpg").write_text("dummy")

    # 5 images in class_multiple
    for i in range(5):
        (cls_b / f"img{i}.jpg").write_text("dummy")

    out_dir = tmp_path / "split"
    split_dataset(str(src_dir), str(out_dir), train_ratio=0.8, seed=42)

    # Class with 1 image must have 1 in train, 0 in val
    train_single = list((out_dir / "train" / "class_single").glob("*.jpg"))
    val_single = list((out_dir / "val" / "class_single").glob("*.jpg"))
    assert len(train_single) == 1
    assert len(val_single) == 0

    # Class with 5 images has 4 in train, 1 in val
    train_mult = list((out_dir / "train" / "class_multiple").glob("*.jpg"))
    val_mult = list((out_dir / "val" / "class_multiple").glob("*.jpg"))
    assert len(train_mult) == 4
    assert len(val_mult) == 1


def test_split_dataset_zero_train_ratio(tmp_path):
    src_dir = tmp_path / "raw"
    cls_a = src_dir / "class_a"
    cls_a.mkdir(parents=True)
    (cls_a / "img1.jpg").write_text("dummy")

    out_dir = tmp_path / "split"
    split_dataset(str(src_dir), str(out_dir), train_ratio=0.0, seed=42)

    train_imgs = list((out_dir / "train" / "class_a").glob("*.jpg")) if (out_dir / "train" / "class_a").exists() else []
    val_imgs = list((out_dir / "val" / "class_a").glob("*.jpg"))
    assert len(train_imgs) == 0
    assert len(val_imgs) == 1


def test_count_dataset(tmp_path):
    data_dir = tmp_path / "dataset"
    (data_dir / "c1").mkdir(parents=True)
    (data_dir / "c2").mkdir(parents=True)
    (data_dir / "c1" / "1.png").write_text("a")
    (data_dir / "c1" / "2.jpg").write_text("b")
    (data_dir / "c2" / "3.bmp").write_text("c")

    counts = count_dataset(str(data_dir))
    assert counts == {"c1": 2, "c2": 1}


def test_safe_rmtree(tmp_path):
    sub = tmp_path / "to_delete"
    sub.mkdir()
    (sub / "file.txt").write_text("hello")
    assert sub.exists()

    safe_rmtree(sub)
    assert not sub.exists()
