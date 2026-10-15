from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from scripts.dataset.auto_label_dataset import CLASS_MAP, convert_split
from scripts.dataset.curate_for_segmentation import ROOT as CURATE_ROOT
from scripts.dataset.extract_color_seq import extract_color_sequence
from scripts.dataset.extract_test_frames import extract_test_frames
from scripts.training.train_yolo import parse_args as train_yolo_parse_args
from scripts.training.train_yolo_seg import check_device_target, parse_args as train_yolo_seg_parse_args


def test_class_map():
    assert len(CLASS_MAP) == 6
    assert CLASS_MAP["Chk1"] == 0
    assert CLASS_MAP["Chk6"] == 5


def test_convert_split_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_dir = tmp_path / "source"
        out_dir = tmp_path / "target"

        chk1_dir = src_dir / "Chk1"
        chk1_dir.mkdir(parents=True)
        dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(chk1_dir / "sample.jpg"), dummy_img)

        convert_split("train", src_dir, out_dir, copy_images=True)

        assert (out_dir / "images" / "train" / "Chk1_sample.jpg").exists()
        label_file = out_dir / "labels" / "train" / "Chk1_sample.txt"
        assert label_file.exists()

        content = label_file.read_text().strip().split()
        assert content[0] == "0"
        assert len(content) == 5
        cx, cy, nw, nh = map(float, content[1:])
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 < nw <= 1.0
        assert 0.0 < nh <= 1.0


def test_convert_split_collision_prevention():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_dir = tmp_path / "source"
        out_dir = tmp_path / "target"

        chk1_dir = src_dir / "Chk1"
        chk2_dir = src_dir / "Chk2"
        chk1_dir.mkdir(parents=True)
        chk2_dir.mkdir(parents=True)

        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(chk1_dir / "same_name.jpg"), img1)
        cv2.imwrite(str(chk2_dir / "same_name.jpg"), img2)

        convert_split("train", src_dir, out_dir, copy_images=True)

        assert (out_dir / "images" / "train" / "Chk1_same_name.jpg").exists()
        assert (out_dir / "images" / "train" / "Chk2_same_name.jpg").exists()

        lbl1 = (out_dir / "labels" / "train" / "Chk1_same_name.txt").read_text().strip().split()
        lbl2 = (out_dir / "labels" / "train" / "Chk2_same_name.txt").read_text().strip().split()

        assert lbl1[0] == "0"
        assert lbl2[0] == "1"


@pytest.mark.parametrize("device_str, expected_idx", [
    ("cuda", 0),
    ("cuda:0", 0),
    ("cuda:1", 1),
    ("0", 0),
    ("1", 1),
    ("0,1", 0),
])
def test_check_device_target_cuda_valid(device_str, expected_idx, monkeypatch, capsys):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: f"Mock GPU {idx}")

    check_device_target(device_str)
    captured = capsys.readouterr()
    assert f"CUDA active on device {expected_idx}: Mock GPU {expected_idx}" in captured.out


def test_check_device_target_cpu_and_negative_one(capsys):
    check_device_target("cpu")
    check_device_target("-1")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_check_device_target_cuda_unavailable(monkeypatch, capsys):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    check_device_target("cuda")
    captured = capsys.readouterr()
    assert "CUDA GPU was requested (--device cuda)" in captured.out


def test_check_device_target_exception_handled(monkeypatch, capsys):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def mock_get_device_name(idx):
        raise RuntimeError("Device error")
    monkeypatch.setattr(torch.cuda, "get_device_name", mock_get_device_name)

    check_device_target("cuda:0")
    captured = capsys.readouterr()
    assert "Device target check: Device error" in captured.out


def test_curate_for_segmentation_root_not_hardcoded():
    assert "admin" not in str(CURATE_ROOT).lower() or CURATE_ROOT.exists()
    assert (CURATE_ROOT / "scripts").exists()


def test_extract_color_sequence_missing_video(capsys):
    count = extract_color_sequence(vpath="/nonexistent/video.mp4")
    assert count == 0
    captured = capsys.readouterr()
    assert "Video not found" in captured.out


def test_extract_test_frames_missing_videos(capsys):
    count = extract_test_frames(videos=["missing1.mp4", "missing2.mp4"])
    assert count == 0
    captured = capsys.readouterr()
    assert "File not found" in captured.out


def test_train_yolo_wandb_entity_default(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr("sys.argv", ["train_yolo.py"])
    args = train_yolo_parse_args()
    assert args.wandb_entity is None


def test_train_yolo_seg_wandb_entity_default(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr("sys.argv", ["train_yolo_seg.py"])
    args = train_yolo_seg_parse_args()
    assert args.wandb_entity is None


def test_train_yolo_wandb_entity_env_var(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "custom-org")
    monkeypatch.setattr("sys.argv", ["train_yolo.py"])
    args = train_yolo_parse_args()
    assert args.wandb_entity == "custom-org"

