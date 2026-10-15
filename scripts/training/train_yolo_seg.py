"""
train_yolo_seg.py — Train YOLO11-seg for Origami Checkpoint Segmentation
========================================================================
Trains Ultralytics YOLO11s-seg Instance Segmentation for origami checkpoint detection.
Features:
- Dual Box + Polygon Mask segmentation loss
- Multi-color paper augmentations (HSV jitter, mosaic, flips)
- Automated metric logging (mAP50-seg, mAP50-95-seg)
- Weights & Biases (W&B) integration with configurable entity
- Deterministic reproducibility flags (seed, amp)
"""

import argparse
import os
from pathlib import Path
from typing import Any, Union
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

# Default data augmentation parameters for multi-color origami paper
AUGMENTATION_CONFIG: dict[str, Any] = {
    "hsv_h": 0.50,         # Hue jitter: shifts paper colors across spectrum
    "hsv_s": 0.70,         # Saturation jitter
    "hsv_v": 0.40,         # Value/Lighting jitter
    "degrees": 15.0,       # Rotation angle (+/- deg)
    "translate": 0.10,     # Translation (+/- fraction)
    "scale": 0.20,         # Scale gain (+/- fraction)
    "shear": 5.0,          # Shear angle (+/- deg)
    "perspective": 0.0005, # Perspective distortion
    "flipud": 0.0,         # No upside-down flip (camera is fixed top-down/front)
    "fliplr": 0.5,         # Horizontal mirror flip
    "mosaic": 1.0,         # 4-image mosaic augmentation probability
    "mixup": 0.10,         # Image blend probability
}


def parse_cache_arg(value: str) -> Union[str, bool]:
    """Parse cache argument into valid Ultralytics cache option."""
    val_lower = value.strip().lower()
    if val_lower in {"false", "0", "no", "none"}:
        return False
    if val_lower in {"true", "1", "yes"}:
        return True
    if val_lower in {"ram", "disk"}:
        return val_lower
    raise argparse.ArgumentTypeError(
        f"Invalid cache value '{value}'. Choose from 'ram', 'disk', or 'False'."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for YOLO11 segmentation training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11-seg for origami checkpoint segmentation."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "configs" / "origami_seg_v3_session_split.yaml"),
        help="Path to dataset yaml file (default: origami_seg_v3_session_split.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s-seg.pt",
        help="Pretrained segmentation model (e.g. yolo11s-seg.pt, yolo11n-seg.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image resolution (default: 640).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0 for Windows CUDA stability).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience epochs (default: 20).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(ROOT / "checkpoints" / "yolo_runs"),
        help="Directory to save training checkpoints.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="origami_yolo11s-seg",
        help="Experiment run name (default: origami_yolo11s-seg).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on (e.g. '0' for CUDA GPU, 'cpu').",
    )
    parser.add_argument(
        "--cache",
        type=parse_cache_arg,
        default=False,
        help="Cache images ('ram', 'disk', or 'False').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic reproducibility (default: 0).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Automatic Mixed Precision (AMP) training (default: True).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gamisenpai-origami-yolo",
        help="Weights & Biases project name (leave empty to disable).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.getenv("WANDB_ENTITY", "puwit-tubt-kmutt"),
        help="Weights & Biases entity/username (defaults to $WANDB_ENTITY or 'puwit-tubt-kmutt').",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="",
        help="Weights & Biases run name (defaults to --name).",
    )
    return parser.parse_args()


def check_device_target(requested_device: str) -> None:
    if requested_device not in {"cpu", "-1"}:
        try:
            import torch
            if not torch.cuda.is_available():
                print(
                    "\n" + "!" * 70 + "\n"
                    f"[WARNING] CUDA GPU was requested (--device {requested_device}), "
                    "but torch.cuda.is_available() is False!\n"
                    "Training will likely fall back to CPU or fail.\n"
                    "If on NVIDIA RTX 50 series, check docs/GPU_SETUP_AND_TROUBLESHOOTING_GUIDE_TH.md\n"
                    + "!" * 70 + "\n"
                )
            else:
                dev_part = requested_device.split(",")[0].strip()
                dev_digits = "".join(filter(str.isdigit, dev_part))
                first_device_idx = int(dev_digits) if dev_digits else 0
                gpu_name = torch.cuda.get_device_name(first_device_idx)
                print(f"[INFO] CUDA active on device {first_device_idx}: {gpu_name}")
        except (ImportError, ValueError, Exception) as err:
            print(f"[INFO] Device target check: {err}")


def summarize_and_log_metrics(results: Any, save_dir: Path, wb_run: Any = None) -> None:
    """Extract, display, and log final evaluation metrics from YOLO results."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS SUMMARY")
    print("=" * 60)

    metrics_dict = getattr(results, "results_dict", {}) or {}
    if metrics_dict:
        box_p = metrics_dict.get("metrics/precision(B)", 0.0)
        box_r = metrics_dict.get("metrics/recall(B)", 0.0)
        box_map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
        box_map = metrics_dict.get("metrics/mAP50-95(B)", 0.0)

        seg_p = metrics_dict.get("metrics/precision(M)", 0.0)
        seg_r = metrics_dict.get("metrics/recall(M)", 0.0)
        seg_map50 = metrics_dict.get("metrics/mAP50(M)", 0.0)
        seg_map = metrics_dict.get("metrics/mAP50-95(M)", 0.0)

        print(f"Box  Metrics : Precision={box_p:.4f}, Recall={box_r:.4f}, mAP50={box_map50:.4f}, mAP50-95={box_map:.4f}")
        print(f"Mask Metrics : Precision={seg_p:.4f}, Recall={seg_r:.4f}, mAP50={seg_map50:.4f}, mAP50-95={seg_map:.4f}")

        if wb_run:
            try:
                wb_run.summary.update({
                    "final/box_precision": box_p,
                    "final/box_recall": box_r,
                    "final/box_mAP50": box_map50,
                    "final/box_mAP50-95": box_map,
                    "final/mask_precision": seg_p,
                    "final/mask_recall": seg_r,
                    "final/mask_mAP50": seg_map50,
                    "final/mask_mAP50-95": seg_map,
                })
            except Exception as e:
                print(f"[INFO] W&B summary update skipped (run may have been finalized by Ultralytics): {e}")
    else:
        print("[INFO] Summary metrics dictionary was not returned.")

    best_weight = save_dir / "weights" / "best.pt"
    if best_weight.exists():
        print(f"Best checkpoint : {best_weight}")
    print(f"Output directory: {save_dir}")
    print("=" * 60)


def main() -> None:
    """Main training execution function."""
    args = parse_args()
    wb_run = None

    if args.wandb_project:
        try:
            import wandb
            wb_name = args.wandb_name if args.wandb_name else args.name
            wb_run = wandb.init(
                project=args.wandb_project,
                name=wb_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
            print(f"W&B Initialized: Project={args.wandb_project}, Entity={args.wandb_entity}, Name={wb_name}")
            print(f"W&B Dashboard URL: {wb_run.url}")
        except ImportError:
            print("[INFO] wandb package not installed; skipping W&B logging.")
        except Exception as e:
            print(f"[WARNING] W&B initialization skipped or offline: {e}")

    print("=" * 60)
    print(f"Starting YOLO11-seg Training: {args.model}")
    print(f"Dataset config: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}, Patience: {args.patience}")
    print(f"Device: {args.device}, AMP: {args.amp}, Seed: {args.seed}")
    print("=" * 60)

    check_device_target(args.device)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device if args.device else 0,
        cache=args.cache,
        seed=args.seed,
        amp=args.amp,
        plots=True,
        save=True,
        val=True,
        **AUGMENTATION_CONFIG,
    )

    # Use save_dir from results directly (Ultralytics auto-increments folder name if it already exists)
    if results is not None:
        save_dir = Path(results.save_dir)
        summarize_and_log_metrics(results, save_dir, wb_run)
        print(f"\n[SUCCESS] Segmentation training completed. Best model saved in {save_dir}")
    else:
        print("\n[WARNING] Training returned None — training may have been interrupted or failed.")

    if wb_run:
        try:
            wb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
