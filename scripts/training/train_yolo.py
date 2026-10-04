"""
train_yolo.py — Train YOLOv8 / YOLOv11 for Multi-Colored Origami
================================================================
Trains Ultralytics YOLO with custom environmental & color augmentations
to ensure robustness across paper colors, lighting conditions, and cluttered desks.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for origami step detection.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "configs" / "origami_yolo.yaml"),
        help="Path to dataset yaml file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base pretrained model (e.g. yolov8n.pt, yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default 2 to prevent Windows shared memory issues).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt checkpoint.",
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
        default="origami_yolo_run1",
        help="Experiment name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on (e.g. '0' or 'cpu'). Default auto-select.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        help="Cache images ('ram', 'disk', or False) for zero-disk-latency GPU training.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gamisenpai-origami-yolo",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="",
        help="Weights & Biases run name. Defaults to --name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wb_run = None
    if args.wandb_project:
        import wandb
        wb_name = args.wandb_name if args.wandb_name else args.name
        wb_run = wandb.init(
            project=args.wandb_project,
            name=wb_name,
            entity="puwit-tubt-kmutt",
            config=vars(args),
        )
        print(f"W&B Initialized: Entity=puwit-tubt-kmutt, Project={args.wandb_project}, Name={wb_name}")
        print(f"W&B Dashboard URL: {wb_run.url}")

    print("=" * 60)
    print(f"Starting YOLO Origami Training with model: {args.model}")
    print(f"Data config: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}, Workers: {args.workers}")
    print("=" * 60)

    if args.resume:
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Cannot resume, last checkpoint not found: {last_ckpt}")
        print(f"Resuming training from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            project=args.project,
            name=args.name,
            device=args.device if args.device else None,
            cache=args.cache if args.cache.lower() != "false" else False,
            amp=True,
            # Multi-color paper augmentations:
            hsv_h=0.50,         # Hue jitter: shifts hue dramatically to generalize across paper colors
            hsv_s=0.70,         # Saturation jitter: simulates faded paper vs vibrant paper
            hsv_v=0.40,         # Value/Brightness jitter: handles shadow/lighting shifts
            degrees=15.0,       # Random rotation
            translate=0.10,     # Random translation
            scale=0.20,         # Random zoom
            shear=5.0,          # Shear distortion
            perspective=0.0005, # Perspective distortion
            flipud=0.0,         # Origami orientation matters, no vertical flip
            fliplr=0.5,         # Horizontal flip (mirror fold symmetry)
            mosaic=1.0,         # 4-image mosaic for complex cluttered backgrounds
            mixup=0.10,         # Blends pairs of images
            plots=True,
            save=True,
            val=True,
        )

    print(f"\n[SUCCESS] YOLO training completed. Best model saved in {args.project}/{args.name}")
    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    main()
