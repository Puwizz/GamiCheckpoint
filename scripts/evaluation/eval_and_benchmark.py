"""
eval_and_benchmark.py — Unified Evaluation & Suppression Benchmark for YOLO Runs
==================================================================================
Evaluates any YOLO11-seg run on:
1. Validation set (unseen session_004 daylight bedroom + 25 authentic backgrounds)
2. Test set (unseen session_006 glossy floor tiles + 21 authentic backgrounds)
3. False-Positive suppression benchmark on pure background / negative samples:
   - Measures False Positive Rate (FPR) at conf=0.25, 0.50, 0.70
   - Measures Max False Positive Confidence on hard negative items (stickers, keyboards, floors)
4. Compares against baseline runs (Run 9 and Run 10).
5. Generates Markdown evaluation report and JSON summary.

Usage:
    python scripts/yolo/eval_and_benchmark.py --name origami_yolo_run11_yolo11s-seg_v3_bg --data configs/origami_seg_v3_bg_balanced.yaml
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark YOLO Runs.")
    parser.add_argument(
        "--name",
        type=str,
        default="origami_yolo_run11_yolo11s-seg_v3_bg",
        help="Experiment run name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to trained weights. If omitted, auto-discovers from runs/ or checkpoints/.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "configs" / "origami_seg_v3_bg_balanced.yaml"),
        help="Path to dataset yaml.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image resolution.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run evaluation on.",
    )
    return parser.parse_args()


def evaluate_split(model: YOLO, data_yaml: str, split: str, imgsz: int, device: str) -> dict:
    print(f"\n" + "=" * 60)
    print(f"Evaluating split: {split.upper()} on {model.model_name or 'model'}")
    print("=" * 60)
    val_results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=device,
        plots=False,
        workers=0,
        save_json=False,
    )

    box_map50 = float(val_results.box.map50) * 100
    box_map50_95 = float(val_results.box.map) * 100
    box_p = float(val_results.box.mp) * 100
    box_r = float(val_results.box.mr) * 100

    seg_map50 = float(val_results.seg.map50) * 100
    seg_map50_95 = float(val_results.seg.map) * 100
    seg_p = float(val_results.seg.mp) * 100
    seg_r = float(val_results.seg.mr) * 100

    per_class_seg = {}
    class_names = val_results.names
    for i, name in class_names.items():
        if i < len(val_results.seg.maps):
            per_class_seg[name] = {
                "map50": round(float(val_results.seg.maps[i]) * 100, 2),
            }

    metrics = {
        "box_mAP50": round(box_map50, 2),
        "box_mAP50_95": round(box_map50_95, 2),
        "box_precision": round(box_p, 2),
        "box_recall": round(box_r, 2),
        "seg_mAP50": round(seg_map50, 2),
        "seg_mAP50_95": round(seg_map50_95, 2),
        "seg_precision": round(seg_p, 2),
        "seg_recall": round(seg_r, 2),
        "per_class_seg": per_class_seg,
    }
    return metrics


def benchmark_background_suppression(model: YOLO, data_yaml_path: Path, imgsz: int, device: str) -> dict:
    print(f"\n" + "=" * 60)
    print("Benchmarking False Positive Suppression on Pure Background Samples...")
    print("=" * 60)

    # Search for all 0-byte background images across dataset test set and background_candidates
    bg_image_paths = []
    
    # 1. From dataset test set
    import yaml
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg.get("path", ROOT / "data" / "origami_seg_v2_bg_balanced"))
    
    test_labels_dir = dataset_root / "labels" / "test"
    test_imgs_dir = dataset_root / "images" / "test"
    if test_labels_dir.exists():
        for label_path in test_labels_dir.glob("*.txt"):
            if label_path.stat().st_size == 0:
                for ext in [".jpg", ".png", ".jpeg"]:
                    img_path = test_imgs_dir / f"{label_path.stem}{ext}"
                    if img_path.exists():
                        bg_image_paths.append(img_path)
                        break

    # 2. Add candidates from 02_captures_hard_negatives (stickers, keyboards, fan, clothing)
    hard_neg_dir = ROOT / "data" / "background_candidates" / "02_captures_hard_negatives"
    if hard_neg_dir.exists():
        for img_path in hard_neg_dir.rglob("*.jpg"):
            bg_image_paths.append(img_path)

    bg_image_paths = list(set(bg_image_paths))
    print(f"Total background test images to benchmark: {len(bg_image_paths)}")

    fp_results = {
        "total_bg_images": len(bg_image_paths),
        "detections_at_010": 0,
        "detections_at_025": 0,
        "detections_at_050": 0,
        "max_conf_observed": 0.0,
        "worst_false_positive": None,
    }

    for img_path in bg_image_paths:
        res = model.predict(source=str(img_path), conf=0.10, imgsz=imgsz, device=device, verbose=False)[0]
        boxes = res.boxes
        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            max_c = float(np.max(confs))
            if max_c > fp_results["max_conf_observed"]:
                fp_results["max_conf_observed"] = round(max_c, 4)
                cls_id = int(boxes.cls[np.argmax(confs)].item())
                fp_results["worst_false_positive"] = {
                    "image": img_path.name,
                    "class": res.names.get(cls_id, str(cls_id)),
                    "conf": round(max_c, 4),
                }

            fp_results["detections_at_010"] += 1
            if max_c >= 0.25:
                fp_results["detections_at_025"] += 1
            if max_c >= 0.50:
                fp_results["detections_at_050"] += 1

    total = len(bg_image_paths) if len(bg_image_paths) > 0 else 1
    fp_results["clean_rate_at_025"] = round((1.0 - fp_results["detections_at_025"] / total) * 100, 2)
    fp_results["clean_rate_at_050"] = round((1.0 - fp_results["detections_at_050"] / total) * 100, 2)

    print(f"Max False Positive Conf Observed: {fp_results['max_conf_observed'] * 100:.2f}%")
    print(f"Clean Negative Rate at Conf=0.25: {fp_results['clean_rate_at_025']}%")
    print(f"Clean Negative Rate at Conf=0.50: {fp_results['clean_rate_at_050']}%")
    return fp_results


def generate_benchmark_report(run_name: str, val_metrics: dict, test_metrics: dict, fp_metrics: dict, output_path: Path):
    run9_test_map50 = 92.75
    run10_test_map50 = 92.40

    report_md = f"""# รายงานผลการประเมินและเปรียบเทียบ {run_name}
## (Model Benchmark & False Positive Suppression Report)

**โครงการ:** GamiSenpai (Interactive Origami AI Tutor)  
**ชื่อการทดลอง:** `{run_name}`  
**เกณฑ์การแบ่ง:** Strict Zero-Leakage Balanced Session Split (Val: session_004 | Test: session_006)

---

## 1. ผลการเปรียบเทียบตัวชี้วัด (Benchmark Comparison)

| ตัวชี้วัด (Metric) | Run 9 (No BG)<br>v1 Balanced | Run 10 (+BG)<br>v2 Balanced | {run_name}<br>(Current Run) | ผลต่างเทียบ Run 10 | การวิเคราะห์ทางวิศวกรรม |
|---|:---:|:---:|:---:|:---:|---|
| **Val Set Mask mAP50** | 95.44% | 93.82% | **{val_metrics['seg_mAP50']:.2f}%** | {val_metrics['seg_mAP50'] - 93.82:+.2f}% | ประเมินบน session_004 (Unseen Bedroom) |
| **Val Set Mask mAP50-95** | 76.50% | 89.13% | **{val_metrics['seg_mAP50_95']:.2f}%** | {val_metrics['seg_mAP50_95'] - 89.13:+.2f}% | ความคมชัดของขอบ Polygon Mask |
| **Val Set Mask Precision** | 97.64% | 94.70% | **{val_metrics['seg_precision']:.2f}%** | {val_metrics['seg_precision'] - 94.70:+.2f}% | ความแม่นยำ ปราศจาก False Positive |
| **Val Set Mask Recall** | 94.06% | 87.10% | **{val_metrics['seg_recall']:.2f}%** | {val_metrics['seg_recall'] - 87.10:+.2f}% | ความสามารถในการตรวจพบกระดาษ |
| **Test Set Mask mAP50** | 92.75% | 92.40% | **{test_metrics['seg_mAP50']:.2f}%** | {test_metrics['seg_mAP50'] - run10_test_map50:+.2f}% | ประเมินบน session_006 (Unseen Glossy Floor) |
| **Test Set Mask mAP50-95** | 71.30% | 82.80% | **{test_metrics['seg_mAP50_95']:.2f}%** | {test_metrics['seg_mAP50_95'] - 82.80:+.2f}% | คุณภาพ Mask บนพื้นผิวมันสะท้อนแสง |
| **Test Set Mask Precision** | 94.48% | 91.50% | **{test_metrics['seg_precision']:.2f}%** | {test_metrics['seg_precision'] - 91.50:+.2f}% | Precision บนห้องที่ไม่เคยเห็นมาก่อน |
| **Test Set Mask Recall** | 88.10% | 87.20% | **{test_metrics['seg_recall']:.2f}%** | {test_metrics['seg_recall'] - 87.20:+.2f}% | Recall บนห้องที่ไม่เคยเห็นมาก่อน |

---

## 2. ผลการทดสอบการขจัด False Positive บนสิ่งรบกวน (Hard Negative Suppression)

ทดสอบกับตัวอย่าง Background ของจริง ({fp_metrics['total_bg_images']} ตัวอย่าง) เช่น สติ๊กเกอร์สีขาวบนคีย์บอร์ดโน้ตบุ๊ก, ปุ่มคีย์บอร์ด, พัดลม, เตียงนอน และสิ่งของในห้อง:

* **Clean Suppression Rate (at Conf = 0.25):** **`{fp_metrics['clean_rate_at_025']}%`**
* **Clean Suppression Rate (at Conf = 0.50):** **`{fp_metrics['clean_rate_at_050']}%`**
* **Max Confidence Observed บนสิ่งรบกวน:** **`{fp_metrics['max_conf_observed'] * 100:.2f}%`**
* **Worst False Positive Case:** `{fp_metrics['worst_false_positive']}`
"""
    output_path.write_text(report_md, encoding="utf-8")
    print(f"\n[REPORT SAVED] {output_path}")


def main():
    args = parse_args()
    run_name = args.name
    model_path = Path(args.model) if args.model else None

    if not model_path or not model_path.exists():
        candidates = [
            ROOT / "runs" / "segment" / "checkpoints" / "yolo_runs" / run_name / "weights" / "best.pt",
            ROOT / "runs" / "segment" / run_name / "weights" / "best.pt",
            ROOT / "checkpoints" / "yolo_runs" / run_name / "weights" / "best.pt",
        ]
        found = False
        for c in candidates:
            if c.exists():
                model_path = c
                found = True
                break
        if not found:
            print(f"[ERROR] Could not find best.pt in candidates: {candidates}")
            return

    canonical = ROOT / "checkpoints" / "yolo_runs" / run_name / "weights" / "best.pt"
    if model_path != canonical:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, canonical)
        print(f"[INFO] Copied {model_path} -> {canonical}")
        model_path = canonical

    print(f"[INFO] Evaluating model: {model_path}")
    model = YOLO(str(model_path))
    val_metrics = evaluate_split(model, args.data, "val", args.imgsz, args.device)
    test_metrics = evaluate_split(model, args.data, "test", args.imgsz, args.device)
    fp_metrics = benchmark_background_suppression(model, Path(args.data), args.imgsz, args.device)

    all_results = {
        "run_name": run_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fp_metrics": fp_metrics,
    }

    report_path = ROOT / "docs" / f"{run_name.upper()}_BENCHMARK_REPORT.md"
    generate_benchmark_report(run_name, val_metrics, test_metrics, fp_metrics, report_path)

    json_path = ROOT / "docs" / f"{run_name}_metrics.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[JSON SAVED] {json_path}")


if __name__ == "__main__":
    main()
