# scripts/legacy — MobileNetV2 Archive

โค้ดในโฟลเดอร์นี้เป็น **archive** ของ pipeline เดิมที่ใช้ MobileNetV2 เป็น backbone
สำหรับจำแนกขั้นตอน Origami (แบบ classification)

ปัจจุบันโปรเจกต์ได้เปลี่ยนมาใช้ **YOLOv11-seg** แทนทั้งหมด (ดู `src/yolo_verifier.py`)

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | ที่มาเดิม | คำอธิบาย |
|---|---|---|
| `model_mobilenet.py` | `src/model.py` | นิยาม `StepModel` (MobileNetV2) และ `create_model()` |
| `train_mobilenet.py` | `scripts/training/train.py` | Training loop สำหรับ MobileNet |
| `evaluate_mobilenet.py` | `scripts/evaluation/evaluate.py` | Evaluation script สำหรับ `.pth` checkpoint |
| `run_inference_mobilenet.py` | `scripts/utils/run_inference.py` | Real-time inference ผ่านกล้องด้วย MobileNet |
| `build_non_origami_dataset.py` | `scripts/dataset/build_non_origami_dataset.py` | สกัด background/negative class (`Chk0_NonOrigami`) สำหรับ MobileNet (SCR-02 obsolete) |
| `debug_step_confusion.py` | `tools/debug_step_confusion.py` | Confusion matrix debugger สำหรับ MobileNet `.pth` |

## Checkpoint ที่เก็บไว้

ไฟล์ `.pth` ใน `checkpoints/` ยังถูกเก็บไว้เป็น reference
แต่ไม่ได้โหลดโดย `figure_registry.py` อีกต่อไป

## วิธีรัน (ถ้าจำเป็น)

```bash
# ต้องมี torchvision ติดตั้งอยู่
uv run python scripts/legacy/train_mobilenet.py --help
uv run python scripts/legacy/evaluate_mobilenet.py --model checkpoints/origami_step6_final.pth
```
