# GamiSenpai: AI Engineering Knowledge & Know-How Master Playbook
## คู่มือวิศวกรรมปัญญาประดิษฐ์เชิงลึก (End-to-End AI Engineering Specification & Knowledge Base)

> **สำหรับ:** AI Engineer / Computer Vision Researcher  
> **โปรเจกต์:** GamiSenpai (GamiCheckpoint) — Interactive Real-time Origami Verification  
> **โมเดลปัจจุบัน (SOTA):** Ultralytics YOLO11s-seg (Instance Segmentation) — Run 11 (`origami_yolo_run11_yolo11s-seg_v3_bg`)  
> **วัตถุประสงค์ของเอกสารนี้:** รวบรวมองค์ความรู้ (Knowledge) ทักษะทางเทคนิค (Know-How) ตรรกะคณิตศาสตร์ สถิติตัวเลข แผนผังไฟล์โค้ดที่เกี่ยวข้อง และคู่มือปฏิบัติการ (Step-by-Step Playbook) ด้าน AI และ Computer Vision ทั้งหมดของโปรเจกต์นี้ เพื่อให้ AI Engineer ที่กลับมาอ่านสามารถเข้าใจเบื้องหลังอย่างลึกซึ้ง และปฏิบัติตาม (Follow & Reproduce) ได้ทันที 100%

---

## สารบัญ (Table of Contents)

0. [แผนผังไฟล์โค้ดและส่วนประกอบทางเทคนิคที่เกี่ยวข้อง (Codebase Architecture & File Mapping)](#0-แผนผังไฟล์โค้ดและส่วนประกอบทางเทคนิคที่เกี่ยวข้อง-codebase-architecture--file-mapping)
1. [ภาพรวมไปป์ไลน์ AI ระดับสถาปัตยกรรม (High-Level AI Architecture)](#1-ภาพรวมไปป์ไลน์-ai-ระดับสถาปัตยกรรม-high-level-ai-architecture)
2. [กระบวนทัศน์โมเดล: ทำไมต้อง Instance Segmentation (Vision Paradigm Shift)](#2-กระบวนทัศน์โมเดล-ทำไมต้อง-instance-segmentation-vision-paradigm-shift)
3. [วิศวกรรมชุดข้อมูลและการกำจัด Data Leakage ขั้นเด็ดขาด (Data Engineering & Zero-Leakage Protocol)](#3-วิศวกรรมชุดข้อมูลและการกำจัด-data-leakage-ขั้นเด็ดขาด-data-engineering--zero-leakage-protocol)
4. [เกราะป้องกันภาพหลอน: Hard-Negative Ingestion & Background Hardening](#4-เกราะป้องกันภาพหลอน-hard-negative-ingestion--background-hardening)
5. [วิวัฒนาการการทดลองและการฝึกสอน (The 3 Pillar Experimental Runs)](#5-วิวัฒนาการการทดลองและการฝึกสอน-the-3-pillar-experimental-runs)
6. [การตั้งค่า Hyperparameters และกลยุทธ์ Data Augmentation](#6-การตั้งค่า-hyperparameters-และกลยุทธ์-data-augmentation)
7. [ระบบคัดกรองความมั่นใจ 3 ชั้นและการตัดเฟรมเสีย (Inference Gating & Decision Pipeline)](#7-ระบบคัดกรองความมั่นใจ-3-ชั้นและการตัดเฟรมเสีย-inference-gating--decision-pipeline)
8. [คู่มือปฏิบัติการทีละสเต็ปสำหรับ AI Engineer (Step-by-Step Operational Runbook)](#8-คู่มือปฏิบัติการทีละสเต็ปสำหรับ-ai-engineer-step-by-step-operational-runbook)
9. [บทเรียนทางวิศวกรรมและกับดักที่ห้ามทำ (AI Post-Mortem & Anti-Patterns)](#9-บทเรียนทางวิศวกรรมและกับดักที่ห้ามทำ-ai-post-mortem--anti-patterns)
10. [เอกสารอ้างอิงและประวัติศาสตร์ (Cross References)](#10-เอกสารอ้างอิงและประวัติศาสตร์-cross-references)

---

## 0. แผนผังไฟล์โค้ดและส่วนประกอบทางเทคนิคที่เกี่ยวข้อง (Codebase Architecture & File Mapping)

เพื่อให้ AI Engineer และผู้พัฒนาสามารถเชื่อมโยงแนวคิดทางทฤษฎีกับซอร์สโค้ดจริงใน Repository ได้อย่างรวดเร็ว ตารางด้านล่างสรุปความสัมพันธ์ของไฟล์โค้ดทุกส่วนที่ขับเคลื่อนระบบ AI ของโปรเจกต์นี้:

| หมวดหมู่ (Domain) | ไฟล์โค้ดที่เกี่ยวข้อง (Source Code Files) | ความรับผิดชอบหลักและฟังก์ชันสำคัญ (Key Functions & Scope) |
|---|---|---|
| **Real-time Inference Engine** | • [`src/yolo_verifier.py`](../../src/yolo_verifier.py)<br>• [`src/inference.py`](../../src/inference.py) | • `YoloStepVerifier`: โหลดน้ำหนัก YOLO11s-seg, รองรับ FP16 Half-Precision อัตโนมัติบน CUDA<br>• `verify_bgr()`: รับเฟรมภาพเว็บแคม (1280x720 BGR) ทำ Letterbox ปรับขนาดเป็น 640x640 และสั่ง Inference<br>• `_filter_detections()`: คัดกรอง 3-Tier Gating (Area $\ge$ 1.5%, Conf $\ge$ 55%, Top-gap $\ge$ 8%)<br>• `assess_frame_readiness()`: กรองเฟรมมืดหรือภาพไร้รายละเอียด (เช่น มือปิดกล้อง) ก่อนส่งเข้าโมเดล |
| **Catalog & Figure Topology** | • [`src/figure_registry.py`](../../src/figure_registry.py)<br>• [`src/step_config.py`](../../src/step_config.py)<br>• [`configs/figures/crane.json`](../../configs/figures/crane.json) | • `FigureRegistry`: จัดการแคช Verifier แบบหลายตัว (Multi-Figure) และโหลดคอนฟิกโมเดลของแต่ละรูปทรง<br>• `STEP_CLASSES` & `step_number()`: กำหนดคลาส Topology ของรอยพับ รองรับการขยายตัวโอริกามิใหม่ |
| **Foreground & Fallback Preprocessing** | • [`src/preprocessing.py`](../../src/preprocessing.py) | • `AutoCropForeground`: อัลกอริทึมสกัดกระดาษ Foreground<br>• `_detect_bbox_by_white_paper()` & Contour Fallback |
| **Web API Backend Service** | • [`app/app.py`](../../app/app.py)<br>• [`gamisenpai.py`](../../gamisenpai.py) | • Route `/api/predict`: รับอัปโหลดรูปภาพ multipart และเรียกโมเดลผ่าน Verifier<br>• `run_in_threadpool()`: ออฟโหลด Inference แบบ Non-blocking ไม่ให้กระทบ Event Loop<br>• Streaming Size Guard (10MB), CORS Middleware และ Error Response Schema |
| **Dataset Acquisition & Preprocessing** | • [`scripts/dataset/collect_data.py`](../../scripts/dataset/collect_data.py)<br>• [`scripts/dataset/prepare_v3_bg_dataset.py`](../../scripts/dataset/prepare_v3_bg_dataset.py)<br>• [`scripts/dataset/extract_background_candidates.py`](../../scripts/dataset/extract_background_candidates.py)<br>• [`scripts/dataset/extract_color_seq.py`](../../scripts/dataset/extract_color_seq.py)<br>• [`scripts/dataset/extract_test_frames.py`](../../scripts/dataset/extract_test_frames.py)<br>• [`scripts/dataset/auto_label_dataset.py`](../../scripts/dataset/auto_label_dataset.py)<br>• [`scripts/dataset/curate_for_segmentation.py`](../../scripts/dataset/curate_for_segmentation.py) | • สกัดภาพ 3.33 FPS จากไฟล์วิดีโอดิบของแต่ละ Physical Session<br>• จัดแบ่ง Partition ชุดข้อมูลตาม Physical Session (70/17/13) ไร้ Data Leakage<br>• สกัดภาพ Hard-Negative พร้อมสร้าง Label 0-byte สำหรับ Background Penalization<br>• ป้องกันชื่อไฟล์ชนกันด้วย Class Prefix เพื่อไม่ให้ Label ถูกบันทึกทับ |
| **Model Training & Hyperparameters** | • [`scripts/training/train_yolo_seg.py`](../../scripts/training/train_yolo_seg.py)<br>• [`scripts/training/train_yolo.py`](../../scripts/training/train_yolo.py)<br>• [`configs/origami_seg_v3_bg_balanced.yaml`](../../configs/origami_seg_v3_bg_balanced.yaml) | • สคริปต์ฝึกสอนหลัก YOLO11s-seg (AMP FP16, Patience 20, Batch 16, ImgSz 640)<br>• ตั้งค่า Data Augmentations (HSV Jitter, Multi-scale, Mosaic, FlipLR, ปิด FlipUD)<br>• จัดการอุปกรณ์ฝึกสอน `--device` (CUDA/CPU) และส่ง Telemetry ไปยัง Weights & Biases |
| **Evaluation, Auditing & Visualization** | • [`scripts/evaluation/audit_dataset_leakage.py`](../../scripts/evaluation/audit_dataset_leakage.py)<br>• [`scripts/evaluation/eval_and_benchmark.py`](../../scripts/evaluation/eval_and_benchmark.py)<br>• [`scripts/utils/visualize_per_class.py`](../../scripts/utils/visualize_per_class.py)<br>• [`scripts/dataset/visualize_labels.py`](../../scripts/dataset/visualize_labels.py) | • `audit_dataset_leakage.py`: ตรวจสอบ 1:1, Session Isolation, 0-byte BG และ MD5 Exclusivity<br>• `eval_and_benchmark.py`: รัน Test Set Benchmark และ False Positive Stress Test 217 ตัวอย่าง<br>• สคริปต์ Visualization: รองรับทั้ง Bounding Box และ Segmentation Polygon โดยไม่เกิด ValueError |
| **Frontend HUD & Camera Stream** | • [`frontend/src/hud-overlay.ts`](../../frontend/src/hud-overlay.ts)<br>• [`frontend/src/camera.ts`](../../frontend/src/camera.ts)<br>• [`frontend/src/tutorial-coordinator.ts`](../../frontend/src/tutorial-coordinator.ts)<br>• [`frontend/src/main.ts`](../../frontend/src/main.ts) | • `DetectionOverlayRenderer`: แมปพิกัด BBox/Polygon จากโมเดลเข้ากับ Canvas พร้อม LERP Smoothing<br>• `CameraStreamAdapter`: จัดการ MediaStream Tracks, ตรวจสอบ Luma/Contrast และคืนทรัพยากรกล้อง<br>• `TutorialCoordinator`: ควบคุมการเปลี่ยนก้าวพับ, ตัวจับเวลา Escape Hatch, และยกเลิกการโพลภาพเมื่อเสร็จสิ้น |
| **Automated Testing & CI Pipeline** | • [`tests/test_yolo_verifier.py`](../../tests/test_yolo_verifier.py)<br>• [`tests/test_api_endpoints.py`](../../tests/test_api_endpoints.py)<br>• [`tests/test_visualization.py`](../../tests/test_visualization.py)<br>• [`tests/test_yolo_pipeline.py`](../../tests/test_yolo_pipeline.py)<br>• [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) | • ชุดทดสอบอัตโนมัติ 110 ข้อ (Backend) และ 31 ข้อ (Frontend)<br>• CI Workflow บน GitHub Actions ตรวจสอบคุณภาพโค้ดอัตโนมัติบน Python 3.11 และ Node 20 |

---

## 1. ภาพรวมไปป์ไลน์ AI ระดับสถาปัตยกรรม (High-Level AI Architecture)

ระบบ AI ในโปรเจกต์นี้มีหน้าที่ตรวจจับและยืนยัน **ก้าวรอยพับสำคัญ (Milestone Checkpoints)** ของการพับกระดาษโอริกามิจากกล้องเว็บแคมของผู้ใช้ โดยรับภาพ RGB/BGR เข้ามา แล้วประเมินว่ากระดาษอยู่ในขั้นตอนใด (Chk1 ถึง Chk6) พร้อมส่งพิกัด Polygon Mask และ Bounding Box กลับไปแสดงผลบนหน้าจอ

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`app/app.py`](../../app/app.py): Route `/api/predict` รับภาพจากเว็บแคม และออฟโหลดการทำนายไปยัง Threadpool
> - [`src/yolo_verifier.py`](../../src/yolo_verifier.py): คลาส `YoloStepVerifier` สำหรับทำ Letterbox Resizing, Forward Pass และ Decision Gating
> - [`src/inference.py`](../../src/inference.py): ฟังก์ชัน `assess_frame_readiness()` และคลาสส่งผลลัพธ์ `VerificationResult`
> - [`configs/figures/crane.json`](../../configs/figures/crane.json): นิยามไฟล์โมเดลและคลาส Topology ของนกกระเรียน

```mermaid
flowchart TD
    subgraph Ingestion["1. Data Ingestion & Preprocessing"]
        Cam["Webcam Frame (1280x720 BGR)"]
        LumaGate{"Assess Frame Readiness\n(Mean Luma >= 22, Std >= 6)\n[src/inference.py]"}
        Letterbox["Letterbox Resizing\n(Maintain Aspect Ratio to 640x640)\n[src/yolo_verifier.py]"}
    end

    subgraph DeepModel["2. Neural Network Forward Pass"]
        YOLO["YOLO11s-seg Backbone & Neck\n(C3k2, SPPF, C2PSA, Proto-Net)\n[checkpoints/yolo_runs/.../best.pt]"]
        BoxHead["Bounding Box & Class Head"]
        MaskHead["Proto Mask Coefficients Head"]
    end

    subgraph PostProcessing["3. 3-Tier Decision Gating Engine [src/yolo_verifier.py]"]
        MinArea{"Gate 1: Min Area Ratio >= 1.5%?"}
        MinConf{"Gate 2: Confidence >= 55%?"}
        TopGap{"Gate 3: Top-1 vs Top-2 Gap >= 8%?"}
        Decide["Emit Structured Decision\n(recognized, step, bbox, mask polygon)\n[src/inference.py: VerificationResult]"]
    end

    Cam --> LumaGate
    LumaGate -- "Dark/Covered Frame" --> RejectCam["Reject: camera_placeholder_or_too_dark"]
    LumaGate -- "Valid Frame" --> Letterbox
    Letterbox --> YOLO
    YOLO --> BoxHead & MaskHead
    BoxHead & MaskHead --> MinArea
    MinArea -- "Noise < 1.5% Area" --> Discard["Discard Background Noise"]
    MinArea -- "Candidate Box" --> MinConf
    MinConf -- "< 55% Conf" --> LowConf["Reject: low_confidence"]
    MinConf -- ">= 55% Conf" --> TopGap
    TopGap -- "< 8% Gap" --> Ambiguous["Reject: ambiguous_step"]
    TopGap -- ">= 8% Gap" --> Decide
```

### คลาสเป้าหมาย 6 ขั้นตอน (Origami Crane 6-Step Topology):
* **Chk1 (`Square Base`):** ฐานสี่เหลี่ยมจัตุรัสประกบ 4 ชั้น (กากบาท + ทแยง)
* **Chk2 (`Kite Folds`):** พับสามเหลี่ยมว่าว ขอบข้างสองด้านพับชิดแนวกลาง
* **Chk3 (`Bird Base`):** คลี่กลีบพับ (Petal Fold) รีดแบนเป็นข้าวหลามตัดยาว ปลายล่างแยก 2 แฉก
* **Chk4 (`Neck & Tail Setup`):** ดึงปลายล่างทั้งสองข้างเฉียงขึ้นเฉียงออกเป็นคอและหาง (คอยังตรง)
* **Chk5 (`Head Fold`):** หักพับปลายคอด้านหนึ่งลงมาเพื่อทำหัวนก (Fine-grained geometry)
* **Chk6 (`Completed Crane`):** ดึงปีกกางออก เป่าลม/จัดทรงตัวนกกระเรียนเสร็จสมบูรณ์

---

## 2. กระบวนทัศน์โมเดล: ทำไมต้อง Instance Segmentation (Vision Paradigm Shift)

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`src/yolo_verifier.py`](../../src/yolo_verifier.py): คลาส `YoloStepVerifier` ใช้งานผลลัพธ์ Segmentation (Boxes + Masks)
> - [`src/figure_registry.py`](../../src/figure_registry.py): โรงงาน `FigureRegistry.get_verifier()` ที่เปลี่ยนผ่านจาก `.pth` Classification สู่ `.pt` Segmentation
> - [`src/model.py`](../../src/model.py): สถาปัตยกรรมเดิม MobileNetV2 Image Classification ที่ถูกยกเลิกเนื่องจากปัญหา Shortcut Learning

### 2.1 ความล้มเหลวของ Image Classification (ยุค MobileNetV2)
ในระยะแรก โปรเจกต์เริ่มด้วยการเทรน Image Classification โดยส่งภาพตัดทั้งกรอบขนาด $224 \times 224$ เข้าโมเดล:
1. **Shortcut Learning (Background Bias):** กระดาษโอริกามิกินพื้นที่บนจอเพียง 20–40% อีก 60–80% คือพื้นโต๊ะ กำแพง และแสงไฟ โมเดลจึงแอบจดจำว่า "ถ้าโต๊ะลายไม้สีเข้ม = Chk1" หรือ "ถ้าแสงไฟนีออน = Chk3"
2. **ภาพรวมทั้งเฟรมไม่มี Spatial Awareness:** โมเดลไม่รู้ตำแหน่งว่ากระดาษอยู่ตรงไหน หากผู้ใช้ถือมือบังครึ่งหนึ่ง หรือมีของวางข้างๆ การทำนายจะพังทลายทันที
3. **ขาดความโปร่งใส (Explainability):** ระบบไม่สามารถวาดกรอบยืนยันให้ผู้ใช้เห็นได้ว่าระบบกำลัง "มองเห็น" ตัวกระดาษจริงหรือไม่

### 2.2 การเปลี่ยนผ่านสู่ Instance Segmentation (YOLO11s-seg)
เราจึงปฏิรูปสถาปัตยกรรมสู่ **YOLO11s-seg (Instance Segmentation)** ด้วยเหตุผลทางคณิตศาสตร์และวิศวกรรม:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}$$

* **Mask Loss ($\mathcal{L}_{\text{mask}}$) บังคับ Attention:** Mask Loss (BCE ผ่าน Prototype Mask Network) คิดค่าความผิดพลาดเฉพาะพิกเซลที่อยู่ใน Polygon ของตัวกระดาษเท่านั้น **ทำให้ Gradient ไม่ถูกรบกวนจากพิกเซลพื้นหลัง**
* **Dual Representation:** ได้ทั้ง Bounding Box สำหรับกรองขนาดพื้นที่ และ Polygon Mask สำหรับวาดเส้นขอบเรืองแสงบนเว็บแคมแบบ Real-time
* **ความเร็วระดับ Edge Inference:** YOLO11s-seg ใช้เวลาประมาณ 10–18 ms ต่อเฟรมบน GPU (CUDA FP16) และ 40–60 ms บน Modern CPU ซึ่งตอบโจทย์ Real-time Web Loop

---

## 3. วิศวกรรมชุดข้อมูลและการกำจัด Data Leakage ขั้นเด็ดขาด (Data Engineering & Zero-Leakage Protocol)

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`scripts/dataset/collect_data.py`](../../scripts/dataset/collect_data.py): สคริปต์สกัดเฟรมจากวิดีโอการบันทึกด้วยความถี่คงที่ 3.33 FPS
> - [`scripts/dataset/prepare_v3_bg_dataset.py`](../../scripts/dataset/prepare_v3_bg_dataset.py): สคริปต์จัดสรร Partition ชุดข้อมูลแบบ Session Isolation (70/17/13)
> - [`scripts/evaluation/audit_dataset_leakage.py`](../../scripts/evaluation/audit_dataset_leakage.py): ระบบตรวจสอบ Zero-Leakage Audit 4 ขั้นตอนอัตโนมัติ

### 3.1 ธรรมชาติของข้อมูลวิดีโอ (Video Slicing Dynamics)
ข้อมูลดิบไม่ได้มาจากการถ่ายภาพนิ่ง แต่มาจากการบันทึกวิดีโอการพับกระดาษจริง แล้วนำมาตัดเป็นเฟรมทุกๆ **0.3 วินาที (Frame Rate 3.33 FPS)** ด้วย [`scripts/dataset/collect_data.py`](../../scripts/dataset/collect_data.py):
* **ข้อดี:** ได้มุมมอง การเอียง การหมุน และระยะห่างที่สมจริงขณะมนุษย์กำลังพับ
* **ข้อควรระวังขั้นวิกฤต:** เฟรมที่ห่างกัน 0.3 วินาที ($t$ และ $t+0.3\text{s}$) มีความคล้ายคลึงของพิกเซล (Perceptual Similarity) สูงกว่า **95%**

### 3.2 หลุมพราง Uniform Random Split (Run 5–6)
เมื่อนำข้อมูลทั้งหมดไปสับสุ่ม (Flat Random Split 70% Train, 20% Val, 10% Test):
* เฟรม $t$ อยู่ใน Train แต่เฟรม $t+0.3\text{s}$ ไปโผล่ใน Val
* **ตัวเลขลวงตา:** ได้ Validation mAP สูงถึง **97.2%**
* **ความล้มเหลวในการใช้งานจริง:** เมื่อผู้ใช้เปิดกล้องในห้องใหม่ โมเดลทำนายไม่ได้เลย เพราะโมเดลไม่ได้เรียนรู้งานพับ แต่ **Overfit สภาพแวดล้อมเฉพาะห้องเดิม**

### 3.3 โครงสร้าง Balanced 70/17/13 Physical Session Split
เราจัดระเบียบโครงสร้างใหม่ใน [`scripts/dataset/prepare_v3_bg_dataset.py`](../../scripts/dataset/prepare_v3_bg_dataset.py) โดยแบ่งกลุ่มตาม **สภาพแวดล้อมทางกายภาพ (Physical Recording Sessions)** อย่างเด็ดขาด:

| Session ID | สภาพแวดล้อม (Physical Environment) | จำนวนภาพดิบ | จัดสรรอยู่ใน Split | สัดส่วน |
|---|---|:---:|:---:|:---:|
| `session_001` | Xplor Computer Lab (กลางคืน, ไฟนีออนขาว, โต๊ะสีดำ) | 238 | **TRAIN** | |
| `session_002` | Xplor Computer Lab (กลางวัน, แสงธรรมชาติ, โต๊ะสีดำ) | 243 | **TRAIN** | **70.0%** |
| `session_003` | Bedroom (กลางคืน, ไฟห้องเพดาน, มุมมองพื้นห้อง) | 1,441 | **TRAIN** | (3,296 ภาพ |
| `session_005` | Bedroom (กลางคืน, โคมไฟสีส้ม Warm Light คอนทราสต์สูง) | 1,374 | **TRAIN** | + 352 BG) |
| `session_004` | Bedroom (กลางวัน, แสงแดดส่องตรงจากหน้าต่างกระจกดำ) | 811 | **VAL (Unseen)** | **17.2%** (356 ภาพ) |
| `session_006` | Daylight Highly Glossy Floor Tiles (กระเบื้องเงาสะท้อนแสง) | 600 | **TEST (Unseen)** | **12.7%** (279 ภาพ) |

```
DATASET DISTRIBUTION (configs/origami_seg_v3_bg_balanced.yaml):
├── TRAIN (3,648 imgs): S1 + S2 + S3 + S5 (3,296 origami) + 352 Authentic BG (9.65%)
├── VAL   (356 imgs):   S4 Unseen Bedroom (331 origami) + 25 BG
└── TEST  (279 imgs):   S6 Unseen Glossy Floor (258 origami) + 21 BG
```

### 3.4 ระบบตรวจสอบ Zero-Leakage Audit 4 ขั้นตอน
ระบบมีสคริปต์ตรวจสอบความสะอาดของชุดข้อมูลอัตโนมัติใน [`scripts/evaluation/audit_dataset_leakage.py`](../../scripts/evaluation/audit_dataset_leakage.py):
1. **1:1 Alignment:** จำนวนไฟล์ภาพ (`.jpg`, `.png`) และไฟล์ Label (`.txt`) ในแต่ละ Split ต้องตรงกัน 1 ต่อ 1 ไร้ไฟล์กำพร้า
2. **Session Isolation:** สกัด Regex `^(session_\d+)` จากชื่อไฟล์ ตรวจสอบว่าไม่มี Session ID เดียวกันปรากฏข้ามโฟลเดอร์ Split
3. **0-Byte Verification:** ไฟล์ Label ที่เป็นภาพพื้นหลัง (Background) ต้องมีขนาดไฟล์เท่ากับ **0 bytes พอดี**
4. **Cryptographic MD5 Hash Exclusivity:** คำนวณ MD5 Checksum ของทุกไฟล์ภาพใน Train, Val, Test และหาค่า Intersection:

$$\text{Set}(\text{MD5}_{\text{train}}) \cap \text{Set}(\text{MD5}_{\text{val}}) = \emptyset$$
$$\text{Set}(\text{MD5}_{\text{train}}) \cap \text{Set}(\text{MD5}_{\text{test}}) = \emptyset$$

---

## 4. เกราะป้องกันภาพหลอน: Hard-Negative Ingestion & Background Hardening

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`scripts/dataset/extract_background_candidates.py`](../../scripts/dataset/extract_background_candidates.py): สคริปต์สกัดภาพ Hard-Negative (คีย์บอร์ด, สติ๊กเกอร์, เสื้อผ้า) และสร้าง Label ว่างเปล่า 0-byte
> - [`scripts/dataset/extract_color_seq.py`](../../scripts/dataset/extract_color_seq.py): สกัดเฟรมจากวิดีโอทดสอบกระดาษสีเพื่อวิเคราะห์ False Positives
> - [`configs/origami_seg_v3_bg_balanced.yaml`](../../configs/origami_seg_v3_bg_balanced.yaml): กำหนดเส้นทางชุดข้อมูลที่มีภาพพื้นหลังผสานอยู่

### 4.1 ปัญหา False Positives รุนแรงในโลกจริง
กระดาษโอริกามิเป็นวัตถุที่มีขอบคม สันพับ และโทนสีขาว/สว่าง เมื่อนำโมเดลยุค Run 7–9 ไปทดสอบในห้องจริง โมเดลตรวจจับวัตถุอื่นผิดพลาด:
* **สติ๊กเกอร์สีขาวบนคีย์บอร์ดโน้ตบุ๊ก** $\rightarrow$ โมเดลทักว่าเป็น `Chk1` หรือ `Chk2` ด้วยความมั่นใจสูงถึง **82.4%**
* **ปุ่มคีย์บอร์ดสีดำ/ขาว, เสื้อผ้า, ปลอกหมอน, พัดลม** $\rightarrow$ ก่อให้เกิด False Positive รบกวน

### 4.2 กลไก 0-Byte Label ใน Ultralytics YOLO
ในมาตรฐานของ YOLO การใส่ภาพที่มีเฉพาะพื้นหลังโดยไม่มีวัตถุ ทำได้โดยการสร้างไฟล์ `.txt` ที่มีขนาด **0 bytes (Empty File)** ผ่านฟังก์ชัน `save_bg_sample()` ใน [`scripts/dataset/extract_background_candidates.py`](../../scripts/dataset/extract_background_candidates.py):
* เมื่อคำนวณ Loss ฟังก์ชัน Classification และ Box Regression จะสร้าง **Negative Gradient Penalization**:
  "หากเห็นพื้นผิวเหล่านี้ จงอย่าส่งค่า Bounding Box ใดๆ ออกมา"

```text
data/origami_seg_v3_bg_balanced/
├── images/train/
│   ├── session_001_frame_01.jpg
│   └── bg_laptop_keyboard_white_sticker_01.jpg  <-- ภาพสิ่งรบกวน
└── labels/train/
    ├── session_001_frame_01.txt                 <-- พิกัด Polygon
    └── bg_laptop_keyboard_white_sticker_01.txt  <-- 0 BYTES (EMPTY)
```

### 4.3 สัดส่วนทองคำของ Background Ratio
ตามเอกสาร Ultralytics Best Practices แนะนำให้ใส่ภาพพื้นหลังในสัดส่วน $1 - 10\%$:
* ในชุด Train ของ Run 11: มีภาพ Background แท้จริง **352 ภาพ จากทั้งหมด 3,648 ภาพ** คิดเป็น **$9.65\%$**
* **ผลลัพธ์เชิงตัวเลข:**
  * **Clean Suppression Rate (at Conf = 0.25):** เพิ่มขึ้นจาก $0\%$ สู่ **$100.0\%$**
  * **Max False Positive Confidence:** ลดลงจาก $82.4\%$ เหลือเพียง **$15.04\%$**
  * เมื่อตั้งเกณฑ์ระบบไว้ที่ $\ge 55\%$ ใน [`src/yolo_verifier.py`](../../src/yolo_verifier.py) ทำให้เกิด **Safety Margin กว้างถึง $40\%$** ป้องกันการหลอน 100%

---

## 5. วิวัฒนาการการทดลองและการฝึกสอน (The 3 Pillar Experimental Runs)

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`scripts/evaluation/eval_and_benchmark.py`](../../scripts/evaluation/eval_and_benchmark.py): สคริปต์สำหรับรันการประเมินผลบน Unseen Test Set (`session_006`) และวัด Suppression Rate บนภาพรบกวน 217 ตัวอย่าง
> - [`scripts/utils/visualize_per_class.py`](../../scripts/utils/visualize_per_class.py): สคริปต์ตรวจสอบการวาดกล่อง/Polygon ตัวแทนของทั้ง 6 คลาส
> - [`scripts/dataset/visualize_labels.py`](../../scripts/dataset/visualize_labels.py): สคริปต์สุ่มตรวจความถูกต้องของ Ground Truth Annotations

| มิติการวัดผล (Evaluation Metrics) | Run 9 (Baseline)<br>v1 Balanced Split | Run 10 (+BG Hardening)<br>v2 Balanced + Hard-Neg | Run 11 (Current SOTA)<br>v3 Pre-Aug + BG Balanced |
|---|:---:|:---:|:---:|
| **แนวคิดการทดลอง** | Zero-Leakage ดั้งเดิม ไม่มีภาพ BG | ใส่ภาพฉากหลัง Hard-Negative 352 รูป | ผสาน Pre-Augment หมุนหลายมุม + BG |
| **จำนวนภาพ Train** | 1,374 ภาพ | 2,361 ภาพ (รวม BG) | **3,648 ภาพ** |
| **Test Set Mask mAP50 (`session_006`)** | 92.75% | 92.40% | **93.45% (All-Time SOTA)** |
| **Test Set Mask mAP50-95** | 71.30% | 82.80% (+11.5%) | **84.57% (+13.3% vs Run 9)** |
| **Test Set Mask Precision** | 94.48% | 91.50% | **90.99%** |
| **Test Set Mask Recall** | 88.10% | 87.20% | **88.68%** |
| **Step 5 (Chk5: Head Fold) mAP50** | 81.52% | 77.20% | **87.40% (+10.2% Breakthrough)** |
| **Max False Positive Confidence** | 82.40% (หลงคีย์บอร์ด) | **10.39%** | **15.04%** |
| **Suppression Rate (at Conf $\ge$ 0.25)** | 0% (มีกรอบลวง) | **100.0%** | **100.0%** |

### บทวิเคราะห์การก้าวกระโดดของ Run 11:
1. **การทะลวงคอขวด Chk5 (Head Fold):** ใน Run 10 พบว่า Chk5 คะแนนตกลงเหลือ 77.2% เพราะรอยพับหัวนกเป็นเพียงการหักมุมปลายแหลมเล็กๆ ของคอ (Chk4 คือคอตรง Chk5 คือหักหัวนก) เมื่อมีภาพพื้นหลังเข้ามา โมเดลเกิดความลังเล
2. **พลังของ Pre-Augmentation:** การเติมภาพที่ผ่านการหมุนแบบ Geometric Multi-Angle และ Mirror Flip ในชุด Train v3 ทำให้โมเดลเห็นรอยพับปลายแหลมในทุกมุมระนาบ ส่งผลให้ Chk5 กระโดดขึ้นเป็น **87.40% (+10.2%)**
3. **mAP50-95 สูงถึง 84.57%:** แสดงถึงความคมชัดของขอบ Polygon Mask ที่แนบสนิทกับตัวกระดาษบนพื้นผิวกระเบื้องเงาสะท้อนแสง (Glossy Floor Tiles)

---

## 6. การตั้งค่า Hyperparameters และกลยุทธ์ Data Augmentation

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`scripts/training/train_yolo_seg.py`](../../scripts/training/train_yolo_seg.py): ฟังก์ชัน `main()` และการกำหนดอาร์กิวเมนต์ `--device`, `--amp`, `--batch`, `--imgsz`
> - [`configs/origami_seg_v3_bg_balanced.yaml`](../../configs/origami_seg_v3_bg_balanced.yaml): ระบุโครงสร้างเส้นทางข้อมูลและจำนวนคลาส

### 6.1 ตาราง Hyperparameters สำคัญ (Production Configuration):
* **Base Architecture:** `yolo11s-seg.pt` (Small Segmentation model, 35M parameters, balance ความเร็วและความแม่นยำ)
* **Image Size (`imgsz`):** `640` (ปรับด้วย Letterboxing)
* **Batch Size (`batch`):** `16`
* **Training Epochs (`epochs`):** `100` (มี Early Stopping `patience=20`)
* **Precision (`amp`):** `True` (Automatic Mixed Precision FP16 บน CUDA)
* **Seed:** `0` (Deterministic Reproducibility)
* **Workers:** `0` (ตั้งเป็น 0 สำหรับ Windows PyTorch ป้องกันปัญหา Shared Memory Deadlock)

### 6.2 การปรับแต่ง Data Augmentation สำหรับงานโอริกามิใน `train_yolo_seg.py`:

```python
AUGMENTATION_CONFIG = {
    # Color & Lighting Invariance: ป้องกันโมเดลจำสีกระดาษเฉพาะเจาะจง
    "hsv_h": 0.50,         # Hue Jitter กว้าง 50% เพื่อให้กระดาษเปลี่ยนสีข้ามสเปกตรัม (ขาว, แดง, ฟ้า, เหลือง)
    "hsv_s": 0.70,         # Saturation Jitter 70% รองรับกระดาษสีซีดและสีสด
    "hsv_v": 0.40,         # Value/Brightness Jitter 40% รองรับสภาพแสงจ้าและแสงสลัว

    # Geometric Transformations: รักษาระนาบกล้อง Top-down/Frontal
    "degrees": 15.0,       # หมุนเอียง +/- 15 องศา (ไม่หมุน 180 เพราะทิศทางของรอยพับสำคัญ)
    "translate": 0.10,     # ขยับตำแหน่ง +/- 10%
    "scale": 0.20,         # ย่อ-ขยายระยะใกล้ไกล +/- 20%
    "shear": 5.0,          # ดึงมุมเอียง +/- 5 องศา
    "perspective": 0.0005, # การบิดเบือนมุมมองสามมิติเล็กน้อย
    "flipud": 0.0,         # ปิดการพลิกกลับหัว (ผู้ใช้พับบนโต๊ะ ห้ามกลับหัวบนล่าง)
    "fliplr": 0.5,         # พลิกซ้าย-ขวา 50% (รอยพับนกกระเรียนมีความสมมาตรซ้ายขวา)

    # Multi-Image Context:
    "mosaic": 1.0,         # รวม 4 ภาพต่อเฟรม เพื่อให้โมเดลคุ้นเคยกับวัตถุขนาดเล็กและหลากหลายบริบท
    "mixup": 0.10,         # ซ้อนทับภาพแบบโปร่งแสง 10%
}
```

---

## 7. ระบบคัดกรองความมั่นใจ 3 ชั้นและการตัดเฟรมเสีย (Inference Gating & Decision Pipeline)

การนำผลจากโมเดล YOLO ไปใช้ตรงๆ ในเว็บแอปพลิเคชันจะทำให้ผู้ใช้เกิดความสับสน (Jittering, Flickering, False Alarm) เราจึงออกแบบ **Decision & Gating Pipeline** ไว้ใน [`src/inference.py`](../../src/inference.py) และ [`src/yolo_verifier.py`](../../src/yolo_verifier.py):

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้อง:**
> - [`src/inference.py`](../../src/inference.py): ฟังก์ชัน `assess_frame_readiness()` ตรวจสอบค่า Luma/Contrast และคลาส `VerificationResult`
> - [`src/yolo_verifier.py`](../../src/yolo_verifier.py): เมธอด `verify_bgr()` และ `_filter_detections()` บังคับใช้เกณฑ์ Area Ratio, Confidence และ Top-Gap
> - [`frontend/src/hud-overlay.ts`](../../frontend/src/hud-overlay.ts): คลาส `DetectionOverlayRenderer` แปลง RoI/Mask ไปวาดบนหน้าจอ
> - [`tests/test_yolo_verifier.py`](../../tests/test_yolo_verifier.py): ชุดทดสอบ Unit Tests ยืนยันตรรกะของ Ambiguity Gating และ Confidence Thresholds

### 7.1 ด่านที่ 0: Hardware Frame Readiness Assessment ([`src/inference.py`](../../src/inference.py))
ก่อนส่งเข้าโมเดล YOLO ระบบจะตรวจสอบคุณภาพพิกเซลของภาพเว็บแคมก่อน:
```python
mean_luma = float(np.mean(gray))
contrast = float(np.std(gray))
dark_ratio = float(np.mean(gray < 25))
```
* **ตัด Placeholder ของกล้องเสมือน (Iriun Webcam):** ถ้า `mean_luma < 22` และ `dark_ratio > 0.70` $\rightarrow$ ปฏิเสธทันทีด้วยสถานะ `camera_placeholder_or_too_dark` (ไม่ให้โมเดลเดาจากหน้าจอดำ)
* **ตัดภาพไร้รายละเอียด (เช่น มือปิดเลนส์):** ถ้า `contrast < 6` $\rightarrow$ ปฏิเสธด้วยสถานะ `frame_has_too_little_detail`

### 7.2 ด่านที่ 1: Minimum Area Ratio Gating ($\ge 1.5\%$) ([`src/yolo_verifier.py`](../../src/yolo_verifier.py))
เมื่อ YOLO ตรวจพบ Bounding Box ใดๆ จะต้องผ่านการตรวจสัดส่วนพื้นที่ต่อเฟรมกล้อง:

$$\text{Area Ratio} = \frac{\text{Width}_{\text{box}} \times \text{Height}_{\text{box}}}{\text{Width}_{\text{frame}} \times \text{Height}_{\text{frame}}} \ge 0.015 \quad (1.5\%)$$

* **เหตุผล:** ในสภาพแวดล้อมจริง อาจมีเศษกระดาษสีขาวเล็กๆ อยู่ไกลๆ หรือสติ๊กเกอร์เล็กบนตู้เสื้อผ้า หากพื้นที่ต่ำกว่า $1.5\%$ จะถูกตัดทิ้งทันทีว่าเป็น Background Noise

### 7.3 ด่านที่ 2: Minimum Confidence Threshold ($\ge 55\%$) ([`src/yolo_verifier.py`](../../src/yolo_verifier.py))
กำหนดค่าความเชื่อมั่นขั้นต่ำของคลาสอันดับหนึ่ง:

$$\text{Confidence}(c_1) \ge 0.55 \quad (55.0\%)$$

* **เหตุผล:** จากผล Benchmark ของ Run 11 วัตถุรบกวนที่ร้ายแรงที่สุด (Worst Case: สติ๊กเกอร์คีย์บอร์ด) โมเดลให้ความมั่นใจสูงสุดเพียง **$15.04\%$**
* ดังนั้น เกณฑ์ $55\%$ จึงมี **Safety Margin สูงถึง $39.96\%$** ทำให้มั่นใจได้ $100\%$ ว่าจะไม่มีภาพหลอนหลุดไปถึงผู้ใช้

### 7.4 ด่านที่ 3: Ambiguity Gating / Top-Gap Threshold ($\ge 8\%$) ([`src/yolo_verifier.py`](../../src/yolo_verifier.py))
เมื่อผู้ใช้กำลังเปลี่ยนผ่านระหว่างขั้นตอน เช่น กำลังจะพับจาก Chk4 (ดึงคอ) ไป Chk5 (พับหัวนก) รอยพับจะอยู่กึ่งกลาง:

$$\text{Top Gap} = \left(\text{Conf}_{\text{Top 1}} - \text{Conf}_{\text{Top 2}}\right) \times 100\% \ge 8.0\%$$

* **หากคะแนนสูสีกัน:** เช่น Chk4 ได้ 63% และ Chk5 ได้ 59% (ผลต่างคือ 4% ซึ่ง $< 8\%$)
* ระบบจะปรับสถานะ `recognized = False` พร้อมส่งเหตุผล `recognition_reason = "ambiguous_step"`
* **ผลลัพธ์เชิง UX:** ผู้ใช้ไม่สามารถ "โกง" หรือผ่านขั้นตอนได้ หากยังพับไม่เรียบร้อยและคมชัดจริง

---

## 8. คู่มือปฏิบัติการทีละสเต็ปสำหรับ AI Engineer (Step-by-Step Operational Runbook)

หากต้องการเทรนโมเดลใหม่ หรือต้องการเพิ่มตัวโอริกามิชนิดใหม่ ให้ทำตามคู่มือปฏิบัติด้านล่างนี้:

### สเต็ปที่ 1: การเตรียม Dataset และสกัดวิดีโอ (Data Preparation)
1. บันทึกวิดีโอการพับกระดาษแยกตามสถานที่ (เช่น `session_007_lab`, `session_008_room`)
2. วางวิดีโอลงใน `data/raw_videos/` แล้วรันสคริปต์สกัดเฟรม (3.33 FPS) ด้วย [`scripts/dataset/collect_data.py`](../../scripts/dataset/collect_data.py):
   ```powershell
   python scripts/dataset/collect_data.py --input-dir data/raw_videos --fps 3.33
   ```
3. นำภาพไปทำ Polygon Annotation บน Roboflow หรือ CVAT ตามคลาส `Chk1` ถึง `Chk6`
4. ดาวน์โหลด Label ออกมาในรูปแบบ **YOLOv11 Segmentation (.txt)**

### สเต็ปที่ 2: การสร้างชุดข้อมูลและผสานภาพพื้นหลัง (Hard-Negative Merging)
รันสคริปต์ [`scripts/dataset/prepare_v3_bg_dataset.py`](../../scripts/dataset/prepare_v3_bg_dataset.py) เพื่อสร้างชุดข้อมูล Zero-Leakage พร้อมผสานภาพพื้นหลัง 0-byte:
```powershell
# สร้างชุดข้อมูล v3 พร้อมสัดส่วน 70/17/13 และ Hard-Negatives
python scripts/dataset/prepare_v3_bg_dataset.py
```
* สคริปต์จะสร้างโฟลเดอร์ `data/origami_seg_v3_bg_balanced/`
* และสร้างไฟล์คอนฟิก [`configs/origami_seg_v3_bg_balanced.yaml`](../../configs/origami_seg_v3_bg_balanced.yaml)

### สเต็ปที่ 3: ตรวจสอบ Zero-Leakage ทางคณิตศาสตร์ (Mandatory Audit)
**ข้อบังคับ:** ห้ามเริ่มเทรนโมเดลเด็ดขาดหากยังไม่ผ่าน Audit ด้วย [`scripts/evaluation/audit_dataset_leakage.py`](../../scripts/evaluation/audit_dataset_leakage.py):
```powershell
python scripts/evaluation/audit_dataset_leakage.py --dir data/origami_seg_v3_bg_balanced
```
* ตรวจสอบว่าผลลัพธ์ขึ้น `ALL AUDIT CHECKS PASSED: DATASET IS 100% CLEAN OF DATA LEAKAGE!` (0 hash collisions, 0 session leaks)

### สเต็ปที่ 4: เริ่มการฝึกสอนโมเดล (Model Training)
รันคำสั่งเทรน YOLO11s-seg ด้วยสคริปต์ [`scripts/training/train_yolo_seg.py`](../../scripts/training/train_yolo_seg.py):
```powershell
python scripts/training/train_yolo_seg.py `
  --data configs/origami_seg_v3_bg_balanced.yaml `
  --model yolo11s-seg.pt `
  --epochs 100 `
  --batch 16 `
  --imgsz 640 `
  --patience 20 `
  --device 0 `
  --amp `
  --name origami_yolo_run11_yolo11s-seg_v3_bg
```
* น้ำหนักโมเดลที่ดีที่สุดจะถูกบันทึกไว้ที่:
  `checkpoints/yolo_runs/origami_yolo_run11_yolo11s-seg_v3_bg/weights/best.pt`

### สเต็ปที่ 5: การประเมินผลและการวัด False Positive Benchmark
รันการประเมินผลบน Test Set ที่ไม่เคยเห็นมาก่อน พร้อมทดสอบกับภาพสิ่งรบกวน 217 ตัวอย่างด้วย [`scripts/evaluation/eval_and_benchmark.py`](../../scripts/evaluation/eval_and_benchmark.py):
```powershell
python scripts/evaluation/eval_and_benchmark.py `
  --name origami_yolo_run11_yolo11s-seg_v3_bg `
  --data configs/origami_seg_v3_bg_balanced.yaml `
  --model checkpoints/yolo_runs/origami_yolo_run11_yolo11s-seg_v3_bg/weights/best.pt `
  --device 0
```
* ตรวจสอบว่า **Clean Suppression Rate (Conf=0.50)** ต้องได้ **$100.0\%$**
* และ **Max False Positive Confidence** ต้องต่ำกว่า **$20.0\%$**

### สเต็ปที่ 6: การติดตั้งโมเดลเข้าสู่ระบบ Production (Zero-Downtime Deployment)
1. ตรวจสอบว่าไฟล์ `best.pt` อยู่ในโฟลเดอร์ `checkpoints/`
2. อัปเดตพาธในไฟล์คอนฟิก [`configs/figures/crane.json`](../../configs/figures/crane.json):
   ```json
   {
     "id": "crane",
     "model_path": "checkpoints/yolo_runs/origami_yolo_run11_yolo11s-seg_v3_bg/weights/best.pt",
     "total_steps": 6
   }
   ```
3. สั่งรันเซิร์ฟเวอร์ `FastAPI` ผ่านสคริปต์สตาร์ทเตอร์ [`start_server.bat`](../../start_server.bat) หรือ [`gamisenpai.py`](../../gamisenpai.py):
   ```powershell
   python gamisenpai.py
   ```
4. ทดสอบความพร้อมผ่าน API:
   ```bash
   curl http://localhost:8000/api/model-info?figure=crane
   ```

---

## 9. บทเรียนทางวิศวกรรมและกับดักที่ห้ามทำ (AI Post-Mortem & Anti-Patterns)

> 📂 **ไฟล์โค้ดหลักที่เกี่ยวข้องกับการป้องกันข้อผิดพลาด:**
> - ป้องกัน Leakage: [`scripts/evaluation/audit_dataset_leakage.py`](../../scripts/evaluation/audit_dataset_leakage.py)
> - ป้องกันภาพหลอน: [`scripts/dataset/extract_background_candidates.py`](../../scripts/dataset/extract_background_candidates.py)
> - ป้องกันสัดส่วนภาพเพี้ยน: [`src/yolo_verifier.py`](../../src/yolo_verifier.py) (ฟังก์ชัน Letterbox)
> - ป้องกันการกลับหัวรอยพับ: [`scripts/training/train_yolo_seg.py`](../../scripts/training/train_yolo_seg.py) (`flipud: 0.0`)

### 9.1 กับดักที่ 1: "The Metric Mirage" (ภาพลวงตาของ mAP 98%)
* **ความผิดพลาด:** การนำภาพจากวิดีโอไปแบ่ง Train/Val/Test แบบ Flat Random ทำให้ภาพเฟรมติดกันกระจายไปทุก Split
* **ผลลัพธ์:** ได้ Validation mAP สูงลิบ 98% แต่พอผู้ใช้จริงเปิดกล้องในห้องใหม่ โมเดลพังทลายลง 100%
* **กฎทอง:** **"แบ่งตาม Physical Session สถานที่จริงเสมอ และรัน MD5 Exclusivity Hash Check ด้วย [`scripts/evaluation/audit_dataset_leakage.py`](../../scripts/evaluation/audit_dataset_leakage.py) ทุกครั้ง"**

### 9.2 กับดักที่ 2: ละเลยภาพพื้นหลังล้วน (Pure Negative Omission)
* **ความผิดพลาด:** เทรนโมเดลด้วยภาพที่มีกระดาษโอริกามิ 100% โดยไม่มีภาพพื้นหลังเปล่าเลย
* **ผลลัพธ์:** โมเดลจะกลายเป็น "Unconditional Guesser" เมื่อเห็นอะไรที่มีสีขาวหรือทรงเรขาคณิต (เช่น สติ๊กเกอร์คีย์บอร์ด, พัดลม) มันจะบังคับทายว่าเป็นรอยพับก้าวใดก้าวหนึ่งเสมอ
* **กฎทอง:** **"ต้องใส่ภาพ Hard-Negative (0-byte labels) อย่างน้อย 8–10% ในชุด Train เสมอ โดยดึงจาก [`scripts/dataset/extract_background_candidates.py`](../../scripts/dataset/extract_background_candidates.py)"**

### 9.3 กับดักที่ 3: การยืดภาพแบบไม่รักษาสัดส่วน (Direct Image Stretch vs Letterboxing)
* **ความผิดพลาด:** เว็บแคมผู้ใช้ส่งภาพสัดส่วน $16:9$ ($1280 \times 720$) หรือ $4:3$ เข้ามา แล้วจับ Resize ตรงๆ เข้าสู่ $640 \times 640$
* **ผลลัพธ์:** กระดาษโอริกามิจะถูกบีบอัดจนเสียรูปทรงเรขาคณิต เช่น สามเหลี่ยมว่าว (Chk2) ถูกบีบจนแคบกลายเป็นฐานนก (Chk3) ทำให้โมเดลสับสน
* **กฎทอง:** **"ต้องใช้ Letterboxing ใน [`src/yolo_verifier.py`](../../src/yolo_verifier.py) (รักษาสัดส่วนเดิมแล้วเติมขอบ Padding เทา) ทั้งในขั้นตอนการเทรนและขั้นตอน Inference บนเว็บแอปเสมอ"**

### 9.4 กับดักที่ 4: สลับหัวท้ายด้วย Upside-Down Flip (FlipUD)
* **ความผิดพลาด:** เปิดการ Augment พลิกภาพแนวดิ่ง (`flipud: 0.5`)
* **ผลลัพธ์:** ในศิลปะการพับกระดาษ ทิศทางของปลายเปิดและปลายแหลมมีความหมายอย่างยิ่ง (เช่น ปลายเปิดต้องชี้ลงด้านล่าง) การกลับหัวภาพทำให้โมเดลสับสนระหว่าง "ฐานบน" กับ "ปลายล่าง"
* **กฎทอง:** **"ตั้ง `flipud: 0.0` ใน [`scripts/training/train_yolo_seg.py`](../../scripts/training/train_yolo_seg.py) เสมอ และอนุญาตเฉพาะการพลิกแนวนอน (`fliplr: 0.5`) เท่านั้น"**

---

## 10. เอกสารอ้างอิงและประวัติศาสตร์ (Cross References)
* ประวัติการทดลองและ Benchmark รายงานฉบับเต็ม: [`../ORIGAMI_YOLO_RUN11_YOLO11S-SEG_V3_BG_BENCHMARK_REPORT.md`](../ORIGAMI_YOLO_RUN11_YOLO11S-SEG_V3_BG_BENCHMARK_REPORT.md)
* ประวัติวิวัฒนาการการแบ่งชุดข้อมูล: [`../DATASET_SPLIT_AND_LEAKAGE_HISTORY.md`](../DATASET_SPLIT_AND_LEAKAGE_HISTORY.md)
* สถาปัตยกรรมระบบรวมทั้งระบบ: [`../README.md`](README.md)
* คอนฟิกชุดข้อมูล SOTA: [`../../configs/origami_seg_v3_bg_balanced.yaml`](../../configs/origami_seg_v3_bg_balanced.yaml)
* ซอร์สโค้ด Inference Engine: [`../../src/inference.py`](../../src/inference.py) และ [`../../src/yolo_verifier.py`](../../src/yolo_verifier.py)
* เว็บเซิร์ฟเวอร์หลัก: [`../../app/app.py`](../../app/app.py)
* ระบบ CI Workflow อัตโนมัติ: [`../../.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
