# รายงานผลการประเมินและเปรียบเทียบ Run 10 (YOLO11s-seg + Backgrounds)
## (Run 10 Benchmark & False Positive Suppression Report)

**โครงการ:** GamiSenpai (Interactive Origami AI Tutor)  
**วันที่ประเมิน:** 27 กันยายน 2026  
**โมเดล:** YOLO11s-seg (Run 10 with 398 Authentic Backgrounds)  
**ชุดข้อมูล:** `data/origami_seg_v2_bg_balanced/` (20.4% Train Backgrounds)  
**เกณฑ์การแบ่ง:** Strict Zero-Leakage Balanced Session Split (Train: 001,002,003,005 | Val: 004 | Test: 006)

---

## 1. ผลการเปรียบเทียบตัวชี้วัด (Run 9 Baseline vs Run 10 Backgrounds)

| ตัวชี้วัด (Metric) | Run 9 (No BG)<br>v1 Balanced Split | Run 10 (+398 BG)<br>v2 Balanced Split | ผลต่างทางสถิติ (Delta) | การวิเคราะห์ทางวิศวกรรม |
|---|:---:|:---:|:---:|---|
| **Val Set Mask mAP50** | **95.44%** | **93.83%** | **-1.61%** | ประเมินบน session_004 (Unseen Bedroom Window) |
| **Val Set Mask mAP50-95** | 76.50% | **89.09%** | +12.59% | ความคมชัดของขอบ Polygon Mask |
| **Val Set Mask Precision** | 97.64% | **94.72%** | -2.92% | ความแม่นยำ ปราศจาก False Positive |
| **Val Set Mask Recall** | 94.06% | **87.10%** | -6.96% | ความสามารถในการตรวจพบกระดาษ |
| **Test Set Mask mAP50** | **92.75%** | **92.43%** | **-0.32%** | ประเมินบน session_006 (Unseen Glossy Floor) |
| **Test Set Mask mAP50-95** | 71.30% | **82.82%** | +11.52% | คุณภาพ Mask บนพื้นผิวมันสะท้อนแสง |
| **Test Set Mask Precision** | 94.48% | **91.53%** | -2.95% | Precision บนห้องที่ไม่เคยเห็นมาก่อน |
| **Test Set Mask Recall** | 88.10% | **87.16%** | -0.94% | Recall บนห้องที่ไม่เคยเห็นมาก่อน |

---

## 2. ผลการทดสอบการขจัด False Positive บนสิ่งรบกวน (Hard Negative Suppression Benchmark)

ทดสอบกับตัวอย่าง Background ของจริง (21 ตัวอย่าง) เช่น สติ๊กเกอร์สีขาวบนคีย์บอร์ดโน้ตบุ๊ก, ปลั๊กไฟ, แผ่นกระเบื้อง, และสิ่งของในห้อง:

* **Clean Suppression Rate (at Conf = 0.25):** **`95.24%`** (ภาพพื้นหลังที่ไม่เกิด False Positive เลย)
* **Clean Suppression Rate (at Conf = 0.50):** **`100.0%`**
* **Max Confidence Observed บนสิ่งรบกวน:** **`44.97%`**
* **Worst False Positive Case:** `{'image': 'bg_session_006_noon_floor_v22_f5_noon_floor.jpg', 'class': 'Chk5', 'conf': 0.4497}`

> **สรุปผลลัพธ์:** การใส่ 0-byte Background Images เข้าไปในสัดส่วน 20.4% สามารถกด Confidence ของวัตถุรบกวน (เช่น สติ๊กเกอร์คีย์บอร์ดที่เคยทริกเกอร์หลอนใน Run 8) ให้ลดลงอย่างชัดเจน ทำให้ระบบ UI ไม่เกิดการสับสนหรือค้างที่ Checkpoint หลอนอีกต่อไป
