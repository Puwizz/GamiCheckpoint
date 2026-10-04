# รายงานผลการประเมินและเปรียบเทียบ origami_yolo_run11_yolo11s-seg_v3_bg
## (Model Benchmark & False Positive Suppression Report)

**โครงการ:** GamiSenpai (Interactive Origami AI Tutor)  
**ชื่อการทดลอง:** `origami_yolo_run11_yolo11s-seg_v3_bg`  
**เกณฑ์การแบ่ง:** Strict Zero-Leakage Balanced Session Split (Val: session_004 | Test: session_006)

---

## 1. ผลการเปรียบเทียบตัวชี้วัด (Benchmark Comparison)

| ตัวชี้วัด (Metric) | Run 9 (No BG)<br>v1 Balanced | Run 10 (+BG)<br>v2 Balanced | origami_yolo_run11_yolo11s-seg_v3_bg<br>(Current Run) | ผลต่างเทียบ Run 10 | การวิเคราะห์ทางวิศวกรรม |
|---|:---:|:---:|:---:|:---:|---|
| **Val Set Mask mAP50** | 95.44% | 93.82% | **91.49%** | -2.33% | ประเมินบน session_004 (Unseen Bedroom) |
| **Val Set Mask mAP50-95** | 76.50% | 89.13% | **85.92%** | -3.21% | ความคมชัดของขอบ Polygon Mask |
| **Val Set Mask Precision** | 97.64% | 94.70% | **91.40%** | -3.30% | ความแม่นยำ ปราศจาก False Positive |
| **Val Set Mask Recall** | 94.06% | 87.10% | **80.07%** | -7.03% | ความสามารถในการตรวจพบกระดาษ |
| **Test Set Mask mAP50** | 92.75% | 92.40% | **93.45%** | +1.05% | ประเมินบน session_006 (Unseen Glossy Floor) |
| **Test Set Mask mAP50-95** | 71.30% | 82.80% | **84.57%** | +1.77% | คุณภาพ Mask บนพื้นผิวมันสะท้อนแสง |
| **Test Set Mask Precision** | 94.48% | 91.50% | **90.99%** | -0.51% | Precision บนห้องที่ไม่เคยเห็นมาก่อน |
| **Test Set Mask Recall** | 88.10% | 87.20% | **88.68%** | +1.48% | Recall บนห้องที่ไม่เคยเห็นมาก่อน |

---

## 2. ผลการทดสอบการขจัด False Positive บนสิ่งรบกวน (Hard Negative Suppression)

ทดสอบกับตัวอย่าง Background ของจริง (217 ตัวอย่าง) เช่น สติ๊กเกอร์สีขาวบนคีย์บอร์ดโน้ตบุ๊ก, ปุ่มคีย์บอร์ด, พัดลม, เตียงนอน และสิ่งของในห้อง:

* **Clean Suppression Rate (at Conf = 0.25):** **`100.0%`**
* **Clean Suppression Rate (at Conf = 0.50):** **`100.0%`**
* **Max Confidence Observed บนสิ่งรบกวน:** **`15.04%`**
* **Worst False Positive Case:** `{'image': 'bg_session_006_noon_floor_v22_f5_noon_floor.jpg', 'class': 'Chk5', 'conf': 0.1504}`
