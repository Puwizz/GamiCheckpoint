"""
sync_background_labels.py — Sync Background Labels and Refresh Review Artifacts
================================================================================
Deletes orphan label files (where user deleted the .jpg image), verifies 1:1 match,
and regenerates BACKGROUND_REVIEW_REPORT.md and review_gallery.html.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = ROOT / "data" / "background_candidates"


def main():
    print("=" * 60)
    print("SYNCING BACKGROUND LABELS AND REFRESHING ARTIFACTS")
    print("=" * 60)

    total_imgs = 0
    total_lbls = 0
    all_categories = []

    for p in sorted(TARGET_DIR.glob("*/*")):
        img_dir = p / "images"
        lbl_dir = p / "labels"
        if not (img_dir.exists() and lbl_dir.exists()):
            continue

        imgs = sorted(list(img_dir.glob("*.jpg")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        # Check and delete any remaining orphan labels
        img_stems = set(f.stem for f in imgs)
        for l in lbls:
            if l.stem not in img_stems:
                l.unlink()

        # Re-fetch after cleanup
        imgs = sorted(list(img_dir.glob("*.jpg")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))
        zero_lbls = [f for f in lbls if f.stat().st_size == 0]

        assert len(imgs) == len(lbls) == len(zero_lbls), (
            f"Mismatch in {p.name}: {len(imgs)} imgs vs {len(lbls)} lbls vs {len(zero_lbls)} 0-byte lbls"
        )

        if len(imgs) > 0:
            rel_name = str(p.relative_to(TARGET_DIR)).replace("\\", "/")
            total_imgs += len(imgs)
            total_lbls += len(lbls)
            all_categories.append({
                "rel_path": rel_name,
                "count": len(imgs),
                "sample_paths": [str(f.relative_to(TARGET_DIR)).replace("\\", "/") for f in imgs],
            })
            print(f" [OK] {rel_name}: {len(imgs)} images, {len(lbls)} labels")
        else:
            print(f" [EMPTY] {p.name}: 0 images, 0 labels")

    # 1. Update Markdown Report
    md_content = f"""# รายงานสรุปชุดข้อมูลภาพพื้นหลัง (Background Candidates Review Report)
## สำหรับงานแก้ปัญหา False Positive / ภาพหลอนในโครงการ GamiSenpai (ฉบับอัปเดตหลังผู้ใช้คัดเลือก)

**วันที่อัปเดต:** 27 กันยายน 2026  
**สถานะการแยกข้อมูล:** Staged for Human Review (ผ่านการคัดกรองแล้ว ยังไม่ได้กระจายเข้า Session-Split Train/Val/Test)  
**โฟลเดอร์หลัก:** `data/background_candidates/`  
**จำนวนภาพ Background ทั้งหมดที่คงเหลือ:** **{total_imgs} ภาพ**  
**จำนวนไฟล์ Label (0-byte text files):** **{total_lbls} ไฟล์ (สมบูรณ์ 100% ตรงกัน 1:1)**  

---

### 1. สรุปจำนวนภาพคงเหลือแยกตามหมวดหมู่และสภาพแวดล้อมจริง

| หมวดหมู่ (Folder Path) | แหล่งที่มา (Source) | จำนวนภาพ | ลักษณะของภาพ Background / Hard Negative |
|---|---|:---:|---|
"""
    for cat in all_categories:
        desc = ""
        if "laptop" in cat["rel_path"]:
            desc = "โน้ตบุ๊ก คีย์บอร์ด ทัชแพด บานพับ และสติกเกอร์สีขาว"
        elif "bedroom" in cat["rel_path"]:
            desc = "ห้องนอน พัดลม เตียงนอน เสื้อสีขาว ตู้เสื้อผ้า (คัดกรองแล้วโดยผู้ใช้)"
        elif "session_001" in cat["rel_path"]:
            desc = "แผ่นรองตัดเขียว และโต๊ะแล็บ Xplor ตอนกลางคืน"
        elif "session_002" in cat["rel_path"]:
            desc = "แผ่นรองตัดเขียว และโต๊ะแล็บ Xplor ตอนกลางวัน"
        elif "session_003" in cat["rel_path"]:
            desc = "พื้นกระเบื้องห้องนอน แสงไฟนีออนกลางคืน (คัดกรองแล้วโดยผู้ใช้)"
        elif "session_004" in cat["rel_path"]:
            desc = "โต๊ะสีดำ ผิวหยาบ แสงแดดธรรมชาติข้างหน้าต่าง (คัดกรองแล้วโดยผู้ใช้)"
        elif "session_005" in cat["rel_path"]:
            desc = "โต๊ะสีส้ม และผนังห้องนอน แสงไฟนีออนกลางคืน"
        elif "session_006" in cat["rel_path"]:
            desc = "พื้นกระเบื้องสะท้อนแสงแดดจัดตอนเที่ยง (คัดกรองแล้วโดยผู้ใช้)"

        md_content += f"| `{cat['rel_path']}` | `{cat['rel_path'].split('/')[0]}` | **{cat['count']}** | {desc} |\n"

    md_content += f"""| **รวมทั้งสิ้น (Grand Total)** | **ทุกโฟลเดอร์หลังผู้ใช้คัดเลือก** | **{total_imgs} ภาพ** | **ลบไฟล์ Label กำพร้าออกเรียบร้อย 1:1 สมบูรณ์** |

---

### 2. ผลการตรวจสอบความถูกต้อง (Quality & Label Integrity Audit)

- [x] **1:1 Image-to-Label Alignment:** ทุกภาพมีไฟล์ `.txt` ประจำตัวชื่อตรงกัน 1:1 (ไม่มีไฟล์กำพร้าตกค้าง)
- [x] **Zero-Byte File Integrity:** ไฟล์ Label ทุกไฟล์มีขนาด **0 Bytes (ว่างเปล่า 100%)** ถูกต้องตามสเปก YOLO Background
- [x] **No Premature Split Distribution:** ข้อมูลทั้งหมดยังคงถูกจัดเก็บแยกในโฟลเดอร์สำหรับรีวิว ไม่มีการกระจายเข้า `train`, `val`, `test` จนกว่าผู้ใช้จะสั่ง
- [x] **Orphan Labels Cleared:** ไฟล์ Label ที่ภาพถูกลบไปแล้ว ถูกจัดการลบทิ้งเรียบร้อย

---

### 3. ช่องทางการตรวจสอบภาพ

1. **เปิดดูโฟลเดอร์โดยตรงผ่าน Windows Explorer:**
   `C:\\Users\\admin\\OneDrive\\Desktop\\ACS_KMUTT\\GamiSenpai\\data\\background_candidates`
2. **เปิดดูผ่านเบราว์เซอร์ (Visual Gallery):**
   เปิดไฟล์ `data/background_candidates/review_gallery.html` ใน Microsoft Edge เพื่อดูภาพทั้งหมดที่คงเหลือ ({total_imgs} ภาพ)
"""

    report_path = TARGET_DIR / "BACKGROUND_REVIEW_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f" [OK] Updated Markdown report: {report_path}")

    # 2. Update HTML Gallery
    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GamiSenpai - Background Candidates Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121212; color: #f0f0f0; margin: 0; padding: 24px; }}
        h1 {{ color: #4ade80; margin: 0 0 8px 0; font-size: 1.8em; }}
        .header {{ background: #1e1e1e; padding: 24px; border-radius: 12px; margin-bottom: 25px; border-left: 6px solid #4ade80; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
        .header p {{ color: #a3a3a3; margin: 0 0 16px 0; font-size: 1.05em; }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 15px; }}
        .stat-item {{ background: #2a2a2a; padding: 10px 18px; border-radius: 8px; border: 1px solid #3d3d3d; font-size: 1.0em; }}
        .stat-item b {{ color: #4ade80; }}
        .category-card {{ background: #1e1e1e; padding: 22px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
        .category-title {{ color: #38bdf8; font-size: 1.25em; margin-top: 0; margin-bottom: 16px; border-bottom: 2px solid #333; padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ background: #0284c7; color: white; padding: 3px 12px; border-radius: 20px; font-size: 0.75em; font-weight: bold; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; }}
        .img-item {{ background: #141414; border-radius: 8px; overflow: hidden; border: 1px solid #333; transition: transform 0.15s, border-color 0.15s; }}
        .img-item:hover {{ transform: scale(1.03); border-color: #4ade80; }}
        .img-item img {{ width: 100%; height: 135px; object-fit: cover; display: block; }}
        .img-item p {{ margin: 6px 8px; font-size: 0.72em; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GamiSenpai — คลังภาพ Background สำหรับตรวจสอบ (Staging Gallery)</h1>
        <p>ภาพพื้นหลังและ Hard Negatives ทั้งหมด {total_imgs} ภาพ จัดเก็บแยกโฟลเดอร์ (อัปเดตหลังคัดเลือกและลบ Label กำพร้าแล้ว)</p>
        <div class="stats">
            <div class="stat-item">ภาพคงเหลือ: <b>{total_imgs} ภาพ</b></div>
            <div class="stat-item">ไฟล์ Label (0-byte): <b>{total_lbls} ไฟล์ (สมบูรณ์ 100%)</b></div>
            <div class="stat-item">สถานะ: <b>ซิงค์ Label และภาพตรงกัน 1:1</b></div>
        </div>
    </div>
"""
    for cat in all_categories:
        html_content += f"""
    <div class="category-card">
        <div class="category-title">
            <span>📁 {cat['rel_path']}</span>
            <span class="badge">{cat['count']} ภาพ</span>
        </div>
        <div class="grid">
"""
        for sp in cat["sample_paths"]:
            stem = sp.split("/")[-1]
            html_content += f"""
            <div class="img-item">
                <a href="{sp}" target="_blank" title="คลิกเพื่อดูภาพขนาดเต็ม">
                    <img src="{sp}" alt="{stem}" loading="lazy">
                </a>
                <p title="{stem}">{stem}</p>
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""
    gallery_path = TARGET_DIR / "review_gallery.html"
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" [OK] Updated Interactive HTML gallery: {gallery_path}")
    print(f"\n[DONE] Grand Total: {total_imgs} images, {total_lbls} labels, 100% matched.")


if __name__ == "__main__":
    main()
