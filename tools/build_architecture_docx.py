from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "GamiSenpai_Project_Architecture_TH.docx"
SVG_PATH = ROOT / "artifacts" / "GamiSenpai_Project_Architecture_Diagram.svg"
MD_PATH = ROOT / "artifacts" / "GamiSenpai_Project_Architecture_TH.md"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_widths(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.insert(0, tbl_w)
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(sum(widths)))

    tbl_grid = tbl.tblGrid
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        tbl_grid.append(grid_col)

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths[index]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_run(paragraph, text, bold=False, color=None, size=None, font="Tahoma"):
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if size:
        run.font.size = Pt(size)
    return run


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    add_run(p, text, bold=True)
    return p


def add_paragraph(doc, text, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.2
    add_run(p, text)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    add_run(p, text)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    add_run(p, text)
    return p


def style_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Tahoma"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.2

    for name, size, color in [
        ("Heading 1", 16, "2E74B5"),
        ("Heading 2", 13, "2E74B5"),
        ("Heading 3", 12, "1F4D78"),
    ]:
        st = doc.styles[name]
        st.font.name = "Tahoma"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(footer, "GamiSenpai Project Architecture", color="666666", size=8)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = header
        set_cell_shading(cell, "E8EEF5")
        for p in cell.paragraphs:
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.font.name = "Tahoma"
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
                r.font.bold = True
                r.font.size = Pt(9.5)

    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = value

    set_table_widths(table, widths)
    for row in table.rows[1:]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = "Tahoma"
                    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
                    r.font.size = Pt(9)
    return table


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    style_document(doc)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(3)
    add_run(title, "เอกสารสถาปัตยกรรมโปรเจกต์ GamiSenpai", bold=True, color="0B2545", size=22)

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(14)
    add_run(
        subtitle,
        "สรุป Tech stack, data flow, โครงสร้างโฟลเดอร์ และหน้าที่ของไฟล์สำคัญในระบบตรวจการพับโอริกามิแบบ real-time",
        color="555555",
        size=10.5,
    )

    add_heading(doc, "ภาพรวมระบบ", 1)
    add_paragraph(
        doc,
        "GamiSenpai เป็นระบบ AI สำหรับตรวจ checkpoint การพับนกกระเรียนแบบ real-time ผ่านหน้าเว็บ โดยผู้ใช้เปิดกล้องใน browser "
        "แล้ว frontend จะส่งภาพไปยัง FastAPI backend เพื่อให้โมเดล PyTorch ทำนายว่าอยู่ในขั้นไหนและผ่านหรือยัง",
    )
    add_bullet(doc, "Runtime side: Web Demo + Inference")
    add_bullet(doc, "Offline side: Data Collection + Dataset Preparation + Training + Evaluation")

    add_heading(doc, "Tech Stack", 1)
    add_table(
        doc,
        ["หมวด", "Technology"],
        [
            ("Backend API", "Python 3.11+, FastAPI, Uvicorn, python-multipart"),
            ("AI / Deep Learning", "PyTorch, TorchVision, MobileNetV2"),
            ("Computer Vision", "OpenCV, Pillow, NumPy"),
            ("Evaluation / Visualization", "scikit-learn, matplotlib, seaborn, tqdm"),
            ("Frontend", "HTML, CSS, Vanilla JavaScript, Browser Camera APIs"),
            ("Environment / Tooling", ".venv, Jupyter, ipykernel"),
        ],
        [2500, 6860],
    )

    add_heading(doc, "Runtime Flow", 1)
    for text in [
        "ผู้ใช้เปิดเว็บที่ http://127.0.0.1:8000 และอนุญาตสิทธิ์กล้อง",
        "Frontend ใน app/static เปิดกล้องผ่าน getUserMedia และจับภาพเป็นช่วง ๆ",
        "script.js ส่ง frame ไปที่ POST /api/predict ด้วย FormData",
        "app/app.py รับไฟล์ภาพ, แปลงเป็น PIL image, แล้ว preprocess ด้วย val_transform",
        "โมเดลจาก src/model.py พร้อม checkpoint origami_model_final.pth ทำ inference",
        "Backend ส่ง prediction JSON กลับไปให้ frontend เพื่ออัปเดต guided step และ confidence",
    ]:
        add_number(doc, text)

    add_heading(doc, "Training Pipeline", 1)
    for text in [
        "src/data_collection.py ใช้เก็บภาพแต่ละ step และแยก correct / incorrect",
        "src/prepare_dataset.py จัดโครงสร้างข้อมูลให้เป็น train / val แบบ ImageFolder",
        "src/train.py train โมเดลและบันทึก best / final checkpoints",
        "src/evaluate.py สร้าง confusion matrix และ classification report",
        "plots/ เก็บกราฟ training_history.png และ confusion_matrix.png",
    ]:
        add_number(doc, text)

    add_heading(doc, "โฟลเดอร์และหน้าที่", 1)
    add_table(
        doc,
        ["โฟลเดอร์ / ไฟล์", "หน้าที่"],
        [
            ("app/", "Web application layer: FastAPI backend และ static frontend"),
            ("src/", "ML logic, preprocessing, data pipeline, training, inference"),
            ("Checkpoints/", "โมเดลที่ train แล้วและ reference images"),
            ("data/", "raw dataset, processed dataset, reference data"),
            ("plots/", "ผลการ visualize หลัง train และ evaluate"),
            ("artifacts/", "เอกสาร, logs, review outputs, generated files"),
            ("tools/", "helper scripts สำหรับสร้างเอกสารหรือ utility เพิ่มเติม"),
            ("tests/", "พื้นที่สำหรับ automated tests; ตอนนี้ยังเป็น placeholder เป็นหลัก"),
            ("main.py", "stub entry file, ยังไม่ได้เป็น runtime หลักของระบบ"),
            ("start_server.bat / .ps1", "helper สำหรับเปิด demo web บน Windows"),
        ],
        [2900, 6460],
    )

    add_heading(doc, "ไฟล์สำคัญ", 1)
    add_table(
        doc,
        ["ไฟล์", "บทบาท"],
        [
            ("app/app.py", "โหลดโมเดล, เสิร์ฟ index.html, เปิด /api/predict"),
            ("app/static/index.html", "หน้าเว็บ UI หลักของ demo"),
            ("app/static/script.js", "camera control, request loop, guided step flow"),
            ("app/static/style.css", "layout และ visual styling ของหน้าเว็บ"),
            ("src/model.py", "นิยาม OrigamiClassifier ที่ใช้ MobileNetV2"),
            ("src/preprocessing.py", "กำหนด transforms สำหรับ train / val / inference"),
            ("src/train.py", "training loop และการบันทึก checkpoint"),
            ("src/evaluate.py", "ประเมินโมเดลและสร้าง confusion matrix"),
            ("src/inference.py", "standalone webcam inference ผ่าน OpenCV"),
            ("src/data_collection.py", "เครื่องมือเก็บ dataset"),
            ("src/prepare_dataset.py", "แปลง dataset ให้พร้อม train"),
            ("src/test_camera.py", "ตรวจกล้องและ OpenCV local"),
        ],
        [3000, 6360],
    )

    add_heading(doc, "ข้อสังเกตทางสถาปัตยกรรม", 1)
    add_bullet(doc, "โครงสร้างแยก app และ src ค่อนข้างชัด ทำให้ web layer กับ ML layer ไม่ปนกันมาก")
    add_bullet(doc, "checkpoint เป็นจุดเชื่อมสำคัญระหว่าง training pipeline และ runtime demo")
    add_bullet(doc, "app/app.py ใช้วิธี append src เข้า sys.path ซึ่งทำงานได้ แต่ยังไม่ใช่ packaging ที่สะอาดที่สุด")
    add_bullet(doc, "src/prepare_dataset.py มี path ตัวอย่างที่อาจไม่ตรงกับโฟลเดอร์จริงใน repo ปัจจุบัน จึงควรระวังเวลาใช้งาน")
    add_bullet(doc, "ใน repo มีหลายโฟลเดอร์เชิงทดลองหรือสำรอง เช่น Check_Dataset, Dataset_backUp, Testing จึงควรอธิบาย source-of-truth ให้ชัดเวลา onboarding คนใหม่")

    add_heading(doc, "ไฟล์ประกอบที่สร้างเพิ่ม", 1)
    add_bullet(doc, f"Markdown version: {MD_PATH}")
    add_bullet(doc, f"Canva-ready diagram: {SVG_PATH}")
    add_paragraph(
        doc,
        "หมายเหตุ: ใน session นี้ยังไม่ได้ต่อ Canva โดยตรง จึงยังแก้บน Canva ให้ไม่ได้ แต่ไฟล์ SVG ที่แนบไว้สามารถ import เข้า Canva แล้วจัดวางต่อได้ทันที",
        after=8,
    )

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
