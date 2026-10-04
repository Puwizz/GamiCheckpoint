from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "GamiSenpai_System_Startup_Guide_TH.docx"


PROJECT_PATH = str(ROOT)
LOCAL_URL = "http://127.0.0.1:8000"
START_CMD = (
    r".\.venv\Scripts\python.exe -m uvicorn app.app:app "
    r"--host 127.0.0.1 --port 8000 --log-level warning"
)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
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
        for idx, cell in enumerate(row.cells):
            cell.width = Pt(widths[idx] / 20)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths[idx]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_run(paragraph, text, bold=False, color=None):
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Tahoma"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def add_paragraph(doc, text="", style=None, bold_prefix=None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.25
    if bold_prefix and text.startswith(bold_prefix):
        add_run(p, bold_prefix, bold=True)
        add_run(p, text[len(bold_prefix) :])
    else:
        add_run(p, text)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    add_run(p, text)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    add_run(p, text)
    return p


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Inches(0.15)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    run.font.size = Pt(9.5)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(7 if level == 1 else 5)
    run = p.add_run(text)
    run.font.name = "Tahoma"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    run.font.color.rgb = RGBColor.from_string("2E74B5" if level <= 2 else "1F4D78")
    return p


def style_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Tahoma"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for name, size, color in [
        ("Heading 1", 16, "2E74B5"),
        ("Heading 2", 13, "2E74B5"),
        ("Heading 3", 12, "1F4D78"),
    ]:
        st = styles[name]
        st.font.name = "Tahoma"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(color)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("GamiSenpai System Startup Guide")
    run.font.name = "Tahoma"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string("666666")


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    style_document(doc)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(3)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tr = title.add_run("คู่มือการเปิดระบบ GamiSenpai")
    tr.font.name = "Tahoma"
    tr._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    tr.font.size = Pt(22)
    tr.font.bold = True
    tr.font.color.rgb = RGBColor.from_string("0B2545")

    sub = doc.add_paragraph()
    sub.paragraph_format.space_after = Pt(14)
    sr = sub.add_run("สำหรับเปิด Demo Web ตรวจการพับนกกระเรียนด้วย AI บนเครื่อง Windows ของโปรเจกต์")
    sr.font.name = "Tahoma"
    sr._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
    sr.font.size = Pt(10.5)
    sr.font.color.rgb = RGBColor.from_string("555555")

    add_heading(doc, "สรุปแบบเร็วที่สุด", 1)
    add_number(doc, "ดับเบิลคลิกไฟล์ start_server.bat ในโฟลเดอร์โปรเจกต์")
    add_number(doc, f"เปิดเว็บเบราว์เซอร์ไปที่ {LOCAL_URL}")
    add_number(doc, "กด Refresh Cameras แล้วเลือกกล้องที่ต้องการ")
    add_number(doc, "กด Start Camera และเริ่มทดสอบจาก Step 1")

    add_heading(doc, "ตำแหน่งโปรเจกต์และไฟล์สำคัญ", 1)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "รายการ"
    table.rows[0].cells[1].text = "ตำแหน่ง / ความหมาย"
    for c in table.rows[0].cells:
        set_cell_shading(c, "E8EEF5")
        for p in c.paragraphs:
            for r in p.runs:
                r.font.name = "Tahoma"
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
                r.font.bold = True
    rows = [
        ("โฟลเดอร์โปรเจกต์", PROJECT_PATH),
        ("ไฟล์เปิดระบบแบบดับเบิลคลิก", rf"{PROJECT_PATH}\start_server.bat"),
        ("ไฟล์เปิดระบบด้วย PowerShell", rf"{PROJECT_PATH}\start_server.ps1"),
        ("หน้าเว็บหลังเปิดระบบ", LOCAL_URL),
        ("โมเดลที่ระบบโหลด", rf"{PROJECT_PATH}\Checkpoints\origami_model_final.pth"),
    ]
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
    set_table_widths(table, [2700, 6660])
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = "Tahoma"
                    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
                    r.font.size = Pt(9.5)

    add_heading(doc, "วิธีเปิดระบบแบบแนะนำ", 1)
    add_paragraph(doc, "ใช้วิธีนี้เมื่อต้องการเปิดระบบเพื่อทดสอบ Demo Web อย่างรวดเร็ว")
    add_number(doc, rf"ไปที่โฟลเดอร์ {PROJECT_PATH}")
    add_number(doc, "ดับเบิลคลิก start_server.bat")
    add_number(doc, "รอให้หน้าต่าง Command Prompt/PowerShell เปิดค้างไว้")
    add_number(doc, f"เปิดเว็บที่ {LOCAL_URL}")
    add_number(doc, "ห้ามปิดหน้าต่าง server ระหว่างทดสอบ เพราะเว็บจะหยุดทำงานทันที")

    add_heading(doc, "วิธีเปิดระบบผ่าน PowerShell", 1)
    add_paragraph(doc, "ถ้าต้องการรันเองหรือแก้ปัญหา ให้เปิด PowerShell แล้วใช้คำสั่งด้านล่าง")
    add_code(doc, f"cd {PROJECT_PATH}")
    add_code(doc, START_CMD)

    add_heading(doc, "ขั้นตอนใช้งานหลังเปิดหน้าเว็บ", 1)
    for step in [
        f"เข้า {LOCAL_URL}",
        "ถ้าใช้มือถือเป็นกล้อง ให้เปิด Iriun Webcam ทั้งบนมือถือและบนคอมก่อน",
        "กด Refresh Cameras",
        "เลือกกล้องที่ต้องการ เช่น Iriun Webcam",
        "กด Start Camera",
        "เริ่มจาก Step 1 และรอให้ระบบปลดล็อก Step ถัดไปเมื่อผ่านแล้ว",
        "ถ้าต้องเริ่มใหม่ ให้กด Reset",
    ]:
        add_number(doc, step)

    add_heading(doc, "Checklist ก่อนทดสอบ", 1)
    for item in [
        "ไฟล์ Checkpoints/origami_model_final.pth ยังอยู่ในโปรเจกต์",
        "virtual environment .venv ติดตั้ง dependency แล้ว",
        "กล้องหรือ Iriun Webcam เชื่อมต่อเรียบร้อย",
        "Browser อนุญาตสิทธิ์กล้องแล้ว",
        "แสงพอ เห็นขอบกระดาษและรอยพับชัด",
    ]:
        add_bullet(doc, item)

    add_heading(doc, "ปัญหาที่พบบ่อยและวิธีแก้", 1)
    problems = doc.add_table(rows=1, cols=3)
    problems.style = "Table Grid"
    headers = ["อาการ", "สาเหตุที่เป็นไปได้", "วิธีแก้"]
    for i, h in enumerate(headers):
        problems.rows[0].cells[i].text = h
        set_cell_shading(problems.rows[0].cells[i], "E8EEF5")
    data = [
        ("เข้าเว็บไม่ได้", "server ยังไม่รัน หรือหน้าต่าง server ถูกปิด", "รัน start_server.bat ใหม่ แล้วเปิดเว็บอีกครั้ง"),
        ("โมเดลโหลดไม่ได้", "ไฟล์ checkpoint หายหรืออยู่ผิดที่", "ตรวจว่า Checkpoints/origami_model_final.pth ยังมีอยู่"),
        ("ไม่เห็นกล้อง", "ยังไม่ได้อนุญาตกล้อง หรือ Iriun ยังไม่เชื่อม", "กด Refresh Cameras, อนุญาตกล้อง, แล้วเปิด Iriun ใหม่"),
        ("เปิด .ps1 ไม่ได้", "Windows policy บล็อก PowerShell script", "ใช้ start_server.bat แทน"),
        ("AI ทายผิดบ่อย", "แสง/มุม/รอยพับไม่ชัด หรือข้อมูลฝึกยังไม่หลากหลาย", "จัดแสงให้เห็นขอบชัด ถ่ายให้กระดาษเต็มเฟรม และเก็บ dataset เพิ่ม"),
    ]
    for row in data:
        cells = problems.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
    set_table_widths(problems, [2100, 3300, 3960])
    for row in problems.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = "Tahoma"
                    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Tahoma")
                    r.font.size = Pt(9)
                    if row == problems.rows[0]:
                        r.font.bold = True

    add_heading(doc, "วิธีหยุดระบบ", 1)
    add_bullet(doc, "ถ้ารันผ่าน PowerShell หรือ Command Prompt ให้กด Ctrl + C")
    add_bullet(doc, "ถ้าเปิดจาก start_server.bat ให้ปิดหน้าต่าง server ได้เลย")
    add_bullet(doc, "หลังหยุดระบบแล้ว หน้าเว็บที่ 127.0.0.1:8000 จะใช้งานไม่ได้จนกว่าจะเปิด server ใหม่")

    add_heading(doc, "หมายเหตุสำหรับการ Demo", 1)
    add_paragraph(
        doc,
        "ระบบถูกออกแบบให้พาผู้ใช้ผ่าน Step ทีละขั้น ถ้า Step ปัจจุบันยังไม่ผ่าน ระบบจะยังไม่ปลดล็อก Step ถัดไป "
        "จึงควรเริ่มจาก Step 1 ทุกครั้ง และวางกระดาษให้อยู่ในเฟรมชัดเจนก่อนเริ่มทดสอบ",
    )

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
