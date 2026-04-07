#!/usr/bin/env python3
"""
Fill Chapter 5 with real experimental data and charts.
"""

import zipfile
import os
import shutil
import re
from lxml import etree
from PIL import Image

# Paths
DOCX_PATH = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿.docx'
OUTPUT_PATH = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿_更新版.docx'

# Image sources - CORRECT paths
EXP_BASE = '/Volumes/Samsung/Projects/robotagent/experiments/results'

# Namespaces
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
CT_NS = 'http://schemas.openxmlformats.org/package/2006/content-types'
PKG_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'

def w(tag):
    return f'{{{W_NS}}}{tag}'

def create_image_paragraph_node(rel_id, width_px, height_px, desc=''):
    """Create a paragraph containing an inline image."""
    MAX_WIDTH_INCHES = 6.0
    aspect = height_px / width_px if width_px > 0 else 0.5
    max_w_px = MAX_WIDTH_INCHES * 96
    if width_px > max_w_px:
        width_px = max_w_px
    height_px = int(width_px * aspect)

    cx = int(width_px / 96 * 914400)
    cy = int(height_px / 96 * 914400)

    p_elem = etree.Element(w('p'))
    ppr = etree.SubElement(p_elem, w('pPr'))
    jc = etree.SubElement(ppr, w('jc'))
    jc.set(w('val'), 'center')

    r_elem = etree.SubElement(p_elem, w('r'))
    rpr = etree.SubElement(r_elem, w('rPr'))
    etree.SubElement(rpr, w('noProof'))

    drawing = etree.SubElement(r_elem, w('drawing'))

    inline_xml = f'''<wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
                xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                distT="0" distB="0" distL="114300" distR="114300">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="999" name="Image" descr="{desc}"/>
        <wp:cNvGraphicFramePr/>
        <a:graphic>
            <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic>
                    <pic:nvPicPr>
                        <pic:cNvPr id="999" name="Image" descr="{desc}"/>
                        <pic:cNvPicPr/>
                    </pic:nvPicPr>
                    <pic:blipFill>
                        <a:blip r:embed="{rel_id}"/>
                        <a:stretch><a:fillRect/></a:stretch>
                    </pic:blipFill>
                    <pic:spPr>
                        <a:xfrm>
                            <a:off x="0" y="0"/>
                            <a:ext cx="{cx}" cy="{cy}"/>
                        </a:xfrm>
                        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                        <a:noFill/>
                        <a:ln><a:noFill/></a:ln>
                    </pic:spPr>
                </pic:pic>
            </a:graphicData>
        </a:graphic>
    </wp:inline>'''

    inline_elem = etree.fromstring(inline_xml)
    drawing.append(inline_elem)
    return p_elem


def create_caption_paragraph(text):
    """Create a centered caption paragraph."""
    p_elem = etree.Element(w('p'))
    ppr = etree.SubElement(p_elem, w('pPr'))
    jc = etree.SubElement(ppr, w('jc'))
    jc.set(w('val'), 'center')
    r = etree.SubElement(p_elem, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), '宋体')
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '21')
    t = etree.SubElement(r, w('t'))
    t.text = text
    return p_elem


def create_rag_table():
    """Create RAG comparison table."""
    tbl = etree.Element(w('tbl'))

    tblPr = etree.SubElement(tbl, w('tblPr'))
    tblStyle = etree.SubElement(tblPr, w('tblStyle'))
    tblStyle.set(w('val'), 'TableGrid')
    tblW = etree.SubElement(tblPr, w('tblW'))
    tblW.set(w('w'), '0')
    tblW.set(w('type'), 'auto')
    tblBorders = etree.SubElement(tblPr, w('tblBorders'))
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = etree.SubElement(tblBorders, w(border_name))
        border.set(w('val'), 'single')
        border.set(w('sz'), '4')
        border.set(w('color'), '000000')

    tblGrid = etree.SubElement(tbl, w('tblGrid'))
    for _ in range(4):
        etree.SubElement(tblGrid, w('gridCol')).set(w('w'), '1800')

    tr = etree.SubElement(tbl, w('tr'))
    trPr = etree.SubElement(tr, w('trPr'))
    trHeight = etree.SubElement(trPr, w('trHeight'))
    trHeight.set(w('val'), '400')
    trHeight.set(w('hRule'), 'atLeast')

    headers = ['评价指标', '基线模型(%)', 'Agentic RAG(%)', '变化(百分点)']
    for h in headers:
        tc = etree.SubElement(tr, w('tc'))
        tcPr = etree.SubElement(tc, w('tcPr'))
        tcW = etree.SubElement(tcPr, w('tcW'))
        tcW.set(w('w'), '1800')
        tcW.set(w('type'), 'dxa')
        shd = etree.SubElement(tcPr, w('shd'))
        shd.set(w('val'), 'clear')
        shd.set(w('color'), 'auto')
        shd.set(w('fill'), 'D9D9D9')

        p = etree.SubElement(tc, w('p'))
        pPr = etree.SubElement(p, w('pPr'))
        jc = etree.SubElement(pPr, w('jc'))
        jc.set(w('val'), 'center')
        r = etree.SubElement(p, w('r'))
        rPr = etree.SubElement(r, w('rPr'))
        rFonts = etree.SubElement(rPr, w('rFonts'))
        rFonts.set(w('eastAsia'), '宋体')
        rFonts.set(w('ascii'), '宋体')
        etree.SubElement(rPr, w('b'))
        sz = etree.SubElement(rPr, w('sz'))
        sz.set(w('val'), '21')
        t = etree.SubElement(r, w('t'))
        t.text = h

    data = [
        ('相关性 (Relevance)', '95.2', '90.4', '-4.8'),
        ('准确性 (Accuracy)', '74.5', '71.8', '-2.7'),
        ('完整性 (Completeness)', '68.3', '76.2', '+7.9'),
        ('引证质量 (Citation)', '28.1', '63.0', '+34.9'),
        ('整体得分 (Overall)', '66.7', '75.1', '+8.4'),
    ]
    for row_data in data:
        tr = etree.SubElement(tbl, w('tr'))
        trPr = etree.SubElement(tr, w('trPr'))
        trHeight = etree.SubElement(trPr, w('trHeight'))
        trHeight.set(w('val'), '360')
        trHeight.set(w('hRule'), 'atLeast')

        for j, cell_text in enumerate(row_data):
            tc = etree.SubElement(tr, w('tc'))
            tcPr = etree.SubElement(tc, w('tcPr'))
            tcW = etree.SubElement(tcPr, w('tcW'))
            tcW.set(w('w'), '1800')
            tcW.set(w('type'), 'dxa')

            p = etree.SubElement(tc, w('p'))
            pPr = etree.SubElement(p, w('pPr'))
            if j > 0:
                jc = etree.SubElement(pPr, w('jc'))
                jc.set(w('val'), 'center')

            r = etree.SubElement(p, w('r'))
            rPr = etree.SubElement(r, w('rPr'))
            rFonts = etree.SubElement(rPr, w('rFonts'))
            rFonts.set(w('eastAsia'), '宋体')
            rFonts.set(w('ascii'), '宋体')
            sz = etree.SubElement(rPr, w('sz'))
            sz.set(w('val'), '21')
            t = etree.SubElement(r, w('t'))
            t.text = cell_text
            if cell_text.startswith('+'):
                color = etree.SubElement(rPr, w('color'))
                color.set(w('val'), '00B050')
                etree.SubElement(rPr, w('b'))

    return tbl


def modify_paragraph_text(p_elem, new_text):
    """Replace all runs in a paragraph with new text runs."""
    for r_elem in p_elem.findall(w('r')):
        p_elem.remove(r_elem)

    sentences = re.split(r'([。；！？])', new_text)
    for sent in sentences:
        if sent.strip():
            r_elem = etree.SubElement(p_elem, w('r'))
            rPr = etree.SubElement(r_elem, w('rPr'))
            rFonts = etree.SubElement(rPr, w('rFonts'))
            rFonts.set(w('eastAsia'), '宋体')
            sz = etree.SubElement(rPr, w('sz'))
            sz.set(w('val'), '24')
            t = etree.SubElement(r_elem, w('t'))
            t.text = sent


def main():
    print("Starting Chapter 5 fill process...")

    with zipfile.ZipFile(DOCX_PATH, 'r') as zin:
        file_contents = {}
        for name in zin.namelist():
            file_contents[name] = zin.read(name)

    # Image files to add - CORRECTED paths
    images_to_add = [
        (f'{EXP_BASE}/academic_rag_comparison/figures/comparison_summary.png', 'RAG对比总览图'),
        # success_rate_curve.png is in task_attempt_analysis, not experience_transferability
        (f'{EXP_BASE}/task_attempt_analysis/figures/success_rate_curve.png', 'Pass@k成功率曲线'),
        # by_difficulty.png is in experience_transferability, not task_attempt_analysis
        (f'{EXP_BASE}/experience_transferability/figures/by_difficulty.png', 'GRPO按难度提升图'),
    ]

    print("Checking image paths:")
    for img_path, desc in images_to_add:
        exists = os.path.exists(img_path)
        print(f"  {img_path}: {'EXISTS' if exists else 'NOT FOUND'}")

    ct_xml = file_contents['[Content_Types].xml']
    ct_root = etree.fromstring(ct_xml)

    rels_xml = file_contents['word/_rels/document.xml.rels']
    rels_root = etree.fromstring(rels_xml)

    max_rid = 0
    for rel in rels_root.findall(f'.//{{{PKG_NS}}}Relationship'):
        rid = rel.get('Id')
        if rid:
            match = re.search(r'rId(\d+)', rid)
            if match:
                max_rid = max(max_rid, int(match.group(1)))

    image_rel_ids = []

    for img_path, img_desc in images_to_add:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        max_rid += 1
        new_rid = f'rId{max_rid}'

        existing_images = [n for n in file_contents.keys() if n.startswith('word/media/') and not n.endswith('/')]
        existing_nums = []
        for name in existing_images:
            match = re.search(r'image(\d+)\.(\w+)', name)
            if match:
                existing_nums.append(int(match.group(1)))
        next_num = max(existing_nums) + 1 if existing_nums else 1

        ext = os.path.splitext(img_path)[1].lower()
        if ext == '.jpeg':
            ext = '.jpeg'
        new_media_name = f'image{next_num}{ext}'

        ct_map = {'.png': 'image/png', '.jpeg': 'image/jpeg', '.jpg': 'image/jpeg'}
        ct = ct_map.get(ext, 'image/png')

        part_name = f'/word/media/{new_media_name}'
        override = etree.SubElement(ct_root, f'{{{CT_NS}}}Override')
        override.set('PartName', part_name)
        override.set('ContentType', ct)

        new_rel = etree.SubElement(rels_root, f'{{{PKG_NS}}}Relationship')
        new_rel.set('Id', new_rid)
        new_rel.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image')
        new_rel.set('Target', f'media/{new_media_name}')

        with open(img_path, 'rb') as f:
            img_data = f.read()
        file_contents[f'word/media/{new_media_name}'] = img_data

        image_rel_ids.append((new_rid, img_path))
        print(f"Added image: {new_media_name} with relId {new_rid}")

    file_contents['[Content_Types].xml'] = etree.tostring(ct_root, xml_declaration=True, encoding='UTF-8', standalone=True)
    file_contents['word/_rels/document.xml.rels'] = etree.tostring(rels_root, xml_declaration=True, encoding='UTF-8', standalone=True)

    doc_xml = file_contents['word/document.xml']
    root = etree.fromstring(doc_xml)
    body = root.find(f'.//{w("body")}')
    paragraphs = body.findall(f'.//{w("p")}')

    rag_content = """本节实验对比了基线模型（无RAG辅助）与Agentic RAG系统（带知识检索增强）的回答质量。实验使用100条机器人仿真领域的专业查询，涵盖任务规划、工具调用、仿真参数设置等典型场景。评价指标包括相关性（Relevance）、准确性（Accuracy）、完整性（Completeness）和引证质量（Citation）四个维度，每项满分5分，整体得分取加权平均并换算为百分制。

实验结果如表5-1所示。基线模型的相关性得分最高（95.2%），说明模型本身对领域问题有较好的语义理解能力；然而在完整性和引证质量上表现较弱，分别为68.3%和28.1%，反映出缺乏领域知识时模型容易产生幻觉或不完整的规划。Agentic RAG系统整体得分达到75.1%，较基线提升8.4个百分点，其中引证质量提升最为显著（从28.1%提升至63.0%，提升34.9%），完整性也有7.9%的提升。这表明知识检索模块有效帮助模型获取了领域专业知识，提升了规划方案的完整性和依据可靠性。

值得注意的是，Agentic RAG在相关性和准确性上有小幅下降（分别下降4.8%和2.7%），这是因为检索增强引入了外部知识后，模型的回答风格和依据来源发生了变化，整体仍然保持在较高水平（90.4%和71.8%）。"""

    passk_content = """本节实验分析了系统完成仿真任务的成功率，采用Pass@k指标进行评估。Pass@k表示在每个任务最多允许k次尝试的条件下，至少有1次成功的任务比例。本实验基于80条仿真任务指令（包括推箱、抓取、路径规划等典型操作），统计了不同尝试次数下的累计成功率。

实验结果如图5-1所示。系统在单次尝试下成功率为56.25%，随着尝试次数增加，成功率逐步提升：2次尝试后达到71.25%，3次尝试后达到82.05%，4次尝试后达到87.18%。平均每任务尝试次数为3.95次，说明多尝试机制对复杂任务有明显的容错效果。

从曲线形态来看，成功率在第1至第2次、第2至第3次之间提升幅度较大（约15个百分点），第3至第4次提升幅度减缓（约5个百分点）。这表明大多数任务在前3次尝试中能够完成，额外的第4次尝试主要用于处理极端边界情况。根据实验结果，建议对简单任务配置1-2次尝试次数，对复杂任务配置3-4次尝试次数，以在成功率和执行效率之间取得平衡。"""

    grpo_content = """本节实验评估了Training-free GRPO经验强化机制对任务完成质量的提升效果。实验设置20条不同难度的仿真任务指令，分为简单（Easy）、中等（Medium）、困难（Hard）三个等级。实验组采用有经验的Agent（加载历史成功轨迹总结），对照组采用无经验的Agent，对比两组的任务得分差异。

实验结果如图5-2所示。从整体提升来看，有经验Agent的平均得分提升为1.1分（满分10分制），正向提升比例为50%，说明经验总结机制对部分任务有明显的帮助。按难度分项统计：简单任务平均提升2.667分，提升幅度最大；中等任务平均提升1.429分；困难任务提升最小，仅0.4分。这一结果符合预期：简单和中等任务有明确的可套用模式，经验总结能帮助Agent快速定位关键步骤和参数；困难任务受限于模型自身能力和环境复杂度，经验的作用相对有限。

上述结果表明，Training-free GRPO适合用于提升中低难度任务的完成质量，对高难度任务建议配合模型能力升级或更丰富的经验库迭代策略。"""

    para_map = {}
    for i, p in enumerate(paragraphs):
        texts = p.findall(f'.//{w("t")}')
        full_text = ''.join([t.text for t in texts if t.text])

        if '本节实验对比了模型在没有RAG' in full_text:
            para_map['5.1_content'] = (i, p)
        elif '本节实验分析了模型对100个仿真任务' in full_text:
            para_map['5.2_content'] = (i, p)
        elif '本节实验评估了training-free GRPO' in full_text:
            para_map['5.3_content'] = (i, p)

    print(f"Found paragraphs to modify: {list(para_map.keys())}")
    print(f"Found images: {len(image_rel_ids)}")

    if '5.1_content' in para_map:
        idx, p = para_map['5.1_content']
        modify_paragraph_text(p, rag_content)
        print(f"Modified paragraph {idx} (5.1 content)")

    if '5.2_content' in para_map:
        idx, p = para_map['5.2_content']
        modify_paragraph_text(p, passk_content)
        print(f"Modified paragraph {idx} (5.2 content)")

    if '5.3_content' in para_map:
        idx, p = para_map['5.3_content']
        modify_paragraph_text(p, grpo_content)
        print(f"Modified paragraph {idx} (5.3 content)")

    # Re-find paragraphs after modification
    body_children = list(body)

    for i, child in enumerate(body_children):
        if child.tag == w('p'):
            texts = child.findall(f'.//{w("t")}')
            full_text = ''.join([t.text for t in texts if t.text])
            if 'Agentic RAG系统整体得分达到75.1%' in full_text:
                caption = create_caption_paragraph('表5-1 RAG性能对比实验结果')
                table = create_rag_table()
                body.insert(i + 1, caption)
                body.insert(i + 2, table)
                print(f"Inserted table after paragraph {i}")
                break

    body_children = list(body)
    for i, child in enumerate(body_children):
        if child.tag == w('p'):
            texts = child.findall(f'.//{w("t")}')
            full_text = ''.join([t.text for t in texts if t.text])
            if '系统在单次尝试下成功率' in full_text and len(image_rel_ids) > 1:
                rel_id, img_path = image_rel_ids[1]
                try:
                    with Image.open(img_path) as img:
                        w_px, h_px = img.size
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
                    w_px, h_px = 800, 400

                img_p = create_image_paragraph_node(rel_id, w_px, h_px, 'Pass@k成功率曲线')
                caption = create_caption_paragraph('图5-1 不同尝试次数下的累计成功率')
                body.insert(i + 1, img_p)
                body.insert(i + 2, caption)
                print(f"Inserted Pass@k image after paragraph {i}")
                break

    body_children = list(body)
    for i, child in enumerate(body_children):
        if child.tag == w('p'):
            texts = child.findall(f'.//{w("t")}')
            full_text = ''.join([t.text for t in texts if t.text])
            if '有经验Agent的平均得分提升为1.1分' in full_text and len(image_rel_ids) > 2:
                rel_id, img_path = image_rel_ids[2]
                try:
                    with Image.open(img_path) as img:
                        w_px, h_px = img.size
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
                    w_px, h_px = 800, 400

                img_p = create_image_paragraph_node(rel_id, w_px, h_px, 'GRPO按难度提升')
                caption = create_caption_paragraph('图5-2 Training-free GRPO在不同难度任务上的提升效果')
                body.insert(i + 1, img_p)
                body.insert(i + 2, caption)
                print(f"Inserted GRPO image after paragraph {i}")
                break

    doc_xml_final = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    file_contents['word/document.xml'] = doc_xml_final

    with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_contents.items():
            zout.writestr(name, data)

    print(f"\nDone! Output saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
