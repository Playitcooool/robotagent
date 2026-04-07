#!/usr/bin/env python3
"""
Add mathematical formulas for evaluation metrics to Chapter 5.
Fixes the formula insertion properly.
"""

import zipfile
import re
from lxml import etree

# Paths
INPUT_PATH = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿_更新版.docx'
OUTPUT_PATH = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿_更新版.docx'

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def w(tag):
    return f'{{{W_NS}}}{tag}'

def create_plain_paragraph(text):
    """Create a simple paragraph with body text formatting."""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    ind = etree.SubElement(pPr, w('ind'))
    ind.set(w('firstLineChars'), '200')
    ind.set(w('firstLine'), '480')
    r = etree.SubElement(p, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), '宋体')
    rFonts.set(w('ascii'), '宋体')
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '24')
    t = etree.SubElement(r, w('t'))
    t.text = text
    return p


def create_formula_para(parts):
    """Create a centered formula paragraph from a list of (text, bold, italic) tuples."""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    jc = etree.SubElement(pPr, w('jc'))
    jc.set(w('val'), 'center')
    spacing = etree.SubElement(pPr, w('spacing'))
    spacing.set(w('before'), '60')
    spacing.set(w('after'), '60')

    for text, bold, italic in parts:
        r = etree.SubElement(p, w('r'))
        rPr = etree.SubElement(r, w('rPr'))
        rFonts = etree.SubElement(rPr, w('rFonts'))
        rFonts.set(w('eastAsia'), 'Times New Roman')
        rFonts.set(w('ascii'), 'Times New Roman')
        if bold:
            etree.SubElement(rPr, w('b'))
            etree.SubElement(rPr, w('bCs'))
        if italic:
            etree.SubElement(rPr, w('i'))
            etree.SubElement(rPr, w('iCs'))
        sz = etree.SubElement(rPr, w('sz'))
        sz.set(w('val'), '26')  # 13pt
        szCs = etree.SubElement(rPr, w('szCs'))
        szCs.set(w('val'), '26')
        t = etree.SubElement(r, w('t'))
        t.text = text
    return p


def create_description_para(text):
    """Create a paragraph with description text."""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    ind = etree.SubElement(pPr, w('ind'))
    ind.set(w('firstLineChars'), '200')
    ind.set(w('firstLine'), '480')
    r = etree.SubElement(p, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), '宋体')
    rFonts.set(w('ascii'), '宋体')
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '24')
    t = etree.SubElement(r, w('t'))
    t.text = text
    return p


def main():
    print("Fixing mathematical formulas in Chapter 5...")

    # Read all files
    with zipfile.ZipFile(INPUT_PATH, 'r') as zin:
        file_contents = {}
        for name in zin.namelist():
            file_contents[name] = zin.read(name)

    # Parse document
    doc_xml = file_contents['word/document.xml']
    root = etree.fromstring(doc_xml)
    body = root.find(f'.//{w("body")}')

    # Collect all paragraphs to process
    all_elements = list(body)
    para_indices = {}
    for i, elem in enumerate(all_elements):
        if elem.tag == w('p'):
            texts = elem.findall(f'.//{w("t")}')
            full_text = ''.join([t.text for t in texts if t.text])
            para_indices[i] = full_text

    # Find formula paragraphs that need to be fixed
    bad_indices = []
    for i, full_text in para_indices.items():
        if 'S = (R + A + C + Cit) / 4 × 20' in full_text:
            bad_indices.append(i)
        elif 'P(pass@k) = 1' in full_text and '−' in full_text:
            bad_indices.append(i)
        elif 'Δ = Sexp' in full_text and '−' in full_text:
            bad_indices.append(i)

    # Remove bad formula paragraphs and their description lines
    # Also remove duplicate description paragraphs
    bad_indices = sorted(set(bad_indices))
    print(f"Found {len(bad_indices)} bad formula paragraphs at indices: {bad_indices[:10]}")

    # Remove elements at bad indices (in reverse order to maintain indices)
    for i in reversed(bad_indices):
        elem = all_elements[i]
        if elem.getparent() is not None:
            elem.getparent().remove(elem)
        print(f"Removed paragraph at index {i}")

    # Re-scan to find insertion points
    all_elements = list(body)
    insert_points = {}

    for i, elem in enumerate(all_elements):
        if elem.tag == w('p'):
            texts = elem.findall(f'.//{w("t")}')
            full_text = ''.join([t.text for t in texts if t.text])

            # Find the original text paragraphs that should have formulas after them
            if '每项满分5分' in full_text and '百分制' in full_text:
                insert_points['rag'] = i
            elif 'Pass@k表示在每个任务最多允许k次尝试' in full_text:
                insert_points['passk'] = i
            elif '对比两组的任务得分差异' in full_text:
                insert_points['grpo'] = i

    print(f"Insert points: {insert_points}")

    # Insert new formula paragraphs
    insertions = []

    # RAG formula
    if 'rag' in insert_points:
        idx = insert_points['rag']
        insertions.append((idx + 1, create_description_para('其中，整体得分计算公式为：')))
        insertions.append((idx + 2, create_formula_para([
            ('S', True, False), (' = (', False, False),
            ('R', False, True), (' + ', False, False),
            ('A', False, True), (' + ', False, False),
            ('C', False, True), (' + ', False, False),
            ('Cit', False, True), (') / 4 × 20', False, False)
        ])))
        insertions.append((idx + 3, create_description_para('式中：R为相关性得分，A为准确性得分，C为完整性得分，Cit为引证质量得分，满分均为5分。')))

    # Pass@k formula
    if 'passk' in insert_points:
        idx = insert_points['passk']
        insertions.append((idx + 1, create_description_para('累计成功率的计算公式为：')))
        insertions.append((idx + 2, create_formula_para([
            ('P(pass@', True, False),
            ('k', False, True),  # k in italic
            (') = 1 − (1 − ', False, False),
            ('p', False, True),
            (')', False, False),
            ('ᵏ', True, False)  # superscript k
        ])))
        insertions.append((idx + 3, create_description_para('式中：p为单次尝试的成功率，k为允许的最大尝试次数。')))

    # GRPO formula
    if 'grpo' in insert_points:
        idx = insert_points['grpo']
        insertions.append((idx + 1, create_description_para('任务得分提升计算公式为：')))
        insertions.append((idx + 2, create_formula_para([
            ('Δ', True, False),
            (' = ', False, False),
            ('S', False, True),
            ('exp', False, False),
            (' − ', False, False),
            ('S', False, True),
            ('base', False, False)
        ])))
        insertions.append((idx + 3, create_description_para('式中：S_exp为有经验Agent的得分，S_base为无经验Agent的得分，Δ > 0表示经验提升有效。')))

    # Sort by position (descending) and insert
    insertions.sort(key=lambda x: x[0], reverse=True)
    for pos, elem in insertions:
        body.insert(pos, elem)

    print(f"Inserted {len(insertions)} new paragraphs")

    # Serialize and save
    doc_xml_final = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    file_contents['word/document.xml'] = doc_xml_final

    with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_contents.items():
            zout.writestr(name, data)

    print(f"\nDone! Fixed formulas in {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
