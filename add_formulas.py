#!/usr/bin/env python3
"""
Add mathematical formulas for evaluation metrics to Chapter 5.
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

def create_paragraph_with_text(text, center=False):
    """Create a simple paragraph with text."""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    if center:
        jc = etree.SubElement(pPr, w('jc'))
        jc.set(w('val'), 'center')
    else:
        ind = etree.SubElement(pPr, w('ind'))
        ind.set(w('firstLineChars'), '200')
        ind.set(w('firstLine'), '480')

    r = etree.SubElement(p, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), 'Times New Roman')
    rFonts.set(w('ascii'), 'Times New Roman')
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '24')
    szCs = etree.SubElement(rPr, w('szCs'))
    szCs.set(w('val'), '24')
    t = etree.SubElement(r, w('t'))
    t.text = text
    return p


def create_formula_run(text, base_text=None):
    """Create a run with formula text, optionally appended to base_text."""
    r = etree.Element(w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), 'Times New Roman')
    rFonts.set(w('ascii'), 'Times New Roman')
    # Slightly larger font for formulas
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '26')
    szCs = etree.SubElement(rPr, w('szCs'))
    szCs.set(w('val'), '26')
    t = etree.SubElement(r, w('t'))
    t.text = text
    return r


def create_rag_formula_paragraph():
    """Create a centered formula paragraph for RAG overall score."""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    jc = etree.SubElement(pPr, w('jc'))
    jc.set(w('val'), 'center')
    spacing = etree.SubElement(pPr, w('spacing'))
    spacing.set(w('before'), '80')
    spacing.set(w('after'), '80')

    # "S = "
    r1 = create_formula_run('S')
    rPr1 = r1.find(w('rPr'))
    etree.SubElement(rPr1, w('b'))
    etree.SubElement(rPr1, w('bCs'))
    p.append(r1)

    # " = "
    r2 = create_formula_run(' = ')
    p.append(r2)

    # "("
    r3 = create_formula_run('(')
    p.append(r3)

    # R
    r4 = create_formula_run('R')
    rPr4 = r4.find(w('rPr'))
    etree.SubElement(rPr4, w('i'))  # Italic for variables
    etree.SubElement(rPr4, w('iCs'))
    p.append(r4)

    # +
    r5 = create_formula_run(' + ')
    p.append(r5)

    # A
    r6 = create_formula_run('A')
    rPr6 = r6.find(w('rPr'))
    etree.SubElement(rPr6, w('i'))
    etree.SubElement(rPr6, w('iCs'))
    p.append(r6)

    # +
    r7 = create_formula_run(' + ')
    p.append(r7)

    # C
    r8 = create_formula_run('C')
    rPr8 = r8.find(w('rPr'))
    etree.SubElement(rPr8, w('i'))
    etree.SubElement(rPr8, w('iCs'))
    p.append(r8)

    # +
    r9 = create_formula_run(' + ')
    p.append(r9)

    # Cit
    r10 = create_formula_run('Cit')
    rPr10 = r10.find(w('rPr'))
    etree.SubElement(rPr10, w('i'))
    etree.SubElement(rPr10, w('iCs'))
    p.append(r10)

    # ") / 4 × 20"
    r11 = create_formula_run(') / 4 × 20')
    p.append(r11)

    return p


def create_passk_formula_paragraph():
    """Create formula: P(pass@k) = 1 - (1-p)^k"""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    jc = etree.SubElement(pPr, w('jc'))
    jc.set(w('val'), 'center')
    spacing = etree.SubElement(pPr, w('spacing'))
    spacing.set(w('before'), '80')
    spacing.set(w('after'), '80')

    # P(pass@k)
    r1 = etree.SubElement(p, w('r'))
    rPr1 = etree.SubElement(r1, w('rPr'))
    rFonts = etree.SubElement(rPr1, w('rFonts'))
    rFonts.set(w('eastAsia'), 'Times New Roman')
    rFonts.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr1, w('b'))
    etree.SubElement(rPr1, w('bCs'))
    sz = etree.SubElement(rPr1, w('sz'))
    sz.set(w('val'), '26')
    szCs = etree.SubElement(rPr1, w('szCs'))
    szCs.set(w('val'), '26')
    t1 = etree.SubElement(r1, w('t'))
    t1.text = 'P'

    r2 = etree.SubElement(p, w('r'))
    rPr2 = etree.SubElement(r2, w('rPr'))
    rFonts2 = etree.SubElement(rPr2, w('rFonts'))
    rFonts2.set(w('eastAsia'), 'Times New Roman')
    rFonts2.set(w('ascii'), 'Times New Roman')
    sz2 = etree.SubElement(rPr2, w('sz'))
    sz2.set(w('val'), '26')
    szCs2 = etree.SubElement(rPr2, w('szCs'))
    szCs2.set(w('val'), '26')
    t2 = etree.SubElement(r2, w('t'))
    t2.text = '(pass@'

    # k as superscript-like (using regular k, Word will display)
    r3 = etree.SubElement(p, w('r'))
    rPr3 = etree.SubElement(r3, w('rPr'))
    rFonts3 = etree.SubElement(rPr3, w('rFonts'))
    rFonts3.set(w('eastAsia'), 'Times New Roman')
    rFonts3.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr3, w('i'))
    etree.SubElement(rPr3, w('iCs'))
    sz3 = etree.SubElement(rPr3, w('sz'))
    sz3.set(w('val'), '26')
    szCs3 = etree.SubElement(rPr3, w('szCs'))
    szCs3.set(w('val'), '26')
    t3 = etree.SubElement(r3, w('t'))
    t3.text = 'k'

    r4 = etree.SubElement(p, w('r'))
    rPr4 = etree.SubElement(r4, w('rPr'))
    rFonts4 = etree.SubElement(rPr4, w('rFonts'))
    rFonts4.set(w('eastAsia'), 'Times New Roman')
    rFonts4.set(w('ascii'), 'Times New Roman')
    sz4 = etree.SubElement(rPr4, w('sz'))
    sz4.set(w('val'), '26')
    szCs4 = etree.SubElement(rPr4, w('szCs'))
    szCs4.set(w('val'), '26')
    t4 = etree.SubElement(r4, w('t'))
    t4.text = ') = 1 − (1 − '

    # p
    r5 = etree.SubElement(p, w('r'))
    rPr5 = etree.SubElement(r5, w('rPr'))
    rFonts5 = etree.SubElement(rPr5, w('rFonts'))
    rFonts5.set(w('eastAsia'), 'Times New Roman')
    rFonts5.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr5, w('i'))
    etree.SubElement(rPr5, w('iCs'))
    sz5 = etree.SubElement(rPr5, w('sz'))
    sz5.set(w('val'), '26')
    szCs5 = etree.SubElement(rPr5, w('szCs'))
    szCs5.set(w('val'), '26')
    t5 = etree.SubElement(r5, w('t'))
    t5.text = 'p'

    r6 = etree.SubElement(p, w('r'))
    rPr6 = etree.SubElement(r6, w('rPr'))
    rFonts6 = etree.SubElement(rPr6, w('rFonts'))
    rFonts6.set(w('eastAsia'), 'Times New Roman')
    rFonts6.set(w('ascii'), 'Times New Roman')
    sz6 = etree.SubElement(rPr6, w('sz'))
    sz6.set(w('val'), '26')
    szCs6 = etree.SubElement(rPr6, w('szCs'))
    szCs6.set(w('val'), '26')
    t6 = etree.SubElement(r6, w('t'))
    t6.text = ')'

    r7 = etree.SubElement(p, w('r'))
    rPr7 = etree.SubElement(r7, w('rPr'))
    rFonts7 = etree.SubElement(rPr7, w('rFonts'))
    rFonts7.set(w('eastAsia'), 'Times New Roman')
    rFonts7.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr7, w('b'))
    etree.SubElement(rPr7, w('bCs'))
    sz7 = etree.SubElement(rPr7, w('sz'))
    sz7.set(w('val'), '26')
    szCs7 = etree.SubElement(rPr7, w('szCs'))
    szCs7.set(w('val'), '26')
    t7 = etree.SubElement(r7, w('t'))
    t7.text = 'ᵏ'

    return p


def create_grpo_formula_paragraph():
    """Create formula: Δ = S_exp - S_base"""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    jc = etree.SubElement(pPr, w('jc'))
    jc.set(w('val'), 'center')
    spacing = etree.SubElement(pPr, w('spacing'))
    spacing.set(w('before'), '80')
    spacing.set(w('after'), '80')

    # Δ (Greek letter)
    r1 = etree.SubElement(p, w('r'))
    rPr1 = etree.SubElement(r1, w('rPr'))
    rFonts = etree.SubElement(rPr1, w('rFonts'))
    rFonts.set(w('eastAsia'), 'Times New Roman')
    rFonts.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr1, w('b'))
    etree.SubElement(rPr1, w('bCs'))
    sz = etree.SubElement(rPr1, w('sz'))
    sz.set(w('val'), '26')
    szCs = etree.SubElement(rPr1, w('szCs'))
    szCs.set(w('val'), '26')
    t1 = etree.SubElement(r1, w('t'))
    t1.text = 'Δ'

    #  =
    r2 = etree.SubElement(p, w('r'))
    rPr2 = etree.SubElement(r2, w('rPr'))
    rFonts2 = etree.SubElement(rPr2, w('rFonts'))
    rFonts2.set(w('eastAsia'), 'Times New Roman')
    rFonts2.set(w('ascii'), 'Times New Roman')
    sz2 = etree.SubElement(rPr2, w('sz'))
    sz2.set(w('val'), '26')
    szCs2 = etree.SubElement(rPr2, w('szCs'))
    szCs2.set(w('val'), '26')
    t2 = etree.SubElement(r2, w('t'))
    t2.text = ' = S'

    # exp subscript
    r3 = etree.SubElement(p, w('r'))
    rPr3 = etree.SubElement(r3, w('rPr'))
    rFonts3 = etree.SubElement(rPr3, w('rFonts'))
    rFonts3.set(w('eastAsia'), 'Times New Roman')
    rFonts3.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr3, w('i'))
    etree.SubElement(rPr3, w('iCs'))
    sz3 = etree.SubElement(rPr3, w('sz'))
    sz3.set(w('val'), '26')
    szCs3 = etree.SubElement(rPr3, w('szCs'))
    szCs3.set(w('val'), '26')
    t3 = etree.SubElement(r3, w('t'))
    t3.text = 'exp'

    #  - S
    r4 = etree.SubElement(p, w('r'))
    rPr4 = etree.SubElement(r4, w('rPr'))
    rFonts4 = etree.SubElement(rPr4, w('rFonts'))
    rFonts4.set(w('eastAsia'), 'Times New Roman')
    rFonts4.set(w('ascii'), 'Times New Roman')
    sz4 = etree.SubElement(rPr4, w('sz'))
    sz4.set(w('val'), '26')
    szCs4 = etree.SubElement(rPr4, w('szCs'))
    szCs4.set(w('val'), '26')
    t4 = etree.SubElement(r4, w('t'))
    t4.text = ' − S'

    # base subscript
    r5 = etree.SubElement(p, w('r'))
    rPr5 = etree.SubElement(r5, w('rPr'))
    rFonts5 = etree.SubElement(rPr5, w('rFonts'))
    rFonts5.set(w('eastAsia'), 'Times New Roman')
    rFonts5.set(w('ascii'), 'Times New Roman')
    etree.SubElement(rPr5, w('i'))
    etree.SubElement(rPr5, w('iCs'))
    sz5 = etree.SubElement(rPr5, w('sz'))
    sz5.set(w('val'), '26')
    szCs5 = etree.SubElement(rPr5, w('szCs'))
    szCs5.set(w('val'), '26')
    t5 = etree.SubElement(r5, w('t'))
    t5.text = 'base'

    return p


def main():
    print("Adding improved mathematical formulas to Chapter 5...")

    # Read all files
    with zipfile.ZipFile(INPUT_PATH, 'r') as zin:
        file_contents = {}
        for name in zin.namelist():
            file_contents[name] = zin.read(name)

    # Parse document
    doc_xml = file_contents['word/document.xml']
    root = etree.fromstring(doc_xml)
    body = root.find(f'.//{w("body")}')
    paragraphs = body.findall(f'.//{w("p")}')

    # Find insertion points and replace old formulas with proper ones
    for i, p in enumerate(paragraphs):
        texts = p.findall(f'.//{w("t")}')
        full_text = ''.join([t.text for t in texts if t.text])

        if 'S = (R + A + C + Cit) / 4 × 20' in full_text and '式中' not in full_text:
            # This is the old RAG formula - replace with proper one
            # First, add the description paragraph
            desc_p = create_paragraph_with_text('其中，整体得分计算公式为：')
            body.insert(i, desc_p)

            # Now replace this paragraph with the proper formula
            formula_p = create_rag_formula_paragraph()
            p.getparent().replace(p, formula_p)
            print(f"Replaced RAG formula at original para {i}")
            break

    # Re-get paragraphs
    paragraphs = body.findall(f'.//{w("p")}')

    for i, p in enumerate(paragraphs):
        texts = p.findall(f'.//{w("t")}')
        full_text = ''.join([t.text for t in texts if t.text])

        if 'P(pass@k) = 1 - (1 - p)k' in full_text:
            # Replace with proper Pass@k formula
            desc_p = create_paragraph_with_text('累计成功率的计算公式为：')
            body.insert(i, desc_p)

            formula_p = create_passk_formula_paragraph()
            p.getparent().replace(p, formula_p)
            print(f"Replaced Pass@k formula at original para {i}")
            break

    # Re-get paragraphs
    paragraphs = body.findall(f'.//{w("p")}')

    for i, p in enumerate(paragraphs):
        texts = p.findall(f'.//{w("t")}')
        full_text = ''.join([t.text for t in texts if t.text])

        if 'Δ = Sexp - Sbase' in full_text or ('Δ = S' in full_text and 'exp' in full_text and 'base' in full_text):
            # Replace with proper GRPO formula
            desc_p = create_paragraph_with_text('任务得分提升计算公式为：')
            body.insert(i, desc_p)

            formula_p = create_grpo_formula_paragraph()
            p.getparent().replace(p, formula_p)
            print(f"Replaced GRPO formula at original para {i}")
            break

    # Serialize and save
    doc_xml_final = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    file_contents['word/document.xml'] = doc_xml_final

    with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_contents.items():
            zout.writestr(name, data)

    print(f"\nDone! Improved formulas added to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
