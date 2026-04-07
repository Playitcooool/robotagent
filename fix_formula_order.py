#!/usr/bin/env python3
"""
Fix formula ordering in Chapter 5.
Move formula S = (R+A+C+Cit)/4×20 to after the RAG table.
Also remove duplicate explanation paragraph.
"""

import zipfile
from lxml import etree

INPUT_PATH = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿_更新版.docx'
OUTPUT_PATH = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿_更新版.docx'

ns_w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def main():
    print("Fixing formula ordering in Chapter 5...")

    with zipfile.ZipFile(INPUT_PATH, 'r') as zin:
        file_contents = {}
        for name in zin.namelist():
            file_contents[name] = zin.read(name)

    doc_xml = file_contents['word/document.xml']
    root = etree.fromstring(doc_xml)
    body = root.find('.//{%s}body' % ns_w)
    body_children = list(body)

    # Find the formula, table, and explanation paragraphs
    formula_para = None
    formula_idx = None
    explanation_para = None
    explanation_idx = None
    dup_explanation_para = None
    dup_explanation_idx = None
    table_idx = None

    for i, child in enumerate(body_children):
        if child.tag.replace('{%s}' % ns_w, '') == 'p':
            texts = child.findall('.//{%s}t' % ns_w)
            text = ''.join([t.text for t in texts if t.text])
            if text == 'S = (R + A + C + Cit) / 4 × 20':
                formula_para = child
                formula_idx = i
                print(f"Found formula at body index {i}")
            elif text == '式中：R为相关性得分，A为准确性得分，C为完整性得分，Cit为引证质量得分，满分均为5分。':
                if explanation_para is None:
                    explanation_para = child
                    explanation_idx = i
                    print(f"Found explanation at body index {i}")
                else:
                    dup_explanation_para = child
                    dup_explanation_idx = i
                    print(f"Found DUPLICATE explanation at body index {i}")
        elif child.tag.replace('{%s}' % ns_w, '') == 'tbl':
            # Check if this is the RAG comparison table
            inner_paras = child.findall('.//{%s}p' % ns_w)
            if inner_paras:
                first_texts = inner_paras[0].findall('.//{%s}t' % ns_w)
                first_text = ''.join([t.text for t in first_texts if t.text])
                if '评价指标' in first_text:
                    table_idx = i
                    print(f"Found RAG table at body index {i}")

    if formula_para is None or explanation_para is None or table_idx is None:
        print(f"ERROR: formula={formula_para is None}, table={table_idx is None}, explanation={explanation_para is None}")
        return

    # Remove formula from its current position
    body.remove(formula_para)
    print(f"Removed formula from index {formula_idx}")

    # Re-index after removal
    body_children = list(body)

    # Find the new index of the table and explanation after removal
    new_table_idx = body_children.index(formula_para) if formula_para in body_children else None
    # Actually formula_para was removed, so we need to find where the table is now
    new_table_idx = None
    for i, child in enumerate(body_children):
        if child.tag.replace('{%s}' % ns_w, '') == 'tbl':
            inner_paras = child.findall('.//{%s}p' % ns_w)
            if inner_paras:
                first_texts = inner_paras[0].findall('.//{%s}t' % ns_w)
                first_text = ''.join([t.text for t in first_texts if t.text])
                if '评价指标' in first_text:
                    new_table_idx = i
                    print(f"Table is now at index {i}")
                    break

    if new_table_idx is None:
        print("ERROR: Could not find table after removal")
        return

    # Insert formula AFTER the table
    body.insert(new_table_idx + 1, formula_para)
    print(f"Inserted formula at index {new_table_idx + 1}")

    # Remove duplicate explanation if found
    if dup_explanation_para is not None:
        body.remove(dup_explanation_para)
        print(f"Removed duplicate explanation from index {dup_explanation_idx}")

    # Verify the new order
    body_children = list(body)
    print("\nNew order (body indices 260-270):")
    for i in range(max(0, 260), min(275, len(body_children))):
        child = body_children[i]
        tag = child.tag.replace('{%s}' % ns_w, '')
        if tag == 'p':
            texts = child.findall('.//{%s}t' % ns_w)
            text = ''.join([t.text for t in texts if t.text])[:60]
            print(f"  {i}: <p> {text if text else '[empty]'}")
        elif tag == 'tbl':
            print(f"  {i}: <tbl> [TABLE with评价指标]")

    # Save
    doc_xml_final = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    file_contents['word/document.xml'] = doc_xml_final

    with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_contents.items():
            zout.writestr(name, data)

    print(f"\nDone! Fixed formula ordering in {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
