import re
from lxml import etree

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def w(tag):
    return f'{{{W_NS}}}{tag}'

def create_body_para(text, bold=False, center=False, first_line_indent=True, font_size='24', font='宋体'):
    """Create a body paragraph with Chinese typography.

    Args:
        text: the paragraph text
        bold: whether to bold the text
        center: whether to center align (for captions)
        first_line_indent: whether to add 2-em first line indent
        font_size: half-points (24 = 12pt)
        font: font family for eastAsia
    """
    nsmap = {'w': W_NS}
    para = etree.Element(w('p'), nsmap=nsmap)

    # Create paragraph properties
    pPr = etree.SubElement(para, w('pPr'))

    if center:
        # Center alignment for captions
        ind = etree.SubElement(pPr, w('ind'))
        ind.set(w('firstLineChars'), '200')
        ind.set(w('firstLine'), '480')

        jc = etree.SubElement(pPr, w('jc'))
        jc.set(w('val'), 'center')
    else:
        # Body paragraph with first-line indent
        ind = etree.SubElement(pPr, w('ind'))
        ind.set(w('firstLineChars'), '200')
        ind.set(w('firstLine'), '480')

    # Create run element
    run = etree.SubElement(para, w('r'))

    # Add run properties to run
    run_rPr = etree.SubElement(run, w('rPr'))

    run_rFonts = etree.SubElement(run_rPr, w('rFonts'))
    run_rFonts.set(w('eastAsia'), font)
    run_rFonts.set(w('ascii'), font)

    if bold:
        etree.SubElement(run_rPr, w('b'))
        etree.SubElement(run_rPr, w('bCs'))

    run_sz = etree.SubElement(run_rPr, w('sz'))
    run_sz.set(w('val'), font_size)
    run_szCs = etree.SubElement(run_rPr, w('szCs'))
    run_szCs.set(w('val'), font_size)

    # Create text element
    t = etree.SubElement(run, w('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text

    return para

def create_heading_para(text, level=3):
    """Create a section heading paragraph (no indent, bold, 黑体).

    Args:
        text: heading text like "4.5.3 前端仿真渲染技术"
        level: outline level (3 = 3rd level heading)
    """
    nsmap = {'w': W_NS}
    para = etree.Element(w('p'), nsmap=nsmap)

    # Create paragraph properties
    pPr = etree.SubElement(para, w('pPr'))

    # Outline level for Word's navigation
    outlineLvl = etree.SubElement(pPr, w('outlineLvl'))
    outlineLvl.set(w('val'), str(level))

    # Keep heading with next paragraph
    etree.SubElement(pPr, w('keepNext'))

    # Create run properties
    rPr = etree.SubElement(pPr, w('rPr'))

    # Font: 黑体 for both eastAsia and ascii
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), '黑体')
    rFonts.set(w('ascii'), '黑体')

    # Bold
    etree.SubElement(rPr, w('b'))
    etree.SubElement(rPr, w('bCs'))

    # Size: 28 half-points = 14pt
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '28')
    szCs = etree.SubElement(rPr, w('szCs'))
    szCs.set(w('val'), '28')

    # Create run element
    run = etree.SubElement(para, w('r'))

    # Add run properties to run
    run_rPr = etree.SubElement(run, w('rPr'))

    run_rFonts = etree.SubElement(run_rPr, w('rFonts'))
    run_rFonts.set(w('eastAsia'), '黑体')
    run_rFonts.set(w('ascii'), '黑体')

    etree.SubElement(run_rPr, w('b'))
    etree.SubElement(run_rPr, w('bCs'))

    run_sz = etree.SubElement(run_rPr, w('sz'))
    run_sz.set(w('val'), '28')
    run_szCs = etree.SubElement(run_rPr, w('szCs'))
    run_szCs.set(w('val'), '28')

    # Create text element
    t = etree.SubElement(run, w('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text

    return para

if __name__ == '__main__':
    # Test create_body_para
    p = create_body_para('测试正文段落，含中文和English。', bold=False)
    xml = etree.tostring(p, encoding='unicode', pretty_print=True)
    assert 'w:ind' in xml, "Should have w:ind for first-line indent"
    assert '宋体' in xml, "Should use 宋体 font"

    # Test centered body para
    p2 = create_body_para('图5-1 测试', center=True)
    xml2 = etree.tostring(p2, encoding='unicode', pretty_print=True)
    assert 'center' in xml2, "Should be centered"

    # Test heading para
    h = create_heading_para('4.5.3 前端仿真渲染技术', level=3)
    hxml = etree.tostring(h, encoding='unicode', pretty_print=True)
    assert '黑体' in hxml, "Should use 黑体"
    assert 'outlineLvl' in hxml, "Should have outlineLvl"
    assert 'keepNext' in hxml, "Should have keepNext"

    print("All tests passed!")
