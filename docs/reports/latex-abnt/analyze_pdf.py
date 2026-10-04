import sys

import pymupdf

sys.stdout.reconfigure(encoding='utf-8')

def analyze_document(pdf_path):
    doc = pymupdf.open(pdf_path)
    print(f"=== DETAILED PDF ANALYSIS FOR {pdf_path} ===")
    print(f"Total Pages: {len(doc)}\n")

    # 1. Font Analysis across the whole document
    doc_fonts = set()
    for i, page in enumerate(doc):
        page_fonts = page.get_fonts()
        for f in page_fonts:
            # f is (xref, ext, type, basefont, name, encoding)
            doc_fonts.add((f[3], f[2]))

    print("--- FONTS USED IN DOCUMENT ---")
    for f_name, f_type in sorted(doc_fonts):
        print(f"  Font: {f_name:40s} | Type: {f_type}")
    print()

    # 2. Check for blank or near-blank pages
    print("--- PAGE-BY-PAGE AUDIT ---")
    for i, page in enumerate(doc):
        text = page.get_text()
        words = text.split()
        rect = page.rect
        blocks = page.get_text("blocks")

        # Margins: A4 is 595.28 x 841.89 points
        # ABNT margins: top 3cm = 85.04pt, bottom 2cm = 56.69pt, left 3cm = 85.04pt, right 2cm = 56.69pt
        # So text width = 595.28 - 85.04 - 56.69 = 453.55pt (x from 85.04 to 538.59)
        # Text height = 841.89 - 85.04 - 56.69 = 700.16pt (y from 85.04 to 785.20)

        # Check text bounds (excluding headers/footers)
        content_blocks = [b for b in blocks if b[4].strip() and not (b[3] < 70 or b[1] > 800)]

        min_x = min([b[0] for b in content_blocks]) if content_blocks else 0
        max_x = max([b[2] for b in content_blocks]) if content_blocks else 0
        min_y = min([b[1] for b in content_blocks]) if content_blocks else 0
        max_y = max([b[3] for b in content_blocks]) if content_blocks else 0

        flags = []
        if len(words) < 20:
            flags.append(f"FEW_WORDS ({len(words)})")
        if max_x > 545: # overflow right margin (538.6 + small tolerance)
            flags.append(f"OVERFLOW_RIGHT (max_x={max_x:.1f})")
        if min_x < 80 and i >= 2: # left margin too small
            flags.append(f"UNDERFLOW_LEFT (min_x={min_x:.1f})")

        flag_str = " | ".join(flags) if flags else "OK"
        print(f"Page {i+1:2d}: Words={len(words):4d}, Blocks={len(blocks):2d}, x=[{min_x:5.1f}, {max_x:5.1f}], y=[{min_y:5.1f}, {max_y:5.1f}] -> {flag_str}")
        if flags:
            for b in content_blocks:
                if b[2] > 545:
                    print(f"    Overflow text snippet: {b[4][:80].strip()!r}")
                if len(words) < 20:
                    print(f"    Few words text: {b[4][:80].strip()!r}")

if __name__ == "__main__":
    analyze_document("main.pdf")
