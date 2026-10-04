
import fitz


def inspect_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")

    issues = []

    for i, page in enumerate(doc):
        page_num = i + 1
        text = page.get_text()
        words = text.split()
        num_words = len(words)

        # Check empty or near-empty pages
        if num_words < 15:
            issues.append(f"Page {page_num}: Nearly empty ({num_words} words): {text.strip()[:100]}")

        # Check for undefined references/citations
        if "??" in text:
            issues.append(f"Page {page_num}: Found '??' (undefined reference or citation)")
            for line in text.split('\n'):
                if "??" in line:
                    issues.append(f"   Line: {line.strip()}")
        if "[?]" in text:
            issues.append(f"Page {page_num}: Found '[?]'")

        # Check font usage
        fonts = page.get_fonts()
        # Report fonts if needed

        # Check rect and layout
        rect = page.rect
        # Text bounding boxes
        blocks = page.get_text("blocks")
        # Check if text overflows page width (a4 is 595.3 x 841.9 pt)
        for b in blocks:
            x0, y0, x1, y1, btext, bno, btype = b
            if x1 > rect.width + 5:
                issues.append(f"Page {page_num}: Block overflows right margin: x1={x1:.1f} > {rect.width:.1f}: {btext[:40]}")
            if y1 > rect.height + 5:
                issues.append(f"Page {page_num}: Block overflows bottom margin: y1={y1:.1f} > {rect.height:.1f}: {btext[:40]}")

    print("\n--- ISSUES FOUND ---")
    for issue in issues:
        print(issue)

    print("\n--- PAGE SUMMARY ---")
    for i in range(len(doc)):
        page = doc[i]
        first_line = ""
        for line in page.get_text().split("\n"):
            if line.strip():
                first_line = line.strip()
                break
        print(f"Page {i+1:2d} | Words: {len(page.get_text().split()):4d} | Starts with: {first_line[:50]}")

if __name__ == "__main__":
    inspect_pdf("main.pdf")
