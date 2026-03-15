import argparse
from pathlib import Path
import sys


def is_pdf_header_ok(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False


def can_open_with_pdfium(path: Path) -> bool:
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        _ = len(doc)
        doc.close()
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check PDFs under a directory and delete corrupted ones."
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="/Volumes/Samsung/Projects/robotagent/arxiv_pdfs_filtered",
        help="Directory containing PDFs to check.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report bad PDFs, do not delete.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"[ERROR] pdf_dir not found: {pdf_dir}")
        return 1

    pdfs = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print("[DONE] no pdfs found")
        return 0

    bad = []
    for pdf in pdfs:
        header_ok = is_pdf_header_ok(pdf)
        if not header_ok:
            bad.append((pdf, "bad_header"))
            continue
        if not can_open_with_pdfium(pdf):
            bad.append((pdf, "pdfium_open_failed"))

    if not bad:
        print("[DONE] no corrupted pdfs found")
        return 0

    for path, reason in bad:
        if args.dry_run:
            print(f"[BAD] {path.name} ({reason})")
            continue
        try:
            path.unlink()
            print(f"[DELETED] {path.name} ({reason})")
        except Exception as e:
            print(f"[ERROR] failed to delete {path.name}: {e}")

    print(f"[SUMMARY] total={len(pdfs)} bad={len(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
