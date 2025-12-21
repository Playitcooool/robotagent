# split_pdfs.py
import os
import shutil
from pathlib import Path

PDF_DIR = Path("arxiv_pdfs")
BATCH_DIR = Path("batches")
BATCH_SIZE = 1  # ⭐⭐⭐ 推荐：3～5（pipeline）

BATCH_DIR.mkdir(exist_ok=True)

pdfs = sorted([p for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf"])

print(f"Total PDFs: {len(pdfs)}")
print(f"Batch size: {BATCH_SIZE}")

for i in range(0, len(pdfs), BATCH_SIZE):
    batch_pdfs = pdfs[i:i + BATCH_SIZE]
    batch_id = i // BATCH_SIZE
    batch_path = BATCH_DIR / f"batch_{batch_id:03d}"
    batch_path.mkdir(exist_ok=True)

    for pdf in batch_pdfs:
        shutil.copy2(pdf, batch_path / pdf.name)

    print(f"Created {batch_path} ({len(batch_pdfs)} PDFs)")
