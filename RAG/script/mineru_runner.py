import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_ROOT = Path("/Volumes/Samsung/Projects/robotagent/arxiv_pdfs_filtered")
OUTPUT_ROOT = BASE_DIR / "output"
LOG_ROOT = BASE_DIR / "logs"

MAX_RETRY = 3
SLEEP_BETWEEN = 5  # seconds
ERROR_PATTERNS = [
    "traceback",
    "error",
    "exception",
    "aborted",
    "cannot import name",
]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def get_pdfs():
    if not PDF_ROOT.exists():
        return []
    return sorted([p for p in PDF_ROOT.iterdir() if p.suffix.lower() == ".pdf"])


def has_new_outputs(before_snapshot: set[Path]) -> bool:
    after_snapshot = set(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.exists() else set()
    new_entries = [p for p in after_snapshot - before_snapshot if p.exists()]
    if new_entries:
        return True
    # fallback: some backends may overwrite existing files instead of creating new folders
    return any(OUTPUT_ROOT.rglob("*.md")) or any(OUTPUT_ROOT.rglob("*.json"))


def log_has_fatal_error(log_file: Path) -> bool:
    if not log_file.exists():
        return True
    text = log_file.read_text(errors="ignore").lower()
    return any(pat in text for pat in ERROR_PATTERNS)


def run_pdf(pdf: Path):
    if not pdf.exists():
        print(f"[SKIP] missing {pdf}")
        return

    env = {
        **dict(**subprocess.os.environ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    log_file = LOG_ROOT / f"{pdf.stem}.log"
    retry_file = LOG_ROOT / f"{pdf.stem}.retry"
    done_flag = LOG_ROOT / f"{pdf.stem}.done"

    retry = int(retry_file.read_text()) if retry_file.exists() else 0
    if retry >= MAX_RETRY:
        print(f"[SKIP] {pdf.name} exceeded max retries")
        return

    print(f"[RUN] {pdf.name} (retry {retry})")

    before_snapshot = set(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.exists() else set()

    with log_file.open("w") as lf:
        proc = subprocess.run(
            [
                "mineru",
                "-p",
                str(pdf),
                "-o",
                str(OUTPUT_ROOT),
                "--backend",
                "pipeline",
            ],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )

    output_ok = has_new_outputs(before_snapshot)
    fatal_in_log = log_has_fatal_error(log_file)
    success = (proc.returncode == 0) and output_ok and (not fatal_in_log)

    if success:
        done_flag.touch()
        if retry_file.exists():
            retry_file.unlink()
        print(f"[OK] {pdf.name} finished")
    else:
        retry += 1
        retry_file.write_text(str(retry))
        reason = []
        if proc.returncode != 0:
            reason.append(f"exit={proc.returncode}")
        if not output_ok:
            reason.append("no_output_artifacts")
        if fatal_in_log:
            reason.append("fatal_error_in_log")
        print(f"[FAIL] {pdf.name}, retry {retry} ({', '.join(reason)})")

    # 给 macOS 回收内存一点时间
    time.sleep(SLEEP_BETWEEN)


def main():
    pdfs = get_pdfs()
    if not pdfs:
        print("[DONE] no pdfs found")
        return
    for pdf in pdfs:
        run_pdf(pdf)


if __name__ == "__main__":
    main()
