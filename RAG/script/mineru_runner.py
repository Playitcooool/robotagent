import subprocess
import time
from pathlib import Path

BATCH_ROOT = Path("batches")
OUTPUT_ROOT = Path("output")
LOG_ROOT = Path("logs")

MAX_RETRY = 3
SLEEP_BETWEEN = 5  # seconds

OUTPUT_ROOT.mkdir(exist_ok=True)
LOG_ROOT.mkdir(exist_ok=True)

def get_batches():
    return sorted([
        p for p in BATCH_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("batch_")
    ])

def is_done(batch):
    return (LOG_ROOT / f"{batch.name}.done").exists()

def run_batch(batch: Path):
    log_file = LOG_ROOT / f"{batch.name}.log"
    retry_file = LOG_ROOT / f"{batch.name}.retry"

    retry = int(retry_file.read_text()) if retry_file.exists() else 0
    if retry >= MAX_RETRY:
        print(f"[SKIP] {batch.name} exceeded max retries")
        return

    print(f"[RUN] {batch.name} (retry {retry})")

    env = {
        **dict(**subprocess.os.environ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    with log_file.open("w") as lf:
        proc = subprocess.run(
            [
                "mineru",
                "-p", str(batch),
                "-o", str(OUTPUT_ROOT),
                "--backend", "pipeline"
            ],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env
        )

    if proc.returncode == 0:
        (LOG_ROOT / f"{batch.name}.done").touch()
        if retry_file.exists():
            retry_file.unlink()
        print(f"[OK] {batch.name} finished")
    else:
        retry += 1
        retry_file.write_text(str(retry))
        print(f"[FAIL] {batch.name}, retry {retry}")

    # 给 macOS 回收内存一点时间
    time.sleep(SLEEP_BETWEEN)

def main():
    while True:
        batches = get_batches()
        pending = [b for b in batches if not is_done(b)]

        if not pending:
            print("[DONE] All batches processed")
            break

        for batch in pending:
            run_batch(batch)

if __name__ == "__main__":
    main()
