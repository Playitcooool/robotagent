import gc
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

# 内存监控阈值 (MB)，超过此值会触发警告
MEMORY_WARNING_THRESHOLD = 8000  # 8GB
# 内存危险阈值 (MB)，超过此值会暂停处理
MEMORY_DANGER_THRESHOLD = 12000  # 12GB

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def get_pdfs(skip_done: bool = True):
    """获取要处理的 PDF 列表

    Args:
        skip_done: 是否跳过已处理的文件（通过检查 output 目录中是否有对应输出文件夹）
    """
    if not PDF_ROOT.exists():
        return []

    all_pdfs = sorted([p for p in PDF_ROOT.iterdir() if p.suffix.lower() == ".pdf"])

    if not skip_done:
        return all_pdfs

    # 过滤掉已完成的（通过检查 output 目录中是否有对应的输出文件夹）
    pending = []
    for pdf in all_pdfs:
        # mineru 的输出是按 PDF 文件名（不含扩展名）创建的文件夹
        output_folder = OUTPUT_ROOT / pdf.stem
        if not output_folder.exists():
            pending.append(pdf)

    # 按修改时间排序（从最近修改的开始）
    pending.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return pending


def has_new_outputs(before_snapshot: set[Path]) -> bool:
    after_snapshot = set(OUTPUT_ROOT.iterdir()) if OUTPUT_ROOT.exists() else set()
    new_entries = [p for p in after_snapshot - before_snapshot if p.exists()]
    if new_entries:
        return True
    # fallback: some backends may overwrite existing files instead of creating new folders
    return any(OUTPUT_ROOT.rglob("*.md")) or any(OUTPUT_ROOT.rglob("*.json"))


def get_memory_usage_mb() -> float:
    """获取当前进程及子进程的内存使用（MB）"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # 如果没有 psutil，使用 resource 模块
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # macOS 上是 KB


def check_memory_and_pause() -> bool:
    """
    检查内存使用情况，如果过高则暂停处理。
    返回 True 表示需要暂停，False 表示可以继续。
    """
    try:
        mem_mb = get_memory_usage_mb()
        print(f"[MEMORY] Current: {mem_mb:.1f} MB")

        if mem_mb > MEMORY_DANGER_THRESHOLD:
            print(f"[MEMORY] WARNING: Memory usage ({mem_mb:.1f} MB) exceeds danger threshold!")
            print(f"[MEMORY] Pausing for 30 seconds to let system recover...")
            time.sleep(30)
            gc.collect()
            mem_after = get_memory_usage_mb()
            print(f"[MEMORY] After cleanup: {mem_after:.1f} MB")
            return True
        elif mem_mb > MEMORY_WARNING_THRESHOLD:
            print(f"[MEMORY] WARNING: Memory usage ({mem_mb:.1f} MB) is high")
            gc.collect()
            time.sleep(10)
        return False
    except Exception as e:
        print(f"[MEMORY] Failed to check memory: {e}")
        return False


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

    # 每次都重新尝试
    print(f"[RUN] {pdf.name}")

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
        print(f"[OK] {pdf.name} finished")
    else:
        reason = []
        if proc.returncode != 0:
            reason.append(f"exit={proc.returncode}")
        if not output_ok:
            reason.append("no_output_artifacts")
        if fatal_in_log:
            reason.append("fatal_error_in_log")
        print(f"[FAIL] {pdf.name} ({', '.join(reason)})")

    # 给 macOS 回收内存一点时间
    time.sleep(SLEEP_BETWEEN)

    # 强制垃圾回收，释放内存
    gc.collect()


def main():
    all_pdfs = get_pdfs(skip_done=False)
    pending_pdfs = get_pdfs(skip_done=True)

    total = len(all_pdfs)
    pending = len(pending_pdfs)

    if total == 0:
        print("[DONE] no pdfs found")
        return

    print(f"[INFO] Total PDFs: {total}, Pending: {pending}, Completed: {total - pending}")

    if not pending_pdfs:
        print("[DONE] all PDFs have been processed")
        return

    # 初始内存检查
    check_memory_and_pause()

    for i, pdf in enumerate(pending_pdfs):
        print(f"[PROGRESS] Processing {i+1}/{pending}: {pdf.name}")
        run_pdf(pdf)

        # 每处理 5 个文件进行一次内存检查
        if (i + 1) % 5 == 0:
            check_memory_and_pause()


if __name__ == "__main__":
    main()
