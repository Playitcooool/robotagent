import gc
import os
import signal
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_ROOT = Path("/Volumes/Samsung/Projects/robotagent/arxiv_pdfs_filtered")
OUTPUT_ROOT = BASE_DIR / "output"
LOG_ROOT = BASE_DIR / "logs"
DONE_ROOT = BASE_DIR / "done_markers"

MAX_RETRY = 3
SLEEP_BETWEEN = 5  # seconds
ERROR_PATTERNS = []
# 每个 PDF 处理超时时间（秒）
PDF_TIMEOUT = 420  # 7 分钟超时

# 内存监控阈值 (MB)，超过此值会触发警告
MEMORY_WARNING_THRESHOLD = 4000  # 4GB
# 内存危险阈值 (MB)，超过此值会暂停处理
MEMORY_DANGER_THRESHOLD = 7000  # 7GB
MEMORY_POLL_INTERVAL = 10  # seconds while a PDF is running
POST_PDF_SLEEP = 20  # seconds to let system recover after each PDF
DEFAULT_VIRTUAL_VRAM_GB = 14  # Conservative default for 24GB unified memory

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def build_output_name_index() -> tuple[set[str], set[str]]:
    """Return (names, stems) of all files/dirs under OUTPUT_ROOT."""
    names: set[str] = set()
    stems: set[str] = set()
    if not OUTPUT_ROOT.exists():
        return names, stems
    for path in OUTPUT_ROOT.rglob("*"):
        try:
            name = path.name
            if name:
                names.add(name)
            stem = path.stem
            if stem:
                stems.add(stem)
        except Exception:
            continue
    return names, stems


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

    # 过滤掉已完成的（通过检查 output 目录中是否有对应输出）
    pending = []
    output_names, output_stems = build_output_name_index()
    for pdf in all_pdfs:
        if output_exists_for_pdf(pdf, output_names, output_stems):
            continue
        pending.append(pdf)

    # 按修改时间排序（从最近修改的开始）
    pending.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return pending


def output_exists_for_pdf(
    pdf: Path,
    output_names: set[str],
    output_stems: set[str],
) -> bool:
    output_folder = OUTPUT_ROOT / pdf.stem
    if output_folder.exists():
        return True
    if pdf.name in output_names:
        return True
    return pdf.stem in output_stems


def get_memory_usage_mb() -> float:
    """获取当前进程及子进程的内存使用（MB）"""
    try:
        import psutil

        process = psutil.Process()
        # 加上所有子进程的内存
        total_mem = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_mem / (1024 * 1024)
    except ImportError:
        # 如果没有 psutil，使用 resource 模块（只统计当前进程）
        print("[MEMORY] psutil not installed; child process memory not included.")
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
            print(
                f"[MEMORY] WARNING: Memory usage ({mem_mb:.1f} MB) exceeds danger threshold!"
            )
            print(f"[MEMORY] Pausing for 30 seconds to let system recover...")
            time.sleep(30)
            gc.collect()
            cleanup_mps_cache()
            mem_after = get_memory_usage_mb()
            print(f"[MEMORY] After cleanup: {mem_after:.1f} MB")
            return True
        elif mem_mb > MEMORY_WARNING_THRESHOLD:
            print(f"[MEMORY] WARNING: Memory usage ({mem_mb:.1f} MB) is high")
            gc.collect()
            cleanup_mps_cache()
            time.sleep(10)
        return False
    except Exception as e:
        print(f"[MEMORY] Failed to check memory: {e}")
        return False


def log_has_fatal_error(log_file: Path) -> bool:
    return False


def cleanup_mps_cache() -> None:
    """If running on Apple Silicon with MPS, try to release cached memory."""
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def terminate_process_tree(proc: subprocess.Popen, timeout: int = 10) -> None:
    """Best-effort terminate a process and its children."""
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        gone, alive = psutil.wait_procs([parent, *children], timeout=timeout)
        if alive:
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            psutil.wait_procs(alive, timeout=timeout)
        return
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: kill the process group if possible
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(proc.pid, signal.SIGKILL)
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_pdf(pdf: Path):
    if not pdf.exists():
        print(f"[SKIP] missing {pdf}")
        return

    mineru_path = subprocess.os.environ.get("MINERU_PATH") or subprocess.os.environ.get(
        "MINERU_BIN"
    )
    if not mineru_path:
        from shutil import which

        mineru_path = which("mineru")
    if not mineru_path:
        print("[ERROR] mineru command not found in PATH (set MINERU_PATH or fix PATH).")
        return

    base_env = {
        **dict(**subprocess.os.environ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    user_set_vram = "MINERU_VIRTUAL_VRAM_SIZE" in base_env
    user_set_device = "MINERU_DEVICE_MODE" in base_env

    def env_for_attempt(attempt: int) -> dict:
        env = dict(base_env)
        # Apple Silicon: prefer MPS unless user explicitly overrides.
        if not user_set_device:
            if attempt >= 3:
                env["MINERU_DEVICE_MODE"] = "cpu"
            else:
                env["MINERU_DEVICE_MODE"] = "mps"
        # Virtual VRAM size: progressively lower on retries unless user overrides.
        if not user_set_vram:
            vram_gb = DEFAULT_VIRTUAL_VRAM_GB
            if attempt == 2:
                vram_gb = max(8, DEFAULT_VIRTUAL_VRAM_GB - 2)
            elif attempt >= 3:
                vram_gb = max(8, DEFAULT_VIRTUAL_VRAM_GB - 4)
            env["MINERU_VIRTUAL_VRAM_SIZE"] = str(vram_gb)
        return env

    log_file = LOG_ROOT / f"{pdf.stem}.log"

    for attempt in range(1, MAX_RETRY + 1):
        print(f"[RUN] {pdf.name} (attempt {attempt}/{MAX_RETRY})")
        # 每个 PDF 开始前都先做一次内存检查
        check_memory_and_pause()
        env = env_for_attempt(attempt)
        device_mode = env.get("MINERU_DEVICE_MODE", "mps")
        vram_size = env.get("MINERU_VIRTUAL_VRAM_SIZE", "unset")
        print(f"[ENV] device={device_mode}, virtual_vram_gb={vram_size}")
        with log_file.open("w") as lf:
            cmd = [
                mineru_path,
                "-p",
                str(pdf),
                "-o",
                str(OUTPUT_ROOT),
                "-b",
                "pipeline",
                "--no-md",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                preexec_fn=lambda: os.nice(10),
            )
            timed_out = False
            try:
                start = time.time()
                while True:
                    if proc.poll() is not None:
                        break
                    # 运行中也持续检查内存
                    check_memory_and_pause()
                    # 超时控制
                    if time.time() - start > PDF_TIMEOUT:
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=PDF_TIMEOUT)
                    time.sleep(MEMORY_POLL_INTERVAL)
            except subprocess.TimeoutExpired:
                timed_out = True
                print(f"[TIMEOUT] {pdf.name} exceeded {PDF_TIMEOUT}s, terminating...")
                terminate_process_tree(proc)

        output_names, output_stems = build_output_name_index()
        output_ok = output_exists_for_pdf(pdf, output_names, output_stems)
        if output_ok:
            print(f"[OK] {pdf.name} finished")
            break

        reason = "no_output_artifacts"
        if timed_out:
            reason = "timeout"
        elif proc.returncode != 0:
            reason = f"exit={proc.returncode}"
        print(f"[WARN] {pdf.name} ({reason}), will retry...")
        try:
            tail = log_file.read_text(errors="ignore").splitlines()[-8:]
            if tail:
                print("[LOG TAIL]")
                for line in tail:
                    print(line[:500])
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN)
    else:
        print(f"[FAIL] {pdf.name} (no_output_artifacts)")

    # 给 macOS 回收内存一点时间
    time.sleep(POST_PDF_SLEEP)

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

    print(
        f"[INFO] Total PDFs: {total}, Pending: {pending}, Completed: {total - pending}"
    )

    if not pending_pdfs:
        print("[DONE] all PDFs have been processed")
        return

    # 初始内存检查
    check_memory_and_pause()

    for i, pdf in enumerate(pending_pdfs):
        print(f"[PROGRESS] Processing {i+1}/{pending}: {pdf.name}")
        run_pdf(pdf)

        # 每处理 2 个文件进行一次内存检查
        if (i + 1) % 2 == 0:
            check_memory_and_pause()


if __name__ == "__main__":
    main()
