#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔记分类器 客户端
拖入 .txt 文件即自动分类，输出到桌面。
"""
import os, sys, time, shutil, subprocess, json
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent
PY = sys.executable
SERVICE = PROJECT / "service.py"
QUEUE = PROJECT / "queue"
RESULTS = PROJECT / "results"
LOCK = PROJECT / "service.lock"
OUTPUT_BASE = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"


def is_service_running():
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().strip())
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x0400, False, pid)
        if h:
            k32.CloseHandle(h)
            return True
    except (ValueError, OSError):
        pass
    LOCK.unlink(missing_ok=True)
    return False


def start_service():
    QUEUE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([PY, str(SERVICE)], creationflags=subprocess.CREATE_NO_WINDOW)
    print("启动分类服务...")
    for _ in range(60):
        time.sleep(1)
        if LOCK.exists():
            try:
                return int(LOCK.read_text().strip())
            except ValueError:
                pass
    print("错误: 服务启动超时")
    sys.exit(1)


def submit_and_wait(files):
    QUEUE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for old in QUEUE.glob("*"):
        old.unlink()
    for old in RESULTS.glob("*"):
        old.unlink()

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / f"{files[0].stem}_分类_{batch_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir.name}/")

    submitted = []
    for fp in files:
        qname = f"{batch_ts}__{fp.stem}.txt"
        submitted.append((qname, str(out_dir)))
    meta = {"batch_ts": batch_ts, "files": {q: s for q, s in submitted}}
    (QUEUE / f"{batch_ts}.meta").write_text(json.dumps(meta), encoding="utf-8")
    time.sleep(0.5)

    for fp, (qname, _) in zip(files, submitted):
        shutil.copy2(fp, QUEUE / qname)
        print(f"  已提交: {fp.name}")

    progress_file = QUEUE / f"{batch_ts}.progress"
    if progress_file.exists():
        progress_file.unlink()

    print()
    print("等待分类完成...")
    print("-" * 50)
    done_count = 0
    timeout = 600
    start = time.time()
    last_progress_size = 0

    while done_count < len(submitted) and (time.time() - start) < timeout:
        time.sleep(0.5)

        if progress_file and progress_file.exists():
            try:
                content = progress_file.read_text(encoding="utf-8")
                if len(content) > last_progress_size:
                    print(content[last_progress_size:], end="")
                    last_progress_size = len(content)
                    sys.stdout.flush()
            except Exception:
                pass

        for marker in RESULTS.glob(f"{batch_ts}_*.done"):
            try:
                marker.unlink()
                done_count += 1
            except Exception:
                pass
        for marker in RESULTS.glob(f"{batch_ts}_*.error"):
            try:
                content = marker.read_text(encoding="utf-8").strip()
                marker.unlink()
                done_count += 1
                print(f"\n  错误: {content}")
            except Exception:
                pass

    if progress_file and progress_file.exists():
        progress_file.unlink()
    (QUEUE / f"{batch_ts}.meta").unlink(missing_ok=True)

    print("-" * 50)
    if done_count < len(submitted):
        print("超时: 部分文件未完成处理")


def main():
    files = [Path(f) for f in sys.argv[1:] if Path(f).suffix.lower() == ".txt"]
    if not files:
        print("错误: 没有有效的 TXT 文件")
        print("用法: 拖入 .txt 文件 或  python client.py file1.txt file2.txt")
        sys.exit(1)

    files = [f for f in files if f.exists()]
    if not files:
        print("没有可用的文件")
        sys.exit(1)

    if not is_service_running():
        pid = start_service()
        print(f"服务已启动 (PID: {pid})")

    print()
    print("=" * 50)
    print(f"提交 {len(files)} 个文件到分类服务")
    print("=" * 50)

    submit_and_wait(files)

    print()
    print("=" * 50)
    print("分类完成!")
    print("服务继续运行，空闲 5 分钟后自动退出。")
    print("=" * 50)


if __name__ == "__main__":
    main()
