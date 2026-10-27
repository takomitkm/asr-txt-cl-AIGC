#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bge-m3-zeroshot-v2.0 — 常驻分类服务（单分类模式）
22 个细分类别 → 9 个输出文件，分差优先 + 优先级兜底
"""
from __future__ import annotations
import argparse, json, logging, os, re, signal, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 路径 ──
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
MODEL_CACHE_DIR = PROJECT_DIR / "cache"
OUTPUT_BASE = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
QUEUE_DIR = PROJECT_DIR / "queue"
RESULTS_DIR = PROJECT_DIR / "results"
LOCK_FILE = PROJECT_DIR / "service.lock"
LOG_FILE = PROJECT_DIR / "service.log"
MODEL_ID = "MoritzLaurer/bge-m3-zeroshot-v2.0"
DEVICE = -1
IDLE_TIMEOUT = 300
POLL_INTERVAL = 1.0

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("service")

# ════════════════════════════════════════════════════════════
#  配置加载
# ════════════════════════════════════════════════════════════

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_candidate_labels(config: dict) -> List[str]:
    return [s["en"] for s in config["subcategories"]]

def build_label_key_map(config: dict) -> Dict[str, str]:
    return {s["en"]: s["id"] for s in config["subcategories"]}

def build_zh_map(config: dict) -> Dict[str, str]:
    return {s["id"]: s["zh"] for s in config["subcategories"]}

def build_output_map(config: dict) -> Dict[str, str]:
    return {s["id"]: s["output"] for s in config["subcategories"]}

def get_output_files(config: dict) -> List[str]:
    return config["output_files"]

# ════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════

def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "gb18030", "gbk", "big5"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"

def read_input(path: Path) -> str:
    enc = detect_encoding(path)
    log.info(f"编码: {enc}")
    with open(path, "r", encoding=enc) as f:
        return f.read()

def split_paragraphs(text: str) -> List[Tuple[int, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.split(r"\n{2,}", text)
    paragraphs: List[Tuple[int, str]] = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            paragraphs.append((i + 1, chunk))
    return paragraphs

def chunk_long_text(text: str, max_chars: int) -> List[str]:
    sentences = re.split(r"(?<=[。！？.!?\n])", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current += sent
    if current.strip():
        chunks.append(current)
    return chunks if chunks else [text]

def load_classifier():
    from transformers import pipeline
    local_model = None
    model_dir_name = f"models--{MODEL_ID.replace('/', '--')}"
    # Check both cache\hub\ and cache\ locations
    for base in (MODEL_CACHE_DIR / "hub" / model_dir_name, MODEL_CACHE_DIR / model_dir_name):
        snapshots_dir = base / "snapshots"
        if snapshots_dir.exists():
            for snap in sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if (snap / "model.safetensors").exists():
                    local_model = snap
                    break
        if local_model:
            break
    if local_model:
        log.info(f"加载本地模型: {local_model}")
    else:
        log.info(f"加载模型: {MODEL_ID}")
    classifier = pipeline(
        "zero-shot-classification",
        model=str(local_model) if local_model else MODEL_ID,
        cache_dir=str(MODEL_CACHE_DIR),
        device=DEVICE,
    )
    log.info("模型加载成功")
    return classifier

# ════════════════════════════════════════════════════════════
#  分类核心
# ════════════════════════════════════════════════════════════

def classify_paragraph(
    classifier, paragraph: str, config: dict,
    label_key_map: Optional[Dict[str, str]] = None,
    max_chars: int = 512,
) -> Dict:
    if label_key_map is None:
        label_key_map = build_label_key_map(config)
    subcats = config["subcategories"]
    chunks = chunk_long_text(paragraph, max_chars)
    is_chunked = len(chunks) > 1
    all_scores: Dict[str, float] = {s["id"]: 0.0 for s in subcats}
    n = len(chunks)
    for chunk in chunks:
        result = classifier(
            chunk,
            config["candidate_labels"],
            hypothesis_template=config["hypothesis_template"],
            multi_label=True,
            device=DEVICE,
        )
        for label, score in zip(result["labels"], result["scores"]):
            key = label_key_map.get(label)
            if key:
                all_scores[key] += score
    for k in all_scores:
        all_scores[k] /= n
    return {"scores": all_scores, "chunked": is_chunked, "chunks": n}

def decide_outputs(
    scores: Dict[str, float],
    config: dict,
    output_map: Dict[str, str],
) -> Tuple[str, List[str], str]:
    """
    单分类决策：
    - 只考虑达到阈值的细分类别
    - 分差 > score_margin → 选最高分
    - 分差 <= score_margin → 按优先级选择
    - 返回: (final_output_file, matched_subcats, note)
    """
    threshold = config["threshold"]
    score_margin = config.get("score_margin", 0.10)
    priority = config.get("priority", [])
    subcats = config["subcategories"]

    above = {sid: v for sid, v in scores.items() if v >= threshold}
    if not above:
        return "", [], "no_category_above_threshold"

    # 按分数排序
    sorted_items = sorted(above.items(), key=lambda x: x[1], reverse=True)
    top1_id = sorted_items[0][0]
    top1_score = sorted_items[0][1]
    top2_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    diff = top1_score - top2_score

    # 分差 > score_margin → 直接选最高分
    if diff > score_margin:
        final_id = top1_id
    else:
        # 分差 <= score_margin → 按优先级选择（只考虑达到阈值的类别）
        above_ids = set(above.keys())
        final_id = None
        for cat_zh in priority:
            # 找到该主分类对应的细分类别
            for s in subcats:
                if s["zh"] == cat_zh and s["id"] in above_ids:
                    final_id = s["id"]
                    break
            if final_id:
                break
        if not final_id:
            final_id = top1_id  # 兜底

    final_output = output_map.get(final_id, "未分类.txt")
    note = "边界样本" if diff <= score_margin else ""
    return final_output, list(above.keys()), note

def format_paragraph(
    idx: int,
    scores: Dict[str, float],
    original_text: str,
    final_output: str,
    matched_subcats: List[str],
    note: str,
    chunked: bool,
    chunk_count: int,
    config: dict,
    zh_map: Dict[str, str],
) -> str:
    subcats = config["subcategories"]
    score_parts = []
    for s in subcats:
        score_parts.append(f"{s['zh']}={scores.get(s['id'], 0.0):.4f}")
    score_line = " | ".join(score_parts)

    matched_zh = "、".join(zh_map.get(sid, sid) for sid in matched_subcats)
    lines: List[str] = []
    lines.append(f"[文段 {idx}]")
    lines.append(f"[评分] {score_line}")
    lines.append(f"[命中] {matched_zh}")
    lines.append(f"[最终] {final_output}")
    if note:
        lines.append(f"[备注] {note}")
    if chunked:
        lines.append(f"[备注] 长文段分块推理，共 {chunk_count} 块")
    lines.append(original_text)
    return "\n".join(lines)

def format_uncategorized_paragraph(
    idx: int,
    scores: Dict[str, float],
    original_text: str,
    config: dict,
) -> str:
    subcats = config["subcategories"]
    score_parts = []
    for s in subcats:
        score_parts.append(f"{s['zh']}={scores.get(s['id'], 0.0):.4f}")
    score_line = " | ".join(score_parts)

    best = max(scores.items(), key=lambda x: x[1])
    lines: List[str] = []
    lines.append(f"[文段 {idx}]")
    lines.append(f"[原因] 最高分 {best[1]:.4f} 未达到阈值 {config['threshold']}")
    lines.append(f"[评分] {score_line}")
    lines.append(original_text)
    return "\n".join(lines)

# ════════════════════════════════════════════════════════════
#  文件处理
# ════════════════════════════════════════════════════════════

def process_batch(classifier, config, label_key_map, input_files: List[Path],
                  output_dir: Path, progress_file: Optional[Path] = None) -> Path:
    """处理多个文件，合并输出到同一个文件夹。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zh_map = build_zh_map(config)
    output_map = build_output_map(config)
    output_files = get_output_files(config)
    subcats = config["subcategories"]

    def write_progress(line: str):
        if progress_file:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    buffers: Dict[str, List[str]] = {f: [] for f in output_files}
    total = 0
    failed = 0
    separator = "\n\n"
    global_idx = 0

    for input_path in input_files:
        file_stem = input_path.stem.split("__", 1)[-1] if "__" in input_path.stem else input_path.stem
        log.info(f"处理文件: {file_stem} -> {output_dir}")
        text = read_input(input_path)
        paragraphs = split_paragraphs(text)
        log.info(f"  共 {len(paragraphs)} 个文段")

        if not paragraphs:
            log.warning(f"  {file_stem} 为空，跳过")
            continue

        write_progress(f"[开始] {file_stem} ({len(paragraphs)} 段)")

        for idx, para_text in paragraphs:
            total += 1
            global_idx += 1
            log.info(f"  [{idx}/{len(paragraphs)}] ({file_stem}) 分类中...")
            try:
                result = classify_paragraph(classifier, para_text, config, label_key_map)
                scores = result["scores"]
                final_output, matched_subcats, note = decide_outputs(scores, config, output_map)
                chunked = result["chunked"]
                chunk_count = result["chunks"]

                score_parts = []
                for s in subcats:
                    score_parts.append(f"{s['zh']}={scores.get(s['id'], 0.0):.2f}")
                score_line = " ".join(score_parts)
                if final_output:
                    write_progress(f"[{global_idx}] {file_stem} 段{idx}: {score_line} -> {final_output}")
                else:
                    write_progress(f"[{global_idx}] {file_stem} 段{idx}: {score_line} -> 未分类")

                source_tag = f"[来源] {file_stem}"

                if not final_output:
                    formatted = format_uncategorized_paragraph(
                        global_idx, scores, para_text, config
                    )
                    formatted = source_tag + "\n" + formatted
                    buffers["未分类.txt"].append(formatted)
                else:
                    formatted = format_paragraph(
                        global_idx, scores, para_text,
                        final_output, matched_subcats, note,
                        chunked, chunk_count, config, zh_map,
                    )
                    formatted = source_tag + "\n" + formatted
                    buffers[final_output].append(formatted)

            except Exception as e:
                failed += 1
                log.error(f"  {file_stem} 段 {idx} 分类失败: {e}")
                write_progress(f"[{global_idx}] {file_stem} 段{idx} ERROR: {str(e)[:60]}")
                err_block = f"[来源] {file_stem}\n[文段 {idx}]\n[错误] {str(e)}\n[原文]\n{para_text}"
                buffers["未分类.txt"].append(err_block)

        write_progress("")

    for fname in output_files:
        filepath = output_dir / fname
        content = separator.join(buffers[fname]) if buffers[fname] else ""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        log.info(f"  {fname}: {len(buffers[fname])} 段")

    counts_parts = []
    for fname in output_files:
        n = len(buffers[fname])
        if n > 0:
            short = fname.replace(".txt", "")
            counts_parts.append(f"{short}={n}")
    counts_str = "  ".join(counts_parts)
    write_progress(f"[完成] {len(input_files)}个文件 总{total}段 成功{total-failed} 失败{failed}  {counts_str}")
    write_progress("")

    log.info(f"输出目录: {output_dir}")
    log.info(f"  总文段: {total}  成功: {total - failed}  失败: {failed}")
    return output_dir

# ════════════════════════════════════════════════════════════
#  服务主循环
# ════════════════════════════════════════════════════════════

def is_running() -> bool:
    if not LOCK_FILE.exists():
        return False
    try:
        pid = int(LOCK_FILE.read_text().strip())
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x0400, False, pid)
        if h:
            k32.CloseHandle(h)
            return True
    except (ValueError, OSError):
        pass
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass
    return False

def get_pid() -> Optional[int]:
    if not LOCK_FILE.exists():
        return None
    try:
        return int(LOCK_FILE.read_text().strip())
    except (ValueError, OSError):
        return None

def main():
    parser = argparse.ArgumentParser(description="bge-m3 分类服务")
    parser.add_argument("--stop", action="store_true", help="停止服务")
    parser.add_argument("--status", action="store_true", help="查看状态")
    parser.add_argument("--idle-timeout", type=int, default=IDLE_TIMEOUT, help="空闲超时秒数")
    args = parser.parse_args()

    if args.status:
        pid = get_pid()
        if pid and is_running():
            print(f"服务运行中 (PID: {pid})")
        else:
            print("服务未运行")
        return

    if args.stop:
        pid = get_pid()
        if pid and is_running():
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"已发送停止信号 (PID: {pid})")
            except OSError as e:
                print(f"停止失败: {e}")
        else:
            print("服务未运行")
        return

    # ── 启动 ──
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 清空旧队列文件（防止残留文件被误处理）
    for old in QUEUE_DIR.glob("*"):
        old.unlink()
        log.info(f"清理旧队列文件: {old.name}")
    for old in RESULTS_DIR.glob("*"):
        old.unlink()
        log.info(f"清理旧结果标记: {old.name}")

    LOCK_FILE.write_text(str(os.getpid()))

    log.info("=" * 60)
    log.info("bge-m3 分类服务启动 (18 细分类别)")
    log.info(f"  PID: {os.getpid()}")
    log.info(f"  队列: {QUEUE_DIR}")
    log.info(f"  输出: {OUTPUT_BASE}")
    log.info(f"  空闲超时: {args.idle_timeout} 秒")
    log.info("=" * 60)

    log.info("加载模型中...")
    classifier = load_classifier()
    config = load_config()
    config["candidate_labels"] = build_candidate_labels(config)
    label_key_map = build_label_key_map(config)
    log.info(f"模型就绪，{len(config['subcategories'])} 个细分类别，开始监听队列...")

    last_activity = time.time()
    running = True

    while running:
        now = time.time()
        idle_time = now - last_activity
        if idle_time > args.idle_timeout:
            log.info(f"空闲超过 {args.idle_timeout} 秒，服务退出")
            break

        queue_files = sorted(QUEUE_DIR.glob("*.txt"))
        meta_files = list(QUEUE_DIR.glob("*.meta"))
        batch_meta = {}
        for mf in meta_files:
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                for qname, outdir in data.get("files", {}).items():
                    batch_meta[qname] = outdir
            except Exception:
                pass

        if queue_files:
            groups: Dict[str, List[Path]] = {}
            for qfile in queue_files:
                outdir = batch_meta.get(qfile.name)
                if not outdir:
                    clean_stem = qfile.stem.split("__", 1)[-1] if "__" in qfile.stem else qfile.stem
                    outdir = str(OUTPUT_BASE / f"{clean_stem}_分类_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                groups.setdefault(outdir, []).append(qfile)

            for outdir, qfiles in groups.items():
                batch_ts = qfiles[0].stem.split("__", 1)[0] if "__" in qfiles[0].stem else ""
                progress_file = QUEUE_DIR / f"{batch_ts}.progress" if batch_ts else None

                try:
                    output_dir = process_batch(classifier, config, label_key_map, qfiles, Path(outdir), progress_file)

                    for qfile in qfiles:
                        done_marker = RESULTS_DIR / (qfile.stem + ".done")
                        if output_dir:
                            done_marker.write_text(str(output_dir), encoding="utf-8")
                        else:
                            done_marker.write_text("SKIPPED", encoding="utf-8")
                        log.info(f"完成: {qfile.name} -> {output_dir}")
                        qfile.unlink()

                    if batch_ts:
                        mf_path = QUEUE_DIR / f"{batch_ts}.meta"
                        if mf_path.exists():
                            try:
                                mf_path.unlink()
                            except Exception:
                                pass

                    last_activity = time.time()

                except Exception as e:
                    log.error(f"处理批次失败: {e}")
                    for qfile in qfiles:
                        done_marker = RESULTS_DIR / (qfile.stem + ".error")
                        done_marker.write_text(str(e), encoding="utf-8")
                        try:
                            qfile.unlink()
                        except OSError:
                            pass
                    last_activity = time.time()

        try:
            for marker in RESULTS_DIR.glob("*.done"):
                if time.time() - marker.stat().st_mtime > 600:
                    marker.unlink()
        except Exception:
            pass

        time.sleep(POLL_INTERVAL)

    log.info("服务退出，清理锁文件...")
    LOCK_FILE.unlink(missing_ok=True)
    log.info("服务已退出")

if __name__ == "__main__":
    main()
