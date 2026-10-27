"""
Universal Blackboard Lecture Digitization Agent (v2 Production Pipeline)
========================================================================
Supports dual-lecture processing for all recorded university math courses:
  - 复分析 (Complex Analysis)
  - 数学分析（荣誉）III (Mathematical Analysis III Honours)
  - 代数拓扑 (Algebraic Topology)

Core Pipeline:
  1. Dense Temporal Sampling (20s @ 4fps = 80 frames)
  2. Fourier Phase Correlation Sub-Pixel Motion Registration (频域重定位配准)
  3. Temporal Persistence & Monotonic Edge Filtering (时序持续性与单调边缘滤波消遮挡)
  4. Robust Chalk Union (稳态粉笔字迹并集)
  5. Stage 1 Parallel VLM Recognition with Strict LaTeX & Diagram Placeholders
  6. Stage 2 Global Note Synthesis with Semantic Deduplication
"""

import os
import sys
import time
import base64
import requests
import json
import argparse
import threading
import subprocess
import warnings
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
NOTES_DIR = os.path.join(DATA_DIR, "notes")

# ============================================================================
# Configuration & Environment
# ============================================================================

def _load_env_file():
    env_path = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env_file()

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

COURSE_SESSIONS = {
    "ca_1019": {
        "course": "复分析",
        "date": "2026-10-19",
        "title": "复分析 (2026-10-19)",
        "output_dir": os.path.join(OUTPUTS_DIR, "ca_1019"),
        "lectures": [
            {
                "id": "lec1",
                "title": "复分析 (第9节) 1555-1650",
                "video_path": os.path.join(ROOT_DIR, "data/videos/复分析/复分析_2026-10-19_复分析 (第9节)_1555-1650.mp4")
            },
            {
                "id": "lec2",
                "title": "复分析 (第10节) 1650-1745",
                "video_path": os.path.join(ROOT_DIR, "data/videos/复分析/复分析_2026-10-19_复分析 (第10节)_1650-1745.mp4")
            }
        ]
    },
    "ca_1021": {
        "course": "复分析",
        "date": "2026-10-21",
        "title": "复分析 (2026-10-21)",
        "output_dir": os.path.join(OUTPUTS_DIR, "ca_1021"),
        "lectures": [
            {
                "id": "lec1",
                "title": "复分析 (第7节) 1355-1450",
                "video_path": os.path.join(ROOT_DIR, "data/videos/复分析/复分析_2026-10-21_复分析 (第7节)_1355-1450.mp4")
            },
            {
                "id": "lec2",
                "title": "复分析 (第8节) 1450-1545",
                "video_path": os.path.join(ROOT_DIR, "data/videos/复分析/复分析_2026-10-21_复分析 (第8节)_1450-1545.mp4")
            }
        ]
    },
    "ma_1019": {
        "course": "数学分析（荣誉）III",
        "date": "2026-10-19",
        "title": "数学分析（荣誉）III (2026-10-19)",
        "output_dir": os.path.join(OUTPUTS_DIR, "ma_1019"),
        "lectures": [
            {
                "id": "lec1",
                "title": "数学分析（荣誉）III (第3节) 0955-1050",
                "video_path": os.path.join(ROOT_DIR, "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-19_数学分析（荣誉）III (第3节)_0955-1050.mp4")
            },
            {
                "id": "lec2",
                "title": "数学分析（荣誉）III (第4节) 1050-1150",
                "video_path": os.path.join(ROOT_DIR, "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-19_数学分析（荣誉）III (第4节)_1050-1150.mp4")
            }
        ]
    },
    "ma_1021": {
        "course": "数学分析（荣誉）III",
        "date": "2026-10-21",
        "title": "数学分析（荣誉）III (2026-10-21)",
        "output_dir": os.path.join(OUTPUTS_DIR, "ma_1021"),
        "lectures": [
            {
                "id": "lec1",
                "title": "数学分析（荣誉）III (第1节) 0755-0850",
                "video_path": os.path.join(ROOT_DIR, "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-21_数学分析（荣誉）III (第1节)_0755-0850.mp4")
            },
            {
                "id": "lec2",
                "title": "数学分析（荣誉）III (第2节) 0850-0945",
                "video_path": os.path.join(ROOT_DIR, "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-21_数学分析（荣誉）III (第2节)_0850-0945.mp4")
            }
        ]
    },
    "topo_1020": {
        "course": "代数拓扑",
        "date": "2026-10-20",
        "title": "代数拓扑 (2026-10-20)",
        "output_dir": os.path.join(OUTPUTS_DIR, "topo_1020"),
        "lectures": [
            {
                "id": "lec1",
                "title": "代数拓扑 (第3节) 0955-1050",
                "video_path": os.path.join(ROOT_DIR, "data/videos/代数拓扑/代数拓扑_2026-10-20_代数拓扑 (第3节)_0955-1050.mp4")
            },
            {
                "id": "lec2",
                "title": "代数拓扑 (第4节) 1050-1150",
                "video_path": os.path.join(ROOT_DIR, "data/videos/代数拓扑/代数拓扑_2026-10-20_代数拓扑 (第4节)_1050-1150.mp4")
            }
        ]
    },
    "topo_1022": {
        "course": "代数拓扑",
        "date": "2026-10-22",
        "title": "代数拓扑 (2026-10-22)",
        "output_dir": os.path.join(OUTPUTS_DIR, "topo_agent"),
        "lectures": [
            {
                "id": "lec1",
                "title": "代数拓扑 (第1节) 0755-0850",
                "video_path": os.path.join(ROOT_DIR, "data/videos/代数拓扑/代数拓扑_2026-10-22_代数拓扑 (第1节)_0755-0850.mp4")
            },
            {
                "id": "lec2",
                "title": "代数拓扑 (第2节) 0850-0945",
                "video_path": os.path.join(ROOT_DIR, "data/videos/代数拓扑/代数拓扑_2026-10-22_代数拓扑 (第2节)_0850-0945.mp4")
            }
        ]
    }
}

# ============================================================================
# Cost Monitor
# ============================================================================

class CostMonitor:
    def __init__(self, max_cost_rmb: float = 3.0):
        self.max_cost = max_cost_rmb
        self.lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cost_rmb = 0.0
        self.aborted = False

    def add_usage(self, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0) -> Tuple[float, bool]:
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_reasoning_tokens += (reasoning_tokens or 0)
            
            step_cost = (prompt_tokens * 2.0 + completion_tokens * 8.0) / 1_000_000.0
            self.total_cost_rmb += step_cost
            if self.total_cost_rmb >= self.max_cost:
                self.aborted = True
            return self.total_cost_rmb, self.aborted

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_cost_rmb": round(self.total_cost_rmb, 4),
                "aborted": self.aborted
            }

# ============================================================================
# Image Extraction & Fourier Registered Persistence Fusion Engine
# ============================================================================

def extract_registered_persistence_fusion(
    video_path: str,
    t_center: float,
    duration_sec: int = 20,
    fps: int = 4
) -> np.ndarray:
    """
    Dense Temporal Sampling + Fourier Phase Correlation Registration
    + Temporal Persistence Filtering + Monotonic Chalk Union.
    Eliminates camera PTZ drift ghosting, teacher occlusion residues, and transient eraser lines.
    """
    t_start = max(0, t_center - duration_sec // 2)
    cmd = [
        'ffmpeg', '-ss', str(t_start), '-t', str(duration_sec), '-i', video_path,
        '-r', str(fps), '-pix_fmt', 'gray', '-f', 'rawvideo', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_data, _ = proc.communicate()
    
    w, h = 1920, 1080
    num_frames = len(raw_data) // (w * h)
    if num_frames == 0:
        raise RuntimeError(f"FFmpeg failed to extract frames around t={t_center}s from {video_path}")
        
    frames = np.frombuffer(raw_data, dtype=np.uint8).reshape((num_frames, h, w))
    # Full active blackboard span across classrooms and PTZ variations: y in [0, 740], x in [200, 1800]
    board_frames = frames[:, 0:740, 200:1800]
    N, bh, bw = board_frames.shape
    
    # 1. Fourier Phase Correlation Alignment
    ref_idx = N // 2
    ref_frame = board_frames[ref_idx].astype(np.float32)
    win = np.outer(np.hanning(bh), np.hanning(bw)).astype(np.float32)
    F_ref = np.fft.fft2(ref_frame * win)
    
    aligned = np.zeros_like(board_frames)
    for i in range(N):
        F_cur = np.fft.fft2(board_frames[i].astype(np.float32) * win)
        cp = (F_ref * np.conj(F_cur)) / (np.abs(F_ref * np.conj(F_cur)) + 1e-12)
        r = np.fft.ifft2(cp).real
        r = np.fft.fftshift(r)
        py, px = np.unravel_index(np.argmax(r), r.shape)
        dy = py - bh // 2
        dx = px - bw // 2
        aligned[i] = np.roll(board_frames[i], shift=(dy, dx), axis=(0, 1))
        
    # 2. Pure Clean Background (30th percentile anchor)
    bg = np.percentile(aligned, 30, axis=0).astype(np.float32)
    aligned_f = aligned.astype(np.float32)
    
    # 3. Temporal Activity (difference between adjacent frames)
    temp_diff = np.abs(aligned_f[1:] - aligned_f[:-1])
    temp_diff = np.pad(temp_diff, ((1, 0), (0, 0), (0, 0)), mode='edge')
    d_pad = np.pad(temp_diff, ((2, 2), (0, 0), (0, 0)), mode='edge')
    mov_activity = (d_pad[0:-4] + d_pad[1:-3] + d_pad[2:-2] + d_pad[3:-1] + d_pad[4:]) / 5.0
    is_static = mov_activity < 12.0
    
    # 4. Spatial Gradient for chalk detection
    grad_y = np.abs(aligned_f[:, 1:, :] - aligned_f[:, :-1, :])
    grad_y = np.pad(grad_y, ((0, 0), (1, 0), (0, 0)))
    grad_x = np.abs(aligned_f[:, :, 1:] - aligned_f[:, :, :-1])
    grad_x = np.pad(grad_x, ((0, 0), (0, 0), (1, 0)))
    spatial_grad = grad_y + grad_x
    
    # 5. Chalk Candidate Mask
    is_chalk_candidate = (aligned_f - bg > 22.0) & (spatial_grad > 14.0) & is_static
    
    # 6. Temporal Persistence & Monotonic Edge Filter
    total_count = np.sum(is_chalk_candidate, axis=0)
    cond_persistent = total_count >= 5
    cond_fresh_write = is_chalk_candidate[-1] & is_chalk_candidate[-2] & is_chalk_candidate[-3]
    cond_fresh_erase = is_chalk_candidate[0] & is_chalk_candidate[1] & is_chalk_candidate[2]
    
    valid_chalk = cond_persistent | cond_fresh_write | cond_fresh_erase
    
    # 7. Robust Intensity Composition (nanmedian over candidate frames)
    chalk_vals = np.where(is_chalk_candidate, aligned_f, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chalk_robust = np.nanmedian(chalk_vals, axis=0)
    chalk_robust = np.where(np.isnan(chalk_robust), bg, chalk_robust)
    
    fused = np.where(valid_chalk, np.maximum(bg, chalk_robust), bg).astype(np.uint8)
    return fused


def _extract_task_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    save_path = task["save_path"]
    frame_id = task["frame_id"]
    t0 = time.time()
    
    if not os.path.exists(save_path):
        fused_board = extract_registered_persistence_fusion(task["video_path"], task["timestamp_sec"])
        Image.fromarray(fused_board).save(save_path, quality=92)
        elapsed = time.time() - t0
        print(f"  [{frame_id}] Extracted & registered in {elapsed:.1f}s")
    else:
        print(f"  [{frame_id}] Using cached frame")
        
    return {
        "lec_id": task["lec_id"],
        "lec_title": task["lec_title"],
        "timestamp_sec": task["timestamp_sec"],
        "timestamp_str": task["timestamp_str"],
        "frame_id": frame_id,
        "image_path": save_path
    }


def prepare_all_frames(
    lectures: List[Dict[str, Any]],
    output_dir: str,
    interval_sec: float = 120.0,
    sample_limit: int = None,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    frames_meta = []
    frames_dir = f"{output_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    print("\n[Phase 1] Extracting Registered Persistence Fusion Frames...")
    tasks = []
    
    for lec in lectures:
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', lec["video_path"]
        ]
        try:
            duration_str = subprocess.check_output(probe_cmd).decode('utf-8').strip()
            duration = float(duration_str)
        except Exception:
            duration = 3300.0
            
        curr = interval_sec
        lec_count = 0
        while curr < duration - 30.0:
            if sample_limit and len(tasks) >= sample_limit:
                break
            frame_id = f"{lec['id']}_{int(curr):04d}s"
            save_path = f"{frames_dir}/{frame_id}.jpg"
            tasks.append({
                "lec_id": lec["id"],
                "lec_title": lec["title"],
                "video_path": lec["video_path"],
                "timestamp_sec": curr,
                "timestamp_str": f"{int(curr//60):02d}:{int(curr%60):02d}",
                "frame_id": frame_id,
                "save_path": save_path
            })
            curr += interval_sec
            lec_count += 1
            
        print(f"  - {lec['title']}: {lec_count} frames scheduled (Duration: {duration/60:.1f} min)")
        if sample_limit and len(tasks) >= sample_limit:
            break
            
    print(f"  Extracting {len(tasks)} frames using {max_workers} worker threads...")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_task_worker, t) for t in tasks]
        for future in as_completed(futures):
            frames_meta.append(future.result())
            
    frames_meta.sort(key=lambda x: (x["lec_id"], x["timestamp_sec"]))
    print(f"  [Phase 1 Completed] {len(frames_meta)} frames ready in {time.time() - t_start:.1f}s\n")
    return frames_meta

# ============================================================================
# Stage 1: Parallel Recognition with Thinking Disabled & Strict LaTeX Prompts
# ============================================================================

def process_single_frame(
    frame_info: Dict[str, Any],
    cost_monitor: CostMonitor,
    stage1_dir: str,
    course_name: str
) -> Dict[str, Any]:
    """Recognize single blackboard frame via deepseek-flash with thinking disabled."""
    if cost_monitor.aborted:
        return {"frame_id": frame_info["frame_id"], "status": "skipped_budget_exceeded"}
        
    json_path = f"{stage1_dir}/{frame_info['frame_id']}.json"
    md_path = f"{stage1_dir}/{frame_info['frame_id']}.md"
    
    # Check cache
    if os.path.exists(json_path) and os.path.exists(md_path):
        with open(json_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "frame_id": frame_info["frame_id"],
            "timestamp_str": frame_info["timestamp_str"],
            "lec_id": frame_info["lec_id"],
            "lec_title": frame_info["lec_title"],
            "status": "cached",
            "content": content,
            "usage": cached_data.get("usage", {})
        }
        
    with open(frame_info["image_path"], "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    prompt_text = (
        f"这是大学数学系《{course_name}》课堂在 {frame_info['timestamp_str']} 时的黑板板书图像。\n"
        "请准确识别并完整转录图中的全部手写数学板书内容（包括引理、定理、定义、命题、推论、证明推导细节、数学公式及中文说明）。\n\n"
        "【LaTeX 公式规范（绝对严格）】：\n"
        "1. 只能使用 $（行内公式）与 $$（行间公式）。严禁使用 \\[、\\]、\\(、\\)。\n"
        "2. 行内公式外侧留空格，内侧紧贴不留空格；行间公式 $$ 独占一行前后各空一行。\n"
        "3. 如果黑板上尚未书写数学内容或黑板为空，请仅回复【黑板暂无板书】，不要转录监控摄像机的时间戳水印。\n\n"
        "【手绘示意图占位符规范（绝对严格）】：\n"
        "对黑板上的手绘示意图（如几何图形、闭合曲线围道、区域映射示意图、复平面图、图表等），必须使用 Markdown 图片语法占位符：\n"
        "![{对图形的简短中文描述}]()\n"
        "例如黑板画了同心圆，必须写入：![同心圆区域示意图]()。\n"
        "严禁写成普通文字描述或括号备注（如严禁写成“（手绘示意图：...）”）。\n"
        "不需要精确内容或精确画，而只需要有该 ![...] 格式的占位符即可。\n"
        "注意：必须严格复刻黑板上的示意图，不可随意增减；纯公式与文字推导绝不添加占位符。\n\n"
        "请直接输出板书转录结果，不得有任何省略或总结概括，不要输出问候寒暄。"
    )
    
    payload = {
        "model": "deepseek-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "thinking": {"type": "disabled"},
        "max_tokens": 2048,
        "temperature": 0.1
    }
    
    t0 = time.time()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        t_elapsed = time.time() - t0
        
        if resp.status_code != 200:
            print(f"[{frame_info['frame_id']}] API Error {resp.status_code}: {resp.text[:150]}")
            return {"frame_id": frame_info["frame_id"], "status": f"error_{resp.status_code}"}
            
        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        
        total_cost, aborted = cost_monitor.add_usage(prompt_tokens, completion_tokens, reasoning_tokens)
        
        content = data["choices"][0]["message"].get("content", "").strip()
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"[{frame_info['frame_id']}] {t_elapsed:.2f}s | "
              f"In:{prompt_tokens} Out:{completion_tokens} | "
              f"Session Cost: ¥{total_cost:.4f}{' [LIMIT REACHED!]' if aborted else ''}")
              
        return {
            "frame_id": frame_info["frame_id"],
            "timestamp_str": frame_info["timestamp_str"],
            "lec_id": frame_info["lec_id"],
            "lec_title": frame_info["lec_title"],
            "status": "success",
            "content": content,
            "usage": usage,
            "elapsed_sec": t_elapsed
        }
    except Exception as e:
        print(f"[{frame_info['frame_id']}] Exception: {str(e)}")
        return {"frame_id": frame_info["frame_id"], "status": "exception", "error": str(e)}


def run_stage1_parallel(
    frames_meta: List[Dict[str, Any]],
    output_dir: str,
    course_name: str,
    max_workers: int = 10,
    cost_monitor: CostMonitor = None
) -> List[Dict[str, Any]]:
    stage1_dir = f"{output_dir}/stage1_transcriptions"
    os.makedirs(stage1_dir, exist_ok=True)
    
    print(f"[Phase 2] Running Stage 1 Recognition ({len(frames_meta)} frames, {max_workers} threads, thinking: disabled)...")
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {
            executor.submit(process_single_frame, fm, cost_monitor, stage1_dir, course_name): fm
            for fm in frames_meta
        }
        for future in as_completed(future_to_frame):
            res = future.result()
            results.append(res)
            if cost_monitor.aborted:
                print(">>> WARNING: Budget limit reached! Terminating pending jobs...")
                break
                
    results.sort(key=lambda x: x.get("frame_id", ""))
    return results

# ============================================================================
# LaTeX Formatting Normalization & Sanitization
# ============================================================================

def enforce_latex_formatting(text: str) -> str:
    text = re.sub(re.escape(r'\[') + r'\s*', '\n\n$$\n', text)
    text = re.sub(r'\s*' + re.escape(r'\]'), '\n$$\n\n', text)
    text = re.sub(re.escape(r'\(') + r'\s*', ' $', text)
    text = re.sub(r'\s*' + re.escape(r'\)'), '$ ', text)
    
    lines = text.split('\n')
    out_lines = []
    in_display = False
    
    for line in lines:
        stripped = line.strip()
        if stripped == '$$':
            if not in_display:
                if out_lines and out_lines[-1].strip() != '':
                    out_lines.append('')
                out_lines.append('$$')
                in_display = True
            else:
                out_lines.append('$$')
                out_lines.append('')
                in_display = False
        else:
            if in_display:
                if stripped == '':
                    continue
                out_lines.append(line)
            else:
                formatted_line = re.sub(r'(?<![\$\s])\$([^\$]+?)\$(?![\$\s])', r' $\1$ ', line)
                formatted_line = re.sub(r' +', ' ', formatted_line)
                out_lines.append(formatted_line)
                
    res = '\n'.join(out_lines)
    res = re.sub(r'\n{3,}', '\n\n', res)
    return res.strip()

# ============================================================================
# Stage 2: Global Deduplication & Unified Synthesis
# ============================================================================

def run_stage2_synthesis(
    stage1_results: List[Dict[str, Any]],
    output_dir: str,
    course_name: str,
    session_title: str,
    cost_monitor: CostMonitor
) -> str:
    stage2_dir = f"{output_dir}/stage2_final"
    os.makedirs(stage2_dir, exist_ok=True)
    
    if cost_monitor.aborted:
        print("Skipping Stage 2 because budget limit was reached in Stage 1.")
        return ""
        
    print(f"\n[Phase 3] Running Stage 2 Global Note Synthesis for {session_title} (effort: low)...")
    
    valid_results = [r for r in stage1_results if r.get("status") in ["success", "cached"] and r.get("content")]
    
    combined_feed = []
    for r in valid_results:
        combined_feed.append(
            f"### 【时间戳 {r['timestamp_str']}】 ({r['lec_title']})\n{r['content']}\n"
        )
    feed_text = "\n---\n".join(combined_feed)
    
    with open(f"{stage2_dir}/raw_stage1_feed.md", "w", encoding="utf-8") as f:
        f.write(feed_text)
        
    synthesis_prompt = f"""以下是大学数学系《{course_name}》课程（{session_title}，共计两节大课约 110 分钟）按每 2 分钟时序抽样转录的全部黑板板书记录序列：

{feed_text}

============================================================
请依据上述全部连续采样的板书时序记录，输出这堂课的全部完整板书笔记。

【LaTeX 数学公式排版规范（绝对严格执行）】：
1. 严禁使用 \\[ 和 \\]，严禁使用 \\( 和 \\)。
2. 任何公式定界符只能且必须使用 $（行内公式）与 $$（行间独立公式）。
3. 行内公式排布规则：$ 外侧必须留有空格，内侧紧贴公式字符不得有空格。
   - 正确示例：设 $x \\in X$ 为拓扑空间中的点
   - 错误示例：设$x \\in X$、设 $ x \\in X $
4. 行间独立公式排布规则：$$ 必须单独占一行，前后必须各有一个空行，$$ 内侧紧贴公式不得有额外空行。
   - 正确示例：

     $$
     \\ker \\partial_n = \\operatorname{{im}} \\partial_{{n+1}}
     $$

【去重与结构整合规范（绝对严格执行）】：
1. 严禁机械重复：因摄像机镜头移动或多次采样而连续出现的同一图表、引理陈述、命题或定义，必须在全局中去重合并，严禁机械重复抄写多遍。
2. 忠实还原黑板授课脉络：
   - 黑板上老师推导过的步骤，必须完整、严谨地还原。
   - 黑板上老师仅列出待证目标但未书写证明的部分，忠实保持黑板原状，不私自伪造编造黑板上不存在的板书证明。
3. 允许且必须结合数学语义上下文，针对黑板部分模糊、反光或字迹遮挡处做精确修正与补齐（例如数学符号上下标、积分限、映射箭头符号等）。
4. 滤除空黑板时期的无意义占位（如【黑板暂无板书】）。
5. 保留并整合正文中的手绘示意图占位符（如 ![{{description}}]()），不随意增减，忠实复刻黑板上的示意图脉络。
6. 输出内容除了黑板完整板书内容外，绝对不要包含任何问候、前言、总结结语或元说明等非板书文字。"""

    payload = {
        "model": "deepseek-flash",
        "messages": [
            {"role": "user", "content": synthesis_prompt}
        ],
        "reasoning_effort": "low",
        "max_tokens": 32768,
        "temperature": 0.1
    }
    
    t0 = time.time()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
        t_elapsed = time.time() - t0
        
        if resp.status_code != 200:
            print(f"Stage 2 API Error {resp.status_code}: {resp.text[:200]}")
            return ""
            
        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        
        total_cost, _ = cost_monitor.add_usage(prompt_tokens, completion_tokens, reasoning_tokens)
        
        raw_final_notes = data["choices"][0]["message"].get("content", "").strip()
        
        with open(f"{stage2_dir}/deepseek_stage2_response.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        final_notes = enforce_latex_formatting(raw_final_notes)
        
        final_notes_path = f"{stage2_dir}/{session_title}_完整板书笔记.md"
        with open(final_notes_path, "w", encoding="utf-8") as f:
            f.write(final_notes)
            
        os.makedirs(NOTES_DIR, exist_ok=True)
        root_notes_path = os.path.join(NOTES_DIR, f"{session_title}_完整板书笔记.md")
        shutil.copy2(final_notes_path, root_notes_path)
        
        print(f"Stage 2 completed in {t_elapsed:.2f}s | "
              f"In:{prompt_tokens} Out:{completion_tokens} (Reasoning:{reasoning_tokens}) | "
              f"Session Total Cost: ¥{total_cost:.4f}")
        print(f"Final lecture notes saved to: {final_notes_path} and {root_notes_path}")
        return root_notes_path
        
    except Exception as e:
        print(f"Stage 2 Exception: {str(e)}")
        return ""


def run_session(
    session_key: str,
    workers: int = 10,
    extract_workers: int = 4,
    max_cost: float = 3.0,
    sample_n: int = None
) -> str:
    session = COURSE_SESSIONS[session_key]
    out_dir = session["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cost_monitor = CostMonitor(max_cost_rmb=max_cost)

    print("\n" + "=" * 70)
    print(f"   STARTING SESSION: {session['title']}")
    print(f"   Output Directory: {out_dir}")
    print(f"   Lectures: {[l['title'] for l in session['lectures']]}")
    print("=" * 70)

    frames_meta = prepare_all_frames(
        lectures=session["lectures"],
        output_dir=out_dir,
        interval_sec=120.0,
        sample_limit=sample_n,
        max_workers=extract_workers
    )

    if not frames_meta:
        print("No frames were extracted.")
        return ""

    stage1_results = run_stage1_parallel(
        frames_meta=frames_meta,
        output_dir=out_dir,
        course_name=session["course"],
        max_workers=workers,
        cost_monitor=cost_monitor
    )

    final_path = run_stage2_synthesis(
        stage1_results=stage1_results,
        output_dir=out_dir,
        course_name=session["course"],
        session_title=session["title"],
        cost_monitor=cost_monitor
    )

    summary = cost_monitor.get_summary()
    print("\n" + "-" * 70)
    print(f"   SESSION SUMMARY: {session['title']}")
    print(f"   - Processed Frames: {len(stage1_results)}")
    print(f"   - Total Tokens: {summary['total_tokens']}")
    print(f"   - Total Cost: ¥{summary['total_cost_rmb']:.4f} RMB")
    print(f"   - Result: {final_path}")
    print("-" * 70 + "\n")
    return final_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Universal Blackboard Digitization Agent")
    parser.add_argument("--session", type=str, default="all",
                        choices=list(COURSE_SESSIONS.keys()) + ["all"],
                        help="Specific session key to process, or 'all'")
    parser.add_argument("--sample-n", type=int, default=None, help="Dry run with only N frames per session")
    parser.add_argument("--workers", type=int, default=10, help="Stage 1 recognition threads")
    parser.add_argument("--extract-workers", type=int, default=4, help="Frame extraction threads")
    parser.add_argument("--max-cost", type=float, default=3.0, help="Budget per session in RMB")
    args = parser.parse_args()

    sessions_to_run = list(COURSE_SESSIONS.keys()) if args.session == "all" else [args.session]

    completed = {}
    for sk in sessions_to_run:
        res = run_session(
            session_key=sk,
            workers=args.workers,
            extract_workers=args.extract_workers,
            max_cost=args.max_cost,
            sample_n=args.sample_n
        )
        completed[sk] = res

    print("\n" + "=" * 70)
    print("   ALL REQUESTED SESSIONS PROCESSED")
    for sk, p in completed.items():
        print(f"   - {COURSE_SESSIONS[sk]['title']}: {p}")
    print("=" * 70)

if __name__ == "__main__":
    main()
