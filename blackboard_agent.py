"""
Blackboard Content Recognition Agent (高校黑板板书数字化解析 Agent)
========================================================================
Architecture:
  1. Classroom ROI Registry: Fixed multi-panel blackboard slab coordinates for
     - 东中院 2-106 (复分析, 数学分析（荣誉）III)
     - 上院 113 (代数拓扑)
  2. Temporal Median Filtering (时域中值滤波):
     Multi-frame median across temporal window (e.g. [-10, -5, 0, 5, 10]s)
     to seamlessly eliminate moving/standing teachers and podium occlusions.
  3. Multi-Slab Independent Processing:
     Each physical blackboard panel is cropped and processed as an independent document section.
  4. Chalkboard Optical Recognition:
     - MFD & Text OCR via Pix2Text (CPU runtime to avoid ONNX/CUDA conflicts)
     - Handwritten Formula OCR via UniMERNet (GPU on GTX 1650, ~412MB VRAM)
     - Inverted chalk contrast (dark background -> light background)
     - Annotation Disentanglement (chalk highlight boxes detected and preserved in metadata/LaTeX)
  5. Diagram & Geometric Structure Preservation:
     - Commutative diagrams, Cayley graphs, simplicial complexes, knots
     - Cropped at original full resolution, saved to assets/, embedded in Markdown
  6. Spatial Reading Order & Markdown Assembler:
     - XY-Cut column projection
     - Vertical overlap line grouping
     - Seamless inline ($...$) and block ($$...$$) math stitching
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure unimernet repo is in path
REPO_PATH = Path("/tmp/unimernet_repo")
if REPO_PATH.exists() and str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

from unimernet.common.config import Config as UnimerConfig
import unimernet.tasks as unimer_tasks
from unimernet.processors import load_processor
from pix2text import Pix2Text


# ============================================================================
# 1. Classroom & Blackboard Panel Registry
# ============================================================================

CLASSROOM_CONFIGS = {
    "dongzhong_2_106": {
        "name": "东中院 2-106",
        "description": "三组横向升降黑板 (左/中/右 × 上/下，共 6 块独立黑板板面)",
        "panels": {
            "left_top": {
                "name": "左侧-上方黑板",
                "bbox": (150, 260, 725, 370),
            },
            "left_bottom": {
                "name": "左侧-下方黑板",
                "bbox": (150, 375, 725, 595),
            },
            "center_top": {
                "name": "中间-上方黑板",
                "bbox": (750, 260, 1175, 370),
            },
            "center_bottom": {
                "name": "中间-下方黑板",
                "bbox": (750, 375, 1175, 595),
            },
            "right_top": {
                "name": "右侧-上方黑板",
                "bbox": (1205, 255, 1630, 370),
            },
            "right_bottom": {
                "name": "右侧-下方黑板",
                "bbox": (1205, 375, 1630, 595),
            },
        }
    },
    "shangyuan_113": {
        "name": "上院 113",
        "description": "阶梯教室 PTZ 云台黑板系统 (主黑板上下分层 + 侧面辅助黑板条)",
        "presets": {
            "preset_right": {
                "left_column": {
                    "name": "左侧辅助黑板条",
                    "bbox": (0, 0, 310, 740),
                },
                "right_top": {
                    "name": "主黑板-上方板面",
                    "bbox": (325, 0, 1515, 270),
                },
                "right_bottom": {
                    "name": "主黑板-下方板面",
                    "bbox": (325, 280, 1515, 735),
                },
            },
            "preset_left": {
                "left_top": {
                    "name": "主黑板-上方板面",
                    "bbox": (168, 0, 1260, 275),
                },
                "left_bottom": {
                    "name": "主黑板-下方板面",
                    "bbox": (168, 295, 1260, 725),
                },
                "right_column": {
                    "name": "右侧辅助黑板条",
                    "bbox": (1285, 0, 1920, 725),
                },
            }
        }
    }
}

COURSE_TO_CLASSROOM = {
    "复分析": "dongzhong_2_106",
    "数学分析（荣誉）III": "dongzhong_2_106",
    "代数拓扑": "shangyuan_113",
}


def detect_shangyuan_preset(frame: np.ndarray) -> str:
    """Auto-detect Shangyuan 113 PTZ camera preset (preset_left vs preset_right)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Wall on right in preset_right has high brightness (> 100)
    right_wall_val = float(np.mean(gray[50:200, 1600:1800]))
    if right_wall_val > 95.0:
        return "preset_right"
    return "preset_left"


# ============================================================================
# 2. Temporal Median Filter (时域中值滤波)
# ============================================================================

def extract_temporal_median(
    video_path: str,
    timestamp_sec: float,
    offsets: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample N video frames around timestamp_sec and compute pixel-wise median.
    Returns: (raw_center_frame, median_filtered_clean_frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    frames = []
    raw_center = None

    for off in offsets:
        t = max(0.0, min(duration - 0.5, timestamp_sec + off))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            if off == 0.0 or raw_center is None:
                raw_center = frame.copy()

    cap.release()

    if not frames:
        raise ValueError(f"Failed to read any frames from {video_path} at {timestamp_sec}s")

    # Pixel-wise temporal median eliminates moving objects (teacher, gestures)
    median_frame = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return raw_center, median_frame


def extract_registered_persistence_fusion(
    video_path: str,
    timestamp_sec: float,
    duration_sec: int = 20,
    fps: int = 4,
    crop_y: Tuple[int, int] = (0, 740),
    crop_x: Tuple[int, int] = (200, 1800)
) -> np.ndarray:
    """
    Advanced v2 Fusion Engine:
    Dense Temporal Sampling + Fourier Phase Correlation Registration
    + Temporal Persistence Filtering + Monotonic Chalk Union.
    Eliminates camera PTZ drift ghosting, teacher occlusion residues, and transient eraser lines.
    """
    import subprocess
    t_start = max(0.0, timestamp_sec - duration_sec // 2)
    cmd = [
        'ffmpeg', '-ss', str(t_start), '-t', str(duration_sec), '-i', video_path,
        '-r', str(fps), '-pix_fmt', 'gray', '-f', 'rawvideo', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_data, _ = proc.communicate()
    
    w, h = 1920, 1080
    num_frames = len(raw_data) // (w * h)
    if num_frames == 0:
        raise RuntimeError(f"FFmpeg failed to extract frames around t={timestamp_sec}s from {video_path}")
        
    frames = np.frombuffer(raw_data, dtype=np.uint8).reshape((num_frames, h, w))
    y1, y2 = crop_y
    x1, x2 = crop_x
    board_frames = frames[:, y1:y2, x1:x2]
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
    
    # 3. Temporal Activity
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
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chalk_robust = np.nanmedian(chalk_vals, axis=0)
    chalk_robust = np.where(np.isnan(chalk_robust), bg, chalk_robust)
    
    fused = np.where(valid_chalk, np.maximum(bg, chalk_robust), bg).astype(np.uint8)
    return fused


# ============================================================================
# 3. Data Structures & Formula/Diagram Heuristics
# ============================================================================

@dataclass
class LayoutBlock:
    id: int
    block_type: Literal["text", "isolated", "embedding", "diagram", "annotation"]
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0
    content: str = "" # Recognized text / LaTeX / image path
    is_boxed: bool = False # Whether the teacher surrounded the formula in a chalk box
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def x_mid(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def y_mid(self) -> float:
        return (self.y1 + self.y2) / 2.0


def has_chalk_writing(crop_bgr: np.ndarray, min_edge_density: float = 0.005, min_std: float = 14.0) -> bool:
    """Detect whether a blackboard panel has chalk writing or is clean/empty"""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    edges = cv2.Canny(gray, 40, 120)
    density = float(np.sum(edges > 0) / max(crop_bgr.shape[0] * crop_bgr.shape[1], 1))
    return density >= min_edge_density and std >= min_std


def detect_formula_box(crop_bgr: np.ndarray) -> bool:
    """
    Check if the formula was highlighted by the teacher with a chalk rectangle or boundary box.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 20 or w < 20:
        return False
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > 0.75 * w and ch > 0.75 * h:
            return True
    return False


# ============================================================================
# 4. UniMERNet Formula Engine
# ============================================================================

class UniMERNetEngine:
    """GPU-accelerated mathematical formula recognizer on GTX 1650 (~412MB VRAM)"""
    def __init__(self, config_path: str = "configs/run_tiny.yaml", device: str = "cuda"):
        self.device = device
        args = argparse.Namespace(cfg_path=config_path, options=None)
        cfg = UnimerConfig(args)
        task = unimer_tasks.setup_task(cfg)
        self.model = task.build_model(cfg).to(self.device).eval()
        self.vis_processor = load_processor(
            'formula_image_eval', 
            cfg.config.datasets.formula_rec_eval.vis_processor.eval
        )
        print(f"[UniMERNet] Initialized on {self.device}. VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    def predict(self, crop_bgr: np.ndarray, is_blackboard: bool = True) -> str:
        """Invert chalkboard colors and predict LaTeX formula"""
        if is_blackboard:
            # Dark chalkboard background with bright chalk -> white paper background with dark strokes
            crop_bgr = 255 - crop_bgr

        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.vis_processor(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate({"image": tensor})
        pred = output["pred_str"][0].strip()
        return pred


# ============================================================================
# 5. Blackboard Agent Core
# ============================================================================

class BlackboardAgent:
    def __init__(
        self,
        unimer_cfg: str = "configs/run_tiny.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        vertical_overlap_thresh: float = 0.45
    ):
        self.device = device
        self.vertical_overlap_thresh = vertical_overlap_thresh
        
        # 1. Formula Engine on GPU
        self.unimer = UniMERNetEngine(config_path=unimer_cfg, device=device)
        
        # 2. Layout & Text OCR Engine on CPU
        self.p2t = Pix2Text(languages=('ch_sim', 'en'), device='cpu')

    @staticmethod
    def is_chalkboard(img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < 115.0

    @staticmethod
    def is_diagram_candidate(crop: np.ndarray, block_type: str) -> bool:
        if block_type in ["image", "diagram", "figure"]:
            return True
        h, w = crop.shape[:2]
        aspect_ratio = w / max(h, 1)
        # Moderate to large square-ish blocks with high non-linear edge loops
        if h > 70 and w > 70 and (0.5 <= aspect_ratio <= 2.2):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            edge_density = np.sum(edges > 0) / (h * w)
            if 0.015 < edge_density < 0.28:
                return True
        return False

    def split_columns(self, boxes: List[LayoutBlock], img_width: int, min_gap: int = 40) -> List[List[LayoutBlock]]:
        """Slice blackboard slab into column bands via 1D projection"""
        if not boxes:
            return []
        proj = np.zeros(img_width, dtype=int)
        for b in boxes:
            proj[max(0, b.x1):min(img_width, b.x2)] += 1

        split_x = []
        in_gap = False
        gap_start = 0
        for x in range(img_width):
            if proj[x] == 0:
                if not in_gap:
                    in_gap = True
                    gap_start = x
            else:
                if in_gap:
                    in_gap = False
                    if (x - gap_start) >= min_gap:
                        split_x.append((gap_start + x) // 2)

        split_x = [0] + split_x + [img_width]
        columns = []
        for i in range(len(split_x) - 1):
            left, right = split_x[i], split_x[i+1]
            col_boxes = [b for b in boxes if left <= b.x_mid < right]
            if col_boxes:
                columns.append(col_boxes)
        return columns

    def group_into_lines(self, col_boxes: List[LayoutBlock]) -> List[List[LayoutBlock]]:
        """Group blocks in a column into horizontal reading lines"""
        sorted_boxes = sorted(col_boxes, key=lambda b: b.y1)
        lines: List[List[LayoutBlock]] = []

        for b in sorted_boxes:
            matched_line = None
            for line in lines:
                for item in line:
                    overlap_y = max(0, min(b.y2, item.y2) - max(b.y1, item.y1))
                    min_h = min(b.height, item.height)
                    if min_h > 0 and (overlap_y / min_h) >= self.vertical_overlap_thresh:
                        matched_line = line
                        break
                if matched_line is not None:
                    break
            
            if matched_line is not None:
                matched_line.append(b)
            else:
                lines.append([b])

        lines.sort(key=lambda line: np.mean([item.y1 for item in line]))
        for line in lines:
            line.sort(key=lambda item: item.x1)
            
        return lines

    def stitch_line_markdown(self, line: List[LayoutBlock]) -> str:
        """Render a single clustered line into Markdown"""
        if len(line) == 1 and line[0].block_type == "diagram":
            return f"\n\n![板书几何/结构示意图]({line[0].content})\n*【板书图形原样提取】*\n\n"

        if len(line) == 1 and line[0].block_type == "isolated":
            latex = line[0].content.strip()
            if line[0].is_boxed:
                latex = f"\\boxed{{ {latex} }}"
            return f"\n$$\n{latex}\n$$\n"

        line_str = ""
        for item in line:
            if item.block_type == "text":
                text = item.content.strip()
                if text:
                    line_str += f"{text} "
            elif item.block_type in ["embedding", "isolated"]:
                latex = item.content.strip()
                if item.is_boxed:
                    latex = f"\\boxed{{{latex}}}"
                if latex:
                    line_str += f"${latex}$ "
            elif item.block_type == "diagram":
                line_str += f"\n\n![板书图形]({item.content})\n\n"
        return line_str.strip()

    def detect_residual_diagrams(
        self,
        panel_bgr: np.ndarray,
        layout_blocks: List[LayoutBlock],
        assets_dir: Path,
        panel_prefix: str
    ) -> List[LayoutBlock]:
        """
        Identify chalkboard diagrams (triangles, simplicial complexes, knots)
        that were not detected by standard MFD/OCR and crop them cleanly.
        """
        h, w = panel_bgr.shape[:2]
        gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -12)

        # Mask out existing detected blocks
        occupied = np.zeros((h, w), dtype=np.uint8)
        for b in layout_blocks:
            cv2.rectangle(occupied, (max(0, b.x1 - 12), max(0, b.y1 - 12)), (min(w, b.x2 + 12), min(h, b.y2 + 12)), 255, -1)

        uncovered = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(occupied))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
        closed = cv2.morphologyEx(uncovered, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extra_diagrams = []
        diag_counter = 0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Require significant size and visual stroke density
            if cw > 65 and ch > 65:
                sub_uncovered = uncovered[y:y+ch, x:x+cw]
                stroke_density = np.sum(sub_uncovered > 0) / (cw * ch)
                if 0.02 < stroke_density < 0.45:
                    pad = 12
                    px1, py1 = max(0, x - pad), max(0, y - pad)
                    px2, py2 = min(w, x + cw + pad), min(h, y + ch + pad)
                    diag_crop = panel_bgr[py1:py2, px1:px2]
                    
                    fig_name = f"{panel_prefix}_diagram_{diag_counter:02d}.png"
                    cv2.imwrite(str(assets_dir / fig_name), diag_crop)
                    
                    extra_diagrams.append(
                        LayoutBlock(
                            id=1000 + diag_counter,
                            block_type="diagram",
                            x1=px1, y1=py1, x2=px2, y2=py2,
                            content=f"assets/{fig_name}",
                            metadata={"source": "residual_detector"}
                        )
                    )
                    diag_counter += 1
                    print(f"    [Residual Diagram] Detected & cropped non-OCR drawing -> assets/{fig_name} ({px2-px1}x{py2-py1})")
        return extra_diagrams

    def process_panel(
        self,
        panel_bgr: np.ndarray,
        panel_name: str,
        assets_dir: Path,
        panel_prefix: str
    ) -> Optional[str]:
        """Process a single cropped blackboard panel"""
        h, w = panel_bgr.shape[:2]
        
        # 1. Blank blackboard detection
        if not has_chalk_writing(panel_bgr):
            print(f"  [{panel_name}] Panel is blank/clean. Skipping.")
            return None

        # 2. Run Fast Decoupled MFD & Text OCR
        # We invoke MFD directly to avoid slow CPU LaTeX-OCR
        pil_panel = Image.fromarray(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))
        mfd_raw = self.p2t.text_formula_ocr.mfd.detect(pil_panel)
        text_raw = self.p2t.text_formula_ocr.text_ocr.ocr(panel_bgr)
        is_board = self.is_chalkboard(panel_bgr)

        layout_blocks: List[LayoutBlock] = []
        formula_boxes_coords: List[Tuple[int, int, int, int]] = []

        # Process Mathematical Formulas & Diagram candidates from MFD
        for idx, item in enumerate(mfd_raw):
            b_type = item.get("type", "isolated")
            box = np.array(item["box"])
            x1 = int(np.clip(np.min(box[:, 0]), 0, w - 1))
            y1 = int(np.clip(np.min(box[:, 1]), 0, h - 1))
            x2 = int(np.clip(np.max(box[:, 0]), 0, w))
            y2 = int(np.clip(np.max(box[:, 1]), 0, h))

            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            formula_boxes_coords.append((x1, y1, x2, y2))
            crop = panel_bgr[y1:y2, x1:x2].copy()

            # Check for Diagram candidate
            if self.is_diagram_candidate(crop, b_type):
                pad = 12
                px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
                fig_crop = panel_bgr[py1:py2, px1:px2]
                fig_name = f"{panel_prefix}_fig_{idx:02d}.png"
                cv2.imwrite(str(assets_dir / fig_name), fig_crop)

                layout_blocks.append(
                    LayoutBlock(
                        id=idx,
                        block_type="diagram",
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        content=f"assets/{fig_name}"
                    )
                )
                print(f"    [Diagram Block] Saved assets/{fig_name} ({px2-px1}x{py2-py1})")

            # Mathematical Formula -> UniMERNet on GPU
            else:
                is_boxed = detect_formula_box(crop)
                latex = self.unimer.predict(crop, is_blackboard=is_board)
                layout_blocks.append(
                    LayoutBlock(
                        id=idx,
                        block_type=b_type,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        content=latex,
                        is_boxed=is_boxed
                    )
                )
                box_flag = " [BOXED/HIGHLIGHTED]" if is_boxed else ""
                print(f"    [Formula:{b_type}{box_flag}] {latex[:65]}...")

        # Process Non-overlapping Text OCR lines
        for t_idx, item in enumerate(text_raw):
            pos = np.array(item["position"])
            tx1 = int(np.clip(np.min(pos[:, 0]), 0, w - 1))
            ty1 = int(np.clip(np.min(pos[:, 1]), 0, h - 1))
            tx2 = int(np.clip(np.max(pos[:, 0]), 0, w))
            ty2 = int(np.clip(np.max(pos[:, 1]), 0, h))
            t_text = item.get("text", "").strip()

            if (tx2 - tx1) < 6 or (ty2 - ty1) < 6 or not t_text:
                continue

            # Check if this text overlaps with an existing formula box
            is_overlap = False
            for (fx1, fy1, fx2, fy2) in formula_boxes_coords:
                inter_x = max(0, min(tx2, fx2) - max(tx1, fx1))
                inter_y = max(0, min(ty2, fy2) - max(ty1, fy1))
                inter_area = inter_x * inter_y
                t_area = (tx2 - tx1) * (ty2 - ty1)
                if t_area > 0 and (inter_area / t_area) > 0.4:
                    is_overlap = True
                    break

            if not is_overlap:
                layout_blocks.append(
                    LayoutBlock(
                        id=500 + t_idx,
                        block_type="text",
                        x1=tx1, y1=ty1, x2=tx2, y2=ty2,
                        content=t_text
                    )
                )

        # 3. Detect Residual Diagrams (uncovered geometric sketches / graphs)
        residual_diagrams = self.detect_residual_diagrams(panel_bgr, layout_blocks, assets_dir, panel_prefix)
        layout_blocks.extend(residual_diagrams)

        if not layout_blocks:
            return None

        # 4. Spatial Sort & Markdown Assembly
        columns = self.split_columns(layout_blocks, w)
        panel_lines = []

        for col_boxes in columns:
            lines = self.group_into_lines(col_boxes)
            for line in lines:
                md_line = self.stitch_line_markdown(line)
                if md_line:
                    panel_lines.append(md_line)

        return "\n\n".join(panel_lines)

    def process_course_video_frame(
        self,
        video_path: str,
        course_name: str,
        timestamp_sec: float,
        output_dir: str = "./output",
        use_v2_fusion: bool = True
    ) -> str:
        """
        Complete processing of a classroom video at timestamp_sec:
          1. Temporal Fusion:
             - v2 Fourier Phase Correlation Registration & Persistence Filter (removes PTZ drift ghosting & occlusions)
             - Or classic Temporal Median Filter
          2. Fixed ROI Multi-Panel cropping for specific course & classroom.
          3. Independent recognition per blackboard panel.
          4. Markdown documentation & asset preservation.
        """
        out_path = Path(output_dir)
        assets_dir = out_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        if course_name not in COURSE_TO_CLASSROOM:
            raise ValueError(f"Unknown course '{course_name}'. Supported: {list(COURSE_TO_CLASSROOM.keys())}")

        classroom_id = COURSE_TO_CLASSROOM[course_name]
        classroom_meta = CLASSROOM_CONFIGS[classroom_id]
        print(f"\n=======================================================")
        print(f" Processing: {course_name} (Classroom: {classroom_meta['name']})")
        print(f" Video: {Path(video_path).name}")
        print(f" Timestamp: {timestamp_sec:.1f}s ({int(timestamp_sec//60)}m{int(timestamp_sec%60)}s)")
        print(f" Pipeline Engine: {'v2 Fourier Registration & Persistence Fusion' if use_v2_fusion else 'v1 Temporal Median'}")
        print(f"=======================================================")

        # Step 1: Fusion / Teacher Occlusion Removal
        if use_v2_fusion:
            print("[Step 1/4] Running v2 Fourier Phase Correlation & Persistence Fusion...")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
            ret, raw_frame = cap.read()
            cap.release()
            if not ret or raw_frame is None:
                raw_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            fused_gray = extract_registered_persistence_fusion(
                video_path, timestamp_sec, crop_y=(0, 1080), crop_x=(0, 1920)
            )
            median_frame = cv2.cvtColor(fused_gray, cv2.COLOR_GRAY2BGR)
        else:
            print("[Step 1/4] Running Temporal Median Filtering across time window...")
            raw_frame, median_frame = extract_temporal_median(video_path, timestamp_sec)

        # Save side-by-side comparison
        vis_comparison = np.hstack([
            cv2.resize(raw_frame, (960, 540)),
            cv2.resize(median_frame, (960, 540))
        ])
        cv2.putText(vis_comparison, "Raw Frame (Teacher Occluded)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(vis_comparison, "v2 Clean Chalkboard" if use_v2_fusion else "Temporal Median (Clean Chalkboard)", (990, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        comp_file = out_path / f"teacher_removal_{course_name}_{int(timestamp_sec)}s.jpg"
        cv2.imwrite(str(comp_file), vis_comparison)
        print(f"  [Teacher Removal] Comparison saved to {comp_file}")

        # Step 2: Panel lookup
        if classroom_id == "dongzhong_2_106":
            panels_dict = classroom_meta["panels"]
        else:
            preset = detect_shangyuan_preset(median_frame)
            print(f"  [Preset Detection] Shangyuan 113 PTZ camera state: {preset}")
            panels_dict = classroom_meta["presets"][preset]

        # Step 3: Multi-Panel processing
        print(f"[Step 2/4] Processing {len(panels_dict)} blackboard panels individually...")
        markdown_sections = []

        for p_id, p_info in panels_dict.items():
            p_name = p_info["name"]
            x1, y1, x2, y2 = p_info["bbox"]
            panel_crop = median_frame[y1:y2, x1:x2].copy()
            
            # Save panel image crop for inspection
            panel_save_file = assets_dir / f"{course_name}_{int(timestamp_sec)}_{p_id}.jpg"
            cv2.imwrite(str(panel_save_file), panel_crop)

            print(f"\n  --- Analyzing Panel: {p_name} ({p_id}) [{x2-x1}x{y2-y1}] ---")
            panel_prefix = f"{course_name}_{int(timestamp_sec)}_{p_id}"
            panel_md = self.process_panel(panel_crop, p_name, assets_dir, panel_prefix)

            if panel_md:
                sec_header = f"### 📌 板面: {p_name} (`{p_id}`)\n> 原始板面切片: `assets/{panel_save_file.name}`\n\n"
                markdown_sections.append(sec_header + panel_md)

        # Step 4: Markdown output
        print(f"\n[Step 4/4] Generating Consolidated Markdown Lecture Notes...")
        time_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
        doc_header = (
            f"# 🎓 课堂板书数字化笔记: {course_name}\n\n"
            f"- **授课教室**: {classroom_meta['name']} (`{classroom_id}`)\n"
            f"- **视频源**: `{Path(video_path).name}`\n"
            f"- **提取时间戳**: `{time_str}` ({timestamp_sec:.1f} 秒)\n"
            f"- **消遮挡处理**: 时域中值滤波 (消除走动讲师与讲台遮挡，恢复被挡粉笔笔迹)\n\n"
            f"![消遮挡前后对比]({comp_file.name})\n\n"
            f"---\n\n"
        )

        full_doc = doc_header + "\n\n---\n\n".join(markdown_sections)
        md_file = out_path / f"{course_name}_{int(timestamp_sec)}s_notes.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(full_doc)

        print(f"[BlackboardAgent] Done! Document written to: {md_file}")
        return full_doc


# ============================================================================
# 6. CLI & Automated Evaluation
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackboard Recognition Agent with Temporal Median Filter & Multi-Panel ROIs")
    parser.add_argument("--course", type=str, choices=list(COURSE_TO_CLASSROOM.keys()), default="代数拓扑", help="Course name")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    parser.add_argument("--time", type=float, default=1500.0, help="Timestamp in seconds")
    parser.add_argument("--output", type=str, default="./output_test", help="Output directory")
    parser.add_argument("--eval-all", action="store_true", help="Run full evaluation on all 3 university courses")
    parser.add_argument("--no-v2", action="store_true", help="Disable v2 Fourier Persistence Fusion and use classic median")
    args = parser.parse_args()

    use_v2 = not args.no_v2

    agent = BlackboardAgent(unimer_cfg="configs/run_tiny.yaml")

    if args.eval_all:
        test_suite = [
            {
                "course": "代数拓扑",
                "video": "data/videos/代数拓扑/代数拓扑_2026-10-22_代数拓扑 (第2节)_0850-0945.mp4",
                "time": 1500.0,
            },
            {
                "course": "复分析",
                "video": "data/videos/复分析/复分析_2026-10-22_复分析 (第9节)_1555-1650.mp4",
                "time": 1200.0,
            },
            {
                "course": "数学分析（荣誉）III",
                "video": "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-21_数学分析（荣誉）III (第2节)_0850-0945.mp4",
                "time": 1500.0,
            },
        ]
        for item in test_suite:
            agent.process_course_video_frame(
                video_path=item["video"],
                course_name=item["course"],
                timestamp_sec=item["time"],
                output_dir=args.output,
                use_v2_fusion=use_v2
            )
    else:
        v_path = args.video
        if not v_path:
            # Pick first available video for course
            c_dir = Path("data/videos") / args.course
            v_list = list(c_dir.glob("*.mp4"))
            if v_list:
                v_path = str(v_list[0])
            else:
                raise ValueError(f"No video found in {c_dir}")

        agent.process_course_video_frame(
            video_path=v_path,
            course_name=args.course,
            timestamp_sec=args.time,
            output_dir=args.output,
            use_v2_fusion=use_v2
        )
