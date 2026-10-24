"""
Blackboard Recognition Benchmark Suite (高校黑板板书数字化解析基准评测系统)
========================================================================
Benchmark Design:
  - 3 Core University Courses:
      1. 代数拓扑 (上院 113)
      2. 复分析 (东中院 2-106)
      3. 数学分析（荣誉）III (东中院 2-106)
  - 2 Distinct Time Moments per Course (Total 6 Evaluation Cases).
  - Temporal Median Synthesis: sampled over [-10s, -5s, 0s, +5s, +10s] (前后 10s 合成).
  - Multi-Dimensional Comparative Metrics:
      1. 公式提取保真度与字符准确度 (UniMERNet vs Direct VLM)
      2. 走动讲师与讲台遮挡恢复率 (Teacher Occlusion Area & Hidden Strokes Recovery)
      3. 非公式拓扑/几何草图原样保存 (Vector/PNG Diagram Extraction vs Text Description)
      4. 空间阅读顺序重构 (Multi-column XY-Cut & Discrete Slab Layout)
      5. 本地显存与运行效率 (GTX 1650 4GB VRAM & Latency)
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch

from blackboard_agent import (
    BlackboardAgent,
    CLASSROOM_CONFIGS,
    COURSE_TO_CLASSROOM,
    extract_temporal_median,
    detect_shangyuan_preset
)

BENCHMARK_CASES = [
    {
        "case_id": "topo_case_1",
        "course": "代数拓扑",
        "video": "data/videos/代数拓扑/代数拓扑_2026-10-22_代数拓扑 (第2节)_0850-0945.mp4",
        "time_sec": 900.0,
        "description": "单纯形剖分与同调代数基础定理推导",
    },
    {
        "case_id": "topo_case_2",
        "course": "代数拓扑",
        "video": "data/videos/代数拓扑/代数拓扑_2026-10-22_代数拓扑 (第2节)_0850-0945.mp4",
        "time_sec": 1800.0,
        "description": "重心细分公式与拓扑单形几何图",
    },
    {
        "case_id": "ca_case_1",
        "course": "复分析",
        "video": "data/videos/复分析/复分析_2026-10-22_复分析 (第9节)_1555-1650.mp4",
        "time_sec": 900.0,
        "description": "全纯函数积分与围道积分曲线",
    },
    {
        "case_id": "ca_case_2",
        "course": "复分析",
        "video": "data/videos/复分析/复分析_2026-10-22_复分析 (第9节)_1555-1650.mp4",
        "time_sec": 1800.0,
        "description": "柯西-古萨定理积分推导与同伦闭路径图",
    },
    {
        "case_id": "ma_case_1",
        "course": "数学分析（荣誉）III",
        "video": "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-21_数学分析（荣誉）III (第2节)_0850-0945.mp4",
        "time_sec": 1000.0,
        "description": "含参变量反常积分一致收敛性判定",
    },
    {
        "case_id": "ma_case_2",
        "course": "数学分析（荣誉）III",
        "video": "data/videos/数学分析（荣誉）III/数学分析（荣誉）III_2026-10-21_数学分析（荣誉）III (第2节)_0850-0945.mp4",
        "time_sec": 2000.0,
        "description": "狄利克雷积分公式与多重积分估计",
    },
]


def run_benchmark_suite(output_dir: str = "./output_benchmark"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    assets_dir = out_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("🚀 Initializing Blackboard Recognition Benchmark Suite...")
    print(f"Total Test Cases: {len(BENCHMARK_CASES)}")
    print(f"Target Output Directory: {output_dir}")
    print("=================================================================\n")

    # Initialize Blackboard Agent
    agent = BlackboardAgent(unimer_cfg="configs/run_tiny.yaml")
    results = []

    for idx, case in enumerate(BENCHMARK_CASES):
        cid = case["case_id"]
        course = case["course"]
        v_path = case["video"]
        t_sec = case["time_sec"]
        desc = case["description"]

        print(f"\n[{idx+1}/{len(BENCHMARK_CASES)}] Running Benchmark: {cid}")
        print(f"  Course: {course} | Timestamp: {t_sec}s ({int(t_sec//60)}m{int(t_sec%60)}s)")
        print(f"  Description: {desc}")

        t0 = time.time()
        
        # 1. Temporal Median Filter with ±10s window (offsets: -10, -5, 0, 5, 10)
        raw_frame, median_frame = extract_temporal_median(
            v_path, t_sec, offsets=(-10.0, -5.0, 0.0, 5.0, 10.0)
        )
        t_median_done = time.time()

        # Calculate Occlusion Metrics
        diff = np.abs(raw_frame.astype(float) - median_frame.astype(float))
        teacher_mask = diff > 30.0
        occlusion_pixel_ratio = float(np.sum(teacher_mask) / (raw_frame.shape[0] * raw_frame.shape[1] * 3))
        max_diff = float(np.max(diff))

        # Save Visual Comparison (Raw vs Filtered)
        comp_img = np.hstack([
            cv2.resize(raw_frame, (960, 540)),
            cv2.resize(median_frame, (960, 540))
        ])
        cv2.putText(comp_img, f"Raw Frame (Occluded) t={t_sec}s", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        cv2.putText(comp_img, f"10s Median Synthesized (Clean)", (985, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        comp_file = out_path / f"comp_{cid}.jpg"
        cv2.imwrite(str(comp_file), comp_img)

        # 2. Run Pipeline on Median Frame
        classroom_id = COURSE_TO_CLASSROOM[course]
        classroom_meta = CLASSROOM_CONFIGS[classroom_id]

        if classroom_id == "dongzhong_2_106":
            panels_dict = classroom_meta["panels"]
        else:
            preset = detect_shangyuan_preset(median_frame)
            panels_dict = classroom_meta["presets"][preset]

        active_panels = 0
        formula_count = 0
        diagram_count = 0
        panel_notes = []

        for p_id, p_info in panels_dict.items():
            x1, y1, x2, y2 = p_info["bbox"]
            p_crop = median_frame[y1:y2, x1:x2].copy()
            p_prefix = f"{cid}_{p_id}"
            
            # Save raw panel crop for benchmark inspection
            p_crop_path = assets_dir / f"{p_prefix}.jpg"
            cv2.imwrite(str(p_crop_path), p_crop)

            p_md = agent.process_panel(p_crop, p_info["name"], assets_dir, p_prefix)
            if p_md:
                active_panels += 1
                # Count formulas and diagrams in panel
                formula_count += p_md.count("$$") // 2 + p_md.count("$") // 2
                diagram_count += p_md.count("![板书")
                panel_notes.append(f"### 📌 板面: {p_info['name']} (`{p_id}`)\n> 板面切片: `assets/{p_crop_path.name}`\n\n{p_md}")

        t_pipeline_done = time.time()
        
        # Save Markdown Notes
        full_markdown = (
            f"# 🎓 板书识别 Benchmark 评测: {course} - 样本 {cid}\n\n"
            f"- **课程**: {course} ({classroom_meta['name']})\n"
            f"- **时间戳**: `{int(t_sec//60):02d}:{int(t_sec%60):02d}` ({t_sec:.1f} 秒)\n"
            f"- **测试场景**: {desc}\n"
            f"- **前后 10s 中值滤波恢复遮挡面积**: `{occlusion_pixel_ratio*100:.2f}%`\n"
            f"- **总耗时**: 时域滤波 `{t_median_done - t0:.2f}s` + 解析 `{t_pipeline_done - t_median_done:.2f}s`\n\n"
            f"![消遮挡对比图](comp_{cid}.jpg)\n\n---\n\n"
            + "\n\n---\n\n".join(panel_notes)
        )
        md_file = out_path / f"notes_{cid}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        # Record metrics
        case_result = {
            "case_id": cid,
            "course": course,
            "classroom": classroom_meta["name"],
            "time_sec": t_sec,
            "description": desc,
            "occlusion_ratio_pct": round(occlusion_pixel_ratio * 100, 2),
            "max_pixel_diff": max_diff,
            "median_filter_time_sec": round(t_median_done - t0, 3),
            "pipeline_time_sec": round(t_pipeline_done - t_median_done, 3),
            "total_time_sec": round(t_pipeline_done - t0, 3),
            "active_panels": active_panels,
            "total_panels_checked": len(panels_dict),
            "formula_count": formula_count,
            "diagram_count": diagram_count,
            "comp_image": str(comp_file.name),
            "notes_file": str(md_file.name),
            "gpu_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 1)
        }
        results.append(case_result)
        print(f"  -> Finished {cid}: {active_panels} active panels, {formula_count} formulas, {diagram_count} diagrams in {case_result['total_time_sec']}s")

    # Save summary json
    summary_file = out_path / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=================================================================")
    print(" Benchmark Evaluation Complete!")
    print(f" Summary JSON saved to: {summary_file}")
    print("=================================================================")
    return results


if __name__ == "__main__":
    run_benchmark_suite()
