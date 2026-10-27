import os
import sys
import time
import base64
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
NOTES_DIR = os.path.join(DATA_DIR, "notes")

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

OUT_DIR = os.path.join(OUTPUTS_DIR, "placeholder_test")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = [
    {"id": "with_diag_1", "expected_diag": True, "title": "单连通星形闭合路径与相交圆盘曲线"},
    {"id": "with_diag_2", "expected_diag": True, "title": "复平面保角映射原象与像区域"},
    {"id": "with_diag_3", "expected_diag": True, "title": "圆盘放射状射线与狭缝区域映射"},
    {"id": "with_diag_4", "expected_diag": True, "title": "同心圆复平面与三角形区域剖分"},
    {"id": "with_diag_5", "expected_diag": True, "title": "闭合积分路径与三角形内切圆几何图"},
    {"id": "no_diag_1", "expected_diag": False, "title": "柯西积分定理文字与导数公式板书"},
    {"id": "no_diag_2", "expected_diag": False, "title": "柯西-古尔萨定理数学定义与长公式证明板书"},
    {"id": "no_diag_3", "expected_diag": False, "title": "复变函数极限解析性质与集合条件推导"},
]

prompt_text = (
    "这是大学数学系课堂的黑板板书图像。\n"
    "请准确识别并完整转录图中的全部手写数学板书内容（包括引理、定理、定义、命题、证明推导细节及中文说明）。\n\n"
    "【LaTeX 公式规范（绝对严格）】：\n"
    "1. 只能使用 $（行内公式）与 $$（行间公式）。严禁使用 \\[、\\]、\\(、\\)。\n"
    "2. 行内公式外侧留空格，内侧紧贴不留空格；行间公式 $$ 独占一行前后各空一行。\n\n"
    "【手绘示意图占位符规范（绝对严格）】：\n"
    "对黑板上的手绘示意图（如几何图形、闭合曲线围道、区域映射示意图等），必须使用 Markdown 图片语法占位符：\n"
    "![{对图形的简短中文描述}]()\n"
    "例如黑板画了同心圆，必须写入：![同心圆区域示意图]()。\n"
    "严禁写成普通文字描述或括号备注（如严禁写成“（手绘示意图：...）”）。\n"
    "不需要精确内容或精确画，而只需要有该 ![...] 格式的占位符即可。\n"
    "注意：必须严格复刻黑板上的示意图，不可随意增减；纯公式与文字推导绝不添加占位符。\n\n"
    "请直接输出板书转录结果，不得有任何省略或总结概括，不要输出问候寒暄。"
)

def process_sample(s):
    sid = s["id"]
    img_path = os.path.join(DATA_DIR, "eval_samples", f"{sid}.jpg")
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
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
        "max_tokens": 2500,
        "temperature": 0.1
    }
    
    t0 = time.time()
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    t_elapsed = time.time() - t0
    
    data = resp.json()
    content = data["choices"][0]["message"].get("content", "").strip()
    
    md_file = f"{OUT_DIR}/{sid}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    placeholders = re.findall(r'!\[(.*?)\]\(\)', content)
    
    return {
        "id": sid,
        "title": s["title"],
        "expected_diag": s["expected_diag"],
        "elapsed_sec": round(t_elapsed, 2),
        "placeholders": placeholders,
        "num_placeholders": len(placeholders),
        "md_file": os.path.abspath(md_file),
        "content_snippet": content[:300].replace('\n', ' ')
    }

print("Running placeholder extraction across 8 samples...")
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_sample, s) for s in SAMPLES]
    for f in as_completed(futures):
        results.append(f.result())

results.sort(key=lambda x: x["id"])

print("\n" + "=" * 80)
print(" EXPERIMENT RESULTS: ![{image description here}]() PLACEHOLDER EXTRACTION")
print("=" * 80)
for r in results:
    status = "✅ 100% 正确" if (r["num_placeholders"] > 0) == r["expected_diag"] else "❌ 错误"
    exp_str = "正样本（含示意图）" if r["expected_diag"] else "负样本（纯公式推导）"
    print(f"[{r['id']}] {status} | {exp_str} | 检出占位符: {r['num_placeholders']} 个 | 耗时: {r['elapsed_sec']}s")
    for p in r["placeholders"]:
        print(f"    -> ![{p}]()")
    print(f"    文档路径: {r['md_file']}")
    print("-" * 80)
