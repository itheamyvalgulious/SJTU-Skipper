# picture_parser

课堂板书数字化流水线：从课堂录像中采样、配准、融合黑板帧，经 VLM 转录后合成为完整的 LaTeX 板书笔记。

## 目录结构

```
.
├── src/                     # 源码
│   ├── blackboard_lecture_agent.py    # 通用多课程多节次流水线（复分析 / 数学分析 / 代数拓扑）
│   ├── topo_lecture_agent.py          # 代数拓扑单次双节专用版
│   └── test_placeholder_experiment.py # 示意图占位符识别实验
├── data/                    # 生成数据（已被 .gitignore 忽略）
│   ├── videos/              # 原始课堂录像
│   ├── outputs/<session>/   # 每条会话的 frames / stage1_transcriptions / stage2_final
│   └── notes/               # 最终板书笔记（Markdown）
├── sjtu-canvas-assistant/   # 课程视频下载子模块
├── .env.example
└── .gitignore
```

`data/` 整体不入库，运行产物与笔记都放在这里，仓库根目录只保留代码与配置。

## 环境准备

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pillow requests
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
ffmpeg 需在 PATH 中
```

## 运行

脚本以仓库根目录为基准定位 `.env`、`data/videos` 与输出目录，因此可在任意目录执行：

```bash
# 处理单条会话（可省略 --session 跑全部）
python src/blackboard_lecture_agent.py --session topo_1022

# 代数拓扑专用脚本
python src/topo_lecture_agent.py
```

可用会话：`ca_1019`、`ca_1021`、`ma_1019`、`ma_1021`、`topo_1020`、`topo_1022`。

输出位置：

- 流水线中间结果与转录：`data/outputs/<session>/`
- 最终板书笔记：`data/notes/<标题>_完整板书笔记.md`（同时在 `data/outputs/<session>/stage2_final/` 留档）