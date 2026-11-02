# SJTU-Skipper / 课堂总 Agent

SJTU-Skipper 仓库包含两条相关但独立的能力：

1. **旧的板书批处理流水线**：`legacy/blackboard_lecture_agent.py`，从课堂录像采样、配准、融合黑板帧，经 VLM 转录后合成 LaTeX 板书笔记。
2. **新的课堂总 Agent 架构**：`src/`，把音频/视频输入、STT、VLM、LLM、行为插件和 Web UI 拆成可独立演进的层；行为层只通过 append-only 流、事件和服务接口与底层交互。

新架构的设计与接口契约见：

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/implementation-plan.md`](docs/implementation-plan.md)
- [`docs/plugin-api.md`](docs/plugin-api.md)

## 新架构快速开始

### 1. 环境

系统需要 Python 3.12、`ffmpeg`/`ffprobe`；`aria2c` 可选。

本项目使用仓库根目录的 `.venv`（Python 3.12）。旧 `.venv` 已删除并重建：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

也可以使用普通 venv：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### 2. 真实运行

```bash
.venv/bin/sjtu-skipper check   # 检查插件拓扑加载
.venv/bin/sjtu-skipper serve   # 启动 Web UI
```

打开页面后：

1. 右侧「设置」填入 Canvas Token、LLM API Key / Base URL；模型在 Agent / VLM 分组中设置。
2. 左侧「课程」刷新课程列表。
3. 选择录播或直播，开始一节课。
4. 左侧「课堂」查看转写、板书、笔记。

Canvas token 与 DeepSeek key 都放在根 `.env`（git-ignored）：

```dotenv
DEEPSEEK_API_KEY=...
# 或使用任意 OpenAI 兼容服务
LLM_BASE_URL=http://127.0.0.1:3000/v1
LLM_API_KEY=...
LLM_MODEL=...
```

设置会持久化到 `data/agent/settings.json`（权限 0600）。

### 3. 命令行

```bash
sjtu-skipper serve [--host 0.0.0.0] [--port 8765] [--data-dir PATH]
sjtu-skipper check

# 兼容旧命令：lecture-agent <同上>
```

## 新架构一览

```text
输入插件（Canvas/SJTU 录播 + 直播 + 手动 URL）
        │  AudioStream / VideoStream（append-only）
        ├──────────────► STT 插件 ──► TextStream(transcript)
        └──────────────► VLM 插件 ──► TextStream(visual)
                                      │
行为层（llm-agent / summarize / 自定义插件）
  事件总线 · 时钟 · 服务注册 · UI · LLM 统一接口
                                      │
Web UI（左右多 tab 边栏 + toolbox + 中间 pages）
```

- **流**：音频是 PCM s16le；视频是 JPEG 帧；文字是不断增长的 `TextItem` 列表。`len(stream)` 是整秒/片段数，精确时长用 `stream.duration`。
- **事件**：`clock.tick`、`session.*`、`stream.appended`、`settings.changed`、`agent.*`、`stt.*`、`vlm.*`、`summarize.*`、`input.*`。
- **插件**：`id` + `requires` 组成 DAG，拓扑序加载；插件之间只通过 `ctx.services` 调用。
- **LLM 层**：统一 `ChatRequest/ChatResponse/StreamEvent`，OpenAI 兼容 provider，流式/非流式、工具调用、图片、重试/超时。
- **UI**：声明式 block + action；插件不写前端代码。

内置插件：

| 插件 | id | 作用 |
|---|---|---|
| settings | `settings` | 设置注册、持久化、设置页 |
| llm | `llm` | Provider 注册、统一 LLM 调用 |
| input | `input` | Canvas/SJTU 录播、直播、ffmpeg 源 |
| stt | `stt` | 音频流 -> 文字流 |
| vlm | `vlm` | 视频流 -> 融合板书 -> 文字流 |
| llm-agent | `llm-agent` | 多会话 REPL Agent，info/read/replace/append |
| summarize | `summarize` | 按 N 字符/秒触发，写 knowledge/gossip/notice |

## 旧板书流水线

旧脚本保持原样，仍可单独使用：

```bash
cp .env.example .env   # 填 DEEPSEEK_API_KEY
source .venv/bin/activate
python legacy/blackboard_lecture_agent.py --session topo_1022
```

产物在 `data/outputs/<session>/` 与 `data/notes/`。新 VLM 插件复用其配准/融合算法与提示词；旧脚本已移入 `legacy/`（本地保留、git-ignored）。

## 目录

```text
src/                              # 新总 Agent（直接位于 src 下）
├── app.py / cli.py / config.py
├── applog.py
├── core/                         # 流、事件、插件、服务、UI、runtime
├── llm/                          # 协议无关 LLM 层
├── plugins/                      # settings/llm/input/stt/vlm/agent/summarize
├── integrations/                 # vendored SJTU Canvas/视频客户端
└── ui/                           # FastAPI + 静态前端
legacy/                           # 旧脚本与旧子模块，git-ignored，仅本地保留
├── blackboard_lecture_agent.py
├── topo_lecture_agent.py
├── test_placeholder_experiment.py
└── sjtu-canvas-assistant/
tests/
docs/
data/                             # 运行产物（gitignore）
```

`sjtu-canvas-assistant` 的必要客户端代码已经 vendor 到 `src/integrations/sjtu_canvas_assistant/`，运行时不依赖 `legacy/`。

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试不访问网络：LLM 用 `httpx.MockTransport` 或测试内的脚本化 provider，Canvas/SJTU 用 fake client。生产代码不包含 mock provider / mock STT 后端。

## 说明

- 旧 `sjtu-canvas-assistant` 子模块已移入 `legacy/`，不再参与构建；必要的客户端代码已 vendor 到 `src/integrations/sjtu_canvas_assistant/`。
- 不修改旧板书脚本（同上）。
- 新版 SJTU 门户（`jy-application-resourcemanage`）的录播/直播适配已随 vendored 客户端放入 `src/integrations/sjtu_canvas_assistant/`，旧子模块不再需要。
- 直播通道优先使用 SJTU 新版门户 `v1/vod_live` + `v1/course_vod_videoinfos`；若接口或鉴权变化，仍可在 UI 中手动填入 HLS/RTMP/HTTP URL。
- 详细接口与扩展方式见 `docs/plugin-api.md`。
