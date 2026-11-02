# 实施计划与并行任务分解

> 依据 `docs/architecture.md`。先冻结接口，再并行开发；接口变更必须同步更新本文件和 `docs/architecture.md`。

## 实现状态（v1 已完成）

- P0 核心框架：完成；流、事件、插件 DAG、服务、UI、runtime 均有单测。
- P1 LLM 层：完成；OpenAI 兼容非流式/流式、工具调用、图片、重试、`extra` 顶层合并。
- P2 settings：完成；JSON 持久化、secret 掩码、表单 UI。
- P3 llm-agent：完成；多会话、REPL、info/read/replace/append、扩展点。
- P4 summarize：完成；文字/音频触发、水位、重试、三文件分类。
- P5 input：完成；Canvas 13 门活跃课程、SJTU 新版录播/直播、ffmpeg 双流、UI。
- P6 STT：完成；none / openai-compatible / faster-whisper 后端，worker 与 UI。
- P7 VLM：完成；picture_parser 融合算法适配、实时窗口、prompt、去重、UI。
- P8 Web UI：完成；FastAPI + 左右多 tab + toolbox + pages + WebSocket + action。
- 全量测试通过；真实 Canvas/SJTU 录播/直播、真实 DeepSeek 文本/视觉调用均已冒烟。

实现中的偏差/说明：

- Python 的 `len(stream)` 必须是 `int`，因此 `len` 返回整秒，亚秒精度用 `stream.duration`。
- 输入插件默认只列 `active_only=True` 的 Canvas 课程（实测 13 门）；历史学期可通过底层 catalog 扩展。
- STT 默认 `none`：只接入音频流、不输出文字；真实使用需选择 `openai` 或安装可选 `faster-whisper` 并配置模型/密钥。
- VLM 默认关闭（`vlm.enabled=false`），在设置里打开后会动态启动 worker；需配置视觉模型（如 `deepseek-flash`）。
- VLM 融合固定使用整帧原始分辨率（`crop=(0,0,1,1)`、`max_width=0`），不裁剪、不缩放、不切面板。
- 直播设备默认取第 0 个（老师机位），可在 `input.live_device` 切换；URL 自带签名，必要时可扩展 headers。
- 旧 `legacy/blackboard_lecture_agent.py` 与旧 `legacy/sjtu-canvas-assistant` 均已移入 git-ignored 的 `legacy/`；输入模块使用 vendored 客户端。

## 0. 环境与基线

- Python：3.12（系统 `/usr/bin/python3.12`）。
- 虚拟环境：`.venv`（Python 3.12；已删除旧的不可用 `.venv` 并重建）。
- 依赖：`fastapi`, `uvicorn[standard]`, `httpx`, `pydantic`, `python-dotenv`, `numpy`, `pillow`, `markdown-it-py`, `beautifulsoup4`, `rich`, `typer`；测试 `pytest`, `pytest-asyncio`。
- 外部命令：`ffmpeg`, `ffprobe`, `aria2c`（可选）。
- 真实冒烟：Canvas token 与 DeepSeek API key 已存在于根 `.env`（git-ignored）。
- 约束：
  - 旧 `sjtu-canvas-assistant` 子模块移入 `legacy/` 并 ignore；不参与构建。
  - `legacy/` 内的旧脚本保留本地，不参与构建。
  - 旧批处理脚本保留；新 VLM 算法在 `plugins.vlm.fusion` 中重写/适配。
  - 所有数学公式按 `$...$` / `$$...$$` 书写。

## 1. 阶段划分

### P0 基础框架（主 agent 亲自完成，其他任务的前置）

文件：

- `pyproject.toml`
- `src/config.py`
- `src/applog.py`
- `src/core/events.py`
- `src/core/streams.py`
- `src/core/services.py`
- `src/core/plugin.py`
- `src/core/ui.py`
- `src/core/meta.py`
- `src/core/runtime.py`
- `tests/test_streams.py`
- `tests/test_events.py`
- `tests/test_plugins.py`
- `tests/test_ui.py`

冻结接口：

- `EventBus`：`define/on/once/off/emit/flush/start/stop`；handler 收 `Event`。
- `StreamHub`：`current`、`start_session/end_session/list_sessions`；`StreamSet` 四流。
- `AudioStream/VideoStream/TextStream`：`__len__`、`append`、`read`、`wait_for`。
- `PluginContext`：`plugin_id`, `data_dir`, `events`, `streams`, `llm`, `ui`, `services`, `meta`, `logger`。
- `Plugin`：`id`, `requires`, `provides`, `setup(ctx)`, `teardown()`。
- `UIManager`：`add_tab/add_toolbox_item/add_page/open_page/close_page/refresh/register_action/dispatch_action/state`。
- `MetaRegistry`：`register(plugin_id, name, provider)` / `snapshot()`。

验收：

- `pytest tests/test_streams.py tests/test_events.py tests/test_plugins.py tests/test_ui.py` 全绿。
- 流前缀不可变、事件合并、拓扑排序/环检测、UI state 序列化有测试。

### P1 LLM 层（并行任务 A）

文件：

- `src/llm/types.py`
- `src/llm/errors.py`
- `src/llm/provider.py`
- `src/llm/openai_compat.py`
- `src/llm/service.py`
- `src/plugins/llm_core/__init__.py`
- `src/plugins/llm_core/plugin.py`
- `tests/test_llm.py`

要求：

- 统一 `Message/ContentPart/ToolDef/ChatRequest/ChatResponse/StreamEvent`。
- OpenAI 兼容：非流式 + SSE 流式 + 工具调用增量 + 图片 data URL。
- 重试：连接错误、超时、408/409/429/5xx；`Retry-After`；指数退避 + jitter；流式首包前可重试。
- `LLMService` profile 解析：`agent` / `vlm` / `stt`；设置回调可注入（不直接依赖 settings 插件）。
- 单元测试用 `httpx.MockTransport` 或测试内脚本化 provider，不访问网络。
- 事件回调：`on_event("llm.request"/"llm.response"/"llm.error", payload)`。

不得修改 P0 文件。

### P2 设置插件（并行任务 B）

文件：

- `src/plugins/settings/__init__.py`
- `src/plugins/settings/plugin.py`
- `src/plugins/settings/store.py`
- `tests/test_settings.py`

要求：

- `SettingsSpec`：key/label/type/default/group/description/options/order/secret/placeholder。
- 类型：string/text/int/float/bool/select/secret/json。
- JSON 持久化，原子写，0600；默认值不落盘直到 set。
- `register/get/set/set_many/all/public_all/specs`。
- secret 公开值掩码；内部 `get` 返回真实值。
- 变更发 `settings.changed`（含 key、old、new）。
- UI：右侧 tab `settings`；toolbox 按 group；page `settings:main` 表单；action `settings.save`。
- 单元测试：默认值、持久化、类型转换、secret 掩码、变更事件、非法值。

不得修改 P0/P1 文件。

### P3 LLM Agent 插件（并行任务 C）

文件：

- `src/plugins/agent/__init__.py`
- `src/plugins/agent/plugin.py`
- `src/plugins/agent/session.py`
- `src/plugins/agent/tools.py`
- `src/plugins/agent/workspace.py`
- `src/plugins/agent/prompts.py`
- `tests/test_agent.py`

要求：

- `AgentService` 注册为 `agent`；多会话。
- `create_session(name, system_prompt, files/workspace, tools, model, max_steps, extra_instructions, parent_id)`。
- `run(session_id, user_input, on_event=None)` REPL 循环；工具调用结果回填；最大步数。
- `create_subsession`、`reset`、`close`、`list_sessions`。
- 工具 `info/read/replace/append`，行为符合架构文档。
- `replace` 唯一性：0 次 / >1 次都报错；>1 返回次数与行号。
- `read`：1-based，`-1` 无界；越界截断；文件不在白名单报错。
- `append`：确保新行。
- 扩展：`register_tool/unregister_tool`、`add_prompt_contributor`、`set_system_prompt`。
- `FileWorkspace`：逻辑名 -> 路径；拒绝 `..`/绝对路径；原子写。
- `info` 汇总 `MetaRegistry` + 会话 + 文件 + 流长度。
- UI：右侧 tab `agent`；page `agent:chat`；action `agent.chat`、`agent.new_session`、`agent.select_session`、`agent.reset`。
- 测试用 `FakeLLM`（实现 `chat` 返回脚本化 tool_calls），不依赖网络。

不得修改 P0/P1/P2 文件。可以假设 settings 存在，但测试可注入 fake settings。

### P4 Summarize 插件（并行任务 D）

文件：

- `src/plugins/summarize/__init__.py`
- `src/plugins/summarize/plugin.py`
- `src/plugins/summarize/prompts.py`
- `tests/test_summarize.py`

要求：

- 依赖 `settings`、`llm-agent`。
- 每个会话创建 `knowledge.md` / `gossip.md` / `notice.md`，带固定头。
- 触发：增量文字 >= trigger_chars；或音频增量 >= trigger_audio_seconds 且文字 >= min_chars。
- 串行 worker；运行中增量留到下一轮；水位只在成功后推进。
- 调用 `agent` 服务创建子会话，工具限定 info/read/replace/append。
- Prompt 包含分类规则、Markdown、LaTeX 公式规范、时间戳、去重要求。
- 失败重试/退避；事件 `summarize.*`。
- UI：左侧 tab `classroom`（若不存在则创建）；toolbox `summarize.notes`；page `summarize:notes` 三个 tab + 「立即总结」。
- 测试：fake streams + scripted agent/LLM，验证触发、水位、文件内容、失败重试。

不得修改 P0/P1/P2/P3 文件。

### P5 输入插件（并行任务 E）

文件：

- `src/integrations/__init__.py`
- `src/integrations/sjtu_canvas.py`
- `src/plugins/input/__init__.py`
- `src/plugins/input/plugin.py`
- `src/plugins/input/catalog.py`
- `src/plugins/input/sources.py`
- `src/plugins/input/ffmpeg.py`
- `tests/test_input.py`

要求：

- `integrations/sjtu_canvas.py`：使用 vendored `integrations/sjtu_canvas_assistant`；封装 `CanvasClient` / `SJTUVideoClient`；每门课一个 client，避免 auth 串课；提供：
  - `list_courses()`
  - `list_recordings(course_id)`
  - `get_recording_url(course_id, video_id)`
  - `list_lives(start, end)`（新版 `/v1/vod_live`）
  - `get_live_devices(live_course_id)`（新版 `/v1/course_vod_videoinfos`）
- 直播/录播都映射为 `SourceSpec`，交给 `SourceManager`。
- `FFmpegSource`：两个子进程；音频 s16le 16k mono；视频 mjpeg 默认 4fps、保持原分辨率（不缩放）；JPEG 边界解析；`-re` 用于录播；直播低延迟；支持 stop/pause/resume；状态事件。
- UI：
  - 左侧 tab `courses`：toolbox `input.courses`（课程录播）。
  - 左侧 tab `live`：toolbox `input.live`（今日直播）、`input.manual`（手动 URL）。
  - 右侧 tab `session`：toolbox `input.session`（当前播放）。
  - 页面 `input:courses`：课程卡片；`input:course:<id>`：只展示录播表。
  - 页面 `input:live`：只展示正在/即将开始的直播。
  - 页面 `input:session`：状态、音视频时长、暂停/继续/停止。
  - 页面 `input:manual`：手动 URL 表单。
- 设置：`input.canvas_token`（secret）、`input.canvas_base_url`、`input.video_fps`、`input.video_width`、`input.audio_rate`、`input.playback_speed`、`input.live_device`、`input.reconnect`。
- 测试：JPEG 解析、SourceManager 单源互斥、catalog 映射（fake clients）、无 token 错误提示；不启动真实 ffmpeg（可用 `echo` 小样本或 fake source）。

不得修改 P0/P1/P2/P3/P4 文件。SJTU 客户端代码已 vendor 到 `src/integrations/sjtu_canvas_assistant/`。

### P6 STT 插件（并行任务 F）

文件：

- `src/plugins/stt/__init__.py`
- `src/plugins/stt/plugin.py`
- `src/plugins/stt/backends.py`
- `src/plugins/stt/audio.py`
- `tests/test_stt.py`

要求：

- 依赖 `settings`、`llm`。
- `OpenAITranscriptionBackend`：WAV + multipart `/audio/transcriptions`；解析 text/segments。
- `FasterWhisperBackend`：可选 import，缺依赖时明确报错。
- `none`：只接入音频流，不输出文字（默认）。
- Worker：按 chunk/overlap 切音频，追加 transcript；处理会话切换/结束 flush。
- 设置：provider/model/language/chunk_seconds/overlap/min_seconds/enabled/base_url/api_key。
- 事件 `stt.segment` / `stt.error`；UI 页 `stt:transcript`（实时文字流）。
- 测试：WAV 编码、切片游标、重叠去重、fake backend、会话切换。

不得修改其他插件文件。

### P7 VLM 插件（并行任务 G）

文件：

- `src/plugins/vlm/__init__.py`
- `src/plugins/vlm/plugin.py`
- `src/plugins/vlm/fusion.py`
- `src/plugins/vlm/prompts.py`
- `tests/test_vlm.py`

要求：

- 依赖 `settings`、`llm`。
- `fusion.fuse_frames(frames, ...)`：从 picture_parser 算法适配；可选的 crop/max_width 仅保留给单元测试，生产 VLM 插件固定 `crop=(0,0,1,1)`、`max_width=0`，即整帧原始分辨率，不做任何裁剪或缩放；帧数不足或解码/融合失败时直接抛错，不静默回退。
- Worker：读 video 窗口，融合，保存 JPEG，调用 `ctx.llm.chat(profile="vlm", ...)`，追加 visual。
- Prompt 继承原管线严格 LaTeX/示意图占位符/空黑板。
- 去重：`difflib.SequenceMatcher` 阈值。
- 设置：enabled/model/window_seconds/fps/max_frames/fusion/dedupe/prompt。
- UI：左侧 tab `classroom`（若不存在）；toolbox `vlm.board`；page `vlm:board` 显示最新融合图 + 文字流。
- 测试：合成帧融合、提示词、去重、worker with FakeLLM。

不得修改其他插件文件。

### P8 Web UI 与 runtime 集成（主 agent + 前端任务 H）

文件：

- `src/ui/server.py`
- `src/ui/static/index.html`
- `src/ui/static/app.js`
- `src/ui/static/styles.css`
- `src/app.py`
- `src/cli.py`
- `src/__main__.py`
- `tests/test_server.py`

要求：

- FastAPI app factory 接收 Runtime；lifespan 启停。
- 路由：`/`、`/api/ui/state`、`/api/ui/action`、`/api/artifacts/{path}`、`/ws`。
- 静态前端：左右多 tab + toolbox + 中间 pages；block 渲染；action 提交；WebSocket 刷新；MathJax。
- CLI：`serve`、`check`。
- 测试：TestClient 取 state、action、artifact 越界拒绝、WebSocket 可选。

## 2. 并行批次

> 每个 subagent 只写自己文件清单内的文件；不要改别人的文件；接口有疑问先在自己的代码里做适配层并记录 TODO。

- **批次 1（P0 完成后）**：A（LLM）、B（settings）、E（input，可先做 catalog/source，不依赖 agent）、F（STT）、G（VLM）。
- **批次 2（A/B 完成后）**：C（agent）、D（summarize）、H（前端静态）。
- **批次 3**：主 agent 集成、端到端测试、真实 Canvas 冒烟、文档收尾。

## 3. 统一约定

- Python 3.12，类型注解完整，`from __future__ import annotations`。
- 不使用 asyncio 作为核心并发模型；核心线程 + 锁；FastAPI 只做 Web 适配。
- 所有时间单位为秒（float）；文本流索引从 0 开始；文件行号从 1 开始。
- 日志用标准 `logging`；不要 print（CLI 除外）。
- 测试不访问网络；LLM 用 `httpx.MockTransport`；Canvas/SJTU 用 fake client。
- 所有新增 public 函数/类写 docstring。
- 不引入重型依赖（transformers/torch）到默认依赖；`faster-whisper` 作为 optional extra。
- 所有文件 UTF-8，LF。

## 4. 测试矩阵

| 模块 | 必测 |
|---|---|
| streams | append/read/len/wait_for/immutability/event coalescing/session switch |
| events | on/once/off/wildcard/priority/exception/flush/clock |
| plugins | topo order/cycle/missing dep/service registry/teardown |
| ui | tabs/items/pages/state/actions/open-close/error |
| settings | defaults/persist/coerce/secret/change/invalid |
| llm | chat/stream/tool calls/image/retry/Retry-After/stream no-retry-after-first-token |
| agent | tools read/replace/append/info, multi-session, subsession, max steps |
| summarize | trigger chars/audio, watermark, failure retry, three files |
| input | JPEG parse, source mutex, catalog mapping, live devices, no token |
| stt | WAV, chunk/overlap, session switch, fake backend |
| vlm | fusion synthetic, prompt, dedupe, fake LLM |
| server | state/action/artifact path/websocket |

## 5. 真实冒烟清单

1. `sjtu-skipper check`：拓扑顺序、插件加载、无异常。
3. `sjtu-skipper serve` + 真实 settings：
   - `input.courses` 能列出 13 门 Canvas 课程。
   - 打开「代数拓扑」页，能列出录播。
   - 选择一条录播，`FFmpegSource` 能持续追加音频/视频；STT/VLM 状态正常（无 API key 时显示错误但不崩）。
   - 今日/近期直播列表可查询；无直播时显示空状态。
4. 真实 LLM：用 DeepSeek key 跑一次 agent `info` + `read` + `append`（低成本），确认工具循环。
5. 真实 VLM：用一条录播的 20 s 窗口，确认融合图和板书文字。
6. 退出：无残留 ffmpeg 进程、线程正常 join。

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| 旧子模块用户改动被覆盖 | 已移入 git-ignored `legacy/`；运行时使用 vendored 客户端 |
| 直播 FLV 需要鉴权 header | `FFmpegSource` 支持 `-headers`；先用门户返回的 token URL，失败再带 `jwt-token`/`Referer` |
| 双 ffmpeg 音视频起点漂移 | 同一 `started_at` 基准；文档标注非严格 A/V sync；后续可换单进程双输出 |
| 实时融合太慢 | 降分辨率、降 fps、可选关闭配准/融合；窗口可配 |
| 流事件过多 | 音视频合并通知；文字流按片段 |
| 插件循环依赖 | 加载时检测并报错；`requires` 只写直接依赖 |
| Agent 长任务阻塞 UI | action 后台线程；会话锁；UI 事件刷新 |
| LLM 流式中断 | 首包前重试；首包后抛错并记录，调用方决定重试 |
| STT/VLM 无可用模型 | STT 默认 none 只消费音频不输出；UI 明确提示；不阻塞其他插件 |

## 7. 交付物

- `src/` 完整实现。
- `tests/` 测试套件。
- `docs/architecture.md`、`docs/implementation-plan.md`、`docs/plugin-api.md`、`README.md` 更新。
- `pyproject.toml`、`.gitignore` 更新。
- 可运行的 `sjtu-skipper serve` / `check`。
