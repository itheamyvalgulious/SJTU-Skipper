# 课堂总 Agent 架构设计

> 状态：v1 设计稿（实现前冻结的接口契约）
>
> 目标：把课堂的音频、视频、语音识别、板书识别、LLM 调用、行为插件和 UI 拆成可独立演进的分层，行为层只通过「流 + 事件 + 服务接口」与底层交互。

## 1. 总体分层

```text
┌──────────────────────────────────────────────────────────────────────┐
│                            UI（Web）                                  │
│  左侧多 tab 边栏 / 右侧多 tab 边栏 / 每个 tab 下有 toolbox           │
│  中间 pages 区域 / 页面内 block 渲染 / action 回传                    │
└───────────────────────────────▲──────────────────────────────────────┘
                                │ UIManager（状态 + action）
┌───────────────────────────────┴──────────────────────────────────────┐
│                         行为层（Behavior）                            │
│  插件：llm-agent / summarize / 自定义行为插件                          │
│  能力：事件总线、流读取、LLM、UI、服务注册与调用                       │
│  依赖：只依赖「流接口 + 事件 + 服务接口 + LLM 接口」                   │
└───────────────▲───────────────────────────────▲──────────────────────┘
                │ TextStream(transcript)         │ TextStream(visual)
┌───────────────┴───────────┐     ┌─────────────┴──────────────────────┐
│  STT 插件                 │     │  VLM 插件（picture_parser 管线）    │
│  AudioStream -> 文字流    │     │  VideoStream -> 文字流              │
└───────────────▲───────────┘     └─────────────▲──────────────────────┘
                │ AudioStream                    │ VideoStream
┌───────────────┴────────────────────────────────┴──────────────────────┐
│ 输入插件（Input）                                                     │
│  录播通道：Canvas + SJTU 视频门户 VOD                                 │
│  直播通道：SJTU 视频门户 vod_live + FLV 设备流 / 手动 HLS/RTMP URL     │
│  ffmpeg -> PCM 音频块 + JPEG 视频帧                                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────┴──────────────────────────────────────┐
│ LLM 层（协议无关）                                                    │
│  Provider 注册表：openai-compatible / 未来 anthropic 等                 │
│  统一 ChatRequest / ChatResponse / StreamEvent                        │
│  重试、超时、流式 SSE、工具调用、图片输入、用量统计                    │
└──────────────────────────────────────────────────────────────────────┘
```

核心原则：

1. **数据平面与控制平面分离**。音视频/文字只通过 append-only 流传递；行为层不接触 Canvas、ffmpeg、SJTU 门户、STT/VLM 实现细节。
2. **插件只通过 ctx 暴露的能力交互**。`PluginContext` 是行为接口的唯一入口。
3. **流前缀不可变**。任何消费者只能追加、读取，不能修改已有片段。
4. **服务按名字发现**。插件之间不互相 import 实现，只通过 `services` 注册/调用。
5. **UI 是声明式的**。插件返回 block 列表和 action id，前端只负责渲染与回传，不执行插件代码。
6. **所有 LLM 请求走 LLM 层**。VLM/Agent/Summarize 不允许各自直连 LLM HTTP；STT 的 Google Speech-to-Text v2 是独立云 API（非 LLM），由 STT 插件用 OAuth 凭据直接调用。

## 2. 流数据平面

### 2.1 通用语义

- 所有流都是 **append-only**：已有片段永不被修改或删除。
- 时间轴使用秒（float），从当前会话开始计为 0。
- 每个流属于一个 `Session`；新会话创建全新的四个流，旧流保留只读（用于历史回放/导出）。
- 消费者使用自己的游标读取；流本身不维护「当前读者」状态。
- 读取超出当前末尾时：`read()` 截断到当前末尾；`wait_for()` 可等待到达指定长度。

### 2.2 AudioStream

```python
class AudioStream:
    sample_rate: int          # 默认 16000
    channels: int             # 默认 1
    sample_width: int         # 默认 2（s16le）
    session_id: str

    def __len__(self) -> int: ...            # 当前时长（整秒；len() 协议限制）
    @property
    def duration(self) -> float: ...         # 当前时长（精确到秒的小数）
    def append(self, pcm: bytes, duration: float | None = None) -> None: ...
    def read(self, start: float, end: float) -> bytes: ...
    def wait_for(self, end: float, timeout: float | None = None) -> bool: ...
```

- `read(l, r)` 返回该时间段的 **PCM 原始字节**（s16le，单声道）。
- Python 的 `len()` 必须返回 `int`，因此 `len(stream)` 是整秒时长；需要亚秒精度时用 `stream.duration`。
- `append` 的时长默认由字节数推算：`len(pcm) / (sample_rate * channels * sample_width)`。
- 输入插件按约 100 ms 一块追加。

### 2.3 VideoStream

```python
class VideoStream:
    codec: str = "mjpeg"
    width: int
    height: int
    fps: float
    session_id: str

    def __len__(self) -> int: ...            # 整秒；精确时长用 .duration
    @property
    def duration(self) -> float: ...
    def append_frame(self, jpeg: bytes, duration: float | None = None) -> None: ...
    def read(self, start: float, end: float) -> bytes: ...   # 拼接的 JPEG 帧字节
    def read_frames(self, start: float, end: float) -> list[VideoFrame]: ...
    def wait_for(self, end: float, timeout: float | None = None) -> bool: ...
```

- 视频流以 **JPEG 帧** 为存储单位。`read(l, r)` 返回区间内所有帧的 JPEG 字节拼接；需要逐帧时使用 `read_frames`。
- 每帧有 `start`、`end`、`data`、`index`。默认帧时长 `1 / fps`。
- 选择 JPEG 而不是 MP4 的原因：无需容器复用/解复用，VLM 管线可直接解码；代价是体积较大；默认 4 fps 以匹配原管线，可按需降低。

### 2.4 TextStream

```python
@dataclass(frozen=True)
class TextItem:
    index: int
    text: str
    start: float | None
    end: float | None
    source: str
    meta: Mapping[str, Any]

class TextStream(Sequence[TextItem]):
    session_id: str
    name: str                 # "transcript" | "visual" | 其他

    def __len__(self) -> int: ...             # 片段数
    def __getitem__(self, i) -> TextItem: ...
    def append(self, text, *, start=None, end=None, source="", meta=None) -> TextItem: ...
    def text_since(self, index: int) -> str: ...
    def char_count_since(self, index: int) -> int: ...
    def snapshot_index(self) -> int: ...
    @property
    def char_count(self) -> int: ...
```

- 文字流是「不断增长的列表」，列表项是 `TextItem`；`len()` 是片段数。
- 片段不可变；`index` 单调递增。
- `text_since(i)` 返回从第 i 项到末尾的拼接文本，便于行为层做增量总结。

### 2.5 StreamHub / StreamSet

```python
@dataclass
class StreamSet:
    session_id: str
    started_at: float
    meta: dict
    audio: AudioStream
    video: VideoStream
    transcript: TextStream
    visual: TextStream

class StreamHub:
    @property
    def current(self) -> StreamSet: ...
    def start_session(self, *, meta=None) -> StreamSet: ...
    def end_session(self, reason="") -> None: ...
    def list_sessions(self) -> list[SessionInfo]: ...
```

- 输入、STT、VLM 都从 `hub.current` 取当前流。
- 行为层通过 `ctx.streams.current` 读取，不调用 `start_session`。
- `start_session` 会发送 `session.started` 事件；`end_session` 发送 `session.stopped`。

### 2.6 流事件

为避免 100 ms 一次的音频追加把事件总线打爆，音视频流做 **合并通知**：

- `audio`：默认每新增 0.5 s 发一次 `stream.appended`。
- `video`：默认每新增 1.0 s 发一次。
- `transcript` / `visual`：每次 `append` 立即发一次。

事件 payload：

```python
@dataclass(frozen=True)
class StreamAppended:
    stream: str          # "audio" | "video" | "transcript" | "visual"
    session_id: str
    old_len: float | int
    new_len: float | int
    delta: float | int
    start: float | int | None
    end: float | int | None
    chars: int = 0       # 仅文字流
    first_index: int = 0 # 仅文字流
```

## 3. 事件总线与时钟

```python
class EventBus:
    def define(self, name: str, description: str = "") -> EventType: ...
    def on(self, type: str, handler: Callable[[Event], None], *, plugin_id="", priority=0) -> Subscription: ...
    def once(...)
    def off(self, subscription) -> None: ...
    def emit(self, type: str, **payload) -> None: ...
    def emit_event(self, event: Event) -> None: ...
    def flush(self, timeout: float = 5.0) -> bool: ...
    def start(self) / stop(self): ...
```

- 事件在独立 dispatcher 线程顺序分发；`emit` 只入队，不阻塞生产者。
- handler 异常被捕获、记录日志，并发出 `event.error`；不会中断其他 handler。
- 支持 `"*"` 通配监听。
- 同一事件、同一插件内按 `priority` 从高到低执行；不同插件按注册顺序。
- `flush()` 用于测试和优雅退出。

内置事件：

| 事件 | 触发 |
|---|---|
| `clock.tick` | 每秒一次，payload: `ts`, `tick`, `interval` |
| `session.started` | 新会话 |
| `session.stopped` | 会话结束 |
| `stream.appended` | 流追加（见上） |
| `settings.changed` | 设置项变化 |
| `ui.changed` | UI 状态变化（节流） |
| `llm.request` / `llm.response` / `llm.error` | LLM 层观测 |
| `agent.run.started` / `agent.message` / `agent.tool_call` / `agent.tool_result` / `agent.run.finished` / `agent.error` | Agent 运行 |
| `vlm.segment` / `vlm.error` | VLM |
| `summarize.started` / `summarize.finished` / `summarize.error` | 整堂课总结插件 |
| `instant-summarize.triggered` / `instant-summarize.updated` / `instant-summarize.error` | 增量总结插件 |
| `input.status` / `input.error` | 输入源 |

插件可以 `define()` 自己的事件类型（仅用于文档/校验），然后 `emit`。

## 4. 插件系统

### 4.1 插件协议

```python
class Plugin:
    id: str
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def setup(self, ctx: PluginContext) -> None: ...
    def teardown(self) -> None: ...
```

- `requires` 是插件 id 列表，形成 DAG。
- `PluginManager` 拓扑排序后依次 `setup`；循环依赖、缺失依赖在加载时报错。
- `teardown` 逆序执行。
- 插件可声明 `provides` 作为文档；实际服务注册用 `ctx.services.register()`。

### 4.2 PluginContext（行为接口）

```python
@dataclass
class PluginContext:
    plugin_id: str
    data_dir: Path
    events: EventBus
    streams: StreamHub          # 只读约定；写入仅限输入/STT/VLM
    llm: LLMService
    ui: UIManager
    services: ServiceRegistry
    meta: MetaRegistry
    logger: logging.Logger

    def service(self, name: str) -> Any: ...
    def register_service(self, name: str, obj: Any) -> None: ...
    def require(self, name: str) -> Any: ...
    def log(self, level, msg, **fields): ...
```

行为插件能看到的能力就是这些；不允许 `ctx.runtime` 之类的后门。

### 4.3 服务注册与插件互调

```python
class ServiceRegistry:
    def register(self, name: str, obj: Any, *, plugin_id: str) -> None: ...
    def get(self, name: str) -> Any | None: ...
    def require(self, name: str) -> Any: ...
    def call(self, name: str, method: str, *args, **kwargs) -> Any: ...
    def names(self) -> list[str]: ...
```

- 服务名用点号命名空间：`settings`, `agent`, `input.catalog`, `stt`, `vlm`, `summarize`, `instant-summarize`。
- 插件声明 `requires` 保证加载顺序；运行时调用前仍应处理服务不存在的情况。
- 服务对象的方法就是插件公开的 API；不要跨插件访问私有字段。

### 4.4 内置插件与加载顺序（示例）

```text
settings
llm
input
stt
vlm
llm-agent
summarize
```

拓扑序由 `requires` 决定，不依赖注册顺序。`llm-agent` 依赖 `llm` + `settings`；`summarize` 与 `instant-summarize` 依赖 `llm-agent`；`vlm` 依赖 `llm` + `settings`；`stt` 依赖 `settings`（注册 `stt.*` 设置项），后端为 Google Cloud Speech-to-Text。

## 5. LLM 层

### 5.1 统一模型

```python
@dataclass(frozen=True)
class TextPart: text: str
@dataclass(frozen=True)
class ImagePart:
    url: str | None = None
    data: bytes | None = None
    mime_type: str = "image/jpeg"
    detail: str = "auto"

@dataclass
class Message:
    role: str                 # system | user | assistant | tool
    content: str | list[TextPart | ImagePart]
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

@dataclass
class ChatRequest:
    model: str | None
    messages: list[Message]
    tools: list[ToolDef] = field(default_factory=list)
    tool_choice: str | dict | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict | None = None
    extra: dict = field(default_factory=dict)
    stream: bool = False

@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage
    raw: dict
    model: str

@dataclass
class StreamEvent:
    type: str                 # "text" | "reasoning" | "tool_call" | "done" | "error"
    text: str = ""
    tool_call: ToolCall | None = None
    response: ChatResponse | None = None
    error: Exception | None = None
```

### 5.2 Provider 接口

```python
class Provider(Protocol):
    name: str
    def chat(self, request: ChatRequest) -> ChatResponse: ...
    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]: ...
    def close(self) -> None: ...
```

首版实现：

- `OpenAICompatibleProvider`
  - 支持 `/chat/completions`、SSE 流式、工具调用、`image_url`（URL 或 data URL）。
  - base URL 归一化：`.../v1`、`.../v1/`、完整 `.../chat/completions` 均可。
  - 额外 header / body 可配置（如 `thinking: disabled`）。

Provider 通过 `ProviderRegistry` 注册，后续可加 Anthropic、Gemini native、vLLM 等。

### 5.3 重试与错误

```python
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 20.0
    jitter: float = 0.2
    retry_statuses: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)
```

- 连接错误、超时、上述 HTTP 状态自动重试。
- `Retry-After` 优先；指数退避 + 抖动。
- 流式请求：**在收到第一个有效增量之前**失败可重试；已开始输出后不再重试，抛 `LLMStreamError`。
- 所有重试记录 `llm.error` 事件，最终失败抛 `LLMRetryExhausted`。

### 5.4 Profile 与配置

- `LLMService.chat(profile="agent" | "vlm" | ..., ...)`。
- Profile 从 settings 读取：
  - `llm.base_url`, `llm.api_key`, `llm.timeout`, `llm.max_retries`, `llm.temperature`, `llm.max_tokens`。
  - 模型不放在 LLM 连接分组：Agent 使用 `agent.model`，VLM 使用 `vlm.model`，Summarize 使用 `summarize.model`，Instant Summarize 使用 `instant-summarize.model`。
  - `llm.extra_headers` / `llm.extra_body`（JSON 字符串）。
- 环境变量 `LLM_BASE_URL` / `LLM_API_KEY` / `OPENAI_*` / `DEEPSEEK_API_KEY` 会在设置注册时写入 `settings.json`，UI 显示实际生效值（密钥掩码），运行时只读设置。
- `llm.mode` 提供两种独立配置：
  - `openai`：使用 `llm.base_url` + `llm.api_key` 的 OpenAI-compatible 接口；
  - `opencode`：在设置页通过 OAuth device flow 关联 OpenCode 套餐，token 只保存在本应用 settings（不读取本机 OpenCode），自动刷新并附加 `x-opencode-org-id`；OpenAI-compatible 端点是 `https://opencode.ai/inference/go/openai/v1`（模型目录里声明的 chat-completions 地址）。
  - **会话亲和**：OpenCode 网关按 conversation id 把请求钉到同一后端，上游 prompt cache 才能命中。agent 每次调用都带自己的会话 id（子会话沿用父会话的根 id），请求头为 `x-opencode-session` / `x-session-affinity` / `X-Session-Id` / `x-opencode-session-id`（+ 父 id）；没有会话 id 的调用（VLM 等）退回 settings 里的固定 id。
- 用量统计：每次响应把 `usage` 记到 `LLMService.usage_by_model`（`requests` / `input_tokens` / `output_tokens` / `cache_read_tokens`），`llm.response` 事件带本次用量与累计值，`usage_snapshot()` 可给 UI 读。缓存命中数兼容三种形状：OpenAI `prompt_tokens_details.cached_tokens`、DeepSeek `prompt_cache_hit_tokens`、OpenCode `opencode.tokens.cache_read`；流式请求会自动加 `stream_options.include_usage`，否则最后一帧不带 usage。
- 思考内容：流式响应里的 `reasoning_content` / `reasoning` 以 `StreamEvent(type="reasoning")` 转发并拼进 `ChatResponse.raw`，正文与思考分开处理。
- 每次请求读取一次配置，设置变更后无需重启。

## 6. UI 层

### 6.1 布局模型

- 左侧边栏：多 tab；每个 tab 下有 toolbox（工具项列表）。
- 右侧边栏：多 tab；每个 tab 下有 toolbox。
- 两侧边栏都可以**整栏收起/展开**（顶栏的 ◧ / ◨ 按钮，状态存 localStorage）；收起后中央区域占满宽度。
- 中间：pages 区域，顶部 page tabs，下面是当前 page 内容。
- 页面内容由 block 列表组成，前端渲染。

```python
class UIManager:
    def add_tab(self, side: Literal["left","right"], id, title, icon=None, order=0) -> TabHandle
    def add_toolbox_item(self, tab_id, id, title, *, render=None, order=0, description="") -> ItemHandle
    def add_page(self, id, title, *, render, icon=None, order=0, closable=True) -> PageHandle
    def remove_page / remove_item / remove_tab
    def open_page(self, id) -> None
    def close_page(self, id) -> None
    def refresh(self, *, reason="") -> None       # 节流 + ui.changed
    def register_action(self, action_id, handler, *, plugin_id) -> None
    def dispatch_action(self, action_id, payload) -> ActionResult
    def state(self) -> dict
```

### 6.2 Block 类型

```python
{"type": "markdown", "text": "..."}
{"type": "html", "html": "..."}                  # 仅可信插件
{"type": "text", "text": "..."}
{"type": "keyvalue", "items": [[k, v], ...]}
{"type": "table", "columns": [...], "rows": [[...]], "actions": [...]}
{"type": "list", "items": [...]}
{"type": "buttons", "buttons": [{"label","action","value","style"}]}
{"type": "form", "fields": [{"name","label","type","value","options","placeholder","secret"}], "submit": {"label","action"}}
{"type": "log", "lines": [...]}
{"type": "image", "src": "/api/artifacts/...", "alt": "..."}
{"type": "video", "src": "..."}
{"type": "tabs", "tabs": [{"title","blocks": [...], "meta": {...}}]}
{"type": "progress", "value": 0.5, "label": "..."}
{"type": "switch", "label": "实时总结", "checked": true, "action": "instant-summarize.toggle", "hint": "..."}
{"type": "qa", "id": "qa-1a2b3c4d", "ts": "...", "status": "done|pending|error",
 "question": "...", "answer": "...", "question_html": "...", "answer_html": "...",
 "path": "...", "delete_action": "reader.qa_delete"}
```

- `switch`：布尔开关，切换后以 `{"enabled": <bool>}` 调用它的 `action`。
- `qa`：笔记里的内联问答对，渲染成可折叠的独立组件（提问 / 回答两侧配色不同），带一键删除按钮。
- `tabs[*].meta`：reader 用它告诉前端当前文档属于谁、选中菜单有哪些项（见 7.10）。

- `render` 回调返回 block 列表（或 `PageContent`）。
- Markdown 在服务端用 `markdown-it-py` 渲染；`$...$` / `$$...$$` 保留给前端 KaTeX。
- 前端不执行插件代码；只处理 `action`。

### 6.3 Action 协议

```json
POST /api/ui/action
{
  "action": "input.play_recording",
  "page_id": "input:course:93578",
  "item_id": null,
  "values": {"course_id": 93578, "video_id": "211449"}
}
```

响应：

```json
{"ok": true, "message": "已开始播放", "refresh": true, "open_page": "input:session"}
```

- 长任务（LLM、下载、网络请求）在后台线程执行；action 立即返回，通过 `ui.changed` 刷新。
- 插件 action 处理器必须捕获异常并返回 `{"ok": false, "message": ...}`。

### 6.4 流控与并发处理

- `core/streams.py` 的 `BacklogGate` 给 `audio`/`video` 各挂一道闸门：reader 在 append 前 `wait(已摄录时长)`，消费者提交窗口后 `release(窗口结束时间)`。录播全速解码时，未处理媒体被限制在 `窗口 × (并发 + 4)` 以内（1.5h 课程不再把 5GB 帧塞进内存）；直播闸门 limit=0，不生效。
- 闸门有 stall 看门狗（默认 300 s）：消费者挂死时自动放行并告警，避免 reader 永久阻塞。
- `core/pipeline.py` 的 `OrderedWindowPool`：线程池并发执行窗口，按窗口序号有序提交；单窗异常封装为 `WindowError` 交给提交回调决定重试/记录。
- 结束屏障：两个 ffmpeg reader 都 EOF（`session.input_done`）→ input 等待 `stt.busy()`/`vlm.busy()` 变 False → 才 `end_session`；`summarize` 在 `session.stopped` 上跑一次全量讲义。录播不触发 instant-summarize。

### 6.5 Web 服务

- `GET /`：静态前端。
- `GET /api/ui/state`：完整 UI 状态 + 运行时状态。
- `POST /api/ui/action`：执行 action。
- `GET /api/artifacts/{path}`：受限于 data_dir 的静态文件（图片、笔记）。
- `GET /api/media?src=...`：视频转发代理。只允许 `*.sjtu.edu.cn` 的 URL，透传 `Range`/`Content-Range`，浏览器不必能直连 `live.sjtu.edu.cn`（VPN 场景）；HLS 播放列表不走代理。
- `WS /ws`：服务端推送 `{"type":"refresh","reason":...}` 与 `{"type":"notify","topic","payload"}`（流式增量用，见 7.10）。
- 前端每 3 秒兜底轮询 state；WebSocket 断开自动重连。
- **登录**：所有 HTTP 路由与 `/ws` 都在签名 cookie 之后。`GET/POST /login` 是唯一公开路由，`POST /logout` 清 cookie；未登录时 `/api/*` 返回 401、页面 303 到 `/login`，WebSocket 直接 4401 关闭。
- 账号在 `data_dir/auth.json`：`users`（`password` 明文或 `password_hash` = `scrypt$salt$hash`）、`session_secret`、`session_ttl_hours`（默认 168）。文件不存在或没有用户时自动创建 `admin` + 随机密码，并把密码打印到日志一次；连续 5 次密码错误锁定该用户名 5 分钟。cookie 为 HttpOnly + SameSite=Lax，`X-Forwarded-Proto: https` 或 HTTPS 下自动加 Secure。

## 7. 内置插件

### 7.1 settings

- 服务名：`settings`。
- 设置项注册：

```python
SettingsSpec(
    key="llm.api_key",
    label="LLM API Key",
    type="secret",
    default="",
    group="LLM",
    description="...",
)
```

- 类型：`string`, `text`, `int`, `float`, `bool`, `select`, `secret`, `json`。
- 持久化到 `data_dir/settings.json`（原子写，权限 0600）。
- `get(key)` 返回真实值；前端表单不会收到 secret 真实值，只显示「已配置（N 字符）」；清除走 `settings.clear_secret`。`public_all()` 对 secret 返回 `"******"` 并带 `is_set`。
- 变更时发 `settings.changed`。
- UI：没有侧边栏 tab，只有一个设置 page（`settings:main`），从菜单栏 / 命令面板（Ctrl+P）打开；page 内按 group 分 tab 渲染表单，action `settings.save`。只有满足 `visible_when` 的字段才会出现。

### 7.2 llm（底层 LLM 插件）

- 服务名：`llm`（`LLMService`）。
- 注册 settings（见 5.4）。
- 注册 `openai` provider。
- 不注册 UI page；设置由 settings 插件统一渲染。

### 7.3 input（输入插件）

- 服务名：`input`。
- 设置项：Canvas token/base URL、SJTU 视频门户、音频采样率、视频 fps/宽度、本地保留录像节数（`input.keep_recordings`）、直播设备选择。
- 处理模式由课程类型决定：**直播 = 1:1 实时**，**录播 = 全速**（无 `-re`，无变速）；`session.meta["mode"]` 记录 `live`/`fast`。
- 依赖 `settings`；使用 vendored `integrations/sjtu_canvas_assistant` 的 Canvas/VOD 能力；直播使用新版门户端点：
  - `GET {portal}/v1/vod_live`（时间范围直播列表）
  - `GET {portal}/v1/course_vod_videoinfos?courseId=...&playType=2`（FLV 设备流）
- 录播通道：
  - Canvas 课程列表 -> SJTU `list_course_videos` -> `get_video_info` -> `get_stream_url`。
  - 本机文件 / HTTP URL 也可作为录播源。
- 直播通道：
  - 今日/近期直播列表 -> 选择设备（老师/课件）-> FLV URL。
  - 也支持手动 HLS/RTMP/HTTP URL。
- 录播先**下载到本地**（`sessions/<sid>/media/<name>.mp4`，`.part` → 原子重命名）再交给 ffmpeg，避免音频/视频两个进程各拉一遍远端；下载失败自动回退为直读远端。本地文件同时用于回看（`/api/artifacts/<sid>/media/<name>`，Starlette `FileResponse` 原生支持 Range）。
- `FFmpegSource` 同时起两个 ffmpeg 子进程：
  - 音频：`-vn -ac 1 -ar 16000 -f s16le pipe:1`
  - 视频：`-an -vf fps=4 -f image2pipe -vcodec mjpeg pipe:1`（默认保持原分辨率）
  - 录播全速解码（无 `-re`）；直播加低延迟参数（天然 1:1）。
  - 解析 JPEG SOI/EOI 边界，逐帧 `append_frame`。
  - 音频、视频两个 reader **都 EOF** 才发 `input.ended` 并置 `session.input_done`；否则全速模式下音频先结束会截断视频与 VLM 结果。
- 一个 `SourceManager` 保证同一时刻最多一个源；支持 pause/resume/stop/seek（seek 仅录播/本地文件）。
- UI：
  - 左侧只有一个可折叠的**课程树**（`tree` block）：课程 → 某次课（录播与直播同一条路径）。
    - 相邻的连续 2–3 节录播自动合并成一节，节点上标注「2 节」；同一门课连排的多节直播也按同样规则合并（间隔不超过 `input.max_gap_minutes`、同日、最多 3 节），展开时跳到「正在直播 / 下一节 / 最后一节」那一节的地址；
    - 正在直播的课程/节次带 `直播中` 徽章并默认展开，一键进入直播；每节课的状态由 begin/end 推导，树里记录最近的边界时刻，到点在时钟 tick 上自动重取课程树（不用手点「刷新」）；
    - 展开状态保存在插件内存，`input.toggle_course` 切换。
  - 右侧 tab「本节课」：toolbox「处理控制」。
  - 页面 `input:session`（标题「课堂」）：标题带 `直播`/`录播` 徽章。
    - **直播**：`player` block 并排播放两路视频——「课堂」（`viewNum=1`，黑板/讲台）与「屏幕」（`viewNum=2`，大屏共享）；没有第二路时显示占位。
    - **录播**：不渲染播放器，只显示下载/处理进度卡片（`progress` block）与「打开录像」链接（本地 mp4，新窗口播放），避免每秒进度刷新重建 `<video>` 打断播放。
    - 前端按 URL 后缀选择播放器：`.m3u8` → hls.js（Safari 用原生 HLS），`.flv` → mpegts.js，其余交给原生 `<video>`。
    - 除 `.m3u8` 外的媒体地址一律改写成 `/api/media?src=...`，由服务端转发；浏览器直连失败时 pane 里会显示红色提示。
    - 屏幕流默认静音，避免两路音频叠加。
  - 点击「课程 → 某次课」只**选中并加载**视频地址（`input.open_class`），不创建会话、不启动 ffmpeg、不自动播放。
  - 「开始处理」按钮（`input.start_processing`）才开始摄取与会话；默认暂停。启动在后台线程里完成，下载几分钟也不会阻塞 HTTP 请求。
  - 「暂停处理」（`input.pause_processing`）：录播暂停 reader（管道反压让 ffmpeg 自动停下）；直播停止摄取但页面继续播放直播画面，恢复时从直播边缘继续。
  - 录播自然播完由 `input.ended` 驱动：先等待 STT/VLM 把队列排空（`_wait_for_consumers`），再切下一节或 `end_session("playback finished")`，保证总结时所有窗口都已提交。
  - 处理完成后按 `input.keep_recordings` 只保留最近 N 节的 `media/` 目录，其余删除（讲义不受影响）。
  - 页面 `input:manual`：手动 URL（调试，仅从命令面板打开）。
- 行为层只看到 `hub.current.audio/video`，看不到 Canvas/SJTU/ffmpeg 对象。

### 7.4 stt（语音识别插件）

- 服务名：`stt`；依赖 `settings`；实现 `enabled()` / `busy()` 供 input 插件做"处理完成"屏障。
- 后端：Google Cloud Speech-to-Text **v2** REST 同步接口 `POST https://{location}-speech.googleapis.com/v2/projects/{project}/locations/{location}/recognizers/_:recognize`（`location=global` 时用 `speech.googleapis.com`）。
- 认证：**OAuth**（v2 明确拒绝 API Key）。支持粘贴 service-account JSON、authorized-user JSON（gcloud ADC 形状）或凭据文件；都不配置时回退 `GOOGLE_APPLICATION_CREDENTIALS` / ADC。`google-auth` 负责签名与刷新 token，HTTP 走 `httpx`。
- 设置项（分组「STT」）：`stt.enabled`（默认 true）、`stt.credentials_json`（secret）、`stt.credentials_file`（可用 `GOOGLE_APPLICATION_CREDENTIALS` 种子）、`stt.project_id`（留空取凭据中的 project）、`stt.location`（默认 `us`；chirp_3 支持 `us`/`eu`/`asia-northeast1`/`asia-southeast1`，**不支持 `global`**；若配成 `global` 会自动改用 `us` 重试）、`stt.model`（默认 `chirp_3`，另有 `chirp_2`/`long`/`short`/`telephony`/`telephony_short`）、`stt.language_code`（默认 `cmn-Hans-CN`）、`stt.chunk_seconds`（默认 20，限制 1–55 秒；v2 Recognize 硬上限 60 秒 / 10 MB）、`stt.enable_punctuation`、`stt.timeout`。
- **不使用「模型自适应」**（Model adaptation）：不上传 phrase sets / custom classes 预制名词；只使用 `stt.model` 指定的模型。
- 音频以 WAV 容器 + `autoDecodingConfig` 发送（chirp 模型首选），`results[*].resultEndOffset` 换算为片段起止时间，追加到 `transcript`（`source="stt"`，`meta` 含 model/confidence/languageCode）。
- **录播（`session.meta["mode"] == "fast"`）**：`stt.concurrency`（默认 4）个窗口并发请求，`core.pipeline.OrderedWindowPool` 保证结果按窗口顺序提交；单片 429/5xx/网络超时自动重试 3 次（指数退避），仍失败则记一次窗口缺口并继续。
- **直播**：单路顺序处理，保持低延迟。
- 会话结束或进程退出时 flush 剩余音频（≥1 秒；录播由并发池负责，事件线程不再 flush）。
- **持久化**：每提交一行就追加到 `sessions/<sid>/transcript.jsonl`（`start`/`end`/`text`），转写不再只活在内存里；summarize 直接读这个文件重建讲义材料。落盘失败只告警，不影响直播。
- **覆盖记录**：每次处理完一个区间就往 `sessions/<sid>/coverage.json` 记一段已处理音频；再次跑同一节课时这些区间会被跳过，直播结束后补跑录播也只会补缺口。
- 事件：`stt.segment`、`stt.error`。
- UI：仅调试 page（`stt:*`），不出现在侧边栏，只能从命令面板（调试 → 语音转写）打开；page 显示状态、模型、音频时长、已转写段数/字数与最近转写文本。

### 7.5 vlm（板书识别插件）

- 服务名：`vlm`。
- 设置项：enabled（默认 true）、model、window_seconds（默认 20，窗口以采样点为中心，前后各 10 s）、cadence_live_seconds（直播采样间隔，默认 180）、cadence_recording_seconds（录播采样间隔，默认 300）、history_windows（默认 3）、fps、max_frames（默认 12）、fusion、dedupe_threshold、完整提示词 prompt。VLM 连接与 API Key 复用 LLM 分组。
- 工作线程：读 `video`，窗口内取帧 -> 融合 -> VLM -> 追加 `visual`。
- **采样节奏**：直播每 3 分钟取一个 20 秒窗口，录播每 5 分钟取一个 20 秒窗口；窗口之间的视频帧不再需要，工作线程会边等边 `trim`，避免把整堂课解码后的 MJPEG 留在内存里。
- **顺序**：两种模式都串行处理（`vlm.concurrency` 保留但固定为 1），保证“最近几次转录”的顺序正确。
- 融合算法来自 `picture_parser`：
  1. 解码 JPEG -> 灰度（保持原始分辨率，不裁剪、不缩放）；
  2. FFT 相位相关配准（可选）；
  3. 30 分位背景 + 时序活动/方差 + 空间梯度 + 粉笔候选掩码；
  4. 时序持续性 + 单调边缘滤波；
  5. 候选帧鲁棒中位数融合。
- 窗口默认 20 s / 4 fps、每窗最多 12 帧；整帧原始分辨率输出，不做裁剪或缩放。
- **VLM 只做观察者**：prompt 要求只转录图中真实可见的内容，不推断、不补全、不纠正，看不清的字符写作 `□`；空黑板回复 `【黑板暂无板书】`。
- **增量转录**：最近 `history_windows` 次转录以纯文本放进 prompt（不附加历史图片）；模型只输出相对最近一次**新增 / 修改 / 擦除**的内容，完全一致时回复 `【无变化】`，插件直接跳过不写入。这样 `visual` 里保存的是板书变更日志，而不是每 20 秒重录一遍整块黑板。
- 去重：与上一条 visual 文本相似度高则跳过（`difflib.SequenceMatcher`，阈值默认 0.92）；输出后处理会折叠连续重复的公式块。
- 保存融合图到 `data_dir/sessions/<session>/vlm/`，UI 可预览。
- **持久化与覆盖记录**：同 STT——每条 visual 追加到 `sessions/<sid>/visual.jsonl`，已采样区间写入 `coverage.json`（`kind="video"`），补跑时跳过已处理区间并在等待期间 `trim` 掉不需要的帧。
- UI 按窗口/段落返回 markdown 块，KaTeX 逐块渲染，避免整页卡死。
- 事件：`vlm.segment`、`vlm.error`。

### 7.6 llm-agent

- 服务名：`agent`（`AgentService`）。
- 设置项：默认模型、最大步数、工具输出上限、系统提示词。
- 工具（每个工具由 `plugins/agent/tools.py` 的具名工厂构建，参数随调用方声明）：
  - `info`：课程信息、日期、会话、流长度、可操作文件列表。
  - `read(file, l, r)`：1-based 行号，`-1` 表示该侧无界；可配置 `max_lines`，单次超出会被截断。
  - `search(keywords, file=?)`：传入若干关键字，返回命中的 `(行号, 该行内容)` 列表；`file` 省略时查全部可读文件。
  - `replace(file, old_str, new_str)`：唯一替换；0 次或多次都报错，多次时返回出现次数和行号。
  - `append(file, text)`：在文件末尾新行追加。
  - `write(file, text)`：整文件覆盖（可选，默认不提供）。
- **Agent 注册接口**：行为插件用 `register_agent(AgentProfile)` 声明一个具名 agent，profile 里包含系统提示、模型、步数 / token / 工具输出上限、`llm_extra`、是否流式，以及**工具声明**（`AgentTool(name, params)`）。`create_session(agent="名字", ...)` 从 profile 创建会话；也可以直接传 `tools=[("read", {"max_lines": 50}), ...]` 在创建时声明。工具参数（例如 read 行数、search 命中上限）只出现在调用方的声明里，不散落在工具实现中。
- 文件：`FileWorkspace`，逻辑名 -> 路径；工具只允许访问会话注册的文件。
- 会话：
  - `AgentSession`：独立消息历史、系统提示、文件、工具、模型、最大步数、锁、是否流式。
  - 可同时运行多个；`create_session` / `create_subsession` / `reset` / `close`。
  - `final_answer_on_limit=True` 时，最后一步会撤掉工具并要求模型直接作答（只读问答用；写作 agent 不启用）。
- 扩展点：
  - `register_agent` / `get_agent` / `unregister_agent`；
  - `register_tool(ToolSpec)` / `unregister_tool`；
  - `add_prompt_contributor(callable)` / 会话级 `extra_instructions` / `set_system_prompt`；
  - `run(session_id, input, on_event=..., stream=?)` 同步；`run_async` 后台线程。
- REPL 循环：messages -> LLM -> tool calls -> tool results -> LLM -> ... -> final。`stream=True` 时走 provider 的流式接口，边生成边发 `agent.delta`（正文）和 `agent.reasoning`（隐藏思考），结束后仍拿到完整的 tool calls。
- 事件：`agent.run.started`、`agent.delta`、`agent.reasoning`、`agent.message`、`agent.tool_call`、`agent.tool_result`、`agent.run.finished`、`agent.error`。
- UI：没有侧边栏 tab，只有 Agent page（`agent:chat`），从菜单栏 / 命令面板打开；page 显示会话列表、对话记录、输入框；action `agent.chat` 后台执行。

### 7.7 summarize（非即时，整堂课）

- 依赖：`settings`、`llm-agent`。
- 设置项：`max_tool_output`（默认 200000，单次 read 上限）、`max_history_chars`（默认 0 = 不裁剪）、`max_tokens`（默认 32000）、`model`、`notes_dir`、`system_prompt`（写作提示词）、`review_enabled`（默认 true）、`reviewer_prompt`、`review_model`（空 = 与写作相同）、`review_max_steps`（默认 0 = 不设上限）、`review_max_tokens`（默认 32000）、`review_max_rounds`（默认 0 = 不设上限）、`retry_max`、`on_stop`（`direct` 直接总结 / `fill` 先补全缺课部分再总结 / `skip` 不生成）、`fill_wait_minutes`（默认 20）、`fill_max_attempts`（默认 6）、`auto_on_stop`（旧开关，`on_stop` 的兼容入口）。
- 每个会话创建三个笔记文件 + 一份只读材料：
  - `note_knowledge.md`：课程知识；
  - `note_gossip.md`：与课程知识无关的内容（世界局势、生活趣事、对学生的期待、AI 对数学的冲击等）；
  - `note_notice.md`：作业、考试、考核等通知；
  - `lecture_source.md`：整堂课的 transcript + visual，按时间合并（`[mm:ss] 讲授：…` / `[mm:ss–mm:ss] 板书：…`）。材料直接由 `transcript.jsonl` / `visual.jsonl` 重建（不是内存流），所以崩溃、提前结束、重复生成都不丢内容；板书是增量日志，prompt 里会提示按时间顺序重建板书状态。若本会话有 `instant_draft.md`（即时笔记合并稿），也会作为组织线索附上。
- 触发：不在课堂中运行。`session.stopped`（`on_stop=direct`）或 UI action `summarize.full` 手动触发；`on_stop=fill` 时先让 input 把本课的录播（可能分多段）补跑一遍，只处理 STT/VLM 没覆盖的时间区间（`coverage.json`），等待期间每 `fill_wait_minutes` 重试一次、最多 `fill_max_attempts` 次，补录结束后自动开始总结。补录产生的独立会话带 `meta.fill_for` 指向原直播会话，讲义会把两边日志按 `media_begin` 对齐合并；超时则退化为直接总结，并在笔记页留下可见警告和「重新补全」按钮（`summarize.fill_again`）。
- 执行（**写作 + 审阅两个独立 agent 上下文**）：
  - **写作 agent**：把整堂课写成 `lecture_source.md`，三份笔记与材料一起放进工作区；先给一条任务提示，之后由模型自己 `info` → `read` 分段读材料 → `append`/`replace` 写笔记。会话 **不设步数上限**（`max_steps=0`）、不裁剪工具结果（`max_history_chars=0`）、`reasoning_effort=max`、`max_tokens=32000`。工具里**不提供 `write`**，只允许追加与局部替换。
  - **审阅 agent**：独立会话、只读工具（`info`/`read`）、同一批文件。检查四件事：老师讲没讲（逐条对照 `lecture_source.md`）、数学正确性、结构完整度、公式与格式；最后输出 `REVIEW: PASS` 或 `REVIEW: FAIL`。
  - **迭代**：审阅不通过就把问题清单原样发回写作 agent，写作 agent 只对问题处做 `replace`/`append`；审阅 agent 带着上一轮的意见继续审，直到 PASS。两个上下文全程不重置。
  - **安全阀**：审阅返回 FAIL 但没有问题清单、写作 agent 收到意见后文件 md5 没变、审阅连续两轮给出完全相同的问题，三者任一触发即记日志停止，避免空转。
  - 关闭审阅时退回旧行为：一次写作，若三份笔记均无变化则用 `RETRY_PROMPT` 重试一次。
- 失败发 `summarize.error`；完成后发 `summarize.finished`。
- UI：右侧「笔记」tab 的 toolbox「课堂讲义」；三个按钮（知识 / 闲话 / 通知）通过 reader 服务打开 `note_*.md`；「生成完整讲义」触发 `summarize.full`；运行中任务条会显示当前阶段（撰写 / 审阅 / 修改）。
- **选中提问** 已迁移到独立的 `ask` 插件，见 7.8。

### 7.8 ask（笔记内问答）

- 依赖：`settings`、`llm-agent`、`reader`；服务名 `ask`。
- 选中菜单：向 reader 注册**全局**菜单项「💬 提问」（不绑定单个页面 / owner，任何笔记页都能用）；弹窗里选择强度：**快速**（instant）或**研究**（research）。
- 两种强度都自动塞入选中位置前后各 2000 词（`ask.before_words` / `ask.after_words`，可配），并把选中片段、历史问答一起组成提示词。模型 `reasoning_effort=max`。
  - **快速**：不注册任何工具，`max_steps=1`，一步直接出结果；输出上限 `ask.instant_max_tokens`（默认 16000）。
  - **研究**：注册 `search` + `read(max_lines=ask.read_max_lines)`（默认 50 行）；提示词要求先 `search` 再针对性 `read`，并写明单次 read 行数上限。输出上限 `ask.research_max_tokens`（默认 131072），步数上限 `ask.research_max_steps`（默认 24，最后一步强制直接作答）。
- 流程：
  - reader 的确定性写入器先在选中段落下方插入 `pending` 问答块（meta 里带 `intensity`），用户立即看到「思考中」；
  - 后台线程通过 agent 插件注册 profile（`ask.instant` / `ask.research`）并创建流式会话；`agent.delta` / `agent.reasoning` / 工具调用被转发成 `ui.notify` 的 `qa.live` 消息，前端在问答块内实时展示思考、工具调用和正文，**全程不触发整页刷新**；
  - 结束时 reader 把最终回答写回问答块，meta 里带 `collapsed=true`（自动折叠）与 `status`；前端收到最终 `qa.live` 后原地替换为渲染好的 HTML 并折叠；
  - 失败时 `status=error`，块保持展开显示原因。
- **折叠持久化**：折叠状态记录在问答块 meta 的 `collapsed` 字段里；前端点折叠按钮时本地切换并调用 `reader.qa_collapse`（`refresh=false`，不触发重渲染），状态随笔记文件保存。
- **流式期间的滚动**：问答块在视口上方时，前端在每次内容增长后按高度差补偿 `scrollTop`，保证屏幕内其它内容不动；整页重渲染（轮询）时也按同样的高度差补偿。
- 日志：每一步的耗时、工具调用参数、可见正文，以及**完整的思考（reasoning）文本**都会写入日志，不截断。

### 7.9 instant-summarize（即时，增量）

- 依赖：`settings`、`llm-agent`。
- 设置项：`enabled`（默认 true，见下方开关）、`trigger_chars`（默认 100000，实际由时间触发）、`trigger_audio_seconds`（默认 600 = 10 分钟）、`min_chars`（默认 80）、`max_steps`（默认 12）、`max_tool_output`（默认 24000）、`model`、`notes_dir`、`system_prompt`、`retry_max`。
- **开关（直播模式）**：笔记面板里是一个 `switch`（`instant-summarize.enabled` / action `instant-summarize.toggle`）。**开**：照常累积并定期总结；**关**：水位直接推到当前时刻——不累积任何内容、不触发总结、会话结束也不 flush，重新打开从打开时刻开始累积。
- 每个会话创建三个文件：`instant_note_knowledge.md`、`instant_note_gossip.md`、`instant_note_notice.md`（与 summarize 共用 `notes/` 目录，互不覆盖）。
- 触发：
  - 监听 `stream.appended`（transcript / visual / audio）；
  - 默认每 10 分钟音频触发一次（`trigger_audio_seconds=600`）；文字增量达到 `trigger_chars` 也可触发，但默认值很大，几乎只按时间走；
  - 串行执行；运行中到达的增量在下一轮处理。
- 执行：
  - 取上次水位之后的文字（语音 + 板书，不注入时间戳），构造中文提示；
  - 创建子会话，工具为 `info/read/replace/append/write`；
  - 要求把新内容**合并进既有大纲**：结构必须清晰，但以增量维护为主——已有章节用 `replace` 补充，新章节用 `append`，只有确需调整结构时才重排受影响章节，不整篇重写；
  - 明确禁止压缩细节、禁止时间戳。
- 成功后推进水位；失败保留水位，指数退避重试；连续失败发 `instant-summarize.error`。
- UI：右侧「笔记」tab 的 toolbox「实时笔记」：一个「实时总结」开关 + 三个文档按钮（通过 reader 服务打开 `instant_note_*.md`）。没有「立即总结」按钮——总结只按时间/阈值自动触发。

### 7.10 reader（文稿阅读）

- 依赖：无（内置插件，始终加载）。
- 服务：`reader.show(title, docs=[{key,label,path}], active=...)`。
- 页面：每个笔记会话一个独立页 `reader:notes:<session_id>`（`reader:main` 作为无会话时的兜底），按文档分 tab 渲染 Markdown（`split_markdown` 分块 + KaTeX）。多门课的讲义可以同时打开、各自独立。
- 实时刷新：agent 每次 `write` / `append` / `replace` 成功后都会 `ui.refresh`，前端通过 WebSocket 立刻重取状态，正在阅读的笔记随写随更新。
- summarize / instant-summarize 的「笔记」工具箱按钮调用各自 action → `reader.show(..., page_key=<notes session>, owner=<plugin id>)`，每门课一个页；不再所有课程共用一页。
- **选中菜单**：用户选中页面正文后会浮出一个小菜单，项由 `reader.register_selection_action(page, item, owner=...)` 注册。`page` 可以是页面 id / page key（按 owner 过滤，只在对应插件提供的文档上出现），也可以是空串 / `"*"`（全局项，任何笔记页都出现）。`ask` 用全局注册提供「💬 提问」。item 形如 `{id,label,icon,kind:"prompt",action,placeholder,intensities}`；`kind="prompt"` 时前端先弹输入框（含强度选择）再连同选中文本一起 post 给插件 action。
- **内联问答块（qa）**：问答对直接写进笔记文件，用一对 HTML 注释包住：`<!-- qa:begin {"id","ts","status","intensity","collapsed"} -->` … `<!-- qa:end -->`，中间是 `**❓ 提问（时间）**`、问题、`**🤖 回答**`、回答（都是 Markdown）。普通 Markdown 阅读器里问题和回答照常可见，只是标记隐藏；reader 把它从 Markdown 流里摘出来渲染成独立组件（两侧不同配色、可折叠、一键删除）。`collapsed` 就是折叠状态的持久化位置（`reader.qa_collapse` 负责改它，`refresh=false`，不触发重渲染）；删除由 `reader.qa_delete` 做确定性的文本切除。reader 还负责把浏览器选中的渲染文本定位回原始 Markdown 段落（精确 → 去空白/去标记 → 行前缀逐级回退），定位失败才附到文末并提示。
- **流式问答**：问答进行中，`ask` 通过 `ui.notify("qa.live", ...)` 把正文/思考增量、工具调用和最终 HTML 推到浏览器；前端按 `data-qa-id` / `data-qa-path` 原地 patch 该块，不做整页刷新。页面轮询命中同一条消息时，key 里忽略 `collapsed`，折叠切换不会触发重渲染。



```text
data/agent/
├── settings.json                 # 设置（0600）
├── auth.json                     # 登录账号 + 会话签名密钥（0600）
├── sessions/
│   └── 20261101-101500-ab12/
│       ├── session.json          # 元信息
│       ├── coverage.json         # 已处理的音频/视频时间区间（断点续跑与补录）
│       ├── transcript.jsonl      # STT 已提交的转写行（每行一条）
│       ├── visual.jsonl          # VLM 已提交的板书增量
│       ├── media/                # 录播本地副本（只保留最近 N 节）
│       ├── notes/
│       │   ├── note_knowledge.md
│       │   ├── note_gossip.md
│       │   ├── note_notice.md
│       │   ├── instant_note_knowledge.md
│       │   ├── instant_note_gossip.md
│       │   ├── instant_note_notice.md
│       │   └── lecture_source.md      # 讲义材料（按时间合并的转写 + 板书）
│       ├── vlm/
│       │   └── 000030-000060.jpg
│       ├── stt/
│       │   └── ...
│       └── events.jsonl          # 可选审计日志
└── cache/
    └── courses.json
```

- 流本身默认不落盘；`transcript.jsonl` / `visual.jsonl` 是转写与板书的持久化副本，进程崩溃或提前结束后重新生成讲义仍然有材料可用。
- 每次会话目录由 `SessionStore` 创建；插件通过 `ctx.data_dir / "sessions" / session_id` 写入自己的产物。
- 旧 `data/outputs`、`data/notes` 保留，不被新架构修改。

## 9. 并发模型

- 事件总线：单 dispatcher 线程，顺序分发。
- 时钟：独立线程，每秒 emit。
- 输入源：每个源两个 ffmpeg 子进程 + 两个读线程（音频/视频）。
- STT/VLM：各自一个 worker 线程，按游标消费流。
- Agent：调用方线程 / 后台线程执行；每个会话有锁，防止并发 run。
- Summarize：独立 worker 线程，串行执行 agent。
- Web：FastAPI/uvicorn 线程池处理 action；UI 状态有锁。
- 所有跨线程共享状态用 `threading.RLock`；流读写用锁 + 不可变 chunk 列表。
- 退出：先停输入源，再停 STT/VLM/Summarize worker，再停插件，最后停时钟/事件总线。

## 10. 错误处理与可观测性

- 每个插件有独立 logger：`lecture_agent.plugin.<id>`。
- 结构化日志：`event`, `plugin`, `session_id`, `error`, `elapsed_ms`。
- 事件 `*.error` 可在 UI 状态栏查看。
- LLM 错误分类：认证、限流、超时、服务端、协议解析、流中断。
- 输入源断开：直播自动重连（指数退避，最大次数可配）；录播结束发 `input.status`。
- STT/VLM 单窗口失败：保留游标并指数退避重试同一窗口，同时发出 `*.error` 并在 UI 显示；不跳过窗口、不静默回退。
- Agent 工具异常：作为 tool 结果返回给模型，不中断循环；连续工具错误超过阈值则终止。

## 11. 安全与隐私

- Token/API key 只存 `settings.json`（0600），UI 掩码显示。
- 日志不打印 token、URL query 中的签名；使用 vendored `integrations/sjtu_canvas_assistant` 的脱敏工具。
- 文件工具限制在会话 workspace，拒绝 `..` 和绝对路径。
- artifact 路由只允许 `data_dir` 下的路径。
- UI 默认只监听 `127.0.0.1`；`--host 0.0.0.0` 时仍受 `auth.json` 登录保护（HTTP + WebSocket 同一道闸门）。
- 单元测试不访问外网；生产代码不包含 mock provider / mock STT 后端。

## 12. 目录结构

```text
src/                       # 新架构直接位于 src 下，不再有 lecture_agent 子目录
├── __main__.py
├── app.py                 # Runtime 构建 + FastAPI app factory
├── cli.py                 # serve / check
├── config.py              # AppConfig、路径、.env
├── applog.py
├── core/
│   ├── events.py
│   ├── streams.py
│   ├── services.py
│   ├── plugin.py
│   ├── ui.py
│   ├── meta.py
│   └── runtime.py
├── llm/
│   ├── types.py
│   ├── errors.py
│   ├── provider.py
│   ├── openai_compat.py
│   └── service.py
├── plugins/
│   ├── builtin.py
│   ├── settings/
│   ├── llm_core/
│   ├── input/
│   ├── stt/
│   ├── vlm/
│   ├── agent/
│   ├── summarize/          # 非即时：整堂课一次性整理
│   ├── instant_summarize/  # 即时：增量合并
├── integrations/
│   ├── opencode_go.py     # OpenCode 套餐 OAuth device flow / token 刷新
│   ├── sjtu_canvas.py     # 客户端池 + 目录封装
│   └── sjtu_canvas_assistant/  # vendored SJTU Canvas/视频客户端
├── ui/
│   ├── server.py
│   └── static/{index.html,app.js,styles.css}
└── util/
    ├── paths.py
    └── text.py
tests/
docs/
```

## 13. 实施顺序

1. **P0 基础**：streams / events / services / plugin / ui / runtime / config / logging。
2. **P1 LLM**：types / provider / openai-compatible / retry / service / tests。
3. **P2 行为基础**：settings / llm-core / llm-agent（含工具与会话）/ tests。
4. **P3 总结**：summarize + scripted LLM 集成测试。
5. **P4 输入**：SJTU catalog（VOD + live）/ ffmpeg source / UI pages / tests。
6. **P5 STT/VLM**：STT 后端（Google Speech-to-Text v2，OAuth）、worker；VLM picture_parser 融合、prompt、tests。
7. **P6 UI**：FastAPI + 静态前端 + WebSocket + 端到端演示。
8. **P7 收尾**：README、架构文档、真实 Canvas 冒烟、代码审查。

## 14. 非目标（v1 不做）

- 不做多用户/权限系统。
- 不做视频持久化回放（只保留会话产物）。
- 不做实时字幕的逐字低延迟（按块离线识别）。
- 不做课堂内容自动推送到 Canvas。
- 不依赖旧的 `legacy/sjtu-canvas-assistant`；输入模块使用 vendored 客户端。
- 旧脚本和旧子模块统一移入 git-ignored 的 `legacy/`；新 VLM 插件复用其算法，但运行时不依赖 `legacy/`。

## 15. 验收标准

- `pytest` 全绿；核心模块单测覆盖流、事件、插件、设置、LLM 重试、Agent 工具、总结、UI、VLM 融合。
- `sjtu-skipper check` 输出正确的拓扑加载顺序。
- `sjtu-skipper serve` 能加载真实 settings；输入插件能列出 Canvas 课程与 SJTU 录播/直播（有 token 时）。
- 输入、STT、VLM 三个模块除流接口外没有互相 import；行为插件没有 import 输入/STT/VLM 实现。
- 所有 LLM 请求（Agent、VLM、Summary）都经过 `LLMService`，无插件直连 LLM HTTP；STT 只直连 Google Speech-to-Text v2（非 LLM，OAuth）。
