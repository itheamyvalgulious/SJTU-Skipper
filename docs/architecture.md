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
6. **所有 LLM 请求走 LLM 层**。STT/VLM/Agent/Summarize 不允许各自直连 HTTP。

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
| `stt.segment` / `stt.error` | STT |
| `vlm.segment` / `vlm.error` | VLM |
| `summarize.triggered` / `summarize.updated` / `summarize.error` | 总结插件 |
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

- 服务名用点号命名空间：`settings`, `agent`, `input.catalog`, `stt`, `vlm`, `summarize`。
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

拓扑序由 `requires` 决定，不依赖注册顺序。`llm-agent` 依赖 `llm` + `settings`；`summarize` 依赖 `llm-agent`；`stt`/`vlm` 依赖 `llm`（复用 provider 配置）。

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

- `LLMService.chat(profile="agent" | "vlm" | "stt" | ..., ...)`。
- Profile 从 settings 读取：
  - `llm.base_url`, `llm.api_key`, `llm.model`, `llm.vlm_model`, `llm.timeout`, `llm.max_retries`, `llm.temperature`, `llm.max_tokens`。
  - `llm.extra_headers` / `llm.extra_body`（JSON 字符串）。
- 没有 settings 时回退环境变量：`LLM_BASE_URL`, `LLM_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`。
- 每次请求读取一次配置，设置变更后无需重启。

## 6. UI 层

### 6.1 布局模型

- 左侧边栏：多 tab；每个 tab 下有 toolbox（工具项列表）。
- 右侧边栏：多 tab；每个 tab 下有 toolbox。
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
{"type": "tabs", "tabs": [{"title","blocks": [...]}]}
{"type": "progress", "value": 0.5, "label": "..."}
```

- `render` 回调返回 block 列表（或 `PageContent`）。
- Markdown 在服务端用 `markdown-it-py` 渲染；`$...$` / `$$...$$` 保留给前端 MathJax。
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

### 6.4 Web 服务

- `GET /`：静态前端。
- `GET /api/ui/state`：完整 UI 状态 + 运行时状态。
- `POST /api/ui/action`：执行 action。
- `GET /api/artifacts/{path}`：受限于 data_dir 的静态文件（图片、笔记）。
- `WS /ws`：服务端推送 `{"type":"refresh","reason":...}`。
- 前端每 3 秒兜底轮询 state；WebSocket 断开自动重连。

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
- `get(key)` 返回真实值；`public_all()` 对 secret 返回 `"******"` 并带 `is_set`。
- 变更时发 `settings.changed`。
- UI：右侧「设置」tab，toolbox 按 group 分组；page 渲染表单，action `settings.save`。

### 7.2 llm（底层 LLM 插件）

- 服务名：`llm`（`LLMService`）。
- 注册 settings（见 5.4）。
- 注册 `openai` provider。
- 不注册 UI page；设置由 settings 插件统一渲染。

### 7.3 input（输入插件）

- 服务名：`input`。
- 设置项：Canvas token/base URL、SJTU 视频门户、音频采样率、视频 fps/宽度、播放速度、直播设备选择。
- 依赖 `settings`；使用 vendored `integrations/sjtu_canvas_assistant` 的 Canvas/VOD 能力；直播使用新版门户端点：
  - `GET {portal}/v1/vod_live`（时间范围直播列表）
  - `GET {portal}/v1/course_vod_videoinfos?courseId=...&playType=2`（FLV 设备流）
- 录播通道：
  - Canvas 课程列表 -> SJTU `list_course_videos` -> `get_video_info` -> `get_stream_url`。
  - 本机文件 / HTTP URL 也可作为录播源。
- 直播通道：
  - 今日/近期直播列表 -> 选择设备（老师/课件）-> FLV URL。
  - 也支持手动 HLS/RTMP/HTTP URL。
- `FFmpegSource` 同时起两个 ffmpeg 子进程：
  - 音频：`-vn -ac 1 -ar 16000 -f s16le pipe:1`
  - 视频：`-an -vf fps=4 -f image2pipe -vcodec mjpeg pipe:1`（默认保持原分辨率）
  - 录播加 `-re` 按实时速度；直播加低延迟参数。
  - 解析 JPEG SOI/EOI 边界，逐帧 `append_frame`。
- 一个 `SourceManager` 保证同一时刻最多一个源；支持 pause/resume/stop/seek（seek 仅录播/本地文件）。
- UI：
  - 左侧 tab「课程」：toolbox「课程录播」「我的课程」。
  - 左侧 tab「直播」：toolbox「今日直播」「手动播放 URL」。
  - 右侧 tab「播放控制」：toolbox「当前播放」。
  - 页面 `input:courses`：课程卡片与录播入口；`input:course:<id>`：只展示该课程的录播表。
  - 页面 `input:live`：只展示今日正在/即将开始的直播表。
  - 页面 `input:session`：播放状态、音频/视频时长、暂停/继续/停止。
  - 页面 `input:manual`：手动 URL 表单。
  - 录播与直播完全分离，不再把直播表嵌入课程录播页。
- 行为层只看到 `hub.current.audio/video`，看不到 Canvas/SJTU/ffmpeg 对象。

### 7.4 stt（语音识别插件）

- 服务名：`stt`。
- 设置项：provider (`none` / `openai` / `faster-whisper`)、model、language、chunk_seconds、overlap、min_seconds、enabled。
- 工作线程：读 `audio`，按 `chunk_seconds` 切片，写临时 WAV，调用后端，追加 `transcript`。
- 后端：
  - `OpenAITranscriptionBackend`：OpenAI 兼容 `/audio/transcriptions`。
  - `FasterWhisperBackend`：可选依赖，本地推理。
  - `none`：只接入音频流，不输出任何文字。
- 重叠去重：保留 0.3–0.5 s 重叠，结果做最长公共前后缀裁剪。
- 事件：`stt.segment`、`stt.error`。
- UI：左侧 tab「课堂」下的 toolbox「转写」；page 显示实时文字流和状态。

### 7.5 vlm（板书识别插件）

- 服务名：`vlm`。
- 设置项：enabled、model、window_seconds、fps、max_frames、fusion、dedupe_threshold、prompt。
- 工作线程：读 `video`，窗口内取帧 -> 融合 -> VLM -> 追加 `visual`。
- 融合算法来自 `picture_parser`：
  1. 解码 JPEG -> 灰度（保持原始分辨率，不裁剪、不缩放）；
  2. FFT 相位相关配准（可选）；
  3. 30 分位背景 + 时序活动/方差 + 空间梯度 + 粉笔候选掩码；
  4. 时序持续性 + 单调边缘滤波；
  5. 候选帧 nanmedian 融合。
- 窗口默认 20 s / 4 fps，整帧原始分辨率输入，不做裁剪或缩放；三个参数均可配置。
- Prompt 继承原管线：严格 LaTeX、示意图占位符 `![描述]()`、空黑板回复 `【黑板暂无板书】`。
- 去重：与上一条 visual 文本相似度高则跳过（`difflib.SequenceMatcher`，阈值默认 0.92）。
- 保存融合图到 `data_dir/sessions/<session>/vlm/`，UI 可预览。
- 事件：`vlm.segment`、`vlm.error`。

### 7.6 llm-agent

- 服务名：`agent`（`AgentService`）。
- 设置项：默认模型、最大步数、工具输出上限、系统提示词。
- 工具：
  - `info`：课程信息、日期、会话、流长度、可操作文件列表。
  - `read(file, l, r)`：1-based 行号，`-1` 表示该侧无界。
  - `replace(file, old_str, new_str)`：唯一替换；0 次或多次都报错，多次时返回出现次数和行号。
  - `append(file, text)`：在文件末尾新行追加。
- 文件：`FileWorkspace`，逻辑名 -> 路径；工具只允许访问会话注册的文件。
- 会话：
  - `AgentSession`：独立消息历史、系统提示、文件、工具、模型、最大步数、锁。
  - 可同时运行多个；`create_session` / `create_subsession` / `reset` / `close`。
- 扩展点：
  - `register_tool(ToolSpec)` / `unregister_tool`；
  - `add_prompt_contributor(callable)` / 会话级 `extra_instructions` / `set_system_prompt`；
  - `create_session` / `create_subsession`；
  - `run(session_id, input, on_event=...)` 同步；`run_async` 后台线程。
- REPL 循环：messages -> LLM -> tool calls -> tool results -> LLM -> ... -> final。
- 事件：`agent.run.started`、`agent.message`、`agent.tool_call`、`agent.tool_result`、`agent.run.finished`、`agent.error`。
- UI：右侧 tab「Agent」；page 显示会话列表、对话记录、输入框；action `agent.chat` 后台执行。

### 7.7 summarize

- 依赖：`settings`、`llm-agent`。
- 设置项：`trigger_chars`（默认 600）、`trigger_audio_seconds`（默认 120）、`min_chars`（默认 80）、`max_steps`、`model`、`notes_dir`、系统提示词。
- 每个会话创建三个文件：
  - `knowledge.md`：课程知识性内容；
  - `gossip.md`：与课程知识无关的内容（世界局势、生活趣事、对学生的期待、AI 对数学的冲击等）；
  - `notice.md`：作业、考试、考核等通知。
- 触发：
  - 监听 `stream.appended`（transcript / visual / audio）；
  - 文字增量达到 `trigger_chars`，或音频增量达到 `trigger_audio_seconds` 且已有足够文字；
  - 串行执行；运行中到达的增量在下一轮处理。
- 执行：
  - 取上次水位之后的文字（语音 + 板书），构造中文提示；
  - 创建子会话，工具限定为 `info/read/replace/append`，文件为三个笔记；
  - 要求 Markdown、`$...$` / `$$...$$`、每条带时间；
  - 要求先 `read` 再 `append`/`replace`，避免重复。
- 成功后推进水位；失败保留水位，指数退避重试；连续失败发 `summarize.error`。
- UI：左侧 tab「课堂」下的 toolbox「笔记」；page 三个 tab 展示文件，支持「立即总结」。

## 8. 数据与持久化

```text
data/agent/
├── settings.json                 # 设置（0600）
├── sessions/
│   └── 20261101-101500-ab12/
│       ├── session.json          # 元信息
│       ├── notes/
│       │   ├── knowledge.md
│       │   ├── gossip.md
│       │   └── notice.md
│       ├── vlm/
│       │   └── 000030-000060.jpg
│       ├── stt/
│       │   └── ...
│       └── events.jsonl          # 可选审计日志
└── cache/
    └── courses.json
```

- 流本身默认不落盘；如需回放，另加 recorder（后续）。
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
- UI 默认只监听 `127.0.0.1`；对外暴露需显式 `--host 0.0.0.0` 并自行加认证。
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
│   ├── summarize/
├── integrations/
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
6. **P5 STT/VLM**：后端、worker、picture_parser 融合、prompt、tests。
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
- 所有 LLM 请求（Agent、VLM、Summary）都经过 `LLMService`，无插件直连 HTTP。
