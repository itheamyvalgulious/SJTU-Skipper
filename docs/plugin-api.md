# 插件开发 API（v1）

本文是实现期的速查表；完整设计见 `docs/architecture.md`。

## 1. 最小插件

```python
from core.plugin import Plugin, PluginContext


class MyPlugin(Plugin):
    id = "my-plugin"
    requires = ("settings", "llm")   # 插件 id，不是服务名

    def setup(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.settings = ctx.require("settings")
        self.llm = ctx.llm            # LLMProxy，调用时解析 "llm" 服务

        ctx.ui.add_tab("right", "my-tab", "我的工具", icon="✦")
        ctx.ui.add_toolbox_item(
            "my-tab", "my-item", "快捷操作",
            render=lambda: [{"type": "text", "text": "hello"}],
        )
        ctx.ui.add_page("my:page", "我的页面", render=self.render_page)
        ctx.ui.register_action("my.action", self.on_action, plugin_id=self.id)
        ctx.events.on("clock.tick", self.on_tick, plugin_id=self.id)

    def render_page(self):
        return [{"type": "markdown", "text": "# 标题\n\n$E=mc^2$"}]

    def on_action(self, payload):
        return {"ok": True, "message": "done"}

    def on_tick(self, event):
        pass

    def teardown(self) -> None:
        pass
```

## 2. 流

```python
streams = ctx.streams.current
audio = streams.audio
video = streams.video
transcript = streams.transcript
visual = streams.visual

audio.duration          # 精确秒数
len(audio)              # 整秒（Python len 协议）
audio.read(0, 10)       # PCM s16le
audio.read_as_wav(0, 10)
audio.wait_for(10, timeout=2)

video.duration
video.read_frames(0, 30)  # list[VideoFrame(start,end,data)]
video.read(0, 30)         # 拼接 JPEG

len(transcript)           # 片段数
transcript[0].text
transcript.text_since(10) # 第 10 项之后的拼接文本
transcript.char_count_since(10)
transcript.snapshot_index()
transcript.append("文字", start=1.0, end=2.0, source="stt")
```

规则：

- 只能追加；已有片段不可变。
- 输入插件启动会话时调用 `ctx.streams.start_session(meta=...)`；行为插件不要调用。
- 音视频 `stream.appended` 是合并通知（默认 0.5s/1s）；文字流每次 append 都通知。

## 3. 事件

```python
ctx.events.on("my.event", handler, plugin_id=ctx.plugin_id, priority=0)
ctx.events.once("my.event", handler)
ctx.events.define("my.event", "说明")
ctx.events.emit("my.event", value=1)
```

内置事件：

- `clock.tick`
- `session.started` / `session.stopped`
- `stream.appended`
- `settings.changed`
- `ui.changed`
- `llm.request` / `llm.response` / `llm.error`
- `agent.run.started` / `agent.message` / `agent.tool_call` / `agent.tool_result` / `agent.run.finished` / `agent.error`
- `stt.segment` / `stt.error`
- `vlm.segment` / `vlm.error`
- `summarize.triggered` / `summarize.updated` / `summarize.error`
- `input.status` / `input.error`

## 4. 服务互调

```python
service = ctx.require("settings")
service.register(spec)
service.get("llm.model")

ctx.register_service("my.service", my_object)
other = ctx.require("summarize")
ctx.services.call("summarize", "trigger_now", reason="manual")
```

约定：

- 服务名点号命名：`settings`, `llm`, `agent`, `input`, `stt`, `vlm`, `summarize`。
- 插件之间不 import 实现；通过服务方法调用。
- `requires` 只保证 setup 顺序，调用时仍要处理服务缺失。

## 5. LLM

```python
response = ctx.llm.chat(
    profile="agent",                 # agent | vlm | stt
    messages=[{"role": "user", "content": "hello"}],
    tools=[...],                     # OpenAI tool schema
    temperature=0.2,
    max_tokens=1024,
)
response.content
response.tool_calls[0].name
response.tool_calls[0].arguments
response.usage

for event in ctx.llm.stream(profile="agent", messages=[...]):
    if event.type == "text":
        print(event.text)
```

- 所有 LLM 请求必须走 `ctx.llm`；不要自己 `httpx.post`。
- 重试/超时/流式解析由 LLM 层负责。
- 配置由 settings 插件持久化，profile 在每次请求时解析。

## 6. UI

```python
tab = ctx.ui.add_tab("left", "courses", "课程", icon="▤", order=10)
item = ctx.ui.add_toolbox_item("courses", "courses.list", "我的课程", render=render_list)
page = ctx.ui.add_page("input:course:123", "课程详情", render=render_page, closable=True)
page.open()
ctx.ui.refresh("reason")
```

Block：

```python
{"type": "markdown", "text": "..."}
{"type": "text", "text": "..."}
{"type": "keyvalue", "items": [["课程", "复分析"], ["日期", "2026-11-01"]]}
{"type": "table", "columns": ["节次", "时间"], "rows": [["第9节", "15:55"]],
 "row_actions": [{"label": "观看", "action": "input.play", "keys": ["video_id"]}]}
{"type": "cards", "items": [{"title": "...", "subtitle": "...", "actions": [...]}]}
{"type": "list", "items": [{"text": "...", "action": "x", "value": {...}}]}
{"type": "buttons", "buttons": [{"label": "刷新", "action": "input.refresh"}]}
{"type": "form", "fields": [
    {"name": "url", "label": "直播 URL", "type": "text", "placeholder": "https://..."}
], "action": "input.play_url", "submit_label": "开始"}
{"type": "log", "lines": [{"time": "12:00", "text": "..."}]}
{"type": "image", "src": "/api/artifacts/<session>/vlm/000030-000060.jpg"}
{"type": "tabs", "tabs": [{"title": "知识", "blocks": [...]}]}
{"type": "progress", "value": 0.42, "label": "处理进度"}
```

Action：

```python
ctx.ui.register_action("my.action", handler, plugin_id=ctx.plugin_id)

def handler(payload: dict):
    # payload 还包含 page_id / item_id
    return {"ok": True, "message": "...", "refresh": True, "open_page": "my:page"}
```

返回 `refresh=True` 会让前端重新拉取 state；长任务请放到后台线程，立即返回。

Web 层还预注册了通用 action `ui.open_page`，插件可以在按钮里这样用：

```python
{"type": "buttons", "buttons": [
    {"label": "打开笔记页", "action": "ui.open_page",
     "value": {"page_id": "summarize:notes"}}
]}
```

## 7. 设置

```python
from plugins.settings import SettingsSpec

settings = ctx.require("settings")
settings.register(SettingsSpec(
    key="my.api_key",
    label="API Key",
    type="secret",
    default="",
    group="My Plugin",
    description="...",
))
value = settings.get("my.api_key")
settings.set("my.api_key", "new")
```

- secret 在 UI/公开输出中掩码；内部 `get` 返回真实值。
- 设置变化发 `settings.changed`。

## 8. Agent

```python
agent = ctx.require("agent")          # llm-agent 插件注册的服务

session = agent.create_session(
    name="my-agent",
    system_prompt="你是课堂助手。",
    files={"notes.md": "/path/notes.md"},
    tools=["info", "read", "replace", "append"],
    max_steps=8,
    extra_instructions=["只修改 notes.md"],
)

result = agent.run(session.id, "请总结这一段", on_event=lambda e: print(e))
sub = agent.create_subsession(session.id, name="child", system_prompt="...")
agent.register_tool(ToolSpec(...))
agent.add_prompt_contributor(lambda session: "全局补充说明")
```

工具契约：

- `info()`：无参数，返回课程/日期/会话/文件/流状态。
- `read(file, l, r)`：1-based 行号；`l` 或 `r` 为 `-1` 表示该侧无界。
- `replace(file, old_str, new_str)`：唯一替换；0 次或多次都报错，多次时报告出现次数与行号。
- `append(file, text)`：在文件末尾新行追加。

## 9. 文件与数据目录

```python
ctx.data_dir                         # data/agent
ctx.session_dir()                    # data/agent/sessions/<current session>
ctx.session_dir("other-session")     # 指定会话
```

产物建议：

- 笔记：`notes/`
- VLM 融合图：`vlm/`
- STT 临时文件：`stt/`

## 10. 测试

```python
from config import load_config
from core.runtime import build_runtime

runtime = build_runtime(load_config(data_dir=tmp_path), plugins=[MyPlugin()])
runtime.start()
...
runtime.stop()
```

- 不访问网络；LLM 用 `tests/support.py` 的 `FakeLLM`。
- 事件异步分发，断言前调用 `runtime.bus.flush(2.0)`。
- 线程型插件在 `teardown` 中必须停止并 join。
