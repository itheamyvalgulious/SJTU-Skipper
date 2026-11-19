# SJTU-Skipper / 课堂总 Agent

SJTU-Skipper 将音视频输入、STT、VLM、LLM、行为插件和 Web UI 分层组织．行为层通过追加流、事件和服务接口访问底层能力．

设计与接口契约见：

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/plugin-api.md`](docs/plugin-api.md)

## 快速开始

### 1. 环境

环境需要 Python 3.13、Bun >=1.4.4、`ffmpeg`/`ffprobe` 和 `uv`．

在仓库根目录创建 `.venv`：

```bash
nix develop
uv sync --locked --extra dev
bun install --frozen-lockfile
```

devShell 提供 Python、Bun、uv、ffmpeg 和动态库路径．非 Nix 系统需安装这些工具，再执行上述依赖安装命令．生产运行从项目 `node_modules` 加载 OMP SDK，版本由 `package.json` 和 `bun.lock` 固定．

### 2. 真实运行

```bash
./start.sh check   # 检查插件拓扑加载
./start.sh         # 启动 Web UI（等价于 serve）
./start.sh --port 9000
```

`start.sh` 检查 Bun 和项目依赖，再设置 `PYTHONPATH=src` 并启动 Python．修改源码后重启服务即可．

服务必须在前台运行才能接收 Ctrl+C：直接在终端执行上面的命令，不要加 `&`、`nohup` 或 `setsid`．后台 shell 启动的进程可能无法接收终端的 Ctrl+C 信号．前台运行时，Ctrl+C 会在 5 秒内退出（`SHUTDOWN_TIMEOUT`，见 `src/config.py`）；再次按 Ctrl+C 可强制退出．

打开页面后：

1. 首次启动生成 `data/agent/auth.json`，用户名为 `admin`，随机密码打印在启动日志中．修改密码后重启服务．浏览器使用签名 cookie 登录，登录有效期由 `session_ttl_hours` 控制．
2. 在「设置」填写 Canvas Token．在「STT」分组选择服务供应商：兼容 HTTP/WebSocket 服务填写地址、语言和课程上下文；Google Cloud Speech-to-Text v2 填写凭据、项目与模型．
3. 在「LLM」按页签配置：「概览」看待处理项，「提供方」接入端点并完成 DeepSeek API Key 或 OpenCode Go 浏览器授权，「模型」先检查端点再挑模型、逐个调整上下文窗口与输入格式，「角色」分配任务模型．
4. 展开左侧课程树，选择录播或直播并开始处理．
5. 在课堂页查看进度和识别内容，在「笔记」面板打开笔记或 Agent 日志．

桌面端左右面板可通过标题栏的 × 收起，通过拖动或双击分隔条调整宽度．窄屏通过「课程」「详情」按钮打开抽屉，按 Esc 关闭．右侧顶部显示选中的课堂．

处理课程时可以选择其他课程．点击「开始处理」将课程加入队列，直播优先．媒体输入按课程排队，语音、板书和笔记任务可同时执行．

直播与录播通过所选供应商转写．兼容服务连续发送最多 200 ms 的音频帧，直播显示 partial，只有 final 写入转写记录；录播使用 offline 模式全速传输，服务负责声学切分、上下文重叠及文本合并，课程名生成一句转写提示．Google Cloud v2 使用 REST 批量识别，按 `stt.window_seconds` 窗口提交，只产生定稿．板书识别按顺序提交，`vlm.prepare_ahead` 控制下个窗口的图像准备．课堂页显示总进度、下载、STT、VLM 和分段笔记进度．`input.keep_recordings` 控制本地录像保留数量．

多节录播共用连续时间轴，进度按各节时长之和计算．处理本节时预下载下一节，节间不等待语音或板书分析清空．全部节次结束后才完成输入尾段．

录播下载与音视频解码共用本地字节缓存．支持 Range 的来源可在完整下载前开始处理．不支持 Range 且索引位于文件尾时，解码等待下载．完整录像下载成功后才显示本地播放链接．

LLM 模型和角色配置保存在 `data/agent/settings.json` 的 `llm.models` 与 `llm.roles` 中．设置页提供图形化编辑与 YAML 编辑器．运行时从 JSON 生成临时文件，退出后清理；LLM 对话不跨服务重启恢复．

设置页按「概览 / 提供方 / 模型 / 角色 / 原始配置」组织，每页只处理一件事：

- 「概览」列出默认模型、提供方与模型数量、缺少的凭据和角色分配；待处理项直接指向要改的页面．
- 「提供方」一张卡片一个端点，端点地址、协议、鉴权与 API Key 在同一张表单里保存；内置预设 `opencode-go`（`https://opencode.ai/inference/go/openai/v1`，OAuth）与 `deepseek`（API Key），也可填写自定义 OpenAI 兼容端点．新提供方保存时请求 `${baseUrl}/models`，把返回的模型写入配置．
- OpenCode Go 用设备授权：点「关联套餐」后，授权链接、验证码与有效期固定显示在「提供方」页顶部（新标签页打开，可复制），并带「重新开始授权」「取消授权」．服务端通常没有浏览器，所以不会替客户端打开页面；授权完成后页面自动刷新．
- 「模型」先选提供方，再「检查端点模型」读取 `${baseUrl}/models`：只列出本地尚未添加的模型，可逐个或全部添加；已配置模型以表格展示上下文、输出、输入格式、推理能力与使用角色，选中一行在下方编辑器里修改，删除前会提示仍在使用该模型的角色．
- 「角色」一表列出设置值与生效模型；选中某个角色后在下方选择模型与思考级别．`board_recognition` 会校验 `image` 输入，缺少时拒绝保存．
- 「原始配置」保留两份 YAML，整体替换并交由运行时校验．
- 端点未公布的限制按默认值落库（上下文 128000、输出 8192、`text` 输入），随后在「模型」逐个调整．OpenCode 等端点公布 `limit.context`、`limit.output`、`modalities.input` 与 `capabilities.reasoning` 时按其填写．
- `llm.models` 使用 OMP `providers` 结构．提供方配置 `baseUrl`、`api`、鉴权和 headers，模型配置 `contextWindow`、`maxTokens`、`input`（`text`/`image`）与 `reasoning`．`api` 接受已安装 OMP SDK 支持的协议，包括 `openai-completions` 和 `openai-responses`．
- `llm.roles` 将职责映射到 `provider/model:level` 或 `@role`．角色包括板书、分段、笔记写作、审阅、问答与上下文压缩．`note_writer` 负责写作和修订，`note_review` 负责独立审阅．未填写的角色继承 `default`．
- 板书角色 `board_recognition` 必须选择支持图像输入的模型．无效选择器、角色循环或不支持的思考级别会阻止保存．
- 默认模型为 `deepseek-flash`．省略 `maxTokens` 时继承 OMP 模型能力信息；显式填写时使用配置值．
- 提供方凭据保存在 `data/agent/auth.json` 的 `llm` 字段中．API Key、OAuth 登录与刷新即时保存，设置页只显示配置状态．无鉴权服务设置 `auth: none`．

## 架构一览

```text
输入插件（Canvas/SJTU 录播 + 直播 + 手动 URL）
        │  AudioStream / VideoStream（append-only）
        ├──────────────► STT 插件 ──► TextStream(transcript)
        └──────────────► VLM 插件 ──► TextStream(visual)
                                      │
行为层（llm-agent / notes / 自定义插件）
  事件总线、时钟、服务注册、UI、项目 OMP SDK 运行时
                                      │
Web UI（左右多 tab 边栏 + toolbox + 中间 pages）
```

- **流**：音频是 PCM s16le；视频是 JPEG 帧；文字是不断增长的 `TextItem` 列表。`len(stream)` 是整秒/片段数，精确时长用 `stream.duration`。
- **事件**：`clock.tick`、`session.*`、`stream.appended`、`settings.changed`、`agent.*`、`stt.*`、`vlm.*`、`notes.*`、`input.*`．
- **插件**：`id` + `requires` 组成 DAG，拓扑序加载；插件之间只通过 `ctx.services` 调用。
- **LLM 层**：Python 通过私有 JSON Lines IPC 调用一个常驻 Bun 进程．OMP SDK 管理模型请求、工具执行、会话与上下文压缩．取消会终止实际模型流和工具执行．
- **UI**：声明式 block + action；插件不写前端代码。

内置插件：

| 插件 ID     | 作用                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| `settings`  | 设置注册、持久化、设置页                                                                                |
| `llm`       | 项目 Bun 运行时、模型角色配置与提供方鉴权                                                               |
| `input`     | Canvas/SJTU 录播、直播、ffmpeg 源                                                                       |
| `stt`       | 供应商适配：兼容 HTTP/WebSocket 服务（流式临时文字、定稿）或 Google Cloud Speech-to-Text v2（批量窗口） |
| `vlm`       | 视频流 -> 融合板书 -> 文字流                                                                            |
| `llm-agent` | SDK 多会话助手，原生 read/grep/glob/edit/write，流式事件与文件授权                                      |
| `notes`     | 整理分段笔记、汇总与审阅，提供笔记页面及课后 direct/fill/skip 策略                                      |
| `ask`       | 笔记内问答：选中提问，快速 / 研究两种强度，流式回答，答案随笔记保存                                     |
| `reader`    | 通用 Markdown 阅读页（每门课程一页），持有选中菜单与内联问答块                                          |

分段核心区间由 `notes.chunk_seconds` 设置，`notes.overlap_seconds` 提供左侧原文上下文．重叠原文不重复计入正文．`notes.concurrency` 限制笔记 agent 并发数．分段预览标为未审阅；`note_review` 统一检查完整笔记的来源覆盖与正确性．

写作者每次 `edit` 或 `write` 成功后，应用在发布锁内更新预览．浏览器可在生成期间阅读已保存文稿．审阅者读取锁保护的固定快照．来源始终只读，完整来源在上下文压缩后仍可重新读取．

日志页和问答执行区可向选中的 Agent 发送多行消息．Enter 换行，Shift+Enter 或 Ctrl+Enter 发送．消息中断目标的模型回合或工具等待，再在同一任务中继续．中断主写作不会取消独立审阅；旧输入的审阅结果不能批准修改后的笔记．

完成的分段、审阅和问答不接受消息．完成的主写作可恢复原会话，读取已发布正文，再执行最终写作、审阅和发布．恢复启动和发布清理期间拒绝发送，并保留输入草稿．日志更新和页面切换保留草稿、选区及输入位置．

## 目录

```text
src/                              # 课堂 Agent
├── app.py / cli.py / config.py
├── applog.py
├── core/                         # 流、事件、插件、服务、UI、runtime
├── agent_runtime/                # Bun OMP SDK、IPC、配置、鉴权与授权工作区
├── llm/                          # Python 角色请求与用量适配
├── plugins/                      # settings/llm/input/stt/vlm/agent/notes
├── integrations/                 # vendored SJTU Canvas/视频客户端
└── ui/                           # FastAPI + 静态前端
tests/
docs/
data/                             # 运行产物（gitignore）
```

## 测试

```bash
.venv/bin/python -m pytest -q
bun run check:types
```

确定性测试使用脚本化运行时与本地请求记录，覆盖配置、IPC、授权和应用行为．真实模型验收单独执行并保存请求与文稿记录．

TypeScript 由项目 devShell 提供；`check:types` 使用根目录 `tsconfig.json`．

## 说明

- 直播通道使用 SJTU 门户 `v1/vod_live` 和 `v1/course_vod_videoinfos`；也可在 UI 中手动填入 HLS/RTMP/HTTP URL．
- 详细接口与扩展方式见 `docs/plugin-api.md`。
