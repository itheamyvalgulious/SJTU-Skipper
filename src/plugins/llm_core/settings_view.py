"""Blocks for the LLM settings page.

The page is a workflow, not a list of fields:

``概览``     what is configured, what still blocks a run
``提供方``   one card per endpoint, credentials included
``模型``     pick a provider, compare its models, edit one
``角色``     pick a role, choose its model
``原始配置`` the two YAML files, as an escape hatch

Rendering is a pure function of the catalog plus the current selection, so the
plugin keeps state and actions while this module keeps layout and wording.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.ui import BlockType, FormInputType

PAGE_ID = "llm:configuration"

# Endpoint presets: protocol, auth mode and URL are decided here so the UI never
# asks the user to remember spellings.
PROVIDER_PRESETS: tuple[tuple[str, str], ...] = (
    ("opencode-go", "OpenCode Go 套餐（浏览器授权）"),
    ("deepseek", "DeepSeek（API Key）"),
    ("custom", "自定义 OpenAI 兼容端点"),
)
PRESETS: dict[str, dict[str, str]] = {
    "opencode-go": {
        "base_url": "https://opencode.ai/inference/go/openai/v1",
        "api": "openai-completions",
        "auth": "oauth",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api": "openai-completions",
        "auth": "apiKey",
    },
    "custom": {"base_url": "", "api": "openai-completions", "auth": "apiKey"},
}
API_OPTIONS: tuple[str, ...] = (
    "openai-completions",
    "openai-responses",
    "openrouter",
    "anthropic-messages",
    "google-generative-ai",
    "ollama-chat",
)
AUTH_OPTIONS: tuple[str, ...] = ("apiKey", "oauth", "none")
AUTH_LABELS: dict[str, str] = {
    "apiKey": "API Key",
    "oauth": "浏览器授权",
    "none": "无需鉴权",
}
INPUT_FORMATS: tuple[str, ...] = ("text", "image")
THINKING_LEVELS: tuple[str, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
ROLE_TITLES: dict[str, str] = {
    "default": "默认",
    "ask_instant": "快速问答",
    "ask_research": "研究问答",
    "note_chunk": "笔记分段",
    "note_writer": "笔记写作",
    "note_review": "笔记审阅",
    "board_recognition": "板书识别",
    "compaction": "上下文压缩",
}


def field(
    name: str,
    label: str,
    value: Any = "",
    *,
    type: FormInputType = FormInputType.STRING,
    input: FormInputType = FormInputType.TEXT,
    **options: Any,
) -> dict[str, Any]:
    condition = options.get("visible_when")
    if condition is not None:
        controller, allowed = condition
        options["visible_when"] = {
            "key": controller,
            "values": [str(item) for item in allowed],
        }
    return {
        "key": name,
        "name": name,
        "label": label,
        "type": type,
        "input": input,
        "value": value,
        **options,
    }


def number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def size(value: Any) -> str:
    """Compact token counts: 1000000 -> 1M, 128000 -> 128K, 4096 -> 4.1K."""

    count = number(value)
    if count <= 0:
        return "—"
    for unit, scale in (("M", 1_000_000), ("K", 1_000)):
        if count >= scale:
            text = f"{count / scale:.1f}".rstrip("0").rstrip(".")
            return f"{text}{unit}"
    return str(count)


def role_titles(roles: list[str]) -> str:
    """Render role ids for messages: 默认（default） reads better than default."""

    return "、".join(
        f"{ROLE_TITLES[role]}（{role}）" if role in ROLE_TITLES else role
        for role in roles
    )


def selector_parts(selector: str) -> tuple[str, str]:
    """Split ``provider/model:level`` without mistaking the model for a level."""

    target, _, level = selector.partition(":")
    return target, level


def providers(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(catalog.get("providers", []))


def resolved(catalog: Mapping[str, Any], role: str) -> dict[str, Any]:
    return dict(catalog.get("roles", {}).get(role, {}))


def roles_using(catalog: Mapping[str, Any], provider: str, model: str) -> list[str]:
    prefix = f"{provider}/{model}"
    roles = [
        role
        for role, selector in catalog.get("selectors", {}).items()
        if selector == prefix or selector.startswith(f"{prefix}:")
    ]
    # ``default`` governs the roles that inherit from it, so lead with it.
    return sorted(roles, key=lambda role: (role != "default", role))


def image_ok(catalog: Mapping[str, Any], role: str) -> bool:
    info = resolved(catalog, role)
    for provider in providers(catalog):
        if provider["id"] != info.get("provider"):
            continue
        for model in provider.get("models", []):
            if model["id"] == info.get("model"):
                return "image" in model.get("input", [])
    return False


def issues(catalog: Mapping[str, Any]) -> list[str]:
    """Everything that still prevents a run, in the order it should be fixed."""

    items = providers(catalog)
    if not items:
        return ["尚未配置提供方：在「提供方」页接入端点．"]
    found: list[str] = []
    for provider in items:
        if provider.get("auth") == "none" or provider.get("configured"):
            continue
        action = (
            "点击「关联套餐」完成浏览器授权"
            if provider.get("auth") == "oauth"
            else "填写 API Key"
        )
        found.append(f"提供方 `{provider['id']}` 未配置凭据：{action}．")
    if "board_recognition" in catalog.get("selectors", {}) and not image_ok(
        catalog, "board_recognition"
    ):
        found.append("板书识别角色使用的模型不支持 `image` 输入．")
    return found


class SettingsView:
    """Render the LLM section for one catalog and one selection."""

    def __init__(
        self,
        catalog: Mapping[str, Any],
        selection: Mapping[str, str],
        checks: Mapping[str, Mapping[str, Any]] | None = None,
        login: Mapping[str, Any] | None = None,
        status: str = "",
    ) -> None:
        self.catalog = catalog
        self.selection = dict(selection)
        self.checks = dict(checks or {})
        # An authorization in flight: {provider, uri, code, remaining}.
        self.login = dict(login) if login else None
        self.login_status = status

    # ---- what is being edited

    def provider(self) -> dict[str, Any] | None:
        items = providers(self.catalog)
        for item in items:
            if item["id"] == self.selection.get("provider"):
                return item
        return items[0] if items else None

    def model(self, provider: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if provider is None:
            return None
        models = list(provider.get("models", []))
        for model in models:
            if model["id"] == self.selection.get("model"):
                return model
        return models[0] if models else None

    def role(self) -> str:
        selectors = self.catalog.get("selectors", {})
        if self.selection.get("role") in selectors:
            return self.selection["role"]
        if "default" in selectors:
            return "default"
        return next(iter(selectors), "")

    # ---- pages

    def blocks(self) -> list[dict[str, Any]]:
        return [
            {"title": "概览", "blocks": self._overview()},
            {"title": "提供方", "blocks": self._provider_page()},
            {"title": "模型", "blocks": self._model_page()},
            {"title": "角色", "blocks": self._role_page()},
            {"title": "原始配置", "blocks": self._raw_page()},
        ]

    def _overview(self) -> list[dict[str, Any]]:
        items = providers(self.catalog)
        models = sum(len(item.get("models", [])) for item in items)
        missing = [
            item["id"]
            for item in items
            if item.get("auth") != "none" and not item.get("configured")
        ]
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.KEYVALUE,
                "items": [
                    ["默认模型", self._selector_text("default")],
                    ["提供方", f"{len(items)} 个"],
                    ["模型", f"{models} 个"],
                    [
                        "凭据",
                        "全部已配置" if not missing else "缺 " + "、".join(missing),
                    ],
                ],
            }
        ]
        lines = ["### 待处理"]
        found = issues(self.catalog)
        if found:
            lines.extend(f"- {item}" for item in found)
        else:
            lines.append("- 无．配置可用于新会话．")
        blocks.append({"type": BlockType.MARKDOWN, "text": "\n".join(lines)})

        lines = [
            "### 角色分配",
            "",
            "| 角色 | 设置 | 生效模型 |",
            "| --- | --- | --- |",
        ]
        for role in self.catalog.get("selectors", {}):
            info = resolved(self.catalog, role)
            label = ROLE_TITLES.get(role, role)
            target = f"`{info.get('provider', '')}/{info.get('model', '')}`"
            selector = str(self.catalog.get("selectors", {}).get(role, "@default"))
            lines.append(f"| {label} `{role}` | `{selector}` | {target} |")
        blocks.append({"type": BlockType.MARKDOWN, "text": "\n".join(lines)})
        return blocks

    def _selector_text(self, role: str) -> str:
        info = resolved(self.catalog, role)
        if not info:
            return "—"
        # Plain text: the key-value block does not render Markdown.
        text = f"{info.get('provider')}/{info.get('model')}"
        selector = str(self.catalog.get("selectors", {}).get(role, ""))
        _target, level = selector_parts(selector)
        if level:
            text += f" · {level}"
        return text

    def _provider_page(self) -> list[dict[str, Any]]:
        items = providers(self.catalog)
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    "### 提供方\n\n"
                    "一个提供方是一组端点、协议与凭据．模型属于提供方，"
                    "在「模型」页比较和编辑．"
                ),
            }
        ]
        blocks.extend(self._login_panel())
        if items:
            blocks.append({"type": BlockType.CARDS, "items": self._cards(items)})
        elif self.login_status:
            blocks.append({"type": BlockType.MARKDOWN, "text": self.login_status})
        blocks.append({"type": BlockType.MARKDOWN, "text": "### 接入新提供方"})
        blocks.append({"type": BlockType.FORM, **self._add_form()})
        return blocks

    def _login_panel(self) -> list[dict[str, Any]]:
        """Keep the authorization link on the page; a toast cannot be copied."""

        if self.login is None:
            if not self.login_status or not providers(self.catalog):
                return []
            return [{"type": BlockType.MARKDOWN, "text": self.login_status}]
        login = self.login
        pid = str(login.get("provider") or "")
        uri = str(login.get("uri") or "")
        code = str(login.get("code") or "")
        minutes = max(1, int(login.get("valid_for") or 0) // 60)
        lines = [
            "### 等待 OpenCode 授权",
            "",
            f"1. 在浏览器打开：[打开授权页面]({uri})（新标签页）",
            f"2. 确认页面显示的验证码：{code}",
            "",
            (
                f"链接与验证码 {minutes} 分钟内有效，授权完成后本页自动刷新；"
                "超时请点「重新开始授权」．"
            ),
            "",
            f"浏览器没有弹出时，复制这个地址：`{uri}`",
            "",
        ]
        return [
            {"type": BlockType.MARKDOWN, "text": "\n".join(lines)},
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {
                        "label": "重新开始授权",
                        "action": "llm.opencode_login",
                        "value": {"provider": pid, "force": True},
                        "class": "btn-solid theme-primary",
                    },
                    {
                        "label": "取消授权",
                        "action": "llm.opencode_cancel",
                        "value": {"provider": pid},
                        "class": "btn-solid theme-secondary",
                    },
                ],
            },
        ]

    def _cards(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for provider in items:
            pid = provider["id"]
            oauth = provider.get("auth") == "oauth"
            configured = bool(provider.get("configured"))
            actions: list[dict[str, Any]] = []
            if oauth:
                actions.append(
                    {
                        "label": "关联套餐",
                        "action": "llm.opencode_login",
                        "value": {"provider": pid},
                        "class": "btn-solid theme-secondary btn-sm",
                    }
                )
            if configured:
                actions.append(
                    {
                        "label": "删除凭据",
                        "action": "llm.logout",
                        "value": {"provider": pid},
                        "class": "btn-solid theme-danger btn-sm",
                    }
                )
            if len(items) > 1:
                actions.append(
                    {
                        "label": "删除提供方",
                        "action": "llm.provider.remove",
                        "value": {"provider": pid},
                        "class": "btn-solid theme-danger btn-sm",
                    }
                )
            cards.append(
                {
                    "id": pid,
                    "title": pid,
                    "subtitle": provider.get("baseUrl") or "未填写端点地址",
                    "meta": [
                        ("协议", provider.get("api", API_OPTIONS[0])),
                        (
                            "鉴权",
                            AUTH_LABELS.get(
                                str(provider.get("auth")), str(provider.get("auth"))
                            ),
                        ),
                        ("凭据", "已配置" if configured else "未配置"),
                        ("模型", f"{len(provider.get('models', []))} 个"),
                    ],
                    "blocks": [
                        {"type": BlockType.FORM, **self._provider_form(provider)}
                    ],
                    "actions": actions,
                }
            )
        return cards

    @staticmethod
    def _provider_form(provider: Mapping[str, Any]) -> dict[str, Any]:
        pid = str(provider["id"])
        return {
            "fields": [
                field(
                    "provider",
                    "提供方",
                    pid,
                    readonly=True,
                    description="用作 roles 前缀，创建后不可改名",
                ),
                field("base_url", "端点地址", provider.get("baseUrl", "")),
                field(
                    "api",
                    "协议",
                    provider.get("api", API_OPTIONS[0]),
                    type=FormInputType.SELECT,
                    input=FormInputType.SELECT,
                    options=list(API_OPTIONS),
                ),
                field(
                    "auth",
                    "鉴权方式",
                    provider.get("auth", AUTH_OPTIONS[0]),
                    type=FormInputType.SELECT,
                    input=FormInputType.SELECT,
                    options=list(AUTH_OPTIONS),
                ),
                field(
                    "key",
                    "API Key",
                    input=FormInputType.PASSWORD,
                    visible_when=("auth", ("apiKey",)),
                    description="留空保持不变",
                ),
            ],
            "submit": {"label": "保存提供方", "action": "llm.provider.save"},
        }

    @staticmethod
    def _add_form() -> dict[str, Any]:
        custom = ("preset", ("custom",))
        return {
            "fields": [
                field(
                    "preset",
                    "预设",
                    PROVIDER_PRESETS[0][0],
                    type=FormInputType.SELECT,
                    input=FormInputType.SELECT,
                    options=[value for value, _label in PROVIDER_PRESETS],
                    description="预设已带端点、协议与鉴权方式；保存后自动获取模型",
                ),
                field(
                    "provider",
                    "标识",
                    "",
                    placeholder="默认为预设名，如 opencode-go",
                    description="作为 roles.yml 的前缀",
                ),
                field(
                    "base_url",
                    "端点地址",
                    "",
                    visible_when=custom,
                    placeholder="https://example.com/v1",
                ),
                field(
                    "api",
                    "协议",
                    API_OPTIONS[0],
                    type=FormInputType.SELECT,
                    input=FormInputType.SELECT,
                    options=list(API_OPTIONS),
                    visible_when=custom,
                ),
                field(
                    "auth",
                    "鉴权方式",
                    AUTH_OPTIONS[0],
                    type=FormInputType.SELECT,
                    input=FormInputType.SELECT,
                    options=list(AUTH_OPTIONS),
                    visible_when=custom,
                ),
                field(
                    "key",
                    "API Key",
                    input=FormInputType.PASSWORD,
                    visible_when=("preset", ("deepseek", "custom")),
                    description="可留空，之后在提供方卡片里补填",
                ),
            ],
            "submit": {"label": "保存并获取模型", "action": "llm.provider.save"},
        }

    def _model_page(self) -> list[dict[str, Any]]:
        items = providers(self.catalog)
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    "### 模型\n\n"
                    "先选提供方，再看端点提供了什么、本地配置了什么，"
                    "最后编辑选中的那一个．"
                ),
            }
        ]
        if not items:
            blocks.append(
                {
                    "type": BlockType.MARKDOWN,
                    "text": "尚无提供方，请先在「提供方」页接入端点．",
                }
            )
            return blocks
        provider = self.provider()
        assert provider is not None
        blocks.append(
            {
                "type": BlockType.ROWS,
                "items": [
                    {
                        "title": item["id"],
                        "subtitle": item.get("baseUrl") or "未填写端点地址",
                        "meta": (
                            f"{len(item.get('models', []))} 个模型 · "
                            f"{AUTH_LABELS.get(str(item.get('auth')), item.get('auth'))} · "
                            f"{'凭据已配置' if item.get('configured') else '凭据未配置'}"
                        ),
                        "active": item["id"] == provider["id"],
                        "action": "llm.select.provider",
                        "value": {"provider": item["id"]},
                        "label": "查看",
                        "class": "btn-outline theme-secondary btn-sm",
                    }
                    for item in items
                ],
            }
        )
        pid = provider["id"]
        blocks.append(
            {
                "type": BlockType.MARKDOWN,
                "text": f"#### `{pid}`\n\n"
                + (
                    f"`{provider['baseUrl']}`"
                    if provider.get("baseUrl")
                    else "未填写端点地址"
                ),
            }
        )
        blocks.append(
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {
                        "label": "检查端点模型",
                        "action": "llm.check",
                        "value": {"provider": pid},
                        "class": "btn-solid theme-secondary",
                    }
                ],
            }
        )
        blocks.extend(self._discovery(provider))
        blocks.extend(self._model_table(provider))
        blocks.extend(self._model_editor(provider))
        return blocks

    def _discovery(self, provider: Mapping[str, Any]) -> list[dict[str, Any]]:
        pid = str(provider["id"])
        check = self.checks.get(pid)
        if check is None:
            return [
                {
                    "type": BlockType.MARKDOWN,
                    "text": (
                        "端点模型尚未检查．点击「检查端点模型」读取 "
                        "`${baseUrl}/models`，只列出本地尚无的模型．"
                    ),
                }
            ]
        if check.get("error"):
            return [
                {
                    "type": BlockType.MARKDOWN,
                    "text": f"检查失败：{check['error']}．检查端点地址与凭据后重试．",
                }
            ]
        known = {model["id"] for model in provider.get("models", [])}
        missing = [item for item in check.get("models", []) if item["id"] not in known]
        if not missing:
            return [
                {
                    "type": BlockType.MARKDOWN,
                    "text": (
                        f"端点返回 {len(check.get('models', []))} 个模型，"
                        "本地已全部添加．"
                    ),
                }
            ]
        return [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    f"端点返回 {len(check.get('models', []))} 个模型，"
                    f"其中 {len(missing)} 个尚未添加："
                ),
            },
            {
                "type": BlockType.TABLE,
                "columns": [
                    {"key": "model", "label": "模型"},
                    {"key": "name", "label": "名称"},
                    {"key": "context", "label": "上下文"},
                    {"key": "input", "label": "输入"},
                ],
                "rows": [
                    {
                        "model": item["id"],
                        "name": item.get("name") or "—",
                        "context": size(item.get("contextWindow")),
                        "input": "/".join(item.get("input", [])),
                    }
                    for item in missing
                ],
                "row_actions": [
                    {
                        "label": "添加",
                        "action": "llm.model.add",
                        "keys": ["model"],
                        "value": {"provider": pid},
                        "class": "btn-solid theme-secondary btn-sm",
                    }
                ],
            },
            {
                "type": BlockType.BUTTONS,
                "buttons": [
                    {
                        "label": f"全部添加（{len(missing)}）",
                        "action": "llm.model.add_all",
                        "value": {"provider": pid},
                        "class": "btn-solid theme-primary",
                    }
                ],
            },
        ]

    def _model_table(self, provider: Mapping[str, Any]) -> list[dict[str, Any]]:
        pid = str(provider["id"])
        models = list(provider.get("models", []))
        blocks: list[dict[str, Any]] = [
            {"type": BlockType.MARKDOWN, "text": "#### 已配置模型"}
        ]
        if not models:
            blocks.append({"type": BlockType.MARKDOWN, "text": "该提供方尚无模型．"})
            return blocks
        rows = []
        for model in models:
            roles = roles_using(self.catalog, pid, model["id"])
            rows.append(
                {
                    "model": model["id"],
                    "context": size(model.get("contextWindow")),
                    "output": size(model.get("maxTokens")),
                    "input": "/".join(model.get("input", [])) or "—",
                    "reasoning": "是" if model.get("reasoning") else "否",
                    "roles": "、".join(roles) if roles else "—",
                }
            )
        blocks.append(
            {
                "type": BlockType.TABLE,
                "columns": [
                    {"key": "model", "label": "模型"},
                    {"key": "context", "label": "上下文"},
                    {"key": "output", "label": "输出"},
                    {"key": "input", "label": "输入"},
                    {"key": "reasoning", "label": "推理"},
                    {"key": "roles", "label": "使用角色"},
                ],
                "rows": rows,
                "row_actions": [
                    {
                        "label": "编辑",
                        "action": "llm.select.model",
                        "keys": ["model"],
                        "value": {"provider": pid},
                        "class": "btn-outline theme-secondary btn-sm",
                    },
                    {
                        "label": "删除",
                        "action": "llm.model.remove",
                        "keys": ["model"],
                        "value": {"provider": pid},
                        "class": "btn-solid theme-danger btn-sm",
                    },
                ],
            }
        )
        return blocks

    def _model_editor(self, provider: Mapping[str, Any]) -> list[dict[str, Any]]:
        model = self.model(provider)
        if model is None:
            return []
        pid, mid = provider["id"], model["id"]
        formats = set(model.get("input", []))
        roles = roles_using(self.catalog, pid, mid)
        hint = f"使用角色：{'、'.join(roles)}" if roles else "当前没有角色使用该模型．"
        return [
            {
                "type": BlockType.MARKDOWN,
                "text": f"#### 编辑 `{pid}/{mid}`\n\n{hint}",
            },
            {
                "type": BlockType.FORM,
                "fields": [
                    field("provider", "提供方", pid, readonly=True),
                    field("model", "模型", mid, readonly=True),
                    field(
                        "context_window",
                        "上下文窗口",
                        number(model.get("contextWindow")),
                        type=FormInputType.NUMBER,
                        input=FormInputType.NUMBER,
                        description="一次请求可容纳的 token 上限；留空沿用端点/默认值",
                    ),
                    field(
                        "max_tokens",
                        "输出上限",
                        number(model.get("maxTokens")),
                        type=FormInputType.NUMBER,
                        input=FormInputType.NUMBER,
                        description="单次回复上限",
                    ),
                    field(
                        "input_text",
                        "输入文本",
                        "text" in formats,
                        type=FormInputType.CHECKBOX,
                        input=FormInputType.CHECKBOX,
                    ),
                    field(
                        "input_image",
                        "输入图像",
                        "image" in formats,
                        type=FormInputType.CHECKBOX,
                        input=FormInputType.CHECKBOX,
                        description="板书识别要求所选模型勾选此项",
                    ),
                    field(
                        "reasoning",
                        "推理模型",
                        bool(model.get("reasoning")),
                        type=FormInputType.CHECKBOX,
                        input=FormInputType.CHECKBOX,
                        description="支持思考级别时，角色可以设置 :high 等后缀",
                    ),
                ],
                "submit": {"label": "保存模型", "action": "llm.model.save"},
            },
        ]

    def _role_page(self) -> list[dict[str, Any]]:
        selectors = dict(self.catalog.get("selectors", {}))
        if not selectors:
            return [{"type": BlockType.MARKDOWN, "text": "尚未配置角色．"}]
        options: list[str] = ["@default"]
        for provider in providers(self.catalog):
            for model in provider.get("models", []):
                options.append(f"{provider['id']}/{model['id']}")
        role = self.role()
        blocks: list[dict[str, Any]] = [
            {
                "type": BlockType.MARKDOWN,
                "text": (
                    "### 角色\n\n"
                    "每个任务选择一个模型与思考级别．选择「编辑」后在下方修改，"
                    "保存即对新会话生效．"
                ),
            }
        ]
        rows = []
        for name in selectors:
            info = resolved(self.catalog, name)
            status = ""
            if name == "board_recognition":
                status = (
                    "图像可用" if image_ok(self.catalog, name) else "缺少 image 输入"
                )
            rows.append(
                {
                    "role": name,
                    "title": ROLE_TITLES.get(name, name),
                    "selector": selectors.get(name, "@default"),
                    "active": f"{info.get('provider')}/{info.get('model')}",
                    "status": status,
                }
            )
        blocks.append(
            {
                "type": BlockType.TABLE,
                "columns": [
                    {"key": "title", "label": "角色"},
                    {"key": "role", "label": "标识"},
                    {"key": "selector", "label": "设置"},
                    {"key": "active", "label": "生效模型"},
                    {"key": "status", "label": "状态"},
                ],
                "rows": rows,
                "row_actions": [
                    {
                        "label": "编辑",
                        "action": "llm.select.role",
                        "keys": ["role"],
                        "class": "btn-outline theme-secondary btn-sm",
                    }
                ],
            }
        )
        if not role:
            return blocks
        selector = selectors.get(role, "@default")
        target, level = selector_parts(selector)
        if target.startswith("@"):
            target = "@default"
            level = level or "off"
        if target not in options:
            target = options[0]
        info = resolved(self.catalog, role)
        note = f"当前生效：`{info.get('provider')}/{info.get('model')}`"
        if role == "board_recognition" and not image_ok(self.catalog, role):
            note += "．板书识别需要勾选 `image` 输入的模型．"
        blocks.append(
            {"type": BlockType.MARKDOWN, "text": f"#### 编辑 `{role}`\n\n{note}"}
        )
        blocks.append(
            {
                "type": BlockType.FORM,
                "fields": [
                    field(
                        "role",
                        "角色",
                        role,
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=list(selectors),
                    ),
                    field(
                        "model",
                        "模型",
                        target,
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=options,
                    ),
                    field(
                        "level",
                        "思考级别",
                        level or "off",
                        type=FormInputType.SELECT,
                        input=FormInputType.SELECT,
                        options=list(THINKING_LEVELS),
                    ),
                ],
                "submit": {"label": "保存角色", "action": "llm.role.save"},
            }
        )
        return blocks

    def _raw_page(self) -> list[dict[str, Any]]:
        lines = [
            "### 原始配置",
            "",
            (
                "上面的页面覆盖常用操作；这里的文本会整体替换两份配置文件，"
                "由运行时校验后原子写入．"
            ),
            "",
        ]
        for provider in providers(self.catalog):
            status = "已配置" if provider.get("configured") else "未配置"
            lines.append(
                f"- `{provider['id']}`：凭据{status}（{provider.get('source')}）"
            )
        if self.login_status:
            lines.extend(["", self.login_status])
        return [
            {"type": BlockType.MARKDOWN, "text": "\n".join(lines)},
            {
                "type": BlockType.FORM,
                "fields": [
                    field(
                        "models_yaml",
                        "models.yml",
                        self.catalog.get("models_yaml", ""),
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                        rows=18,
                    ),
                    field(
                        "roles_yaml",
                        "roles.yml",
                        self.catalog.get("roles_yaml", ""),
                        type=FormInputType.TEXT,
                        input=FormInputType.TEXTAREA,
                        rows=12,
                    ),
                ],
                "submit": {"label": "保存配置", "action": "llm.configure"},
            },
        ]


__all__ = [
    "API_OPTIONS",
    "AUTH_OPTIONS",
    "INPUT_FORMATS",
    "PAGE_ID",
    "PRESETS",
    "PROVIDER_PRESETS",
    "ROLE_TITLES",
    "THINKING_LEVELS",
    "SettingsView",
    "field",
    "image_ok",
    "issues",
    "number",
    "providers",
    "resolved",
    "role_titles",
    "roles_using",
    "selector_parts",
    "size",
]
