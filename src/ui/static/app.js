(() => {
  "use strict";

  const EXECUTION_ROLE = Object.freeze({
    WRITER: "writer",
    QUESTION_ANSWER: "question_answer",
    REVIEWER: "reviewer",
    CHUNK_WRITER: "chunk_writer",
  });
  const TASK_STATE = Object.freeze({
    PENDING: "pending",
    RUNNING: "running",
  });
  const STATUS_LABEL = Object.freeze({
    QUEUED: "排队中",
    RUNNING: "处理中",
    PAUSED: "已暂停",
    FINISHED: "已结束",
    FAILED: "失败",
    CANCELLED: "已取消",
  });

  const BLOCK_TYPE = Object.freeze({
    AGENT_ACTIVITY: "agent_activity",
    TASK_QUEUE: "task_queue",
    QUEUE: "queue",
    HTML: "html",
    MARKDOWN: "markdown",
    TEXT: "text",
    KEYVALUE: "keyvalue",
    TABLE: "table",
    CARDS: "cards",
    ROWS: "rows",
    TREE: "tree",
    LIST: "list",
    BUTTONS: "buttons",
    FORM: "form",
    LOG: "log",
    SCHEMA_FORM: "schema_form",
    IMAGE: "image",
    VIDEO: "video",
    TABS: "tabs",
    DISCLOSURE: "disclosure",
    PLAYER: "player",
    SWITCH: "switch",
    QA: "qa",
    PROGRESS: "progress",
    BADGE: "badge",
  });

  const QA_STATUS = Object.freeze({
    PENDING: "pending",
    DONE: "done",
    ERROR: "error",
  });

  const ASK_INTENSITY = Object.freeze({
    INSTANT: "instant",
    RESEARCH: "research",
  });

  const EXECUTION_BLOCK_KIND = Object.freeze({
    TEXT: "text",
    THINKING: "thinking",
    TOOL: "tool",
    REVIEW: "review",
    USER: "user",
  });

  const EXECUTION_BLOCK_STATUS = Object.freeze({
    STREAMING: "streaming",
    RUNNING: "running",
    COMPLETED: "completed",
    FAILED: "failed",
    CANCELLED: "cancelled",
  });

  const SIDEBAR_SIDE = Object.freeze({
    LEFT: "left",
    RIGHT: "right",
  });

  const HISTORY_MODE = Object.freeze({
    PUSH: "push",
    REPLACE: "replace",
  });

  const TOAST_KIND = Object.freeze({
    DEFAULT: "",
    ERR: "err",
  });

  const THEME_STORAGE_KEY = "sjtu-skipper.theme";
  const THEME_MODE = Object.freeze({
    SYSTEM: "system",
    LIGHT: "light",
    DARK: "dark",
  });
  const THEME_ORDER = Object.freeze([THEME_MODE.SYSTEM, THEME_MODE.LIGHT, THEME_MODE.DARK]);
  const THEME_LABEL = Object.freeze({
    [THEME_MODE.SYSTEM]: "跟随系统",
    [THEME_MODE.LIGHT]: "浅色",
    [THEME_MODE.DARK]: "深色",
  });

  const SETTING_TYPE = Object.freeze({
    STRING: "string",
    TEXT: "text",
    INT: "int",
    FLOAT: "float",
    BOOL: "bool",
    SELECT: "select",
    SECRET: "secret",
    JSON: "json",
    ACTION: "action",
  });

  const FIELD_INPUT = Object.freeze({
    BUTTON: "button",
    PASSWORD: "password",
    TEXTAREA: "textarea",
    SELECT: "select",
    CHECKBOX: "checkbox",
    NUMBER: "number",
    TEXT: "text",
    SUBMIT: "submit",
  });

  const SELECTION_ACTION_KIND = Object.freeze({
    PROMPT: "prompt",
    ACTION: "action",
  });

  const WS_MESSAGE_TYPE = Object.freeze({
    REFRESH: "refresh",
    NOTIFY: "notify",
  });

  const state = {
    ui: null,
    connected: null,
    activeTab: { left: null, right: null },
    blockTabs: {},
    expandedTrees: new Map(),
    pageKey: null,
    playerCache: {},
    paletteItems: [],
    paletteIndex: 0,
    loading: null,
    navigationVersion: 0,
    navigationPending: 0,
    navigationQueue: Promise.resolve(),
    lastLoad: 0,
    ws: null,
    wsRetry: 0,
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k === "dataset") Object.assign(node.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) {
      if (child === null || child === undefined) continue;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  }

  const escapeHtml = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );

  const fmtSeconds = (v) => {
    const n = Number(v || 0);
    if (!Number.isFinite(n)) return "0s";
    const m = Math.floor(n / 60);
    const s = Math.floor(n % 60);
    return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
  };

  const shortId = (id) => (id ? String(id).slice(0, 18) : "-");
  const menuBindings = new Map();
  let toastTimer;

  function bindMenu(button, menu) {
    button._menu = menu;
    const hide = () => {
      menu.classList.remove("show");
      button.setAttribute("aria-expanded", "false");
    };
    const toggle = () => {
      const opening = !menu.classList.contains("show");
      for (const binding of menuBindings.values()) binding.hide();
      if (!opening) return;
      menu.classList.add("show");
      const rect = button.getBoundingClientRect();
      menu.style.left = `${Math.max(4, Math.min(rect.left, innerWidth - menu.offsetWidth - 4))}px`;
      menu.style.top = `${rect.bottom + menu.offsetHeight > innerHeight ? Math.max(4, rect.top - menu.offsetHeight) : rect.bottom}px`;
      button.setAttribute("aria-expanded", "true");
    };
    const keydown = (event) => {
      if (!["ArrowDown", "ArrowUp", "Home", "End", "Escape"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "Escape") {
        hide();
        button.focus();
        return;
      }
      if (!menu.classList.contains("show")) toggle();
      const items = $$('button:not(:disabled), [role="option"]:not([aria-disabled="true"])', menu);
      const index = items.indexOf(document.activeElement);
      const next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? items.length - 1
            : (index + (event.key === "ArrowUp" ? -1 : 1) + items.length) % items.length;
      items[next]?.focus();
    };
    button.addEventListener("click", toggle);
    button.addEventListener("keydown", keydown);
    menu.addEventListener("keydown", keydown);
    const binding = {
      hide,
      dispose: () => {
        hide();
        button.removeEventListener("click", toggle);
        button.removeEventListener("keydown", keydown);
        menu.removeEventListener("keydown", keydown);
        menuBindings.delete(button);
      },
    };
    menuBindings.set(button, binding);
    return binding;
  }

  document.addEventListener("pointerdown", (event) => {
    for (const [button, binding] of menuBindings)
      if (!button.contains(event.target) && !button._menu.contains(event.target)) binding.hide();
  });
  document.addEventListener(
    "scroll",
    (event) => {
      for (const [button, binding] of menuBindings)
        if (event.target.contains(button)) binding.hide();
    },
    true,
  );
  window.addEventListener("resize", () => {
    for (const binding of menuBindings.values()) binding.hide();
  });

  function toast(message, kind = TOAST_KIND.DEFAULT) {
    if (!message) return;
    const node = $("#toast");
    $(".toast-body", node).textContent = message;
    node.className = `toast ${kind === TOAST_KIND.ERR ? "theme-danger" : "theme-secondary"}`;
    clearTimeout(toastTimer);
    node.classList.add("show");
    toastTimer = setTimeout(() => node.classList.remove("show"), 3200);
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    });
    if (resp.status === 401) {
      const next = location.pathname + location.search + location.hash;
      window.location.href = `/login?next=${encodeURIComponent(next)}`;
      throw new Error("未登录");
    }
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        detail = (await resp.json()).detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const type = resp.headers.get("content-type") || "";
    return type.includes("application/json") ? resp.json() : resp.text();
  }

  function navigationFromUrl() {
    const params = new URL(window.location.href).searchParams;
    let pages = [];
    try {
      const value = JSON.parse(params.get("pages") || "[]");
      if (Array.isArray(value)) pages = value.filter((page) => typeof page === "string");
    } catch (_) {}
    return {
      page: params.get("page") || "",
      class: params.get("class") || "",
      doc: params.get("doc") || "",
      pages,
    };
  }

  function syncNavigation(navigation, historyMode) {
    if (!navigation) return;
    const url = new URL(window.location.href);
    for (const key of ["page", "class", "doc"]) {
      const value = navigation[key];
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    }
    url.searchParams.set("pages", JSON.stringify(navigation.pages || []));
    if (url.href !== window.location.href) {
      window.history[`${historyMode}State`](window.history.state, "", url);
    }
  }

  async function loadState({ quiet = false, navigationVersion = null } = {}) {
    if (navigationVersion === null && state.navigationPending) return;
    if (state.loading) {
      await state.loading;
      if (navigationVersion === null) return;
    }
    const version = navigationVersion ?? state.navigationVersion;
    if (version !== state.navigationVersion) return;
    const navigation = navigationFromUrl();
    const query = new URLSearchParams({
      ...navigation,
      pages: JSON.stringify(navigation.pages),
    });
    state.loading = (async () => {
      try {
        const ui = await api(`/api/ui/state?${query}`);
        if (version !== state.navigationVersion) return;
        if (JSON.stringify(navigationFromUrl()) !== JSON.stringify(navigation)) return;
        if (navigationVersion === null && state.navigationPending) return;
        const previousPage = state.ui?.active_page;
        state.ui = ui;
        state.lastLoad = Date.now();
        if (previousPage !== ui.active_page) {
          for (const side of Object.values(SIDEBAR_SIDE)) {
            const tab = (ui[`${side}_tabs`] || []).find((item) =>
              (item.toolbox || []).some((tool) => tool.page === ui.active_page),
            );
            if (tab) state.activeTab[side] = tab.id;
          }
        }
        renderAll();
        setConnection(true);
      } catch (err) {
        setConnection(false);
        if (!quiet) toast(`加载状态失败：${err.message}`, TOAST_KIND.ERR);
      } finally {
        state.loading = null;
      }
    })();
    await state.loading;
  }

  function changeNavigation(callback, historyMode = HISTORY_MODE.PUSH) {
    const version = state.navigationVersion;
    state.navigationPending += 1;
    const task = state.navigationQueue.then(async () => {
      try {
        if (version !== state.navigationVersion) return;
        const result = await callback();
        if (version !== state.navigationVersion) return result;
        syncNavigation(result?.navigation, historyMode);
        if (result?.refresh !== false || result?.navigation) {
          await loadState({ quiet: true, navigationVersion: version });
        }
        return result;
      } finally {
        state.navigationPending -= 1;
      }
    });
    state.navigationQueue = task.catch(() => {});
    return task;
  }

  async function restoreNavigation() {
    state.navigationVersion += 1;
    const navigation = navigationFromUrl();
    try {
      await changeNavigation(
        () =>
          api("/api/ui/navigate", {
            method: "POST",
            body: JSON.stringify(navigation),
          }),
        HISTORY_MODE.REPLACE,
      );
    } catch (err) {
      toast(`加载状态失败：${err.message}`, TOAST_KIND.ERR);
    }
  }

  function setConnection(ok) {
    state.connected = ok;
    const connection = $("#connection");
    if (connection) {
      connection.textContent = ok ? "已连接" : "连接断开";
      const dot = connection.parentElement?.querySelector(".status-dot");
      if (dot) {
        dot.classList.toggle("is-connected", Boolean(ok));
        dot.classList.toggle("is-disconnected", ok === false);
        dot.classList.toggle("is-connecting", ok === null);
      }
    }
  }

  function typeset(node = $("#page-content")) {
    if (!node || !window.renderMathInElement) return;
    if (
      !node.textContent.includes("$") &&
      !node.textContent.includes("\\(") &&
      !node.textContent.includes("\\[")
    ) {
      return;
    }
    try {
      window.renderMathInElement(node, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        ignoredClasses: ["agent-workflow"],
        throwOnError: false,
        strict: false,
      });
    } catch (_) {
      /* one malformed formula must not stop the page */
    }
  }

  function actionButton(
    label,
    action,
    value = undefined,
    cls = "btn-solid theme-secondary btn-sm",
    disabled = false,
  ) {
    const attrs = { class: cls, dataset: { action } };
    if (value !== undefined) attrs.dataset.value = JSON.stringify(value);
    attrs.type = FIELD_INPUT.BUTTON;
    const btn = el("button", { ...attrs, text: String(label) }, []);
    if (disabled) btn.disabled = true;
    return btn;
  }

  function renderLog(block) {
    const pre = el("pre", { class: "log bg-2 border p-3 overflow-auto" });
    const lines = Array.isArray(block.lines) ? block.lines : [];
    for (const line of lines) {
      if (typeof line === "string") {
        pre.appendChild(el("span", { class: "d-block", text: line }));
      } else {
        pre.appendChild(
          el("span", { class: "d-block" }, [
            line.time ? el("span", { class: "fg-2 me-2", text: String(line.time) }) : null,
            document.createTextNode(String(line.text ?? line.message ?? "")),
          ]),
        );
      }
    }
    return pre;
  }

  function renderDisclosure(block, existing = null) {
    const details =
      existing ||
      el("details", { class: "disclosure" }, [
        el("summary"),
        el("div", { class: "disclosure-body" }),
      ]);
    if (!existing) details.open = Boolean(block.open);
    details.firstElementChild.textContent = block.title || "";
    updateBlocks(details.lastElementChild, block.blocks || []);
    details._deferred = details.lastElementChild._deferred;
    return details;
  }

  function renderTable(block, existing = null) {
    const wrap =
      existing ||
      el("div", { class: "table-block" }, [
        el("div", { class: "table-responsive" }, [
          el("table", { class: "table table-hover align-middle" }, [el("thead"), el("tbody")]),
        ]),
        el("p", { class: "fg-2 p-3", text: "暂无数据" }),
        el("div", { class: "table-editor" }),
      ]);
    const table = wrap.firstElementChild.firstElementChild;
    const [thead, tbody] = table.children;
    const columns = Array.isArray(block.columns) ? block.columns : [];
    const rows = Array.isArray(block.rows) ? block.rows : [];
    table.classList.toggle("table-shared-editor", Boolean(block.editor));
    const rowActions = Array.isArray(block.row_actions) ? block.row_actions : [];
    const sharedEditor = Boolean(block.editor);
    const editorHost = wrap.children[2];
    editorHost.hidden = !sharedEditor;
    const selectedKey = rows.some((row) => row.id === wrap._selectedRow)
      ? wrap._selectedRow
      : rows[0]?.id;
    wrap._selectedRow = selectedKey;
    const selectRow = (id) => {
      wrap._selectedRow = id;
      for (const row of tbody.children) {
        const selected = row._rowKey === id;
        row.classList.toggle("editing-row", sharedEditor && selected);
        row.querySelector("[data-edit-row]")?.setAttribute("aria-pressed", String(selected));
      }
      for (const panel of editorHost.children) panel.hidden = panel.dataset.row !== id;
    };
    const previous = new Map(
      Array.from(tbody.children)
        .filter((row) => row._rowKey !== undefined)
        .map((row) => [row._rowKey, row]),
    );
    const wanted = new Set();
    const colLabel = (c) => (typeof c === "string" ? c : (c.label ?? c.key ?? ""));
    const colKey = (c, i) => (typeof c === "string" ? c : (c.key ?? i));
    const headRow = el("tr");
    for (const c of columns) headRow.appendChild(el("th", { scope: "col", text: colLabel(c) }));
    if (rowActions.length || sharedEditor)
      headRow.appendChild(el("th", { scope: "col", text: "操作" }));
    thead.replaceChildren(headRow);
    const ordered = [];
    rows.forEach((row, index) => {
      const rowObj = Array.isArray(row)
        ? Object.fromEntries(columns.map((c, i) => [colKey(c, i), row[i]]))
        : row || {};
      const key = rowObj.id ?? rowObj.key ?? index;
      const tr = previous.get(key) || el("tr");
      tr._rowKey = key;
      tr.replaceChildren();
      for (let i = 0; i < columns.length; i += 1) {
        const c = columns[i];
        const key = colKey(c, i);
        let value = Array.isArray(row) ? row[i] : rowObj[key];
        if (typeof value === "object" && value !== null) value = JSON.stringify(value);
        tr.appendChild(el("td", { text: value ?? "" }));
      }
      if (rowActions.length || sharedEditor) {
        const td = el("td", { class: "text-nowrap" });
        const actionsBox = el("div", { class: "table-actions" });
        if (sharedEditor) {
          const button = el("button", {
            type: "button",
            class: "btn-text btn-icon theme-secondary btn-sm",
            "aria-label": `编辑 ${rowObj.title || rowObj.model || rowObj.role || key}`,
            title: `编辑 ${rowObj.title || rowObj.model || rowObj.role || key}`,
            "aria-pressed": String(key === selectedKey),
            "data-edit-row": "",
          });
          // Bootstrap Icons: pencil-square.
          button.innerHTML =
            '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M15.502 1.94a.5.5 0 0 1 0 .706L14.459 3.69l-2-2L13.502.646a.5.5 0 0 1 .707 0l1.293 1.293zm-1.75 2.456-2-2L4.939 9.21a.5.5 0 0 0-.121.196l-.805 2.414a.25.25 0 0 0 .316.316l2.414-.805a.5.5 0 0 0 .196-.12l6.813-6.814z"/><path fill-rule="evenodd" d="M1 13.5A1.5 1.5 0 0 0 2.5 15h11a1.5 1.5 0 0 0 1.5-1.5v-6a.5.5 0 0 0-1 0v6a.5.5 0 0 1-.5.5h-11a.5.5 0 0 1-.5-.5v-11a.5.5 0 0 1 .5-.5H9a.5.5 0 0 0 0-1H2.5A1.5 1.5 0 0 0 1 2.5z"/></svg>';
          button.addEventListener("click", () => selectRow(key));
          actionsBox.appendChild(button);
        }
        for (const action of [...(rowObj.actions || []), ...rowActions]) {
          let value = action.value ? { ...action.value } : {};
          if (Array.isArray(action.keys)) {
            for (const key of action.keys) value[key] = rowObj[key];
          } else if (!action.value) {
            value = { ...rowObj };
          }
          const baseClass = action.class || "btn-solid theme-secondary btn-sm";
          const btnClass =
            action.icon && !baseClass.includes("btn-icon") ? `${baseClass} btn-icon` : baseClass;
          const button = actionButton(action.label || "执行", action.action, value, btnClass);
          if (action.icon) {
            const label = `${action.label || "执行"} ${rowObj.model || rowObj.title || key}`;
            button.setAttribute("aria-label", label);
            button.title = label;
          }
          if (action.icon === "trash") {
            // Bootstrap Icons: trash.
            button.innerHTML =
              '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0z"/><path d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4zM2.5 3h11V2h-11z"/></svg>';
          }
          if (action.icon === "link") {
            // Bootstrap Icons: link-45deg.
            button.innerHTML =
              '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4.715 6.542 3.343 7.914a3 3 0 1 0 4.243 4.243l1.828-1.829A3 3 0 0 0 8.586 5.5L8 6.086a1 1 0 0 0-.154.199 2 2 0 0 1 .861 3.337L6.88 11.45a2 2 0 1 1-2.83-2.83l.793-.792a4 4 0 0 1-.128-1.287z"/><path d="M6.586 4.672A3 3 0 0 0 7.414 9.5l.775-.776a2 2 0 0 1-.896-3.346L9.12 3.55a2 2 0 1 1 2.83 2.83l-.793.792c.112.42.155.855.128 1.287l1.372-1.372a3 3 0 1 0-4.243-4.243z"/></svg>';
          }
          if (action.icon === "key") {
            // Bootstrap Icons: key.
            button.innerHTML =
              '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M0 8a4 4 0 0 1 7.465-2H14a.5.5 0 0 1 .354.146l1.5 1.5a.5.5 0 0 1 0 .708l-1.5 1.5a.5.5 0 0 1-.708 0L13 9.207l-.646.647a.5.5 0 0 1-.708 0L11 9.207l-.646.647a.5.5 0 0 1-.708 0L9 9.207l-.646.647A.5.5 0 0 1 8 10h-.535A4 4 0 0 1 0 8m4-3a3 3 0 1 0 2.712 4.285A.5.5 0 0 1 7.163 9h.63l.853-.854a.5.5 0 0 1 .708 0l.646.647.646-.647a.5.5 0 0 1 .708 0l.646.647.646-.647a.5.5 0 0 1 .708 0l.646.647.793-.793-1-1h-6.63a.5.5 0 0 1-.451-.285A3 3 0 0 0 4 5"/><path d="M4 8a1 1 0 1 1-2 0 1 1 0 0 1 2 0"/></svg>';
          }
          const before = rowObj.actions?.includes(action)
            ? actionsBox.querySelector("[data-edit-row]")
            : null;
          actionsBox.insertBefore(button, before);
        }
        td.appendChild(actionsBox);
        tr.appendChild(td);
      }
      wanted.add(tr);
      ordered.push(tr);
      if (sharedEditor) {
        let panel = Array.from(editorHost.children).find((node) => node.dataset.row === key);
        if (!panel) {
          panel = el("div", { dataset: { row: key } });
          editorHost.appendChild(panel);
        }
        updateBlocks(panel, rowObj.details || []);
      }
      if (block.expandable && Array.isArray(rowObj.details)) {
        if (!tr._detailsRow) tr._detailsRow = el("tr", { class: "table-details" }, [el("td")]);
        const cell = tr._detailsRow.firstElementChild;
        cell.colSpan = Math.max(1, columns.length + (rowActions.length ? 1 : 0));
        const title = rowObj.title || rowObj.name || rowObj.model || rowObj.role || rowObj.id || "";
        const disclosure = renderDisclosure(
          { title: `编辑 ${title}`, blocks: rowObj.details },
          cell.firstElementChild,
        );
        if (!cell.firstElementChild) cell.appendChild(disclosure);
        wanted.add(tr._detailsRow);
        ordered.push(tr._detailsRow);
      }
    });
    ordered.forEach((row, index) => {
      if (tbody.children[index] !== row)
        moveActivityNode(tbody, row, tbody.children[index] || null);
    });
    for (const row of Array.from(tbody.children)) {
      if (wanted.has(row)) continue;
      destroyPlayers(row);
      row.remove();
    }
    wrap.children[1].hidden = Boolean(rows.length);
    if (sharedEditor) {
      for (const panel of Array.from(editorHost.children))
        if (!rows.some((row) => row.id === panel.dataset.row)) panel.remove();
      selectRow(selectedKey);
    }
    wrap._deferred = ordered.some((row) => row.firstElementChild?.firstElementChild?._deferred);
    return wrap;
  }

  function renderTree(block) {
    const wrap = el("ul", { class: "nav flex-column flex-nowrap gap-0 py-3" });
    const nodes = Array.isArray(block.nodes) ? block.nodes : [];
    if (!nodes.length) {
      wrap.appendChild(
        el("li", {
          class: "fg-2 p-3",
          text: block.empty_text || "暂无内容",
        }),
      );
      return wrap;
    }
    for (const node of nodes) wrap.appendChild(renderTreeNode(node));
    return wrap;
  }

  function renderTreeNode(node, depth = 0) {
    const box = el("li", { class: "nav-item d-block w-100 min-w-0" });
    const children = Array.isArray(node.children) ? node.children : [];
    const canExpand = children.length > 0;
    const empty = !canExpand && !node.action;
    const hasActive = (item) => item.active || (item.children || []).some(hasActive);
    let expanded = state.expandedTrees.get(node.id) ?? children.some(hasActive);
    let kids;
    const toggle = () => {
      expanded = !expanded;
      state.expandedTrees.set(node.id, expanded);
      updateExpansion();
    };
    const updateExpansion = () => {
      if (!canExpand) return;
      row.setAttribute("aria-expanded", String(expanded));
      row.classList.toggle(
        "active",
        Boolean(node.active || (!expanded && children.some(hasActive))),
      );
      if (expanded && !kids) {
        kids = el("ul", { class: "nav flex-column flex-nowrap gap-0" });
        for (const child of children) kids.appendChild(renderTreeNode(child, depth + 1));
        box.appendChild(kids);
      }
      if (kids) kids.hidden = !expanded;
    };
    const row = el("button", {
      class: `nav-link w-100 justify-content-start text-start position-relative ${canExpand || empty || node.active ? "fw-semibold" : ""} ${node.active ? "active" : ""}`,
      type: FIELD_INPUT.BUTTON,
      disabled: empty,
      title: node.label || node.id || "",
      "aria-current": node.active ? "page" : undefined,
    });
    row.style.paddingInlineStart = `${12 + depth * 14}px`;
    if (node.icon) row.appendChild(el("span", { text: node.icon, "aria-hidden": "true" }));
    const copy = el("span", { class: "tree-copy" }, [
      el("span", {
        class: "min-w-0 text-truncate",
        text: node.label || node.id || "",
        title: node.label || node.id || "",
      }),
      node.subtitle
        ? el("span", {
            class: "small fg-2 fw-normal min-w-0 text-truncate",
            text: node.subtitle,
            title: node.subtitle,
          })
        : null,
    ]);
    row.appendChild(copy);
    if (node.badge) {
      row.appendChild(
        el("span", {
          class: `badge flex-shrink-0 ${node.badge_class === "live" ? "theme-primary" : node.badge_class === "paused" ? "theme-warning" : "theme-secondary"}`,
          text: node.badge,
          role: "status",
        }),
      );
    }
    if (canExpand || empty) {
      const caret = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      caret.setAttribute("role", "button");
      caret.setAttribute("aria-label", expanded ? "收起" : "展开");
      caret.setAttribute("class", "tree-caret ms-auto flex-shrink-0");
      caret.setAttribute("width", "14");
      caret.setAttribute("height", "14");
      caret.setAttribute("viewBox", "0 0 32 32");
      caret.setAttribute("fill", "currentColor");
      caret.setAttribute("aria-hidden", "true");
      caret.innerHTML = '<path d="M16 22 6 12 7.4 10.6 16 19.2 24.6 10.6 26 12z"/>';
      caret.addEventListener("click", (event) => {
        if (!canExpand) return;
        event.preventDefault();
        event.stopPropagation();
        toggle();
      });
      row.appendChild(caret);
    }
    row.addEventListener("click", () => {
      if (node.action) runNodeAction(node.action);
      else if (canExpand) toggle();
    });
    box.appendChild(row);
    updateExpansion();
    return box;
  }

  function runNodeAction(spec) {
    if (!spec || !spec.action) return;
    postAction(spec.action, spec.value || {});
  }

  function renderRows(block) {
    const wrap = el("div", { class: "block list-group" });
    for (const item of block.items || []) {
      const row = el("div", {
        class: `list-group-item d-flex flex-wrap align-items-center gap-2 ${item.active ? "active" : ""}`,
      });
      const main = el("div", { class: "flex-grow-1 text-break" }, [
        el("div", { class: "fw-semibold", text: item.title || item.text || "" }),
        item.subtitle ? el("div", { class: "small", text: item.subtitle }) : null,
        item.meta ? el("div", { class: "small", text: item.meta }) : null,
      ]);
      row.appendChild(main);
      if (item.action) {
        row.appendChild(
          actionButton(
            item.label || "打开",
            item.action,
            item.value,
            item.class || "btn-outline theme-primary btn-sm",
          ),
        );
      }
      wrap.appendChild(row);
    }
    if (!wrap.children.length) wrap.appendChild(el("p", { class: "fg-2 p-3", text: "暂无数据" }));
    return wrap;
  }

  function renderCards(block, existing = null) {
    const wrap = existing || el("div", { class: "vstack gap-3" });
    const previous = new Map(Array.from(wrap.children, (card) => [card._cardKey, card]));
    const wanted = new Set();
    (block.items || []).forEach((item, index) => {
      const key = item.id || item.key || `card:${index}`;
      let card = previous.get(key);
      if (!card) {
        card = el("article", { class: "card" }, [
          el("header", { class: "card-header" }),
          el("div", { class: "card-body" }),
          el("footer", { class: "card-footer d-flex flex-wrap gap-2" }),
        ]);
        card._cardKey = key;
      }
      const [head, body, actions] = card.children;
      head.replaceChildren();
      if (item.title) head.appendChild(el("h3", { class: "h5 mb-1", text: item.title }));
      if (item.subtitle)
        head.appendChild(el("p", { class: "small fg-2 mb-0", text: item.subtitle }));
      if (item.meta) {
        const meta = el("div", { class: "d-flex flex-wrap gap-2 small fg-2" });
        for (const entry of Array.isArray(item.meta) ? item.meta : Object.entries(item.meta))
          meta.appendChild(
            el("span", {
              text: Array.isArray(entry) ? `${entry[0]}: ${entry[1]}` : String(entry),
            }),
          );
        head.appendChild(meta);
      }
      updateBlocks(body, item.blocks || []);
      head.hidden = !head.childNodes.length;
      actions.replaceChildren();
      for (const action of item.actions || [])
        actions.appendChild(
          actionButton(
            action.label || "执行",
            action.action,
            action.value,
            action.class || "btn-solid theme-secondary btn-sm",
          ),
        );
      actions.hidden = !actions.childNodes.length;
      wanted.add(card);
      if (wrap.children[index] !== card && !composingWithin(card))
        moveActivityNode(wrap, card, wrap.children[index] || null);
    });
    for (const card of Array.from(wrap.children))
      if (!wanted.has(card) && !composingWithin(card)) {
        destroyPlayers(card);
        card.remove();
      }
    if (!wrap.children.length) wrap.appendChild(el("p", { class: "fg-2 p-3", text: "暂无数据" }));
    wrap._deferred = Array.from(wrap.children).some(
      (card) => !wanted.has(card) || composingWithin(card) || card.children[1]?._deferred,
    );
    return wrap;
  }

  function renderForm(block) {
    const submitSpec = block.submit || {};
    const action = block.action || submitSpec.action || "";
    const form = el("form", { class: "settings-form" });
    if (block.title)
      form.appendChild(el("h3", { class: "settings-form-title", text: block.title }));
    if (block.description)
      form.appendChild(el("p", { class: "form-description", text: block.description }));
    const sections = new Map();
    const sectionFor = (name = "") => {
      if (!sections.has(name)) {
        const section = el("fieldset", { class: "form-section" });
        if (name) section.appendChild(el("legend", { text: name }));
        form.appendChild(section);
        sections.set(name, section);
      }
      return sections.get(name);
    };
    if (action) form.dataset.action = action;
    const conditionalRows = [];
    for (const field of block.fields || []) {
      if (field.hidden) {
        form.appendChild(
          el("input", {
            type: "hidden",
            name: field.name || field.key,
            value: field.value ?? "",
          }),
        );
        continue;
      }
      const section = sectionFor(field.section);
      const row = el("div", { class: "form-field" });
      if (field.visible_when) {
        row._visibleWhen = field.visible_when;
        conditionalRows.push(row);
      }
      const id = `f-${Math.random().toString(36).slice(2)}`;
      const fieldName = field.name || field.key;
      const label = el("label", { for: id, text: field.label || fieldName });
      row.appendChild(label);
      if (field.type === SETTING_TYPE.ACTION || field.input === FIELD_INPUT.BUTTON) {
        const button = actionButton(
          field.button_label || field.label || "执行",
          field.action,
          field.value && typeof field.value === "object" ? field.value : undefined,
          field.button_class || "btn-solid theme-secondary",
        );
        button.id = id;
        row.appendChild(el("div", {}, [button]));
        if (field.description)
          row.appendChild(el("small", { class: "form-text", text: field.description }));
        section.appendChild(row);
        continue;
      }
      const rawType = field.type || (field.secret ? SETTING_TYPE.SECRET : SETTING_TYPE.TEXT);
      const type = field.input || (field.secret ? FIELD_INPUT.PASSWORD : rawType);
      let input;
      if (type === FIELD_INPUT.TEXTAREA) {
        input = el("textarea", {
          class: "form-control",
          id,
          name: fieldName,
          placeholder: field.placeholder || "",
        });
        input.value = field.value ?? "";
        if (field.rows) input.rows = Number(field.rows);
      } else if (type === FIELD_INPUT.SELECT) {
        input = el("select", { class: "form-control", id, name: fieldName });
        const options = field.option_items || field.options || [];
        for (const option of options) {
          const value = typeof option === "object" ? option.value : option;
          const label = typeof option === "object" ? (option.label ?? option.value) : option;
          const opt = el("option", {
            value: String(value),
            text: String(label),
          });
          if (String(value) === String(field.value ?? field.default ?? "")) opt.selected = true;
          input.appendChild(opt);
        }
      } else if (type === FIELD_INPUT.CHECKBOX || rawType === SETTING_TYPE.BOOL) {
        input = el("input", {
          type: FIELD_INPUT.CHECKBOX,
          class: "check",
          id,
          name: fieldName,
        });
        input.checked = Boolean(field.value);
        row.insertBefore(input, label);
      } else {
        input = el("input", {
          class: "form-control",
          id,
          name: fieldName,
          type:
            type === FIELD_INPUT.PASSWORD || field.secret
              ? FIELD_INPUT.PASSWORD
              : type === FIELD_INPUT.NUMBER ||
                  rawType === SETTING_TYPE.INT ||
                  rawType === SETTING_TYPE.FLOAT
                ? FIELD_INPUT.NUMBER
                : FIELD_INPUT.TEXT,
        });
        if (!field.secret && field.value !== undefined && field.value !== null) {
          input.value = field.value;
        }
        if (field.secret) {
          input.placeholder = field.is_set
            ? `已配置（${field.secret_length || "?"} 字符）；留空保持不变`
            : "未配置；粘贴新值";
        } else if (field.placeholder) {
          input.placeholder = field.placeholder;
        }
        if (rawType === SETTING_TYPE.INT || type === FIELD_INPUT.NUMBER) input.step = "1";
        if (rawType === SETTING_TYPE.FLOAT) input.step = "any";
      }
      input.dataset.fieldType = rawType;
      if (field.readonly) {
        if (input.type === FIELD_INPUT.CHECKBOX || input.tagName === "SELECT")
          input.disabled = true;
        else input.readOnly = true;
      }
      if (input.type !== FIELD_INPUT.CHECKBOX) row.appendChild(input);
      if (field.secret) {
        const status = el("span", {
          class: `badge ${field.is_set ? "theme-success" : "theme-secondary"}`,
          role: "status",
          text: field.is_set ? `已配置，${field.secret_length || "?"} 字符` : "未配置",
        });
        const line = el("div", { class: "d-flex flex-wrap align-items-center gap-2" }, [status]);
        if (field.is_set) {
          line.appendChild(
            actionButton(
              "清除",
              "settings.clear_secret",
              { key: fieldName },
              "btn-solid theme-danger btn-sm",
            ),
          );
        }
        row.appendChild(line);
      }
      if (field.description) {
        const descriptionId = `${id}-description`;
        input.setAttribute("aria-describedby", descriptionId);
        row.appendChild(
          el("small", { id: descriptionId, class: "form-text", text: field.description }),
        );
      }
      section.appendChild(row);
    }
    const actions = el("div", { class: "form-actions" });
    const submit = el("button", {
      class: submitSpec.class || "btn-solid theme-primary",
      type: FIELD_INPUT.SUBMIT,
      text: block.submit_label || submitSpec.label || "保存",
    });
    actions.appendChild(submit);
    form.appendChild(actions);
    const applyVisibility = () => {
      const controls = new Map();
      for (const control of Array.from(form.elements)) {
        if (control.name) controls.set(control.name, control);
      }
      for (const row of conditionalRows) {
        const condition = row._visibleWhen;
        const controller = controls.get(condition.key);
        const value = controller
          ? String(controller.type === FIELD_INPUT.CHECKBOX ? controller.checked : controller.value)
          : "";
        const visible = (condition.values || []).map(String).includes(value);
        row.hidden = !visible;
        row.dataset.hidden = visible ? "0" : "1";
      }
      for (const section of sections.values())
        section.hidden = !Array.from(section.querySelectorAll(".form-field")).some(
          (row) => !row.hidden,
        );
    };
    applyVisibility();
    form.addEventListener("change", applyVisibility);
    for (const control of Array.from(form.elements)) {
      if (control.name)
        control._initialValue =
          control.type === FIELD_INPUT.CHECKBOX ? control.checked : control.value;
    }
    return form;
  }

  function retainFormDraft(previous, replacement) {
    const controls = new Map(
      Array.from(replacement.elements)
        .filter((control) => control.name)
        .map((control) => [control.name, control]),
    );
    for (const control of Array.from(previous.elements)) {
      const target = controls.get(control.name);
      if (!target || control.type === "hidden") continue;
      const value = control.type === FIELD_INPUT.CHECKBOX ? control.checked : control.value;
      if (value === control._initialValue) continue;
      if (target.type === FIELD_INPUT.CHECKBOX) target.checked = value;
      else target.value = value;
    }
    replacement.dispatchEvent(new Event("change"));
  }

  function renderSchemaForm(block, existing = null) {
    const form = existing || el("form", { class: "settings-form schema-form" });
    form._schemaBlock = block;
    if (
      existing &&
      (form._dirty || form._saving || form._schemaSignature === JSON.stringify(block))
    )
      return form;
    form._schemaSignature = JSON.stringify(block);
    const disclosureStates = new Map(
      Array.from(form.querySelectorAll("details[data-path]"), (details) => [
        details.dataset.path,
        details.open,
      ]),
    );
    form.replaceChildren();
    const rootSchema = block.schema || {};
    const resolve = (schema) => {
      if (!schema || typeof schema !== "object") return {};
      if (schema.$ref?.startsWith("#/")) {
        let target = rootSchema;
        for (const part of schema.$ref.slice(2).split("/"))
          target = target?.[part.replace(/~1/g, "/").replace(/~0/g, "~")];
        return {
          ...resolve(target),
          ...Object.fromEntries(Object.entries(schema).filter(([key]) => key !== "$ref")),
        };
      }
      if (schema.allOf) {
        const merged = {
          ...schema,
          properties: { ...schema.properties },
          required: [...(schema.required || [])],
        };
        for (const item of schema.allOf) {
          const part = resolve(item);
          Object.assign(merged, part, {
            properties: { ...merged.properties, ...part.properties },
            required: [...merged.required, ...(part.required || [])],
          });
        }
        delete merged.allOf;
        return merged;
      }
      return schema;
    };
    const typeOf = (value) =>
      value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
    const seed = (schema) => {
      schema = resolve(schema);
      if (schema.default !== undefined) return structuredClone(schema.default);
      if (schema.const !== undefined) return structuredClone(schema.const);
      if (schema.enum) return schema.enum[0];
      if (schema.anyOf || schema.oneOf) return seed((schema.anyOf || schema.oneOf)[0]);
      const type = Array.isArray(schema.type) ? schema.type[0] : schema.type;
      return type === "object" || schema.properties
        ? {}
        : type === "array"
          ? []
          : type === "number" || type === "integer"
            ? 0
            : type === "boolean"
              ? false
              : type === "null"
                ? null
                : "";
    };
    const localButton = (label, callback, cls = "btn-solid theme-secondary btn-sm") => {
      const button = el("button", {
        type: "button",
        class: cls,
        text: label,
      });
      button.addEventListener("click", () => {
        callback();
        form._dirty = true;
      });
      return button;
    };
    const field = (rawSchema, value, title, optional = false, path = title) => {
      let schema = resolve(rawSchema);
      const alternatives = schema.oneOf || schema.anyOf;
      if (alternatives?.length && alternatives.every((item) => resolve(item).const !== undefined)) {
        schema = { ...schema, enum: alternatives.map((item) => resolve(item).const) };
        delete schema.oneOf;
        delete schema.anyOf;
      }
      if (schema["x-hidden"]) return { node: null, read: () => value };
      const box = el("fieldset", { class: "schema-field", dataset: { path } });
      const legend = el("legend", { text: schema.title || title });
      box.appendChild(legend);
      const body = el("fieldset", { class: "schema-field-body" });
      const booleanField = schema.type === "boolean";
      const grouped = schema.type === "object" || schema.type === "array" || schema.properties;
      box.classList.toggle("schema-scalar", !grouped);
      if (optional && grouped) {
        const details = el("details", { class: "disclosure", dataset: { path } }, [
          el("summary", { text: schema.title || title }),
        ]);
        details.open = disclosureStates.get(path) || false;
        const content = el("div", { class: "disclosure-body" });
        const enabled = el("input", {
          type: "checkbox",
          class: "check",
          dataset: { path: `${path}:enabled` },
        });
        enabled.checked = value !== undefined;
        const holder = el("div");
        let child;
        const mount = () => {
          if (enabled.checked && !child) {
            child = field(schema, value === undefined ? seed(schema) : value, title, false, path);
            holder.appendChild(child.node);
          }
          holder.hidden = !enabled.checked;
        };
        content.append(
          el("label", { class: "schema-option" }, [enabled, document.createTextNode("启用")]),
          holder,
        );
        details.appendChild(content);
        enabled.addEventListener("change", mount);
        details.addEventListener("toggle", () => {
          if (details.open) mount();
        });
        if (details.open) mount();
        return {
          node: details,
          read: () =>
            !enabled.checked
              ? undefined
              : child
                ? child.read()
                : value === undefined
                  ? seed(schema)
                  : value,
        };
      }
      let omitted = null;
      if (optional && !booleanField) {
        omitted = el("input", { type: "checkbox", class: "check" });
        omitted.checked = grouped ? value !== undefined : value === undefined;
        const label = el("label", { class: "schema-option" }, [
          omitted,
          document.createTextNode(grouped ? "启用" : "使用默认值"),
        ]);
        box.appendChild(label);
        const disabled = () => (grouped ? !omitted.checked : omitted.checked);
        body.disabled = disabled();
        omitted.addEventListener("change", () => {
          body.disabled = disabled();
        });
      }
      if (schema.description)
        box.appendChild(el("p", { class: "form-text", text: schema.description }));
      box.appendChild(body);
      let read;
      const branches =
        schema.oneOf ||
        schema.anyOf ||
        (Array.isArray(schema.type) ? schema.type.map((type) => ({ ...schema, type })) : null);
      const declaredType =
        schema.type || (schema.properties || schema.additionalProperties ? "object" : undefined);
      if (branches || (!declaredType && !schema.enum && schema.const === undefined)) {
        const options =
          branches ||
          ["string", "number", "boolean", "null", "object", "array"].map((type) => ({ type }));
        const selector = el("select", { class: "form-control", "aria-label": `${title} 类型` });
        const initial = value === undefined ? seed(options[0]) : value;
        const matches = (option) => {
          const candidate = resolve(option);
          return candidate.const !== undefined
            ? candidate.const === initial
            : candidate.enum
              ? candidate.enum.includes(initial)
              : (candidate.type || (candidate.properties ? "object" : undefined)) ===
                  typeOf(initial) ||
                (candidate.type === "integer" && Number.isInteger(initial));
        };
        let selected = Math.max(0, options.findIndex(matches));
        options.forEach((option, index) =>
          selector.appendChild(
            el("option", {
              value: String(index),
              text: option.title || option.type || `选项 ${index + 1}`,
            }),
          ),
        );
        selector.value = String(selected);
        const holder = el("div");
        let child;
        const renderChoice = (nextValue) => {
          child = field(options[selected], nextValue, title, false, path);
          holder.replaceChildren(child.node);
        };
        renderChoice(initial);
        selector.addEventListener("change", () => {
          selected = Number(selector.value);
          renderChoice(seed(options[selected]));
        });
        body.append(selector, holder);
        read = () => child.read();
      } else if (declaredType === "object") {
        const object =
          value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
        const properties = schema.properties || {};
        const fields = [];
        for (const [key, property] of Object.entries(properties)) {
          const child = field(
            property,
            object[key],
            key,
            !(schema.required || []).includes(key),
            path ? `${path}.${key}` : key,
          );
          fields.push([key, child]);
          if (child.node) body.appendChild(child.node);
        }
        const rows = [];
        const list = el("div", { class: "schema-entries" });
        const additional =
          typeof schema.additionalProperties === "object" ? schema.additionalProperties : {};
        const addEntry = (key, entryValue) => {
          const row = el("div", { class: "schema-entry" });
          const keyInput = el("input", {
            class: "form-control",
            value: key,
            "aria-label": `${title} 参数名`,
          });
          const child = field(additional, entryValue, "参数值");
          const entry = { row, keyInput, child };
          rows.push(entry);
          row.append(
            keyInput,
            child.node,
            el("div", { class: "form-actions" }, [
              localButton(
                "删除",
                () => {
                  rows.splice(rows.indexOf(entry), 1);
                  row.remove();
                },
                "btn-outline theme-danger btn-sm",
              ),
            ]),
          );
          list.appendChild(row);
        };
        for (const [key, entryValue] of Object.entries(object))
          if (!Object.hasOwn(properties, key)) addEntry(key, entryValue);
        if (schema.additionalProperties !== false || rows.length) {
          body.append(
            el("h4", { class: "fs-md", text: Object.keys(properties).length ? "扩展参数" : title }),
            list,
          );
          body.appendChild(localButton("添加参数", () => addEntry("", seed(additional))));
        }
        read = () => {
          const result = {};
          for (const [key, child] of fields) {
            const entryValue = child.read();
            if (entryValue !== undefined)
              Object.defineProperty(result, key, {
                value: entryValue,
                enumerable: true,
                writable: true,
                configurable: true,
              });
          }
          for (const { keyInput, child } of rows) {
            const key = keyInput.value;
            if (Object.hasOwn(properties, key) || Object.hasOwn(result, key))
              throw new Error(`${title} 的参数名重复：${key}`);
            Object.defineProperty(result, key, {
              value: child.read(),
              enumerable: true,
              writable: true,
              configurable: true,
            });
          }
          return result;
        };
      } else if (declaredType === "array") {
        const list = el("div", { class: "schema-entries" });
        const entries = [];
        const addItem = (itemValue) => {
          const row = el("div", { class: "schema-entry" });
          const child = field(schema.items || {}, itemValue, `${title} 项目`);
          const entry = { row, child };
          entries.push(entry);
          const move = (direction) => {
            const index = entries.indexOf(entry);
            const target = index + direction;
            if (target < 0 || target >= entries.length) return;
            [entries[index], entries[target]] = [entries[target], entries[index]];
            entries.forEach((item, position) => {
              if (list.children[position] !== item.row)
                list.insertBefore(item.row, list.children[position] || null);
            });
          };
          row.append(
            child.node,
            el("div", { class: "form-actions" }, [
              localButton("上移", () => move(-1)),
              localButton("下移", () => move(1)),
              localButton(
                "删除",
                () => {
                  entries.splice(entries.indexOf(entry), 1);
                  row.remove();
                },
                "btn-outline theme-danger btn-sm",
              ),
            ]),
          );
          list.appendChild(row);
        };
        for (const item of Array.isArray(value) ? value : []) addItem(item);
        body.append(
          list,
          localButton("添加项目", () => addItem(seed(schema.items || {}))),
        );
        read = () => entries.map((entry) => entry.child.read());
      } else {
        const options = schema.enum || (schema.const !== undefined ? [schema.const] : null);
        let input;
        if (options || declaredType === "boolean" || declaredType === "null") {
          const values =
            options ||
            (declaredType === "null"
              ? [null]
              : optional
                ? [undefined, true, false]
                : [true, false]);
          input = el("select", { class: "form-control", "aria-label": schema.title || title });
          values.forEach((item, index) =>
            input.appendChild(
              el("option", {
                value: String(index),
                text:
                  item === undefined ? "-" : typeof item === "string" ? item : JSON.stringify(item),
              }),
            ),
          );
          input.value = String(
            Math.max(
              0,
              values.findIndex((item) => item === value),
            ),
          );
          read = () => values[Number(input.value)];
        } else if (declaredType === "number" || declaredType === "integer") {
          input = el("input", {
            type: "number",
            class: "form-control",
            "aria-label": schema.title || title,
            step: declaredType === "integer" ? "1" : "any",
          });
          if (value !== undefined) input.value = String(value);
          if (schema.minimum !== undefined) input.min = String(schema.minimum);
          if (schema.maximum !== undefined) input.max = String(schema.maximum);
          read = () => {
            if (!input.value.trim() || !Number.isFinite(Number(input.value)))
              throw new Error(`${title} 需要填写数字．`);
            return Number(input.value);
          };
        } else {
          const multiline =
            schema["x-multiline"] || (typeof value === "string" && value.includes("\n"));
          input = multiline
            ? el("textarea", {
                class: "form-control",
                rows: 3,
                "aria-label": schema.title || title,
              })
            : el("input", {
                type: "text",
                class: "form-control",
                "aria-label": schema.title || title,
              });
          input.value = value === undefined ? "" : String(value);
          read = () => input.value;
        }
        body.appendChild(input);
        input.dataset.path = path;
        input.name = path;
      }
      return {
        node: box,
        read: () =>
          omitted && (grouped ? !omitted.checked : omitted.checked) ? undefined : read(),
      };
    };
    if (block.title)
      form.appendChild(el("h3", { class: "settings-form-title", text: block.title }));
    if (block.description)
      form.appendChild(el("p", { class: "form-description", text: block.description }));
    const editor = field(rootSchema, block.value, block.title || "配置", false, "");
    const status = el("p", { class: "form-text", role: "status", "aria-live": "polite" });
    const save = el("button", {
      type: "submit",
      class: "btn-solid theme-primary",
      text: block.submit?.label || "保存",
    });
    form.append(editor.node, status, el("div", { class: "form-actions" }, [save]));
    form._dirty = false;
    if (!existing) {
      form.addEventListener("input", () => {
        form._dirty = true;
      });
      form.addEventListener("change", () => {
        form._dirty = true;
      });
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        await form._saveDefinition();
      });
    }
    form._saveDefinition = async () => {
      if (form._saving) return;
      try {
        const definition = editor.read();
        const snapshot = JSON.stringify(definition);
        const spec = form._schemaBlock;
        form._saving = true;
        save.disabled = true;
        status.textContent = "正在保存";
        const result = await postAction(spec.submit.action, { ...spec.context, definition });
        if (result?.ok) {
          form._dirty = JSON.stringify(editor.read()) !== snapshot;
          status.textContent = "已保存";
        } else status.textContent = result?.message || "保存失败，请修改后重试．";
      } catch (error) {
        status.textContent = `保存失败：${error.message}`;
      } finally {
        form._saving = false;
        save.disabled = false;
      }
    };
    return form;
  }

  function renderTabs(block, existing = null) {
    const wrap = existing || el("div", { class: "vstack gap-3" });
    if (!wrap._tabState) {
      const bar = el("div", { class: "nav nav-tabs", role: "tablist" });
      wrap.appendChild(bar);
      wrap._tabState = {
        bar,
        panels: new Map(),
        buttons: new Map(),
        active: null,
      };
    }
    const data = wrap._tabState;
    wrap.classList.toggle("settings-layout", block.layout === "settings");
    wrap.classList.toggle("settings-pages", block.layout === "settings-pages");
    data.bar.setAttribute(
      "aria-orientation",
      block.layout === "settings" ? "vertical" : "horizontal",
    );
    data.bar.setAttribute("aria-label", block.label || block.title || "Tabs");
    const tabs = block.tabs || [];
    data.bar.hidden = Boolean(block.hide_single && tabs.length <= 1);
    const key =
      block.id || block.key || `tabs:${tabs.map((tab) => tab.title || "").join("\u0001")}`;
    const tabKey = (tab, index) => tab.id || tab.key || tab.title || `tab:${index}`;
    const navigation = navigationFromUrl();
    const documentTabs = tabs.some((tab) => tab.meta?.doc_key && tab.meta?.page_id);
    const requested = tabs.find(
      (tab) =>
        navigation.doc &&
        tab.meta?.doc_key === navigation.doc &&
        tab.meta?.page_id === navigation.page,
    );
    const remembered = !documentTabs && tabs.find((tab) => tab.title === state.blockTabs[key]);
    const selected = documentTabs
      ? requested || tabs[0]
      : remembered || tabs.find((tab) => tab.active) || tabs[0];
    let active = selected ? tabKey(selected, tabs.indexOf(selected)) : null;
    const focusedView = Array.from(agentViews.values()).find(
      (view) => wrap.contains(view.root) && agentDraft(view).composing,
    );
    if (focusedView && data.active !== active) active = data.active;
    const show = (target, remember = true) => {
      if (documentTabs) {
        const navigation = navigationFromUrl();
        const selected =
          tabs.find(
            (tab) => tab.meta?.doc_key === navigation.doc && tab.meta?.page_id === navigation.page,
          ) || tabs[0];
        target = selected ? tabKey(selected, tabs.indexOf(selected)) : null;
      }
      const index = tabs.findIndex((tab, index) => tabKey(tab, index) === target);
      if (index < 0) return;
      if (
        Array.from(data.panels.values()).some((panel) => !panel.hidden && composingWithin(panel))
      ) {
        data.deferredTarget = target;
        return;
      }
      for (const view of agentViews.values())
        if (wrap.contains(view.root)) {
          saveAgentEditor(view);
          if (document.activeElement === view.input) view.restoreFocus = true;
        }
      const changed = data.active !== target;
      data.active = target;
      if (remember && !documentTabs) state.blockTabs[key] = tabs[index].title || "";
      for (const [id, panel] of data.panels) panel.hidden = id !== target;
      for (const [id, button] of data.buttons) {
        button.classList.toggle("active", id === target);
        button.setAttribute("aria-selected", String(id === target));
        button.tabIndex = id === target ? 0 : -1;
      }
      const panel = data.panels.get(target);
      for (const view of agentViews.values())
        if (changed && panel.contains(view.root) && view.input.getClientRects().length) {
          restoreAgentEditor(view, remember && Boolean(view.restoreFocus));
          if (remember) view.restoreFocus = false;
        }
      if (remember) typeset(panel);
    };
    data.show = show;
    const wanted = new Set();
    tabs.forEach((tab, index) => {
      const id = tabKey(tab, index);
      wanted.add(id);
      let panel = data.panels.get(id);
      let button = data.buttons.get(id);
      if (!panel) {
        const panelId = `panel-${Math.random().toString(36).slice(2)}`;
        panel = el("div", {
          class: "tabs-body",
          id: panelId,
          role: "tabpanel",
          "aria-labelledby": `${panelId}-tab`,
        });
        button = el("button", {
          class: "tab-btn nav-link",
          id: `${panelId}-tab`,
          role: "tab",
          "aria-controls": panelId,
          type: FIELD_INPUT.BUTTON,
        });
        data.panels.set(id, panel);
        data.buttons.set(id, button);
        wrap.appendChild(panel);
        data.bar.appendChild(button);
        panel.addEventListener("compositionend", () => {
          if (data.deferredTarget !== undefined) {
            const target = data.deferredTarget;
            delete data.deferredTarget;
            data.show(target);
          }
        });
        button.addEventListener("click", async () => {
          const meta = button._tab.meta || {};
          if (!meta.doc_key || !meta.page_id) {
            data.show(id);
            return;
          }
          try {
            await changeNavigation(() =>
              api("/api/ui/navigate", {
                method: "POST",
                body: JSON.stringify({
                  ...navigationFromUrl(),
                  page: meta.page_id,
                  doc: meta.doc_key,
                }),
              }),
            );
          } catch (err) {
            toast(`打开页面失败：${err.message}`, TOAST_KIND.ERR);
          }
        });
        button.addEventListener("keydown", (event) => {
          const vertical = data.bar.getAttribute("aria-orientation") === "vertical";
          const previous = vertical ? "ArrowUp" : "ArrowLeft";
          const next = vertical ? "ArrowDown" : "ArrowRight";
          if (![previous, next, "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const buttons = Array.from(data.bar.children);
          const position = buttons.indexOf(button);
          const index =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? buttons.length - 1
                : (position + (event.key === next ? 1 : -1) + buttons.length) % buttons.length;
          buttons[index].focus();
          buttons[index].click();
        });
        button.addEventListener("focus", () => {
          button.scrollIntoView({ block: "nearest", inline: "nearest" });
        });
      }
      button._tab = tab;
      button.textContent = tab.title || `Tab ${index + 1}`;
      if (data.bar.hidden) {
        panel.removeAttribute("aria-labelledby");
        panel.setAttribute("aria-label", tab.title || block.label || "Tab");
      } else {
        panel.removeAttribute("aria-label");
        panel.setAttribute("aria-labelledby", button.id);
      }
      const meta = tab.meta || {};
      panel.classList.toggle("prose", Boolean(meta.doc_key));
      panel.classList.toggle("w-100", Boolean(meta.doc_key));
      panel.classList.toggle("min-w-0", Boolean(meta.doc_key));
      Object.assign(panel.dataset, {
        docKey: meta.doc_key || "",
        docPath: meta.path || "",
        pageId: meta.page_id || "",
        owner: meta.owner || "",
        selectionActions: JSON.stringify(meta.selection_actions || []),
      });
      updateBlocks(panel, tab.blocks || []);
      for (const content of panel.querySelectorAll(":scope > .markdown")) {
        content.classList.toggle("prose", Boolean(meta.doc_key));
        content.classList.toggle("w-100", Boolean(meta.doc_key));
        content.classList.toggle("min-w-0", Boolean(meta.doc_key));
        content.classList.toggle("flex-shrink-0", Boolean(meta.doc_key));
      }
    });
    for (const [id, panel] of data.panels)
      if (!wanted.has(id) && !composingWithin(panel)) {
        destroyPlayers(panel);
        panel.remove();
        data.buttons.get(id).remove();
        data.panels.delete(id);
        data.buttons.delete(id);
      }
    if (!focusedView) show(active, false);
    wrap._deferred =
      Boolean(focusedView) || Array.from(data.panels.values()).some((panel) => panel._deferred);
    return wrap;
  }

  function playerKey(block) {
    return block.id || "player";
  }

  function attachStream(video, source, startAt) {
    const rawSrc = String(source.src || "");
    if (!rawSrc) return;
    // Relay every media URL through this server: the browser may be on a VPN
    // that cannot reach live.sjtu.edu.cn, while the server can.
    const src = relayedMediaUrl(rawSrc);
    if (rawSrc.includes(".flv") && window.mpegts && window.mpegts.isSupported()) {
      const player = window.mpegts.createPlayer({
        type: "flv",
        isLive: true,
        url: src,
      });
      player.attachMediaElement(video);
      player.load();
      video._mpegts = player;
      return;
    }
    const isHls = rawSrc.includes(".m3u8");
    const nativeHls = video.canPlayType("application/vnd.apple.mpegurl");
    if (isHls && !nativeHls && window.Hls && window.Hls.isSupported()) {
      const hls = new window.Hls({
        startPosition: startAt > 1 ? startAt : -1,
        lowLatencyMode: true,
        maxBufferLength: 30,
      });
      hls.loadSource(src);
      hls.attachMedia(video);
      video._hls = hls;
      return;
    }
    video.src = src;
    if (startAt > 1) {
      video.addEventListener(
        "loadedmetadata",
        () => {
          try {
            video.currentTime = startAt;
          } catch (_) {}
        },
        { once: true },
      );
    }
  }

  // HLS playlists carry absolute segment URLs, so only progressive/FLV media is
  // relayed; those are the formats the SJTU recorder actually returns.
  function relayedMediaUrl(src) {
    if (!src) return src;
    if (String(src).includes(".m3u8")) return src;
    return `/api/media?src=${encodeURIComponent(src)}`;
  }

  function renderPlayer(block) {
    const key = playerKey(block);
    const cache = state.playerCache[key] || {};
    const wrap = el("div", { class: "block player-block vstack gap-3" });
    const finished = Boolean(block.finished);
    const live = Boolean(block.live);
    const processing = Boolean(block.processing);

    const stateText = () => {
      if (finished) return STATUS_LABEL.FINISHED;
      if (block.queued) return live ? `直播，${STATUS_LABEL.QUEUED}` : STATUS_LABEL.QUEUED;
      if (live)
        return processing ? `直播，${STATUS_LABEL.RUNNING}` : `直播，${STATUS_LABEL.PAUSED}`;
      return processing ? STATUS_LABEL.RUNNING : STATUS_LABEL.PAUSED;
    };
    const stateTheme = finished
      ? "success"
      : block.queued
        ? "secondary"
        : processing
          ? "primary"
          : "warning";
    const statusBar = el("div", { class: "player-status-bar" }, [
      el("span", {
        class: `badge theme-${stateTheme}`,
        role: "status",
        text: stateText(),
      }),
    ]);
    wrap.appendChild(statusBar);

    const grid = el("div", { class: "player-grid" });
    const videos = [];
    const sources = Array.isArray(block.sources) ? block.sources : [];

    for (const source of sources) {
      const saved = block.live ? {} : cache[source.key] || {};
      const pane = el("div", { class: "card player-pane min-w-0" });
      pane.appendChild(
        el("div", {
          class: "card-header",
          text: source.label || source.key || "",
        }),
      );
      const video = el("video", {
        class: "player-video",
        controls: "controls",
        playsinline: "",
        preload: "metadata",
      });
      video.muted = Boolean(source.muted);
      video.dataset.srcKey = source.key || "";
      attachStream(video, source, Number(saved.t || 0));
      const remember = () => {
        state.playerCache[key] = state.playerCache[key] || {};
        state.playerCache[key][source.key] = {
          t: Number.isFinite(video.currentTime) ? video.currentTime : 0,
        };
      };
      video.addEventListener("timeupdate", remember);
      video.addEventListener("pause", remember);
      video.addEventListener("seeked", remember);
      const hint = el("div", { class: "player-hint fg-danger", role: "status", hidden: true });
      video.addEventListener("error", () => {
        hint.hidden = false;
        hint.textContent =
          video._mpegts || video._hls
            ? "视频加载失败：转发代理无法连上视频源"
            : "视频无法播放（服务器转发失败）";
      });
      videos.push(video);
      pane.appendChild(el("div", { class: "player-stage" }, [video]));
      pane.appendChild(hint);
      grid.appendChild(pane);
    }

    if (block.screen_missing) {
      const emptyPane = el("div", { class: "card player-pane min-w-0" }, [
        el("div", { class: "card-header", text: "屏幕画面" }),
        el("div", { class: "player-stage-empty", text: "该课次没有屏幕流" }),
      ]);
      grid.appendChild(emptyPane);
    }
    wrap.appendChild(grid);

    const playAll = () => {
      for (const video of videos) {
        if (video.dataset.srcKey === "screen") video.muted = true;
        const tryPlay = () => {
          const p = video.play();
          if (p && p.catch) p.catch(() => {});
        };
        video.addEventListener("canplay", tryPlay, { once: true });
        tryPlay();
      }
    };

    // Live video keeps playing while processing is paused; recordings only
    // play while processing is running.
    if (!finished && (live || processing)) {
      playAll();
    }

    return wrap;
  }

  function renderSwitch(block) {
    const id = `switch-${Math.random().toString(36).slice(2)}`;
    const input = el("input", {
      type: FIELD_INPUT.CHECKBOX,
      role: "switch",
      switch: "",
      id,
    });
    input.checked = Boolean(block.checked);
    input.addEventListener("change", () => {
      postAction(block.action, { enabled: input.checked });
    });
    const content = el("div", { class: "form-field-content" }, [
      el("label", { for: id, text: block.label || "" }),
      block.hint ? el("small", { class: "form-text", text: block.hint }) : null,
    ]);
    return el("div", { class: "block form-field" }, [
      el("div", { class: "switch" }, [input]),
      content,
    ]);
  }

  function qaPreview(text) {
    const flat = String(text || "")
      .replace(/\s+/g, " ")
      .trim();
    return flat.length > 90 ? `${flat.slice(0, 90)}…` : flat || "（提问）";
  }

  function qaLevelLabel(intensity) {
    if (intensity === ASK_INTENSITY.RESEARCH) return "研究";
    if (intensity === ASK_INTENSITY.INSTANT) return "快速";
    return "";
  }

  const qaLiveCache = new Map();

  function qaKey(path, id) {
    return JSON.stringify([path || "", id]);
  }

  function qaActivity(block) {
    return (
      qaLiveCache.get(qaKey(block.path, block.id))?.activity ||
      block.activity || { id: `qa-${block.id}`, agents: [] }
    );
  }

  function renderQA(block, existing = null) {
    const cacheKey = qaKey(block.path, block.id);
    const cached = qaLiveCache.get(cacheKey);
    if (block.status === QA_STATUS.PENDING && cached)
      block =
        cached.status && cached.status !== QA_STATUS.PENDING
          ? { ...block, ...cached }
          : {
              ...cached,
              ...block,
              activity: block.activity || cached.activity,
            };
    const pending = block.status === QA_STATUS.PENDING;
    const retained = retainedQaActivity(block);
    if (pending)
      qaLiveCache.set(cacheKey, {
        ...cached,
        ...block,
        activity: block.activity || qaActivity(block),
      });
    if (!pending && retained)
      qaLiveCache.set(cacheKey, {
        ...cached,
        ...block,
        activity: qaActivity(block),
      });
    if (existing) {
      if (pending) {
        const activity = qaActivity(block);
        const root = renderAgentActivity(activity);
        const view = activityView(activity);
        view.qa = cacheKey;
        const answer = existing.querySelector(".qa-answer");
        if (root.parentNode !== answer) answer.appendChild(root);
      } else finishQaBlock(existing, block);
      return existing;
    }
    const collapsed = Boolean(block.collapsed) && !pending && !retained;
    const wrap = el("section", {
      class: "block qa card min-w-0",
      dataset: {
        qaId: block.id || "",
        qaPath: block.path || "",
        qaStatus: block.status || "",
      },
    });
    const head = el("div", {
      class: "card-header d-flex flex-row flex-wrap align-items-center gap-2",
    });
    head.appendChild(el("span", { class: "badge theme-primary", text: "问" }));
    head.appendChild(
      el("span", { class: "text-break flex-grow-1 min-w-0", text: qaPreview(block.question) }),
    );
    const level = qaLevelLabel(block.intensity);
    if (level) head.appendChild(el("span", { class: "badge theme-secondary", text: level }));
    if (block.ts) head.appendChild(el("time", { class: "small fg-2", text: block.ts }));
    if (pending)
      head.appendChild(
        el("span", {
          class: "qa-pending badge theme-primary",
          role: "status",
          text: STATUS_LABEL.RUNNING,
        }),
      );
    const toggle = el("button", {
      class: "qa-toggle btn-text btn-icon theme-secondary btn-sm",
      type: FIELD_INPUT.BUTTON,
      title: "折叠 / 展开",
      "aria-label": "折叠 / 展开",
      "aria-expanded": String(!collapsed),
    });
    toggle.innerHTML =
      '<svg width="14" height="14" fill="currentColor" viewBox="0 0 16 16" aria-hidden="true"><path fill-rule="evenodd" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708"/></svg>';
    toggle.addEventListener("click", async () => {
      toggle.disabled = true;
      try {
        await postAction(block.collapse_action || "reader.qa_collapse", {
          path: block.path,
          id: block.id,
          collapsed: !wrap.querySelector(".qa-body").hidden,
        });
      } finally {
        toggle.disabled = false;
      }
    });
    const remove = actionButton(
      "删除",
      block.delete_action || "reader.qa_delete",
      { path: block.path, id: block.id },
      "btn-text btn-icon theme-danger btn-sm flex-shrink-0",
    );
    remove.title = "删除";
    remove.setAttribute("aria-label", "删除");
    remove.innerHTML =
      '<svg width="15" height="15" fill="currentColor" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708"/></svg>';
    head.appendChild(el("div", { class: "qa-actions" }, [toggle, remove]));

    const body = el("div", { class: "qa-body card-body", hidden: collapsed });
    const answerBox = el("div", {
      class: `qa-answer ${block.status === QA_STATUS.ERROR ? "fg-danger" : ""}`,
    });
    if (pending || retained) {
      const activity = qaActivity(block);
      answerBox.appendChild(renderAgentActivity(activity));
      activityView(activity).qa = cacheKey;
    }
    if (!pending) {
      answerBox.appendChild(
        el("div", {
          class: "qa-text markdown prose",
          html: block.answer_html || escapeHtml(block.answer || "（无内容）"),
        }),
      );
    }
    body.appendChild(answerBox);

    wrap.appendChild(head);
    wrap.appendChild(body);
    return wrap;
  }

  function qaNode(path, id) {
    return (
      $$(".qa").find(
        (node) => node.dataset.qaId === id && (!path || node.dataset.qaPath === path),
      ) || null
    );
  }

  function withScrollAnchor(node, mutate) {
    const scroller = $("#page-content");
    const before = node.getBoundingClientRect();
    const heightBefore = node.offsetHeight;
    const scrollerTop = scroller ? scroller.getBoundingClientRect().top : 0;
    mutate();
    if (!scroller) return;
    const heightAfter = node.offsetHeight;
    if (before.bottom <= scrollerTop && heightAfter !== heightBefore) {
      // The block is above the viewport: keep the visible content where it is.
      scroller.scrollTop += heightAfter - heightBefore;
    }
  }

  function finishQaBlock(node, payload) {
    node.dataset.qaStatus = payload.status || QA_STATUS.DONE;
    const answerBox = node.querySelector(".qa-answer");
    const activity = node.querySelector(".agent-workflow");
    const view = Array.from(agentViews.values()).find((item) => item.root === activity);
    const retained = retainAgentView(view);
    if (answerBox) {
      answerBox.classList.toggle("fg-danger", payload.status === QA_STATUS.ERROR);
      let answer = answerBox.querySelector(".qa-text");
      if (!answer) {
        answer = el("div", { class: "qa-text markdown prose" });
        answerBox.appendChild(answer);
      }
      const html = payload.answer_html || escapeHtml(payload.answer || "（无内容）");
      if (answer._answerHtml !== html) {
        answer.innerHTML = html;
        answer._answerHtml = html;
        typeset(answer);
      }
      if (view && !retained) disposeAgentView(view.key);
    }
    const pending = node.querySelector(".qa-pending");
    if (pending) pending.remove();
    const collapsed = !retained && payload.collapsed !== false;
    node.querySelector(".qa-body").hidden = collapsed;
    const toggle = node.querySelector(".qa-toggle");
    if (toggle) toggle.setAttribute("aria-expanded", String(!collapsed));
  }

  function applyQaLive(payload) {
    if (!payload || !payload.id) return;
    const key = qaKey(payload.path, payload.id);
    const cached = { ...qaLiveCache.get(key), ...payload };
    const finished = payload.status && payload.status !== QA_STATUS.PENDING;
    qaLiveCache.set(key, cached);
    const node = qaNode(payload.path || "", payload.id);
    const update = () => {
      if (finished) {
        const activityId = payload.activity?.id || `qa-${payload.id}`;
        for (const viewKey of agentViews.keys()) {
          if (JSON.parse(viewKey)[1] === activityId && !retainAgentView(agentViews.get(viewKey)))
            disposeAgentView(viewKey);
        }
        if (node) finishQaBlock(node, payload);
      } else if (payload.activity && node?.dataset.qaStatus === QA_STATUS.PENDING) {
        const answerBox = node.querySelector(".qa-answer");
        if (answerBox) {
          const activity = renderAgentActivity(payload.activity);
          activityView(payload.activity).qa = key;
          if (activity.parentNode !== answerBox) answerBox.appendChild(activity);
        }
      }
    };
    if (node) withScrollAnchor(node, update);
    else update();
  }

  function renderBlocks(blocks) {
    const frag = document.createDocumentFragment();
    for (const block of blocks || []) {
      const node = renderBlock(block);
      if (node) {
        node._blockKey = stableBlockKey(block, frag.childNodes.length);
        node._block = block;
        node._blockSignature = JSON.stringify(block);
      }
      if (node) frag.appendChild(node);
    }
    return frag;
  }

  function stableBlockKey(block, index) {
    return block.type === BLOCK_TYPE.QA
      ? `qa:${qaKey(block.path, block.id)}`
      : `${block.type || BLOCK_TYPE.TEXT}:${block.id || block.key || index}`;
  }

  function composingWithin(node) {
    return Array.from(agentViews.values()).some(
      (view) => node.contains(view.root) && agentDraft(view).composing,
    );
  }

  function moveActivityNode(parent, node, before) {
    const focused = Array.from(agentViews.values()).find(
      (view) => node.contains(view.input) && document.activeElement === view.input,
    );
    if (focused) saveAgentEditor(focused);
    parent.insertBefore(node, before);
    if (focused && !agentDraft(focused).composing) restoreAgentEditor(focused, true);
  }

  function updateBlocks(parent, blocks) {
    if (parent.querySelector("[data-ask-editor]")) {
      parent._deferred = true;
      return false;
    }
    const previous = new Map(Array.from(parent.children, (node) => [node._blockKey, node]));
    for (const block of blocks || []) {
      if (block.type !== BLOCK_TYPE.QA) continue;
      const key = stableBlockKey(block, 0);
      const node = Array.from(parent.querySelectorAll("[data-ask-submitted]")).find(
        (candidate) => candidate._blockKey === key,
      );
      if (node) {
        previous.set(key, node);
        delete node.dataset.askSubmitted;
      }
    }
    const wanted = new Set();
    let deferred = false;
    (blocks || []).forEach((block, index) => {
      const key = stableBlockKey(block, index);
      let node = previous.get(key);
      const signature = JSON.stringify(block);
      if (node) {
        if (block.type === BLOCK_TYPE.AGENT_ACTIVITY) renderAgentActivity(block);
        else if (block.type === BLOCK_TYPE.QA) renderQA(block, node);
        else if (block.type === BLOCK_TYPE.TABS) renderTabs(block, node.firstElementChild);
        else if (block.type === BLOCK_TYPE.CARDS) renderCards(block, node.firstElementChild);
        else if (block.type === BLOCK_TYPE.DISCLOSURE)
          renderDisclosure(block, node.firstElementChild);
        else if (block.type === BLOCK_TYPE.TABLE) renderTable(block, node.firstElementChild);
        else if (block.type === BLOCK_TYPE.SCHEMA_FORM)
          renderSchemaForm(block, node.firstElementChild);
        else if (node._blockSignature !== signature) {
          const focused = document.activeElement;
          if (
            !(node.contains(focused) && ["INPUT", "TEXTAREA", "SELECT"].includes(focused.tagName))
          ) {
            destroyPlayers(node);
            const replacement = renderBlock(block);
            if (replacement) {
              if (block.type === BLOCK_TYPE.FORM)
                retainFormDraft(node.firstElementChild, replacement.firstElementChild);
              node.replaceWith(replacement);
              node = replacement;
            }
          } else deferred = true;
        }
      } else node = renderBlock(block);
      if (!node) return;
      node._blockKey = key;
      node._block = block;
      if (!deferred) node._blockSignature = signature;
      if (node.firstElementChild?._deferred) deferred = true;
      wanted.add(node);
      const current = parent.children[index];
      if (current !== node) {
        if (composingWithin(node) || (current && composingWithin(current))) deferred = true;
        else moveActivityNode(parent, node, current || null);
      }
    });
    for (const node of Array.from(parent.children)) {
      if (wanted.has(node)) continue;
      if (composingWithin(node)) deferred = true;
      else {
        destroyPlayers(node);
        node.remove();
      }
    }
    parent._deferred = deferred;
    return !deferred;
  }

  function destroyPlayers(root) {
    if (!root) return;
    for (const button of $$("[data-menu-toggle]", root)) menuBindings.get(button)?.dispose();
    for (const video of $$("video", root)) {
      if (video._hls) {
        try {
          video._hls.destroy();
        } catch (_) {}
        video._hls = null;
      }
      if (video._mpegts) {
        try {
          video._mpegts.destroy();
        } catch (_) {}
        video._mpegts = null;
      }
    }
  }

  const agentSelections = new Map();
  const agentViews = new Map();
  const agentDrafts = new Map();
  const pageSurfaces = new Map();

  function agentDraft(view, target = view.target) {
    const key = JSON.stringify([view.key, target]);
    if (!agentDrafts.has(key))
      agentDrafts.set(key, {
        value: "",
        start: 0,
        end: 0,
        direction: "none",
        top: 0,
        left: 0,
        revision: 0,
        pending: false,
        composing: false,
      });
    return agentDrafts.get(key);
  }

  function saveAgentEditor(view) {
    if (view.target === undefined || view.editorSuspended || !view.input.getClientRects().length)
      return;
    const draft = agentDraft(view);
    const input = view.input;
    if (draft.value !== input.value) {
      draft.value = input.value;
      draft.revision += 1;
    }
    draft.start = input.selectionStart;
    draft.end = input.selectionEnd;
    draft.direction = input.selectionDirection;
    draft.top = input.scrollTop;
    draft.left = input.scrollLeft;
  }

  function restoreAgentEditor(view, focus = false) {
    const draft = agentDraft(view);
    if (draft.composing) return;
    if (view.input.value !== draft.value) view.input.value = draft.value;
    if (focus) view.input.focus({ preventScroll: true });
    view.input.setSelectionRange(draft.start, draft.end, draft.direction);
    view.input.scrollTop = draft.top;
    view.input.scrollLeft = draft.left;
    const target = view.target;
    const revision = draft.revision;
    const top = draft.top;
    const left = draft.left;
    view.editorRestoring = true;
    requestAnimationFrame(() => {
      if (
        view.target === target &&
        draft.revision === revision &&
        !draft.composing &&
        view.input.getClientRects().length
      ) {
        view.input.scrollTop = top;
        view.input.scrollLeft = left;
      }
      view.editorRestoring = false;
    });
    updateAgentSend(view);
  }

  function retainAgentView(view) {
    if (!view) return false;
    saveAgentEditor(view);
    return Array.from(view.targets.keys()).some((target) => {
      const draft = agentDraft(view, target);
      return draft.value.length > 0 || draft.pending || draft.composing;
    });
  }

  function updateAgentSend(view) {
    const pending = agentDraft(view).pending;
    view.send.disabled = pending;
    view.send.textContent = pending ? "发送中…" : "发送";
  }

  async function sendAgentMessage(view) {
    saveAgentEditor(view);
    const target = view.target;
    const draft = agentDraft(view, target);
    if (draft.pending || draft.composing || !draft.value.trim()) return;
    const sessionId = view.targets.get(target)?.session_id;
    if (!sessionId || !view.sendAction) {
      toast("会话记录不可用", TOAST_KIND.ERR);
      return;
    }
    const message = draft.value;
    const revision = draft.revision;
    const action = view.sendAction;
    const values = { ...view.sendValues, agent_session_id: sessionId, message };
    draft.pending = true;
    updateAgentSend(view);
    try {
      const result = await postAction(action, values, view.page, null, true);
      if (view.target === target) saveAgentEditor(view);
      if (result?.ok && draft.revision === revision && draft.composing)
        draft.clearRevision = revision;
      if (result?.ok && draft.revision === revision && !draft.composing) {
        draft.value = "";
        draft.start = draft.end = draft.top = draft.left = 0;
        draft.direction = "none";
        draft.revision += 1;
        if (view.target === target) restoreAgentEditor(view);
      }
    } finally {
      draft.pending = false;
      if (view.target === target) updateAgentSend(view);
      settleQaEditor(view);
    }
  }

  function activityView(block, page = state.ui.active_page) {
    return agentViews.get(JSON.stringify([page, block.id]));
  }

  function retainedQaActivity(block, page = state.ui.active_page) {
    const activity = qaActivity(block);
    return retainAgentView(activityView(activity, page));
  }

  function settleQaEditor(view) {
    if (!view.qa || retainAgentView(view)) return;
    const cached = qaLiveCache.get(view.qa);
    if (!cached?.status || cached.status === QA_STATUS.PENDING) return;
    const node = view.root.closest(".qa");
    if (node) finishQaBlock(node, cached);
    disposeAgentView(view.key);
  }

  function disposeAgentView(key) {
    const view = agentViews.get(key);
    view?.dispose();
    if (view)
      for (const target of view.targets.keys()) agentDrafts.delete(JSON.stringify([key, target]));
    agentViews.delete(key);
    agentSelections.delete(key);
  }

  function cleanupAgentViews() {
    const active = state.ui.active_page;
    const openPages = new Set(state.ui.open_pages || []);
    if (active) openPages.add(active);
    const pages = new Set((state.ui.pages || []).map((page) => page.id));
    const activeKeys = new Set();
    const qaKeys = new Set();
    const collect = (blocks, page) => {
      for (const block of blocks || []) {
        if (block.type === BLOCK_TYPE.AGENT_ACTIVITY)
          activeKeys.add(JSON.stringify([page, block.id]));
        if (block.type === BLOCK_TYPE.QA) {
          const key = qaKey(block.path, block.id);
          qaKeys.add(key);
          const cached = qaLiveCache.get(key);
          const retained = retainedQaActivity(block, page);
          if (block.status !== QA_STATUS.PENDING && !retained) qaLiveCache.delete(key);
          if (
            retained ||
            (block.status === QA_STATUS.PENDING &&
              (!cached?.status || cached.status === QA_STATUS.PENDING))
          )
            activeKeys.add(JSON.stringify([page, qaActivity(block).id]));
        }
        if (block.type === BLOCK_TYPE.TABS)
          for (const tab of block.tabs || []) collect(tab.blocks, page);
        if (block.type === BLOCK_TYPE.CARDS)
          for (const item of block.items || []) collect(item.blocks, page);
        if (block.type === BLOCK_TYPE.DISCLOSURE) collect(block.blocks, page);
        if (block.type === BLOCK_TYPE.TABLE)
          for (const row of block.rows || []) collect(row.details, page);
      }
    };
    for (const [page, blocks] of Object.entries(state.ui.page_content || {})) collect(blocks, page);
    for (const page of pageSurfaces.keys())
      if (!pages.has(page) || !openPages.has(page)) pageSurfaces.delete(page);
    for (const key of qaLiveCache.keys()) if (!qaKeys.has(key)) qaLiveCache.delete(key);
    for (const [key, view] of agentViews) {
      const [page] = JSON.parse(key);
      if (!pages.has(page) || !openPages.has(page) || (page === active && !activeKeys.has(key))) {
        disposeAgentView(key);
      } else if (page !== active) view.close();
    }
  }

  function executionText(value) {
    if (value === undefined || value === null) return "";
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }

  function executionArgs(value) {
    if (value && typeof value === "object") return value;
    try {
      const parsed = JSON.parse(value || "{}");
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (_) {
      return {};
    }
  }

  function executionNumber(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0;
  }

  function executionTokens(value) {
    return value >= 1000000 ? `${(value / 1000000).toFixed(1)}M` : `${(value / 1000).toFixed(1)}K`;
  }

  function executionToolTitle(block, args) {
    const name = (block.name || "tool").split(/[.:/]/).at(-1);
    const path = args.path || args.file_path || args.file || "";
    if (name === "read") {
      const range = args.offset !== undefined ? ` from line ${args.offset}` : "";
      const count = args.limit !== undefined ? `, ${args.limit} lines` : "";
      const details = block.result?.details;
      const lines = details?.lineCount ?? details?.totalLines;
      return `read ${path || "path unavailable"}${range}${count}${executionNumber(lines) ? ` (${lines} lines)` : ""}`;
    }
    if (name === "wait") {
      const targets =
        args.targets ||
        args.ids ||
        args.handles ||
        block.result?.details?.jobs?.map((job) => job.name || job.id);
      return `wait: ${targets ? executionText(targets).replace(/\s+/g, " ") : args.i || "background work or peer messages"}`;
    }
    if (["grep", "glob"].includes(name))
      return `${name} ${args.pattern || path || ""}${args.pattern && path ? ` in ${path}` : ""}`;
    return `${name}${path ? ` ${path}` : args.i ? `: ${args.i}` : ""}`;
  }

  function renderExecutionBlock(block, row) {
    const args = executionArgs(block.arguments);
    const name = (block.name || "").split(/[.:/]/).at(-1);
    const isRead = block.kind === EXECUTION_BLOCK_KIND.TOOL && name === "read";
    const text = executionText(block.text);
    const latestLine =
      text
        .trim()
        .split(/\n\s*\n/)
        .at(-1)
        ?.split("\n")[0] || "";
    const title =
      block.kind === EXECUTION_BLOCK_KIND.TOOL
        ? executionToolTitle(block, args)
        : block.kind === EXECUTION_BLOCK_KIND.THINKING
          ? `Thinking${latestLine ? `: ${latestLine.replaceAll("**", "")}` : ""}`
          : block.kind === EXECUTION_BLOCK_KIND.REVIEW
            ? `Review report${block.round !== undefined ? ` ${block.round}` : ""}`
            : `${block.kind === EXECUTION_BLOCK_KIND.USER ? "User" : "Text"}${text ? `: ${text.trim().split("\n")[0]}` : ""}`;
    row.className = "execution-block font-monospace border-bottom min-w-0 text-break";
    row.dataset.status = block.status || EXECUTION_BLOCK_STATUS.COMPLETED;
    const header = el(
      isRead ? "div" : "summary",
      { class: "d-flex flex-wrap align-items-center gap-2 p-2" },
      [
        el("span", { class: "flex-grow-1 min-w-0 text-break", text: title }),
        block.status && block.status !== EXECUTION_BLOCK_STATUS.COMPLETED
          ? el("span", {
              class: `badge ${block.status === EXECUTION_BLOCK_STATUS.FAILED ? "theme-danger" : block.status === EXECUTION_BLOCK_STATUS.CANCELLED ? "theme-secondary" : "theme-primary"} font-body`,
              text:
                {
                  [EXECUTION_BLOCK_STATUS.STREAMING]: STATUS_LABEL.RUNNING,
                  [EXECUTION_BLOCK_STATUS.RUNNING]: STATUS_LABEL.RUNNING,
                  [EXECUTION_BLOCK_STATUS.FAILED]: STATUS_LABEL.FAILED,
                  [EXECUTION_BLOCK_STATUS.CANCELLED]: STATUS_LABEL.CANCELLED,
                }[block.status] || block.status,
              role: "status",
            })
          : null,
        el("span", {
          class: "small fg-2",
          text: executionNumber(block.seconds) ? `${block.seconds.toFixed(1)}s` : "",
        }),
      ],
    );
    if (isRead) {
      row.replaceChildren(header);
      return;
    }
    const body = el("div", { class: "execution-body p-3" });
    if (block.kind === EXECUTION_BLOCK_KIND.TOOL) {
      body.append(
        el("h4", { class: "h6", text: "Arguments" }),
        el("pre", { text: executionText(block.arguments) || "Unavailable" }),
      );
      if (block.result !== undefined)
        body.append(
          el("h4", { class: "h6", text: "Result" }),
          el("pre", { text: executionText(block.result) }),
        );
    } else if (block.kind === EXECUTION_BLOCK_KIND.REVIEW) {
      const report = executionArgs(block.report);
      const verdict = report.correct;
      if (typeof verdict === "boolean")
        body.appendChild(
          el("span", {
            class: `badge ${verdict ? "theme-success" : "theme-danger"} font-body`,
            role: "status",
            text: verdict ? "Correct" : "Incorrect",
          }),
        );
      if (block.error) body.appendChild(el("p", { text: block.error }));
      if (typeof verdict === "boolean") {
        if (report.explanation) body.appendChild(el("p", { text: report.explanation }));
      } else if (text) body.appendChild(el("pre", { text }));
      for (const finding of report.findings || []) {
        body.appendChild(
          el("article", { class: "border-start ps-3 mb-3" }, [
            el("strong", {
              text: `${finding.priority !== undefined ? `P${finding.priority} ` : ""}${finding.title || "Finding"}`,
            }),
            el("p", { text: finding.body || finding.explanation || "" }),
            finding.file_path
              ? el("pre", {
                  text: `${finding.file_path}:${finding.line_start}-${finding.line_end}`,
                })
              : null,
          ]),
        );
      }
      if (!Object.keys(report).length && block.report)
        body.appendChild(el("pre", { text: executionText(block.report) }));
    } else body.appendChild(el("pre", { text }));
    const preview = el("pre", { class: "execution-preview w-100 overflow-hidden", text: "" });
    if (block.kind === EXECUTION_BLOCK_KIND.TOOL) {
      if (["grep", "glob"].includes(name) && block.result !== undefined) {
        const result = block.result;
        const output =
          result?.details?.displayContent ??
          (Array.isArray(result?.content)
            ? result.content
                .filter((part) => part.type === FIELD_INPUT.TEXT)
                .map((part) => part.text || "")
                .join("\n")
            : result);
        preview.textContent = executionText(output).split("\n").slice(0, 6).join("\n");
      } else if (
        ["write", "edit", "apply_patch"].includes(name) ||
        [EXECUTION_BLOCK_STATUS.STREAMING, EXECUTION_BLOCK_STATUS.RUNNING].includes(block.status)
      ) {
        const writing = ["write", "edit", "apply_patch"].includes(name);
        const content = executionText(
          writing ? (args.content ?? args.input ?? args.patch ?? block.arguments) : block.arguments,
        );
        preview.textContent = writing
          ? content.split("\n").slice(-5).join("\n")
          : content.split("\n").slice(0, 5).join("\n");
      }
    }
    header.appendChild(preview);
    row.replaceChildren(header, body);
  }

  function executionPosition(view) {
    const { log } = view;
    const edge = log.getBoundingClientRect().top;
    const anchor = Array.from(log.children).find(
      (node) => node.getBoundingClientRect().bottom > edge,
    );
    return {
      top: log.scrollTop,
      bottom: log.scrollHeight - log.clientHeight - log.scrollTop <= 24,
      anchor,
      offset: anchor ? anchor.getBoundingClientRect().top - edge : 0,
    };
  }

  function renderAgentActivity(block) {
    const key = JSON.stringify([state.ui.active_page, block.id]);
    let view = agentViews.get(key);
    if (!view) {
      const root = el("section", {
        class: "block agent-workflow vstack gap-3 flex-grow-1 min-w-0 min-h-0 overflow-hidden",
      });
      const select = el("button", {
        class: "btn-outline theme-secondary",
        type: FIELD_INPUT.BUTTON,
        "aria-label": "查看会话",
        "data-menu-toggle": "",
        "aria-haspopup": "listbox",
        "aria-expanded": "false",
      });
      const menu = el("div", {
        class: "menu execution-session-menu overflow-auto",
        role: "listbox",
        "aria-label": "选择会话",
      });
      document.body.appendChild(menu);
      const metrics = el("div", {
        class: "d-flex flex-wrap justify-content-end ms-auto gap-2 small fg-2",
        "aria-label": "Execution metrics",
      });
      const log = el("div", {
        class: "execution-log border p-2 flex-grow-1 min-h-0 overflow-auto",
        tabindex: "0",
        "aria-label": "Execution blocks",
      });
      const input = el("textarea", {
        class: "form-control flex-grow-1 min-w-0",
        rows: "2",
        "aria-label": "发送给 Agent",
        placeholder: "输入消息给 Agent（Shift+Enter 发送）",
      });
      const send = el("button", {
        type: FIELD_INPUT.BUTTON,
        class: "btn-solid theme-primary flex-shrink-0 text-nowrap",
        text: "发送",
      });
      const composer = el(
        "div",
        { class: "execution-composer d-flex align-items-end gap-2 flex-shrink-0" },
        [input, send],
      );
      root.append(
        el("div", { class: "agent-toolbar d-flex flex-wrap align-items-center gap-2" }, [
          select,
          metrics,
        ]),
        log,
        composer,
      );
      view = {
        key,
        page: state.ui.active_page,
        input,
        send,
        targets: new Map(),
        labels: new Map(),
        root,
        select,
        menu,
        options: new Map(),
        metrics,
        log,
        rows: new Map(),
        positions: new Map(),
        agents: [],
        composer,
      };
      const menuController = bindMenu(select, menu);
      const close = () => menuController.hide();
      view.close = close;
      menu.addEventListener("click", (event) => {
        const option = event.target.closest("[data-target]");
        if (!option) return;
        if (agentDraft(view).composing) {
          view.deferredTarget = option.dataset.target;
          close();
          return;
        }
        agentSelections.set(key, option.dataset.target);
        close();
        showAgentExecution(view, key);
        select.focus();
      });
      view.dispose = () => {
        menuController.dispose();
        menu.remove();
        root.remove();
        view.revision = (view.revision || 0) + 1;
      };
      log.addEventListener("scroll", () => {
        if (!view.updating) view.positions.set(view.target, executionPosition(view));
      });
      for (const type of ["input", "select", "keyup", "scroll"])
        input.addEventListener(type, () => {
          if (type !== "scroll" || !view.editorRestoring) saveAgentEditor(view);
          if (type === "input") settleQaEditor(view);
        });
      input.addEventListener("focus", () => {
        view.wasFocused = true;
      });
      input.addEventListener("blur", (event) => {
        saveAgentEditor(view);
        if (
          event.relatedTarget &&
          $("#page-content").contains(event.relatedTarget) &&
          !event.relatedTarget.closest(".tab-btn")
        )
          view.wasFocused = false;
      });
      input.addEventListener("compositionstart", () => {
        agentDraft(view).composing = true;
      });
      input.addEventListener("compositionend", () => {
        agentDraft(view).composing = false;
        saveAgentEditor(view);
        const draft = agentDraft(view);
        if (draft.clearRevision === draft.revision) {
          draft.value = "";
          draft.start = draft.end = draft.top = draft.left = 0;
          draft.direction = "none";
          draft.revision += 1;
          restoreAgentEditor(view);
        }
        delete draft.clearRevision;
        if (view.deferredTarget !== undefined) {
          agentSelections.set(key, view.deferredTarget);
          delete view.deferredTarget;
          showAgentExecution(view, key);
        }
        state.pageKey = null;
        renderPageContent();
        settleQaEditor(view);
      });
      input.addEventListener("keydown", (event) => {
        if (
          event.key === "Enter" &&
          (event.shiftKey || event.ctrlKey) &&
          !event.isComposing &&
          event.keyCode !== 229 &&
          !agentDraft(view).composing
        ) {
          event.preventDefault();
          sendAgentMessage(view);
        }
      });
      send.addEventListener("click", () => sendAgentMessage(view));
      agentViews.set(key, view);
    }
    view.sendAction = block.send_action;
    view.sendValues = block.send_values || {};
    // No channel means the run is closed: keep the log, hide the composer.
    view.composer.hidden = !view.sendAction;
    view.agents = Array.isArray(block.agents) ? block.agents : [];
    const targets = new Map();
    view.agents.forEach((agent, index) => {
      const target = agent.session_id || `legacy:${agent.id || index}`;
      targets.set(target, agent);
      if (!view.labels.has(target)) {
        const label =
          agent.label ||
          (agent.role === EXECUTION_ROLE.WRITER
            ? "笔记写作"
            : agent.role === EXECUTION_ROLE.REVIEWER
              ? "笔记审阅"
              : agent.role) ||
          "Agent";
        const number =
          Array.from(view.labels.values()).filter((entry) => entry.base === label).length + 1;
        view.labels.set(target, { base: label, number });
      }
    });
    for (const [target, agent] of view.targets) {
      const draft = agentDraft(view, target);
      if (!targets.has(target) && (target === view.target || draft.value.length || draft.pending))
        targets.set(target, { ...agent, closed: true });
    }
    if (!targets.size) targets.set("legacy:unavailable", {});
    view.targets = targets;
    for (const [value, agent] of targets) {
      const label = view.labels.get(value);
      const duplicates =
        label &&
        Array.from(view.labels.values()).filter((entry) => entry.base === label.base).length > 1;
      const text = label
        ? `${label.base}${duplicates ? ` ${label.number}` : ""}${agent.role === EXECUTION_ROLE.REVIEWER ? (agent.review_passed ? "（已通过）" : "（未通过）") : ""}${agent.closed ? "（已关闭）" : ""}`
        : "查看会话";
      let option = view.options.get(value);
      if (!option) {
        option = el("button", {
          type: FIELD_INPUT.BUTTON,
          class: "menu-item",
          role: "option",
          dataset: { target: value },
        });
        view.options.set(value, option);
        view.menu.appendChild(option);
      }
      if (option.textContent !== text) option.textContent = text;
    }
    for (const [value, option] of view.options) {
      if (!targets.has(value)) {
        option.remove();
        view.options.delete(value);
        agentDrafts.delete(JSON.stringify([key, value]));
      }
    }
    showAgentExecution(view, key);
    return view.root;
  }

  function showAgentExecution(view, key) {
    const { select, metrics, log } = view;
    if (view.target !== undefined && log.isConnected)
      view.positions.set(view.target, executionPosition(view));
    view.updating = true;
    const preferred = agentSelections.get(key) ?? view.target;
    const target = view.targets.has(preferred)
      ? preferred
      : Array.from(view.targets).find(([, agent]) => agent.role === EXECUTION_ROLE.WRITER)?.[0] ||
        view.targets.keys().next().value;
    if (view.target !== target) {
      saveAgentEditor(view);
      view.target = target;
      restoreAgentEditor(view);
    }
    updateAgentSend(view);
    agentSelections.set(key, target);
    view.root.dataset.sessionId = view.targets.get(target)?.session_id || "";
    const label = `${view.options.get(target)?.textContent || "查看会话"} ▾`;
    if (select.textContent !== label) select.textContent = label;
    for (const [value, option] of view.options) {
      option.setAttribute("aria-selected", String(value === target));
      option.classList.toggle("selected", value === target);
    }
    const latest = view.targets.get(target) || {};
    const agents = [latest];
    const context =
      executionNumber(latest.context_tokens) &&
      executionNumber(latest.context_window) &&
      latest.context_window > 0
        ? `${((100 * latest.context_tokens) / latest.context_window).toFixed(1)}%/${executionTokens(latest.context_window)}`
        : "unavailable";
    const seconds = (value) => (executionNumber(value) ? `${value.toFixed(1)}s` : "unavailable");
    metrics.replaceChildren(
      ...[
        `${latest.model || "unavailable"}:${latest.thinking || "unavailable"}`,
        `上下文 ${context}`,
        `LLM ${seconds(latest.llm_seconds)}`,
        `工具调用 ${seconds(latest.tool_seconds)}`,
        executionNumber(latest.tokens_per_second)
          ? `${latest.tokens_per_second.toFixed(1)} tok/s`
          : "tok/s unavailable",
      ].map((text) => el("span", { class: "execution-metric", text })),
    );
    const rows = [];
    for (const agent of agents) {
      for (const block of agent.blocks || []) {
        const id = JSON.stringify([agent.id, block.id]);
        const tag =
          block.kind === EXECUTION_BLOCK_KIND.TOOL &&
          (block.name || "").split(/[.:/]/).at(-1) === "read"
            ? "div"
            : "details";
        let cached = view.rows.get(id);
        if (!cached || cached.node.localName !== tag) {
          if (cached) cached.node.remove();
          cached = { node: el(tag), signature: null };
          view.rows.set(id, cached);
        }
        const signature = JSON.stringify(block);
        if (cached.signature !== signature) {
          renderExecutionBlock(block, cached.node);
          cached.signature = signature;
        }
        rows.push(cached.node);
      }
    }
    if (!rows.length)
      rows.push(
        el("p", {
          class: "fg-2",
          text: "Waiting for execution blocks",
        }),
      );
    for (const child of Array.from(log.children)) if (!rows.includes(child)) child.remove();
    rows.forEach((row, index) => {
      if (log.children[index] !== row) log.insertBefore(row, log.children[index] || null);
    });
    const position = view.positions.get(target);
    const restore = () => {
      if (!position || position.bottom) log.scrollTop = log.scrollHeight;
      else if (position.anchor?.parentNode === log) {
        log.scrollTop +=
          position.anchor.getBoundingClientRect().top -
          log.getBoundingClientRect().top -
          position.offset;
      } else log.scrollTop = position.top;
    };
    restore();
    view.updating = false;
    const revision = (view.revision = (view.revision || 0) + 1);
    requestAnimationFrame(() => {
      if (view.target === target && view.revision === revision) {
        restore();
        view.positions.set(target, executionPosition(view));
      }
    });
  }

  function renderQueue(items, existing = null) {
    const list = existing || el("ul", { class: "queue-list list-unstyled mb-0 fs-md lh-base" });
    const rows = new Map(Array.from(list.children, (row) => [row.dataset.queueId, row]));
    const icons = {
      pause:
        "M6 3.5a.5.5 0 0 1 .5.5v8a.5.5 0 0 1-1 0V4a.5.5 0 0 1 .5-.5m4 0a.5.5 0 0 1 .5.5v8a.5.5 0 0 1-1 0V4a.5.5 0 0 1 .5-.5",
      play: "M10.804 8 5 4.633v6.734zM12 8a1 1 0 0 1-.5.866l-6 3.5A1 1 0 0 1 4 11.5v-7a1 1 0 0 1 1.5-.866l6 3.5A1 1 0 0 1 12 8",
      x: "M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708",
    };
    (items || []).forEach((item, index) => {
      const key = String(item.id);
      let row = rows.get(key);
      if (!row) {
        row = el("li", {
          class:
            "position-relative d-flex align-items-center gap-3 px-5 py-4 border-bottom border-subtle",
          dataset: { queueId: key },
        });
      }
      rows.delete(key);
      const signature = JSON.stringify(item);
      if (row.dataset.signature !== signature) {
        row.dataset.signature = signature;
        row.title = item.subtitle || "";
        const content = el("div", { class: "flex-grow-1 min-w-0" }, [
          el("div", { class: "text-break", text: item.title }),
          item.detail ? el("div", { class: "fg-2 text-break", text: item.detail }) : null,
          item.progress ? el("div", { class: "fg-2 mt-1", text: item.progress }) : null,
          item.subtitle
            ? el("span", { class: "visually-hidden", role: "status", text: item.subtitle })
            : null,
        ]);
        row.replaceChildren(
          el("span", {
            class: `position-absolute start-0 top-0 bottom-0 my-4 border-start border-3 border-${item.theme || "secondary"}`,
            "aria-hidden": "true",
          }),
          content,
        );
        if (item.actions?.length) {
          const actions = el("div", {
            class: "queue-actions d-flex flex-row flex-shrink-0 ms-auto",
          });
          for (const action of item.actions) {
            const button = actionButton(
              action.label,
              action.action,
              action.value,
              `btn-text theme-${action.theme || "secondary"} btn-sm p-2 border-0`,
            );
            button.title = action.label;
            button.setAttribute("aria-label", action.label);
            button.innerHTML = `<svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16" aria-hidden="true"><path d="${icons[action.icon]}"/></svg>`;
            actions.appendChild(button);
          }
          row.appendChild(actions);
        }
      }
      if (list.children[index] !== row) list.insertBefore(row, list.children[index] || null);
    });
    for (const row of rows.values()) row.remove();
    return list;
  }

  function renderTaskQueue(existing = null) {
    const tasks = state.ui.tasks || [];
    const pending = tasks.filter((task) => task.state === TASK_STATE.PENDING).length;
    const root =
      existing ||
      el("section", { class: "block card task-queue", "aria-label": "任务队列" }, [
        el("div", { class: "card-header d-flex flex-row align-items-center justify-content-between gap-2" }, [
          el("h3", { class: "fs-md fw-semibold mb-0", text: "任务队列" }),
          el("span", { class: "queue-summary small fg-2" }),
        ]),
        el("div", { class: "card-body p-2" }, [renderQueue([])]),
      ]);
    $(".queue-summary", root).textContent =
      `${STATUS_LABEL.RUNNING} ${tasks.length - pending} 个，${STATUS_LABEL.QUEUED} ${pending} 个`;
    renderQueue(
      tasks.map((task, index) => ({
        id: task.id ?? index,
        title: task.label || "任务",
        detail: task.detail,
        progress: task.session_id ? `会话：${task.session_id}` : "",
        subtitle: task.state === TASK_STATE.PENDING ? STATUS_LABEL.QUEUED : STATUS_LABEL.RUNNING,
        theme: task.state === TASK_STATE.PENDING ? "secondary" : "primary",
      })),
      $(".queue-list", root),
    );
    return root;
  }

  // A link to another site must not navigate the single-page app away.
  function openLinksInNewTab(node) {
    for (const link of node.querySelectorAll("a[href]")) {
      const href = link.getAttribute("href") || "";
      if (/^(https?:)?\/\//i.test(href)) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
    }
    return node;
  }

  function renderBlock(block) {
    if (!block || typeof block !== "object") return null;
    const type = block.type || BLOCK_TYPE.TEXT;
    if (type === BLOCK_TYPE.AGENT_ACTIVITY) return renderAgentActivity(block);
    if (type === BLOCK_TYPE.TASK_QUEUE) return renderTaskQueue();
    if (type === BLOCK_TYPE.QUEUE)
      return el("section", { class: "block sidebar-section queue-panel", "aria-label": block.title }, [
        el(
          "div",
          { class: "d-flex flex-wrap align-items-center justify-content-between gap-2 mb-3" },
          [
            el("h3", { class: "fs-md fw-semibold mb-0", text: block.title }),
            el("span", { class: "small fg-2", text: block.summary }),
          ],
        ),
        renderQueue(block.items),
      ]);
    if (type === BLOCK_TYPE.HTML)
      return openLinksInNewTab(el("div", { class: "block", html: block.html || "" }));
    if (type === BLOCK_TYPE.MARKDOWN) {
      return openLinksInNewTab(
        el("div", {
          class: "block markdown prose",
          html: block.html || escapeHtml(block.text || ""),
        }),
      );
    }
    if (type === BLOCK_TYPE.TEXT)
      return el("pre", {
        class: "block bg-2 border p-3 overflow-auto",
        text: block.text ?? "",
      });
    if (type === BLOCK_TYPE.KEYVALUE) {
      const items = Array.isArray(block.items) ? block.items : Object.entries(block.items || {});
      const grid = el("dl", { class: "keyvalue border" });
      for (const [k, v] of items) {
        grid.appendChild(el("dt", { class: "bg-2 p-2", text: k }));
        grid.appendChild(
          el("dd", {
            class: "p-2 text-break",
            text: typeof v === "object" ? JSON.stringify(v) : String(v ?? ""),
          }),
        );
      }
      return el("div", { class: "block" }, [grid]);
    }
    if (type === BLOCK_TYPE.TABLE) return el("div", { class: "block" }, [renderTable(block)]);
    if (type === BLOCK_TYPE.CARDS) return el("div", { class: "block" }, [renderCards(block)]);
    if (type === BLOCK_TYPE.DISCLOSURE)
      return el("div", { class: "block" }, [renderDisclosure(block)]);
    if (type === BLOCK_TYPE.ROWS) return renderRows(block);
    if (type === BLOCK_TYPE.TREE) return el("div", { class: "block" }, [renderTree(block)]);
    if (type === BLOCK_TYPE.LIST) {
      const list = el("div", { class: "block d-flex flex-wrap gap-2" });
      for (const item of block.items || []) {
        if (typeof item === "string") list.appendChild(el("div", { text: item }));
        else if (item.action)
          list.appendChild(
            actionButton(
              item.text || item.label || "执行",
              item.action,
              item.value,
              item.class || "btn-solid theme-secondary",
            ),
          );
        else list.appendChild(el("div", { text: item.text || JSON.stringify(item) }));
      }
      return list;
    }
    if (type === BLOCK_TYPE.BUTTONS) {
      const row = el("div", { class: "block d-flex flex-wrap gap-2" });
      for (const b of block.buttons || []) {
        row.appendChild(
          actionButton(
            b.label || "执行",
            b.action,
            b.value,
            b.class || "btn-solid theme-secondary",
            Boolean(b.disabled),
          ),
        );
      }
      return row;
    }
    if (type === BLOCK_TYPE.FORM) return el("div", { class: "block" }, [renderForm(block)]);
    if (type === BLOCK_TYPE.SCHEMA_FORM)
      return el("div", { class: "block" }, [renderSchemaForm(block)]);
    if (type === BLOCK_TYPE.LOG) return el("div", { class: "block" }, [renderLog(block)]);
    if (type === BLOCK_TYPE.IMAGE) {
      return el("div", { class: "block" }, [
        el("img", { class: "img-fluid", src: block.src, alt: block.alt || "" }),
      ]);
    }
    if (type === BLOCK_TYPE.VIDEO) {
      const video = el("video", { class: "player-video", controls: "controls", src: block.src });
      return el("div", { class: "block card player-pane min-w-0" }, [
        el("div", { class: "player-stage" }, [video]),
      ]);
    }
    if (type === BLOCK_TYPE.TABS) return el("div", { class: "block" }, [renderTabs(block)]);
    if (type === BLOCK_TYPE.PLAYER) return renderPlayer(block);
    if (type === BLOCK_TYPE.SWITCH) return renderSwitch(block);
    if (type === BLOCK_TYPE.QA) return renderQA(block);
    if (type === BLOCK_TYPE.PROGRESS) {
      const value = Number(block.value ?? 0);
      const pct = value <= 1 ? value * 100 : value;
      const indeterminate = Boolean(block.indeterminate);
      return el("div", { class: "block progress-block" }, [
        el("div", { class: "d-flex justify-content-between small fg-2 mb-2" }, [
          el("span", { text: block.label || "进度" }),
          el("span", {
            class: "fw-semibold fg-body",
            text: indeterminate ? "进行中" : `${Math.round(pct)}%`,
          }),
        ]),
        el(
          "div",
          {
            class: "progress",
            role: "progressbar",
            "aria-label": block.label || "进度",
            "aria-valuenow": indeterminate ? undefined : pct,
            "aria-valuemin": 0,
            "aria-valuemax": 100,
            "aria-valuetext": indeterminate ? "进行中" : undefined,
          },
          [
            el("div", {
              class: `progress-bar${indeterminate ? " progress-bar-indeterminate" : ""}`,
              style: indeterminate
                ? "width:100%"
                : `width:${Math.max(0, Math.min(100, pct))}%`,
            }),
          ],
        ),
      ]);
    }
    if (type === BLOCK_TYPE.BADGE)
      return el("span", {
        class: `badge ${block.class || ""}`,
        text: block.text || "",
      });
    return el("pre", {
      class: "block bg-2 border p-3 overflow-auto",
      text: JSON.stringify(block, null, 2),
    });
  }

  function renderAll() {
    if (!state.ui) return;
    cleanupAgentViews();
    renderMenubar();
    renderSidebar(SIDEBAR_SIDE.LEFT);
    renderSidebar(SIDEBAR_SIDE.RIGHT);
    renderPageTabs();
    const pageChanged = renderPageContent();
    renderStatus();
    if (pageChanged) typeset();
  }

  function commands() {
    return state.ui?.commands || [];
  }

  function themeMode() {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (THEME_ORDER.includes(saved)) return saved;
    } catch (_) {}
    return THEME_MODE.SYSTEM;
  }

  const THEME_ICON_PATH = Object.freeze({
    [THEME_MODE.SYSTEM]:
      '<path d="M8 15A7 7 0 1 0 8 1zm0 1A8 8 0 1 1 8 0a8 8 0 0 1 0 16"/>',
    [THEME_MODE.LIGHT]:
      '<path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6m0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8M8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0m0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13m8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5M3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8m10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0m-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0m9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707M4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708"/>',
    [THEME_MODE.DARK]:
      '<path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>',
  });

  function applyTheme(mode) {
    const root = document.documentElement;
    if (mode === THEME_MODE.SYSTEM) {
      root.style.removeProperty("color-scheme");
      delete root.dataset.theme;
    } else {
      root.style.colorScheme = mode;
      root.dataset.theme = mode;
    }
    try {
      if (mode === THEME_MODE.SYSTEM) localStorage.removeItem(THEME_STORAGE_KEY);
      else localStorage.setItem(THEME_STORAGE_KEY, mode);
    } catch (_) {}
    const btn = $("#theme-btn");
    if (btn) {
      const label = THEME_LABEL[mode] || THEME_LABEL[THEME_MODE.SYSTEM];
      btn.title = `切换外观（${label}）`;
      const textNode = btn.querySelector(".theme-btn-label");
      if (textNode) textNode.textContent = label;
      const svg = btn.querySelector("svg");
      if (svg && THEME_ICON_PATH[mode]) svg.innerHTML = THEME_ICON_PATH[mode];
    }
  }

  function cycleTheme() {
    const current = themeMode();
    const mode = THEME_ORDER[(THEME_ORDER.indexOf(current) + 1) % THEME_ORDER.length];
    applyTheme(mode);
    toast(`外观：${THEME_LABEL[mode]}`);
  }

  function themeCommand() {
    return {
      id: "ui.theme",
      title: `切换外观（当前：${THEME_LABEL[themeMode()]}）`,
      description: "在跟随系统、浅色、深色之间切换",
      category: "界面",
      icon: "circle-half",
      keywords: "theme dark light 深色 浅色 暗色 主题 外观",
      client: "theme.cycle",
    };
  }

  function paletteCommands() {
    return [...commands(), themeCommand()];
  }

  function closeMenus() {
    for (const button of $$("[data-menu-toggle]", $("#menubar"))) menuBindings.get(button)?.hide();
  }

  function runCommand(cmd) {
    closeMenus();
    closePalette();
    if (cmd.client === "theme.cycle") return cycleTheme();
    if (cmd.page) return openPage(cmd.page);
    if (cmd.action) return postAction(cmd.action, {});
    return null;
  }

  /* Bootstrap Icons: https://icons.getbootstrap.com/
   * The MIT License (MIT)
   *
   * Copyright (c) 2019-2024 The Bootstrap Authors
   *
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in
   * all copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   * THE SOFTWARE.
   *
   */
  const COMMAND_ICONS = Object.freeze({
    "list-task":
      '<path fill-rule="evenodd" d="M2 2.5a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5V3a.5.5 0 0 0-.5-.5zM3 3H2v1h1z"/>\n  <path d="M5 3.5a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5M5.5 7a.5.5 0 0 0 0 1h9a.5.5 0 0 0 0-1zm0 4a.5.5 0 0 0 0 1h9a.5.5 0 0 0 0-1z"/>\n  <path fill-rule="evenodd" d="M1.5 7a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H2a.5.5 0 0 1-.5-.5zM2 7h1v1H2zm0 3.5a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5zm1 .5H2v1h1z"/>',
    gear: '<path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492M5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0"/>\n  <path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115z"/>',
    "arrow-clockwise":
      '<path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2z"/>\n  <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466"/>',
    "play-circle":
      '<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/>\n  <path d="M6.271 5.055a.5.5 0 0 1 .52.038l3.5 2.5a.5.5 0 0 1 0 .814l-3.5 2.5A.5.5 0 0 1 6 10.5v-5a.5.5 0 0 1 .271-.445"/>',
    mic: '<path d="M3.5 6.5A.5.5 0 0 1 4 7v1a4 4 0 0 0 8 0V7a.5.5 0 0 1 1 0v1a5 5 0 0 1-4.5 4.975V15h3a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1h3v-2.025A5 5 0 0 1 3 8V7a.5.5 0 0 1 .5-.5"/>\n  <path d="M10 8a2 2 0 1 1-4 0V3a2 2 0 1 1 4 0zM8 0a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V3a3 3 0 0 0-3-3"/>',
    easel:
      '<path d="M8 0a.5.5 0 0 1 .473.337L9.046 2H14a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-1.85l1.323 3.837a.5.5 0 1 1-.946.326L11.092 11H8.5v3a.5.5 0 0 1-1 0v-3H4.908l-1.435 4.163a.5.5 0 1 1-.946-.326L3.85 11H2a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1h4.954L7.527.337A.5.5 0 0 1 8 0M2 3v7h12V3z"/>',
    "circle-half": '<path d="M8 15A7 7 0 1 0 8 1zm0 1A8 8 0 1 1 8 0a8 8 0 0 1 0 16"/>',
  });

  function commandIcon(name) {
    const paths = COMMAND_ICONS[name];
    if (!paths) return null;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "command-icon");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.innerHTML = paths;
    return svg;
  }

  function renderMenubar() {
    const bar = $("#menubar");
    if (!bar) return;
    const signature = JSON.stringify(commands());
    if (bar.dataset.commands === signature) return;
    for (const button of $$("[data-menu-toggle]", bar)) menuBindings.get(button)?.dispose();
    bar.replaceChildren();
    bar.dataset.commands = signature;
    const groups = new Map();
    for (const cmd of commands()) {
      const name = cmd.category || "其他";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(cmd);
    }
    for (const [name, items] of groups) {
      const button = el("button", {
        class: "nav-link",
        type: FIELD_INPUT.BUTTON,
        text: name,
        "data-menu-toggle": "",
        "aria-expanded": "false",
      });
      const menu = el("div", { class: "menu" });
      for (const cmd of items) {
        const item = el(
          "button",
          {
            class: "menu-item command-menu-item",
            type: FIELD_INPUT.BUTTON,
          },
          [commandIcon(cmd.icon), el("span", { text: cmd.title })],
        );
        item.disabled = Boolean(cmd.disabled);
        item.addEventListener("click", () => runCommand(cmd));
        menu.appendChild(item);
      }
      bar.appendChild(el("li", { class: "nav-item position-relative" }, [button, menu]));
      bindMenu(button, menu);
    }
  }

  function paletteMatches(query) {
    const q = String(query || "")
      .trim()
      .toLowerCase();
    const items = paletteCommands();
    if (!q) return items;
    return items.filter((cmd) => {
      const hay =
        `${cmd.title} ${cmd.description || ""} ${cmd.category || ""} ${cmd.keywords || ""}`.toLowerCase();
      return q.split(/\s+/).every((part) => hay.includes(part));
    });
  }

  function renderPalette(query) {
    const list = $("#palette-list");
    if (!list) return;
    list.replaceChildren();
    const items = paletteMatches(query);
    state.paletteItems = items;
    if (state.paletteIndex >= items.length) state.paletteIndex = 0;
    if (!items.length) {
      list.appendChild(el("p", { class: "fg-2 p-3", text: "没有匹配的命令" }));
      return;
    }
    items.forEach((cmd, index) => {
      const row = el(
        "button",
        {
          class: `palette-item ${index === state.paletteIndex ? "active" : ""}`,
          type: FIELD_INPUT.BUTTON,
          role: "option",
          "aria-selected": String(index === state.paletteIndex),
          disabled: Boolean(cmd.disabled),
          "aria-describedby": cmd.description ? `palette-description-${index}` : undefined,
        },
        [
          commandIcon(cmd.icon),
          el("span", { class: "palette-copy" }, [
            el("span", { class: "palette-title", text: cmd.title }),
            cmd.description
              ? el("span", {
                  id: `palette-description-${index}`,
                  class: "palette-description",
                  text: cmd.description,
                })
              : null,
          ]),
          cmd.category ? el("span", { class: "palette-category", text: cmd.category }) : null,
        ],
      );
      row.addEventListener("click", () => runCommand(cmd));
      row.addEventListener("mouseenter", () => {
        state.paletteIndex = index;
        $$(".palette-item", list).forEach((node, i) => {
          node.classList.toggle("active", i === index);
          node.setAttribute("aria-selected", String(i === index));
        });
      });
      list.appendChild(row);
    });
  }

  function openPalette() {
    const palette = $("#palette");
    if (!palette) return;
    if (!palette.open) palette.showModal();
    const input = $("#palette-input");
    input.value = "";
    state.paletteIndex = 0;
    renderPalette("");
    input.focus();
  }

  function closePalette() {
    const palette = $("#palette");
    if (palette?.open) palette.close();
    state.paletteIndex = 0;
  }

  function paletteOpen() {
    const palette = $("#palette");
    return Boolean(palette?.open);
  }

  document.addEventListener("keydown", (event) => {
    if (event.isComposing) return;
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === "p") {
      event.preventDefault();
      if (paletteOpen()) closePalette();
      else openPalette();
      return;
    }
    if (!paletteOpen()) return;
    const items = state.paletteItems || [];
    if (key === "arrowdown") {
      event.preventDefault();
      state.paletteIndex = Math.min(items.length - 1, state.paletteIndex + 1);
      renderPalette($("#palette-input").value);
      const active = $(".palette-item.active");
      if (active) active.scrollIntoView({ block: "nearest" });
    } else if (key === "arrowup") {
      event.preventDefault();
      state.paletteIndex = Math.max(0, state.paletteIndex - 1);
      renderPalette($("#palette-input").value);
      const active = $(".palette-item.active");
      if (active) active.scrollIntoView({ block: "nearest" });
    } else if (key === "enter") {
      event.preventDefault();
      const cmd = items[state.paletteIndex];
      if (cmd) runCommand(cmd);
    }
  });

  document.addEventListener("input", (event) => {
    if (event.target && event.target.id === "palette-input") {
      state.paletteIndex = 0;
      renderPalette(event.target.value);
    }
  });

  function bindPaletteTrigger() {
    $("#theme-btn")?.addEventListener("click", cycleTheme);
    $("#palette-btn")?.addEventListener("click", openPalette);
    for (const button of $$("[data-drawer-target]"))
      button.addEventListener("click", () => {
        const sidebar = $(button.dataset.drawerTarget);
        if (!sidebar.open) sidebar.showModal();
      });
    for (const dialog of $$("dialog"))
      dialog.addEventListener("click", (event) => {
        if (event.target !== dialog) return;
        const rect = dialog.getBoundingClientRect();
        if (
          event.clientX < rect.left ||
          event.clientX > rect.right ||
          event.clientY < rect.top ||
          event.clientY > rect.bottom
        )
          dialog.close();
      });
    $("[data-toast-close]")?.addEventListener("click", () => $("#toast").classList.remove("show"));
    $("#palette")?.addEventListener("close", () => {
      state.paletteIndex = 0;
    });
  }

  const desktopLayout = window.matchMedia("(min-width: 1024px)");

  function sidebarWidth(side) {
    const app = $("#app");
    app.style.removeProperty(`--${side}-col`);
    const base = getComputedStyle(app).getPropertyValue(`--${side}-col`).trim();
    const fallback = base.endsWith("vw") ? (parseFloat(base) * innerWidth) / 100 : parseFloat(base);
    const stored = Number(localStorage.getItem(`lecture-agent.sidebar.${side}.w`));
    return clampWidth(Number.isFinite(stored) && stored > 0 ? stored : fallback);
  }

  function clampWidth(value) {
    const width = $(".workspace").clientWidth;
    return Math.round(Math.min(width / 3, Math.max(width / 6, value)));
  }

  function applySidebarState() {
    const app = $("#app");
    if (!app) return;
    for (const side of Object.values(SIDEBAR_SIDE)) {
      const collapsed = localStorage.getItem(`lecture-agent.sidebar.${side}`) === "true";
      app.classList.toggle(`${side}-collapsed`, collapsed);
      $(`#sidebar-${side}`).inert = desktopLayout.matches && collapsed;
      if (collapsed) app.style.removeProperty(`--${side}-col`);
      else app.style.setProperty(`--${side}-col`, `${sidebarWidth(side)}px`);
    }
  }

  function setSidebarCollapsed(side, collapsed) {
    const app = $("#app");
    if (!app) return;
    app.classList.toggle(`${side}-collapsed`, collapsed);
    $(`#sidebar-${side}`).inert = desktopLayout.matches && collapsed;
    if (collapsed) app.style.removeProperty(`--${side}-col`);
    else app.style.setProperty(`--${side}-col`, `${sidebarWidth(side)}px`);
    localStorage.setItem(`lecture-agent.sidebar.${side}`, String(collapsed));
  }

  function bindSidebarControls() {
    for (const btn of $$(".sidebar-close, .sidebar-restore")) {
      btn.addEventListener("click", () => {
        if (!desktopLayout.matches) {
          const sidebar = $(`#sidebar-${btn.dataset.side}`);
          if (sidebar.open) sidebar.close();
          else sidebar.showModal();
          return;
        }
        const side = btn.dataset.side;
        const collapsed = $("#app").classList.contains(`${side}-collapsed`);
        setSidebarCollapsed(side, !collapsed);
      });
    }
  }

  function bindSidebarResize() {
    for (const handle of $$(".resizer")) {
      const side = handle.dataset.side;
      handle.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || !desktopLayout.matches) return;
        event.preventDefault();
        const app = $("#app");
        if (app.classList.contains(`${side}-collapsed`)) return;
        const workspace = $(".workspace");
        const rect = workspace.getBoundingClientRect();
        handle.setPointerCapture(event.pointerId);
        document.body.classList.add("resizing");
        const move = (ev) => {
          const width =
            side === SIDEBAR_SIDE.LEFT ? ev.clientX - rect.left : rect.right - ev.clientX;
          const value = `${clampWidth(width)}px`;
          app.style.setProperty(`--${side}-col`, value);
        };
        const stop = () => {
          document.body.classList.remove("resizing");
          handle.removeEventListener("pointermove", move);
          handle.removeEventListener("pointerup", stop);
          handle.removeEventListener("pointercancel", stop);
          const value = parseFloat(getComputedStyle(app).getPropertyValue(`--${side}-col`));
          if (Number.isFinite(value) && value > 0) {
            localStorage.setItem(`lecture-agent.sidebar.${side}.w`, String(Math.round(value)));
          }
        };
        handle.addEventListener("pointermove", move);
        handle.addEventListener("pointerup", stop);
        handle.addEventListener("pointercancel", stop);
      });
      handle.addEventListener("dblclick", () => {
        localStorage.removeItem(`lecture-agent.sidebar.${side}.w`);
        if (!$("#app").classList.contains(`${side}-collapsed`)) {
          $("#app").style.setProperty(`--${side}-col`, `${sidebarWidth(side)}px`);
        }
      });
    }
    window.addEventListener("resize", applySidebarState);
    desktopLayout.addEventListener("change", () => {
      for (const side of Object.values(SIDEBAR_SIDE)) {
        const sidebar = $(`#sidebar-${side}`);
        if (sidebar.open) sidebar.close();
      }
      applySidebarState();
    });
  }

  function hideSelectionMenu() {
    $(".selection-popover")?.remove();
  }

  function docContext(node) {
    const element = node && (node.nodeType === 1 ? node : node.parentElement);
    const body = element ? element.closest(".tabs-body") : null;
    if (!body) return null;
    let actions = [];
    try {
      actions = JSON.parse(body.dataset.selectionActions || "[]");
    } catch (_) {
      actions = [];
    }
    if (!actions.length) return null;
    return {
      actions,
      docKey: body.dataset.docKey || "",
      path: body.dataset.docPath || "",
      pageId: body.dataset.pageId || "",
    };
  }

  function showSelectionMenu() {
    const page = $("#page-content");
    if (!page) return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return hideSelectionMenu();
    const text = String(sel).trim();
    if (text.length < 2) return hideSelectionMenu();
    const node = sel.anchorNode;
    if (!node || !page.contains(node)) return hideSelectionMenu();
    const ctx = docContext(node);
    if (!ctx) return hideSelectionMenu();
    const range = sel.getRangeAt(0).cloneRange();

    hideSelectionMenu();
    const content = el("div", { class: "d-flex flex-column" });
    content.addEventListener("mousedown", (event) => event.preventDefault());
    for (const item of ctx.actions) {
      const btn = el("button", {
        class: "menu-item",
        type: FIELD_INPUT.BUTTON,
        text: `${item.icon ? item.icon + " " : ""}${item.label || item.id}`,
      });
      btn.addEventListener("click", () => {
        hideSelectionMenu();
        if (item.kind === SELECTION_ACTION_KIND.PROMPT) {
          openAskEditor(item, ctx, text, range);
          return;
        }
        postAction(item.action, {
          ...(item.value || {}),
          doc_key: ctx.docKey,
          path: ctx.path,
          page_id: ctx.pageId,
          selection: text,
        });
      });
      content.appendChild(btn);
    }
    const rect = range.getBoundingClientRect();
    const popup = el(
      "div",
      {
        class: "popover selection-popover show",
        role: "dialog",
        "aria-label": "选中文字操作",
      },
      [content],
    );
    document.body.appendChild(popup);
    popup.style.left = `${Math.max(4, Math.min(rect.left, innerWidth - popup.offsetWidth - 4))}px`;
    popup.style.top = `${rect.top >= popup.offsetHeight + 8 ? rect.top - popup.offsetHeight - 8 : rect.bottom + 8}px`;
  }

  function openAskEditor(item, ctx, selection, range) {
    const panel = range.endContainer.parentElement?.closest(".tabs-body");
    if (!panel) return;
    const active = panel.querySelector("[data-ask-editor]");
    if (active) {
      active.querySelector("textarea").focus();
      return;
    }
    const markdown = range.endContainer.parentElement?.closest(".markdown");
    if (!markdown) return;
    let anchor =
      range.endContainer.nodeType === Node.ELEMENT_NODE
        ? range.endContainer
        : range.endContainer.parentElement;
    while (anchor.parentElement !== markdown && anchor !== markdown) anchor = anchor.parentElement;
    const editor = el("section", {
      class: "card min-w-0 not-prose",
      dataset: { askEditor: "" },
      "aria-label": "提问",
    });
    const input = el("textarea", {
      class: "form-control",
      rows: "3",
      "aria-label": "问题",
      placeholder: "输入问题",
    });
    const levels = el("select", { class: "form-control w-auto", "aria-label": "强度" });
    const options = item.intensities?.length
      ? item.intensities
      : [
          { value: ASK_INTENSITY.INSTANT, label: "快速" },
          { value: ASK_INTENSITY.RESEARCH, label: "研究" },
        ];
    for (const option of options) {
      const value = typeof option === "object" ? option.value : option;
      const label = typeof option === "object" ? (option.label ?? option.value) : option;
      levels.appendChild(el("option", { value: String(value), text: String(label) }));
    }
    const cancel = el("button", {
      class: "btn-solid theme-secondary btn-sm",
      type: "button",
      text: "取消",
    });
    const send = el("button", {
      class: "btn-solid theme-primary btn-sm",
      type: "submit",
      text: "发送",
    });
    const form = el("form", { class: "card-body d-flex flex-column gap-3" }, [
      el("blockquote", { class: "border-start ps-3 fg-2 mb-0", text: selection }),
      input,
      el("div", { class: "d-flex flex-wrap align-items-center gap-2" }, [
        levels,
        el("div", { class: "ms-auto d-flex gap-2" }, [cancel, send]),
      ]),
    ]);
    editor.append(
      el("div", { class: "card-header flex-row align-items-center gap-2" }, [
        el("span", { class: "badge theme-primary", text: "问" }),
        el("span", { text: "提问" }),
      ]),
      form,
    );
    cancel.addEventListener("click", () => {
      editor.remove();
      state.pageKey = null;
      loadState({ quiet: true });
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (send.disabled) return;
      const question = input.value.trim();
      for (const control of [input, levels, send, cancel]) control.disabled = true;
      try {
        const result = await api("/api/ui/action", {
          method: "POST",
          body: JSON.stringify({
            action: item.action,
            values: {
              ...(item.value || {}),
              doc_key: ctx.docKey,
              path: ctx.path,
              page_id: ctx.pageId,
              selection,
              question,
              intensity: levels.value,
            },
            page_id: ctx.pageId,
            navigation: navigationFromUrl(),
          }),
        });
        if (!result.ok) throw new Error(result.message || "提交问题失败");
        const block = {
          type: BLOCK_TYPE.QA,
          id: result.data.qa_id,
          path: ctx.path,
          question,
          intensity: levels.value,
          status: QA_STATUS.PENDING,
          collapsed: false,
        };
        const answer = renderQA(block);
        answer._blockKey = stableBlockKey(block, 0);
        answer.dataset.askSubmitted = "";
        editor.replaceWith(answer);
        state.pageKey = null;
        await loadState({ quiet: true });
      } catch (error) {
        toast(`提交问题失败：${error.message}`, TOAST_KIND.ERR);
        for (const control of [input, levels, send, cancel]) control.disabled = false;
        input.focus();
      }
    });
    input.addEventListener("keydown", (event) => {
      if (
        event.key === "Enter" &&
        (event.shiftKey || event.ctrlKey) &&
        !event.isComposing &&
        event.keyCode !== 229
      ) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    if (anchor === markdown) markdown.appendChild(editor);
    else anchor.after(editor);
    window.getSelection()?.removeAllRanges();
    input.focus();
  }

  function bindSelectionMenu() {
    document.addEventListener("mouseup", (event) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target && target.closest(".selection-popover")) return;
      if (target && target.closest("[data-ask-editor]")) return;
      window.setTimeout(showSelectionMenu, 0);
    });
    document.addEventListener("selectionchange", () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) hideSelectionMenu();
    });
    document.addEventListener("scroll", hideSelectionMenu, true);

    document.addEventListener("mousedown", (event) => {
      if (!event.target.closest(".selection-popover")) hideSelectionMenu();
    });
  }

  function pickTab(tabs, side) {
    const current = state.activeTab[side];
    if (current && tabs.some((t) => t.id === current)) return current;
    const saved = localStorage.getItem(`lecture-agent.tab.${side}`);
    if (saved && tabs.some((t) => t.id === saved)) return saved;
    return tabs[0]?.id || null;
  }

  function renderSidebarContext() {
    const box = $("#right-context");
    if (!box) return;
    const context = state.ui.context || null;
    if (!context) {
      box.hidden = true;
      box.replaceChildren();
      return;
    }
    box.hidden = false;
    box.replaceChildren(
      el("div", { class: "context-header" }, [
        context.state
          ? el("span", {
              class: "context-dot",
              dataset: { state: context.state },
              "aria-hidden": "true",
            })
          : null,
        el("div", { class: "min-w-0 flex-grow-1" }, [
          el("div", { class: "fw-semibold text-break", text: context.title || "" }),
          context.subtitle
            ? el("div", { class: "small fg-2 mt-1", text: context.subtitle })
            : null,
        ]),
      ]),
    );
  }

  function renderSidebar(side) {
    const tabs = side === SIDEBAR_SIDE.LEFT ? state.ui.left_tabs : state.ui.right_tabs;
    if (side === SIDEBAR_SIDE.LEFT) {
      $("#refresh-courses").hidden = !commands().some(
        (command) => command.action === "input.refresh_courses",
      );
    }
    const tabsEl = $(`#${side}-tabs`);
    const toolboxEl = $(`#${side}-toolbox`);
    const sidebarKey = JSON.stringify([
      tabs,
      state.activeTab[side],
      state.ui.active_page,
      side === SIDEBAR_SIDE.RIGHT ? state.ui.context : null,
    ]);
    if (toolboxEl.dataset.renderKey === sidebarKey) return;
    toolboxEl.dataset.renderKey = sidebarKey;
    const sidebarScroll = toolboxEl.scrollTop;
    if (side === SIDEBAR_SIDE.RIGHT) renderSidebarContext();
    const contextKnown = Object.prototype.hasOwnProperty.call(state.ui, "context");
    const context = side === SIDEBAR_SIDE.RIGHT ? state.ui.context : null;
    if (side === SIDEBAR_SIDE.RIGHT && contextKnown && !context) {
      tabsEl.replaceChildren();
      tabsEl.hidden = true;
      toolboxEl.replaceChildren(
        el("div", { class: "sidebar-empty" }, [
          el("strong", { text: "未选择课次" }),
          el("span", { class: "small fg-2", text: "请在左侧课程列表中选择一节课次" }),
        ]),
      );
      return;
    }
    tabsEl.replaceChildren();
    tabsEl.hidden = (tabs || []).length <= 1;
    const active = pickTab(tabs || [], side);
    state.activeTab[side] = active;
    if (active) localStorage.setItem(`lecture-agent.tab.${side}`, active);
    for (const tab of tabs || []) {
      const btn = el(
        "button",
        {
          class: `nav-link ${tab.id === active ? "active" : ""}`,
          type: FIELD_INPUT.BUTTON,
        },
        [
          tab.icon ? el("span", { "aria-hidden": "true", text: tab.icon }) : null,
          el("span", { text: tab.title || tab.id }),
        ],
      );
      btn.addEventListener("click", () => {
        state.activeTab[side] = tab.id;
        localStorage.setItem(`lecture-agent.tab.${side}`, tab.id);
        renderSidebar(side);
        // Open the tab's primary page so switching tabs never leaves the
        // previous page looking like part of the new tab.
        const first = (tab.toolbox || [])[0];
        const pageId = first ? first.page || null : null;
        if (pageId && state.ui.active_page !== pageId) openPage(pageId);
      });
      tabsEl.appendChild(btn);
    }
    toolboxEl.replaceChildren();
    const current = (tabs || []).find((t) => t.id === active);
    if (!current || !(current.toolbox || []).length) {
      toolboxEl.appendChild(el("p", { class: "fg-2 p-2", text: "这个标签页还没有工具" }));
      return;
    }
    for (const item of current.toolbox) {
      if (!(item.blocks || []).length && !item.title && !item.description && !item.page) continue;
      const panel = el("section", { class: "sidebar-panel" });
      const pageId = item.page || null;
      const title = item.title && item.title !== current.title ? item.title : "";
      if (title || item.description || pageId) {
        const head = el("div", { class: "d-flex flex-wrap align-items-center gap-2 mb-2" }, [
          title ? el("h3", { class: "h6 mb-0", text: title }) : null,
          item.description ? el("small", { class: "fg-2", text: item.description }) : null,
        ]);
        if (pageId) {
          const open = actionButton(
            "打开页面",
            "ui.open_page",
            { page_id: pageId },
            "btn-outline theme-secondary btn-sm ms-auto",
          );
          head.appendChild(open);
        }
        panel.appendChild(head);
      }
      const body = el("div");
      if (item.blocks && item.blocks.length) {
        body.appendChild(
          renderBlocks(item.blocks.filter((block) => block.type !== BLOCK_TYPE.AGENT_ACTIVITY)),
        );
      }
      panel.appendChild(body);
      toolboxEl.appendChild(panel);
    }
    toolboxEl.scrollTop = sidebarScroll;
  }

  function renderPageTabs() {
    const bar = $("#page-tabs");
    bar.replaceChildren();
    const openIds = state.ui.open_pages || [];
    const pages = (state.ui.pages || []).filter(
      (p) => openIds.includes(p.id) || p.id === state.ui.active_page,
    );
    bar.hidden = !pages.length;
    for (const page of pages) {
      const active = page.id === state.ui.active_page;
      const open = el("button", {
        type: FIELD_INPUT.BUTTON,
        text: page.title || page.id,
        "aria-current": active ? "page" : undefined,
      });
      const closeButton = page.closable
        ? el("button", {
            type: FIELD_INPUT.BUTTON,
            class: "page-tab-close",
            text: "×",
            "aria-label": `关闭 ${page.title || page.id}`,
            dataset: { closePage: page.id },
          })
        : null;
      const tab = el(
        "div",
        {
          class: "page-tab nav-link",
        },
        [open, closeButton],
      );
      tab.classList.toggle("active", active);
      tab.addEventListener("click", async (event) => {
        if (event.target.closest("[data-close-page]")) return;
        if (page.id !== state.ui.active_page) await openPage(page.id);
      });
      const close = closeButton;
      if (close) {
        close.addEventListener("click", async (event) => {
          event.stopPropagation();
          await closePage(page.id);
        });
      }
      bar.appendChild(tab);
    }
  }

  function renderWelcomeEmpty() {
    const logo = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    logo.setAttribute("width", "32");
    logo.setAttribute("height", "32");
    logo.setAttribute("viewBox", "0 0 32 32");
    logo.setAttribute("fill", "none");
    logo.innerHTML =
      '<defs><linearGradient id="empty-logo-grad" x1="16" y1="3.5" x2="16" y2="28.5" gradientUnits="userSpaceOnUse"><stop offset="0%" class="brand-logo-stop-1"/><stop offset="50%" class="brand-logo-stop-2"/><stop offset="100%" class="brand-logo-stop-3"/></linearGradient></defs><g class="brand-logo-cloud"><circle cx="8.8" cy="18.8" r="5.6"/><circle cx="16" cy="13.8" r="6.8"/><circle cx="23.2" cy="18.8" r="5.4"/><rect x="8.4" y="15.2" width="15.2" height="9.2" rx="4.6"/></g><path d="M22.2 5.4C20.2 3.4 17.6 4.4 17 8.6L15 23.4C14.4 27.6 11.8 28.6 9.8 26.6" stroke="url(#empty-logo-grad)" stroke-width="3.2" stroke-linecap="round"/>';
    return el("div", { class: "welcome-empty" }, [
      el("div", { class: "welcome-emblem", "aria-hidden": "true" }, [logo]),
      el("h2", { text: "未选择课次" }),
      el("p", { text: "从左侧课程列表选择一节课次，或按 Ctrl+P 打开命令面板" }),
    ]);
  }

  function renderPageContent() {
    const container = $("#page-content");
    const active = state.ui.active_page || "";
    container.classList.toggle(
      "conversation-page",
      active.startsWith("conversation:") || active.startsWith("teaching:"),
    );
    const blocks = active ? state.ui.page_content?.[active] : null;
    const previousPage = container.dataset.pageId;
    const samePage = previousPage === active;
    const doc = navigationFromUrl().doc;
    const sameDocument = container.dataset.doc === doc;
    if (active === "ui:tasks" && samePage) {
      const queue = $(".task-queue", container);
      if (queue) {
        renderTaskQueue(queue);
        return false;
      }
    }
    const key = active
      ? `${active}\u0000${doc}\u0000${JSON.stringify(blocks || [])}`
      : "__welcome__";
    if (samePage && state.pageKey === key) return false;
    if (!samePage && composingWithin(container)) return false;
    const focused = document.activeElement;
    const activityPage =
      blocks?.some(hasActivityBlock) || container.querySelector(".agent-workflow");
    if (
      samePage &&
      sameDocument &&
      !activityPage &&
      container.contains(focused) &&
      ["INPUT", "TEXTAREA", "SELECT"].includes(focused?.tagName)
    )
      return false;
    const qaBefore = new Map();
    if (samePage) {
      const edge = container.getBoundingClientRect().top;
      for (const node of $$(".qa", container))
        qaBefore.set(qaKey(node.dataset.qaPath, node.dataset.qaId), {
          height: node.offsetHeight,
          above: node.getBoundingClientRect().bottom <= edge,
        });
    }
    const scrollTop = container.scrollTop;
    if (!samePage) {
      for (const view of agentViews.values()) {
        if (!container.contains(view.root)) continue;
        saveAgentEditor(view);
        view.positions.set(view.target, executionPosition(view));
        if (focused === view.input || view.wasFocused) view.restoreFocus = true;
        view.editorSuspended = true;
        view.close();
      }
      if (previousPage !== undefined && (state.ui.open_pages || []).includes(previousPage))
        pageSurfaces.set(previousPage, {
          nodes: Array.from(container.childNodes),
          scrollTop,
        });
      container.replaceChildren();
      container.dataset.pageId = active;
      const saved = pageSurfaces.get(active);
      if (saved) for (const node of saved.nodes) container.appendChild(node);
    }
    let complete = true;
    if (!active || !blocks) {
      if (!samePage || !container.children.length) container.replaceChildren(renderWelcomeEmpty());
    } else complete = updateBlocks(container, blocks);
    state.pageKey = complete ? key : null;
    container.scrollTop = samePage ? scrollTop : pageSurfaces.get(active)?.scrollTop || 0;
    if (!samePage) {
      for (const view of agentViews.values()) {
        if (!container.contains(view.root)) continue;
        view.editorSuspended = false;
        if (view.input.getClientRects().length) {
          restoreAgentEditor(view, Boolean(view.restoreFocus));
          view.restoreFocus = false;
        }
      }
    }
    let adjust = 0;
    for (const node of $$(".qa", container)) {
      const before = qaBefore.get(qaKey(node.dataset.qaPath, node.dataset.qaId));
      if (before?.above) adjust += node.offsetHeight - before.height;
    }
    if (adjust) container.scrollTop += adjust;
    if (complete) container.dataset.doc = doc;
    return true;
  }

  function hasActivityBlock(block) {
    return (
      block.type === BLOCK_TYPE.AGENT_ACTIVITY ||
      block.type === BLOCK_TYPE.QA ||
      (block.type === BLOCK_TYPE.TABS &&
        (block.tabs || []).some((tab) => (tab.blocks || []).some(hasActivityBlock))) ||
      (block.type === BLOCK_TYPE.CARDS &&
        (block.items || []).some((item) => (item.blocks || []).some(hasActivityBlock))) ||
      (block.type === BLOCK_TYPE.DISCLOSURE && (block.blocks || []).some(hasActivityBlock)) ||
      (block.type === BLOCK_TYPE.TABLE &&
        (block.rows || []).some((row) => (row.details || []).some(hasActivityBlock)))
    );
  }

  function renderStatus() {
    const status = state.ui.status || {};
    const session = status.session || {};
    const runtime = status.runtime || {};
    const bar = $("#statusbar");
    const items = el("ul", { class: "nav flex-nowrap align-items-center text-nowrap" });
    bar.replaceChildren(items);

    const item = (label, value, page = null) =>
      el("li", { class: "nav-item flex-shrink-0" }, [
        el(
          page ? "a" : "span",
          {
            class: "nav-link d-inline-flex align-items-center gap-2 fg-body",
            href: page ? `/?page=${encodeURIComponent(page)}` : undefined,
            "aria-label": page ? `打开${label}，共 ${value} 个任务` : undefined,
            onclick: page
              ? (event) => {
                  event.preventDefault();
                  openPage(page);
                }
              : undefined,
          },
          [el("span", { text: label }), el("b", { text: String(value) })],
        ),
      ]);

    const dotClass =
      state.connected === null
        ? "is-connecting"
        : state.connected
          ? "is-connected"
          : "is-disconnected";
    const connection = el("li", { class: "nav-item flex-shrink-0" }, [
      el("span", { class: "nav-link d-inline-flex align-items-center gap-2 fg-body" }, [
        el("span", { class: `status-dot ${dotClass}`, "aria-hidden": "true" }),
        el("span", { text: "状态" }),
        el("b", {
          id: "connection",
          role: "status",
          text: state.connected === null ? "连接中" : state.connected ? "已连接" : "连接断开",
        }),
      ]),
    ]);
    items.appendChild(connection);
    items.appendChild(item("会话", shortId(session.session_id)));
    items.appendChild(item("音频", fmtSeconds(session.audio_seconds)));
    items.appendChild(item("视频", fmtSeconds(session.video_seconds)));
    items.appendChild(
      item("转写", `${session.transcript_chars ?? 0} 字 / ${session.transcript_items ?? 0} 段`),
    );
    items.appendChild(
      item("板书", `${session.visual_chars ?? 0} 字 / ${session.visual_items ?? 0} 段`),
    );
    items.appendChild(
      item("插件", (runtime.plugins_loaded || runtime.plugin_order || []).length || 0),
    );
    const tasks = state.ui.tasks || [];
    items.appendChild(item("任务队列", tasks.length, "ui:tasks"));
  }

  async function openPage(pageId) {
    try {
      await changeNavigation(() =>
        api(`/api/ui/pages/${encodeURIComponent(pageId)}/open`, {
          method: "POST",
          body: JSON.stringify({ navigation: navigationFromUrl() }),
        }),
      );
    } catch (err) {
      toast(`打开页面失败：${err.message}`, TOAST_KIND.ERR);
    }
  }

  async function closePage(pageId) {
    try {
      await changeNavigation(() =>
        api(`/api/ui/pages/${encodeURIComponent(pageId)}/close`, {
          method: "POST",
          body: JSON.stringify({ navigation: navigationFromUrl() }),
        }),
      );
    } catch (err) {
      toast(`关闭页面失败：${err.message}`, TOAST_KIND.ERR);
    }
  }

  function formValues(form) {
    const values = {};
    for (const field of Array.from(form.elements)) {
      if (!field.name || field.type === FIELD_INPUT.SUBMIT || field.type === FIELD_INPUT.BUTTON)
        continue;
      if (field.closest('[data-hidden="1"]')) continue;
      if (field.type === FIELD_INPUT.CHECKBOX) {
        values[field.name] = field.checked;
      } else if (field.dataset.fieldType === SETTING_TYPE.JSON) {
        try {
          values[field.name] = JSON.parse(field.value);
        } catch (_) {
          values[field.name] = field.value;
        }
      } else if (
        field.dataset.fieldType === SETTING_TYPE.INT ||
        field.dataset.fieldType === FIELD_INPUT.NUMBER
      ) {
        values[field.name] = field.value === "" ? null : Number.parseInt(field.value, 10);
      } else if (field.dataset.fieldType === SETTING_TYPE.FLOAT) {
        values[field.name] = field.value === "" ? null : Number.parseFloat(field.value);
      } else {
        values[field.name] = field.value;
      }
    }
    return values;
  }

  async function postAction(
    action,
    values = {},
    pageId = null,
    itemId = null,
    independent = false,
  ) {
    try {
      const request = async () => {
        const navigation = navigationFromUrl();
        const result = await api("/api/ui/action", {
          method: "POST",
          body: JSON.stringify({
            action,
            values,
            page_id: pageId || navigation.page || null,
            item_id: itemId || null,
            navigation,
          }),
        });
        if (result.message) toast(result.message, result.ok ? TOAST_KIND.DEFAULT : TOAST_KIND.ERR);
        if (result.open_url && /^https?:\/\//i.test(result.open_url)) {
          const opened = window.open(result.open_url, "_blank", "noopener");
          if (opened) opened.opener = null;
        }
        return result;
      };
      if (!independent) return await changeNavigation(request);
      const navigation = JSON.stringify(navigationFromUrl());
      const version = state.navigationVersion;
      const result = await request();
      if (
        version === state.navigationVersion &&
        !state.navigationPending &&
        navigation === JSON.stringify(navigationFromUrl())
      ) {
        syncNavigation(result?.navigation, HISTORY_MODE.PUSH);
        if (result?.refresh !== false || result?.navigation) await loadState({ quiet: true });
      }
      return result;
    } catch (err) {
      toast(`操作失败：${err.message}`, TOAST_KIND.ERR);
      return { ok: false };
    }
  }

  document.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-action]");
    if (!btn || btn.tagName === "FORM") return;
    event.preventDefault();
    const action = btn.dataset.action;
    if (!action) return;
    let values = {};
    for (const [key, value] of Object.entries(btn.dataset)) {
      if (["action", "value", "pageId", "itemId"].includes(key)) continue;
      values[key] = value;
    }
    if (btn.dataset.value) {
      try {
        values = { ...values, ...JSON.parse(btn.dataset.value) };
      } catch (_) {}
    }
    const form = btn.closest("form");
    if (form) values = { ...formValues(form), ...values };
    postAction(action, values, btn.dataset.pageId || null, btn.dataset.itemId || null);
  });

  document.addEventListener("change", (event) => {
    const input = event.target.closest("input[data-change-action], select[data-change-action]");
    if (!input) return;
    const values =
      input.type === FIELD_INPUT.CHECKBOX
        ? { enabled: input.checked }
        : { [input.name]: input.value };
    postAction(input.dataset.changeAction, values);
  });

  document.addEventListener("submit", async (event) => {
    const form = event.target.closest("form[data-action]");
    if (!form) return;
    event.preventDefault();
    const result = await postAction(form.dataset.action, formValues(form));
    if (form.classList.contains("conv-composer") && result?.ok) {
      const textarea = form.querySelector("textarea");
      if (textarea) {
        textarea.value = "";
        textarea.style.height = "";
        textarea.style.overflowY = "hidden";
      }
    }
  });

  document.addEventListener("keydown", (event) => {
    const textarea = event.target.closest?.(".conv-composer textarea");
    if (!textarea) return;
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      textarea.closest("form")?.requestSubmit();
    }
  });

  document.addEventListener("input", (event) => {
    const textarea = event.target.closest?.(".conv-composer textarea");
    if (!textarea) return;
    textarea.style.height = "auto";
    const height = Math.min(textarea.scrollHeight, 160);
    textarea.style.height = `${height}px`;
    textarea.style.overflowY = textarea.scrollHeight > 160 ? "auto" : "hidden";
  });

  document.addEventListener("click", (event) => {
    const plus = event.target.closest?.(".conv-plus");
    if (plus) {
      event.preventDefault();
      const wrap = plus.closest(".conv-plus-wrap");
      wrap?.classList.toggle("open");
      return;
    }
    if (!event.target.closest?.(".conv-plus-wrap")) {
      for (const wrap of $$(".conv-plus-wrap.open")) wrap.classList.remove("open");
    }
  });

  let refreshTimer = null;
  function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => loadState({ quiet: true }), 180);
  }

  function connectWs() {
    if (state.ws && state.ws.readyState <= 1) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;
    ws.addEventListener("open", () => {
      state.wsRetry = 0;
      setConnection(true);
      ws.send("hello");
      loadState({ quiet: true });
    });
    ws.addEventListener("message", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === WS_MESSAGE_TYPE.REFRESH) scheduleRefresh();
        else if (data.type === WS_MESSAGE_TYPE.NOTIFY && data.topic === "qa.live") {
          applyQaLive(data.payload || {});
        } else if (
          data.type === WS_MESSAGE_TYPE.NOTIFY &&
          data.topic === "ui.toast"
        ) {
          const message = String((data.payload || {}).message || "");
          if (message) toast(message);
        }
      } catch (_) {}
    });
    ws.addEventListener("close", () => {
      setConnection(false);
      state.ws = null;
      const delay = Math.min(8000, 800 * 2 ** state.wsRetry++);
      setTimeout(connectWs, delay);
    });
    ws.addEventListener("error", () => {
      try {
        ws.close();
      } catch (_) {}
    });
  }

  async function boot() {
    applyTheme(themeMode());
    applySidebarState();
    bindSidebarControls();
    bindSidebarResize();
    bindSelectionMenu();
    bindPaletteTrigger();
    window.addEventListener("popstate", () => restoreNavigation());
    await restoreNavigation();
    connectWs();
    setInterval(() => {
      if (Date.now() - state.lastLoad > 2500) loadState({ quiet: true });
    }, 4000);
  }

  document.addEventListener("click", (event) => {
    const filler = event.target?.closest?.("[data-fill-text]");
    if (filler) {
      const text = filler.getAttribute("data-fill-text") || "";
      const root = filler.closest(".block")?.parentElement || document;
      const target = root.querySelector("form textarea, form input[type=text]");
      if (target) {
        target.value = text;
        target.dispatchEvent(new Event("input", { bubbles: true }));
        target.focus();
        event.preventDefault();
        return;
      }
    }
    const link = event.target?.closest?.('a[href^="#"]');
    if (!link) return;
    const href = link.getAttribute("href") || "";
    let id = href.slice(1);
    try {
      id = decodeURIComponent(id);
    } catch (_) {
      // keep the raw fragment
    }
    const target = id ? document.getElementById(id) : null;
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  boot();
})();
