(() => {
  "use strict";

  const state = {
    ui: null,
    activeTab: { left: null, right: null },
    loading: false,
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
    String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));

  const fmtSeconds = (v) => {
    const n = Number(v || 0);
    if (!Number.isFinite(n)) return "0s";
    const m = Math.floor(n / 60);
    const s = Math.floor(n % 60);
    return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
  };

  const shortId = (id) => (id ? String(id).slice(0, 18) : "—");

  function toast(message, kind = "") {
    if (!message) return;
    const node = $("#toast");
    node.textContent = message;
    node.className = `toast show ${kind}`;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      node.className = "toast";
    }, 3200);
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const type = resp.headers.get("content-type") || "";
    return type.includes("application/json") ? resp.json() : resp.text();
  }

  async function loadState({ quiet = false } = {}) {
    if (state.loading) return;
    state.loading = true;
    try {
      const ui = await api("/api/ui/state");
      state.ui = ui;
      state.lastLoad = Date.now();
      renderAll();
      setConnection(true);
    } catch (err) {
      setConnection(false);
      if (!quiet) toast(`加载状态失败：${err.message}`, "err");
    } finally {
      state.loading = false;
    }
  }

  function setConnection(ok) {
    const pill = $("#connection");
    pill.textContent = ok ? "已连接" : "连接断开";
    pill.className = `pill ${ok ? "pill-ok" : "pill-err"}`;
  }

  function typeset() {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([$("#page-content")]).catch(() => {});
    }
  }

  // ------------------------------------------------------------------
  // Blocks
  // ------------------------------------------------------------------
  function actionButton(label, action, value = undefined, cls = "btn btn-sm") {
    const attrs = { class: cls, dataset: { action } };
    if (value !== undefined) attrs.dataset.value = JSON.stringify(value);
    attrs.type = "button";
    return el("button", { ...attrs, text: String(label) }, []);
  }

  function renderLog(block) {
    const pre = el("pre", { class: "log" });
    const lines = Array.isArray(block.lines) ? block.lines : [];
    for (const line of lines) {
      if (typeof line === "string") {
        pre.appendChild(el("span", { class: "log-line", text: line }));
      } else {
        pre.appendChild(el("span", { class: "log-line" }, [
          line.time ? el("span", { class: "ts", text: String(line.time) }) : null,
          document.createTextNode(String(line.text ?? line.message ?? "")),
        ]));
      }
    }
    return pre;
  }

  function renderTable(block) {
    const wrap = el("div", { class: "table-wrap" });
    const table = el("table", { class: "data-table" });
    const columns = Array.isArray(block.columns) ? block.columns : [];
    const rows = Array.isArray(block.rows) ? block.rows : [];
    const rowActions = Array.isArray(block.row_actions) ? block.row_actions : [];

    const colLabel = (c) => (typeof c === "string" ? c : (c.label ?? c.key ?? ""));
    const colKey = (c, i) => (typeof c === "string" ? c : (c.key ?? i));
    const thead = el("thead");
    const tr = el("tr");
    for (const c of columns) tr.appendChild(el("th", { text: colLabel(c) }));
    if (rowActions.length) tr.appendChild(el("th", { text: "操作" }));
    thead.appendChild(tr);
    table.appendChild(thead);

    const tbody = el("tbody");
    for (const row of rows) {
      const rowObj = Array.isArray(row)
        ? Object.fromEntries(columns.map((c, i) => [colKey(c, i), row[i]]))
        : (row || {});
      const tr = el("tr");
      for (let i = 0; i < columns.length; i += 1) {
        const c = columns[i];
        const key = colKey(c, i);
        let value = Array.isArray(row) ? row[i] : rowObj[key];
        if (typeof value === "object" && value !== null) value = JSON.stringify(value);
        tr.appendChild(el("td", { text: value ?? "" }));
      }
      if (rowActions.length) {
        const td = el("td");
        for (const action of rowActions) {
          let value = action.value ? { ...action.value } : {};
          if (Array.isArray(action.keys)) {
            for (const key of action.keys) value[key] = rowObj[key];
          } else if (!action.value) {
            value = { ...rowObj };
          }
          td.appendChild(actionButton(action.label || "执行", action.action, value, action.class || "btn btn-sm"));
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    if (!rows.length) wrap.appendChild(el("div", { class: "empty", text: "暂无数据" }));
    return wrap;
  }

  function renderCards(block) {
    const wrap = el("div", { class: "cards" });
    for (const item of block.items || []) {
      const card = el("div", { class: "card" });
      if (item.title) card.appendChild(el("div", { class: "card-title", text: item.title }));
      if (item.subtitle) card.appendChild(el("div", { class: "card-sub", text: item.subtitle }));
      if (item.meta) {
        const meta = el("div", { class: "card-meta" });
        const entries = Array.isArray(item.meta) ? item.meta : Object.entries(item.meta);
        for (const entry of entries) {
          const text = Array.isArray(entry) ? `${entry[0]}: ${entry[1]}` : String(entry);
          meta.appendChild(el("span", { text }));
        }
        card.appendChild(meta);
      }
      if (item.blocks) card.appendChild(renderBlocks(item.blocks));
      if (item.actions && item.actions.length) {
        const actions = el("div", { class: "card-actions" });
        for (const action of item.actions) {
          actions.appendChild(actionButton(action.label || "执行", action.action, action.value, action.class || "btn btn-sm"));
        }
        card.appendChild(actions);
      }
      wrap.appendChild(card);
    }
    if (!wrap.children.length) wrap.appendChild(el("div", { class: "empty", text: "暂无数据" }));
    return wrap;
  }

  function renderForm(block) {
    const submitSpec = block.submit || {};
    const action = block.action || submitSpec.action || "";
    const form = el("form", { class: "form" });
    if (action) form.dataset.action = action;
    for (const field of block.fields || []) {
      const row = el("div", { class: "form-row" });
      const id = `f-${Math.random().toString(36).slice(2)}`;
      const fieldName = field.name || field.key;
      row.appendChild(el("label", { for: id, text: field.label || fieldName }));
      const rawType = field.type || (field.secret ? "secret" : "text");
      const type = field.input || (field.secret ? "password" : rawType);
      let input;
      if (type === "textarea") {
        input = el("textarea", { class: "textarea", id, name: fieldName, placeholder: field.placeholder || "" });
        input.value = field.value ?? "";
      } else if (type === "select") {
        input = el("select", { class: "select", id, name: fieldName });
        const options = field.option_items || field.options || [];
        for (const option of options) {
          const value = typeof option === "object" ? option.value : option;
          const label = typeof option === "object" ? (option.label ?? option.value) : option;
          const opt = el("option", { value: String(value), text: String(label) });
          if (String(value) === String(field.value ?? field.default ?? "")) opt.selected = true;
          input.appendChild(opt);
        }
      } else if (type === "checkbox" || rawType === "bool") {
        input = el("input", { type: "checkbox", id, name: fieldName });
        input.checked = Boolean(field.value);
      } else {
        input = el("input", {
          class: "input", id, name: fieldName,
          type: type === "password" || field.secret ? "password" : (type === "number" || rawType === "int" || rawType === "float" ? "number" : "text"),
        });
        if (field.value !== undefined && field.value !== null && field.value !== "******") {
          input.value = field.value;
        }
        if (field.placeholder) input.placeholder = field.placeholder;
        if (rawType === "int" || type === "number") input.step = "1";
        if (rawType === "float") input.step = "any";
      }
      input.dataset.fieldType = rawType;
      row.appendChild(input);
      if (field.description) row.appendChild(el("div", { class: "hint", text: field.description }));
      form.appendChild(row);
    }
    const actions = el("div", { class: "form-actions" });
    const submit = el("button", {
      class: "btn btn-primary", type: "submit",
      text: block.submit_label || submitSpec.label || "保存",
    });
    actions.appendChild(submit);
    form.appendChild(actions);
    return form;
  }

  function renderTabs(block) {
    const wrap = el("div", { class: "tabs" });
    const bar = el("div", { class: "tabs-bar" });
    const body = el("div", { class: "tabs-body" });
    const tabs = block.tabs || [];
    let active = 0;
    const show = (index) => {
      active = index;
      $$(".tab-btn", bar).forEach((b, i) => b.classList.toggle("active", i === active));
      body.replaceChildren(renderBlocks(tabs[index]?.blocks || []));
    };
    tabs.forEach((tab, i) => {
      const btn = el("button", { class: "tab-btn", type: "button", text: tab.title || `Tab ${i + 1}` });
      btn.addEventListener("click", () => show(i));
      bar.appendChild(btn);
    });
    wrap.appendChild(bar);
    wrap.appendChild(body);
    if (tabs.length) show(0);
    return wrap;
  }

  function renderBlocks(blocks) {
    const frag = document.createDocumentFragment();
    for (const block of blocks || []) {
      const node = renderBlock(block);
      if (node) frag.appendChild(node);
    }
    return frag;
  }

  function renderBlock(block) {
    if (!block || typeof block !== "object") return null;
    const type = block.type || "text";
    if (type === "html") return el("div", { class: "block", html: block.html || "" });
    if (type === "markdown") {
      return el("div", { class: "block markdown", html: block.html || escapeHtml(block.text || "") });
    }
    if (type === "text") return el("pre", { class: "block pre", text: block.text ?? "" });
    if (type === "keyvalue") {
      const items = Array.isArray(block.items) ? block.items : Object.entries(block.items || {});
      const grid = el("div", { class: "kv-grid" });
      for (const [k, v] of items) {
        grid.appendChild(el("div", { class: "k", text: k }));
        grid.appendChild(el("div", { text: typeof v === "object" ? JSON.stringify(v) : String(v ?? "") }));
      }
      return el("div", { class: "block" }, [grid]);
    }
    if (type === "table") return el("div", { class: "block" }, [renderTable(block)]);
    if (type === "cards") return el("div", { class: "block" }, [renderCards(block)]);
    if (type === "list") {
      const list = el("div", { class: "block btn-row" });
      for (const item of block.items || []) {
        if (typeof item === "string") list.appendChild(el("div", { text: item }));
        else if (item.action) list.appendChild(actionButton(item.text || item.label || "执行", item.action, item.value, item.class || "btn btn-sm"));
        else list.appendChild(el("div", { text: item.text || JSON.stringify(item) }));
      }
      return list;
    }
    if (type === "buttons") {
      const row = el("div", { class: "block btn-row" });
      for (const b of block.buttons || []) {
        row.appendChild(actionButton(b.label || "执行", b.action, b.value, b.class || (b.primary ? "btn btn-primary" : "btn")));
      }
      return row;
    }
    if (type === "form") return el("div", { class: "block" }, [renderForm(block)]);
    if (type === "log") return el("div", { class: "block" }, [renderLog(block)]);
    if (type === "image") {
      return el("div", { class: "block" }, [el("img", { class: "image", src: block.src, alt: block.alt || "" })]);
    }
    if (type === "video") {
      const video = el("video", { controls: "controls", src: block.src });
      return el("div", { class: "block" }, [video]);
    }
    if (type === "tabs") return el("div", { class: "block" }, [renderTabs(block)]);
    if (type === "progress") {
      const value = Number(block.value ?? 0);
      const pct = value <= 1 ? value * 100 : value;
      return el("div", { class: "block" }, [
        el("div", { class: "progress-label" }, [
          el("span", { text: block.label || "进度" }),
          el("span", { text: `${Math.round(pct)}%` }),
        ]),
        el("div", { class: "progress" }, [el("i", { style: `width:${Math.max(0, Math.min(100, pct))}%` })]),
      ]);
    }
    if (type === "badge") return el("span", { class: `badge ${block.class || ""}`, text: block.text || "" });
    return el("pre", { class: "block pre", text: JSON.stringify(block, null, 2) });
  }

  // ------------------------------------------------------------------
  // Layout
  // ------------------------------------------------------------------
  function renderAll() {
    if (!state.ui) return;
    renderSidebar("left");
    renderSidebar("right");
    renderPageTabs();
    renderPageContent();
    renderStatus();
    typeset();
  }

  function pickTab(tabs, side) {
    const current = state.activeTab[side];
    if (current && tabs.some((t) => t.id === current)) return current;
    const saved = localStorage.getItem(`lecture-agent.tab.${side}`);
    if (saved && tabs.some((t) => t.id === saved)) return saved;
    return tabs[0]?.id || null;
  }

  function renderSidebar(side) {
    const tabs = side === "left" ? state.ui.left_tabs : state.ui.right_tabs;
    const tabsEl = $(`#${side}-tabs`);
    const toolboxEl = $(`#${side}-toolbox`);
    tabsEl.replaceChildren();
    const active = pickTab(tabs || [], side);
    state.activeTab[side] = active;
    if (active) localStorage.setItem(`lecture-agent.tab.${side}`, active);
    for (const tab of tabs || []) {
      const btn = el("button", {
        class: `side-tab ${tab.id === active ? "active" : ""}`,
        type: "button",
      }, [
        tab.icon ? el("span", { class: "icon", text: tab.icon }) : null,
        el("span", { text: tab.title || tab.id }),
      ]);
      btn.addEventListener("click", () => {
        state.activeTab[side] = tab.id;
        localStorage.setItem(`lecture-agent.tab.${side}`, tab.id);
        renderSidebar(side);
        // Tab-level page convention: settings -> settings:main.  Other tabs
        // expose explicit buttons/actions instead of auto-navigation.
        if (tab.id === "settings") {
          const page = (state.ui.pages || []).find((p) => p.id === "settings:main");
          if (page) openPage(page.id);
        }
      });
      tabsEl.appendChild(btn);
    }
    toolboxEl.replaceChildren();
    const current = (tabs || []).find((t) => t.id === active);
    if (!current || !(current.toolbox || []).length) {
      toolboxEl.appendChild(el("div", { class: "empty", text: "这个 tab 还没有工具" }));
      return;
    }
    for (const item of current.toolbox) {
      const panel = el("section", { class: "toolbox-item" });
      const pageId = findPageForItem(item);
      const head = el("div", { class: "toolbox-head" }, [
        el("div", { class: "toolbox-title", text: item.title || item.id }),
        item.description ? el("div", { class: "toolbox-desc", text: item.description }) : null,
      ]);
      if (pageId) {
        const open = actionButton("打开", "ui.open_page", { page_id: pageId }, "btn btn-sm btn-ghost");
        open.style.float = "right";
        head.appendChild(open);
      }
      panel.appendChild(head);
      const body = el("div", { class: "toolbox-body" });
      if (item.blocks && item.blocks.length) {
        body.appendChild(renderBlocks(item.blocks));
      } else {
        body.appendChild(el("div", { class: "empty", text: pageId ? "在页面中查看" : "暂无内容" }));
      }
      panel.appendChild(body);
      toolboxEl.appendChild(panel);
    }
  }

  function findPageForItem(item) {
    const pages = state.ui?.pages || [];
    const id = item.id || "";
    const ns = id.split(".")[0];
    if (!ns) return null;
    const exact = `${ns}:${id.split(".").slice(1).join(".")}`;
    if (pages.some((p) => p.id === exact)) return exact;
    const candidates = pages.filter((p) => p.id.startsWith(`${ns}:`));
    if (candidates.length === 1) return candidates[0].id;
    for (const suffix of ["main", "chat", "session", "transcript", "notes", "board"]) {
      const wanted = `${ns}:${suffix}`;
      if (pages.some((p) => p.id === wanted)) return wanted;
    }
    return candidates[0]?.id || null;
  }

  function renderPageTabs() {
    const bar = $("#page-tabs");
    bar.replaceChildren();
    const pages = state.ui.pages || [];
    for (const page of pages) {
      const active = page.id === state.ui.active_page;
      const tab = el("div", { class: `page-tab ${active ? "active" : ""}` }, [
        el("span", { text: page.title || page.id }),
        page.closable ? el("button", { class: "close", type: "button", text: "×", title: "关闭" }) : null,
      ]);
      tab.addEventListener("click", async (event) => {
        if (event.target.classList.contains("close")) return;
        if (page.id !== state.ui.active_page) await openPage(page.id);
      });
      const close = $(".close", tab);
      if (close) {
        close.addEventListener("click", async (event) => {
          event.stopPropagation();
          await closePage(page.id);
        });
      }
      bar.appendChild(tab);
    }
  }

  function renderPageContent() {
    const container = $("#page-content");
    container.replaceChildren();
    const active = state.ui.active_page;
    const blocks = active ? state.ui.page_content?.[active] : null;
    if (!active || !blocks) {
      container.appendChild(el("div", { class: "welcome" }, [
        el("div", { class: "welcome-mark", text: "∮" }),
        el("h2", { text: "开始一节课" }),
        el("p", { text: "左侧「课程」里选择录播或直播；设置里填入 Canvas token 与模型 API key。" }),
        el("p", { class: "muted", text: "也可以运行 sjtu-skipper demo 查看完整链路。" }),
      ]));
      return;
    }
    container.appendChild(renderBlocks(blocks));
  }

  function renderStatus() {
    const status = state.ui.status || {};
    const session = status.session || {};
    const runtime = status.runtime || {};
    const llm = status.llm || {};
    const bar = $("#statusbar");
    bar.replaceChildren();

    const item = (label, value, cls = "") => el("span", { class: "status-item" }, [
      cls ? el("span", { class: `dot ${cls}` }) : null,
      el("span", { text: label }),
      el("b", { text: String(value) }),
    ]);

    bar.appendChild(item("会话", shortId(session.session_id)));
    bar.appendChild(item("音频", fmtSeconds(session.audio_seconds)));
    bar.appendChild(item("视频", fmtSeconds(session.video_seconds)));
    bar.appendChild(item("转写", `${session.transcript_chars ?? 0} 字 / ${session.transcript_items ?? 0} 段`));
    bar.appendChild(item("板书", `${session.visual_chars ?? 0} 字 / ${session.visual_items ?? 0} 段`));
    bar.appendChild(item("插件", (runtime.plugins_loaded || runtime.plugin_order || []).length || 0, runtime.started ? "ok" : ""));
    bar.appendChild(item("LLM", llm.provider || (llm.registered ? "已注册" : "未配置"), llm.registered ? "ok" : ""));

    const sessionPill = $("#session-pill");
    if (session.session_id) {
      sessionPill.textContent = `会话 ${shortId(session.session_id)}`;
      sessionPill.className = "pill pill-ok";
    } else {
      sessionPill.textContent = "无会话";
      sessionPill.className = "pill pill-muted";
    }
  }

  async function openPage(pageId) {
    try {
      await api(`/api/ui/pages/${encodeURIComponent(pageId)}/open`, { method: "POST" });
      await loadState({ quiet: true });
    } catch (err) {
      toast(`打开页面失败：${err.message}`, "err");
    }
  }

  async function closePage(pageId) {
    try {
      await api(`/api/ui/pages/${encodeURIComponent(pageId)}/close`, { method: "POST" });
      await loadState({ quiet: true });
    } catch (err) {
      toast(`关闭页面失败：${err.message}`, "err");
    }
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------
  function formValues(form) {
    const values = {};
    for (const field of Array.from(form.elements)) {
      if (!field.name || field.type === "submit" || field.type === "button") continue;
      if (field.type === "checkbox") {
        values[field.name] = field.checked;
      } else if (field.dataset.fieldType === "json") {
        try { values[field.name] = JSON.parse(field.value); }
        catch (_) { values[field.name] = field.value; }
      } else if (field.dataset.fieldType === "int" || field.dataset.fieldType === "number") {
        values[field.name] = field.value === "" ? null : Number.parseInt(field.value, 10);
      } else if (field.dataset.fieldType === "float") {
        values[field.name] = field.value === "" ? null : Number.parseFloat(field.value);
      } else {
        values[field.name] = field.value;
      }
    }
    return values;
  }

  async function postAction(action, values = {}, pageId = null, itemId = null) {
    try {
      const result = await api("/api/ui/action", {
        method: "POST",
        body: JSON.stringify({
          action,
          values,
          page_id: pageId || state.ui?.active_page || null,
          item_id: itemId || null,
        }),
      });
      if (result.message) toast(result.message, result.ok ? "ok" : "err");
      if (result.open_page) await openPage(result.open_page);
      if (result.refresh !== false) await loadState({ quiet: true });
      return result;
    } catch (err) {
      toast(`操作失败：${err.message}`, "err");
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
      try { values = { ...values, ...JSON.parse(btn.dataset.value) }; } catch (_) { /* keep dataset */ }
    }
    const form = btn.closest("form");
    if (form) values = { ...formValues(form), ...values };
    postAction(action, values, btn.dataset.pageId || null, btn.dataset.itemId || null);
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("form[data-action]");
    if (!form) return;
    event.preventDefault();
    postAction(form.dataset.action, formValues(form));
  });

  // ------------------------------------------------------------------
  // WebSocket / polling
  // ------------------------------------------------------------------
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
    });
    ws.addEventListener("message", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "refresh") scheduleRefresh();
      } catch (_) { /* ignore */ }
    });
    ws.addEventListener("close", () => {
      setConnection(false);
      state.ws = null;
      const delay = Math.min(8000, 800 * 2 ** state.wsRetry++);
      setTimeout(connectWs, delay);
    });
    ws.addEventListener("error", () => {
      try { ws.close(); } catch (_) {}
    });
  }

  function boot() {
    loadState();
    connectWs();
    setInterval(() => {
      if (Date.now() - state.lastLoad > 2500) loadState({ quiet: true });
    }, 4000);
  }

  boot();
})();
