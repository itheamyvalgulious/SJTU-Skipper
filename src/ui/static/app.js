(() => {
  "use strict";

  const state = {
    ui: null,
    activeTab: { left: null, right: null },
    blockTabs: {},
    pageKey: null,
    playerCache: {},
    paletteItems: [],
    paletteIndex: 0,
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
        throwOnError: false,
        strict: false,
      });
    } catch (_) {
      /* one malformed formula must not stop the page */
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

  function renderTree(block) {
    const wrap = el("div", { class: "tree" });
    const nodes = Array.isArray(block.nodes) ? block.nodes : [];
    if (!nodes.length) {
      wrap.appendChild(el("div", { class: "tree-empty", text: block.empty_text || "暂无内容" }));
      return wrap;
    }
    for (const node of nodes) wrap.appendChild(renderTreeNode(node));
    return wrap;
  }

  function renderTreeNode(node) {
    const box = el("div", { class: "tree-node" });
    const children = Array.isArray(node.children) ? node.children : [];
    const hasChildren = children.length > 0;
    const row = el("div", {
      class: `tree-row ${node.active ? "active" : ""} ${node.live ? "live" : ""}`,
    });
    const caret = el("button", {
      class: `tree-caret ${hasChildren ? "" : "leaf"}`,
      type: "button",
      text: node.expanded ? "▾" : "▸",
      title: hasChildren ? (node.expanded ? "折叠" : "展开") : "",
    });
    caret.addEventListener("click", (event) => {
      event.stopPropagation();
      if (hasChildren) runNodeAction(node.toggle);
    });
    row.appendChild(caret);
    if (node.icon) row.appendChild(el("span", { class: "tree-icon", text: node.icon }));
    row.appendChild(el("span", { class: "tree-label", text: node.label || node.id || "" }));
    if (node.subtitle) row.appendChild(el("span", { class: "tree-sub", text: node.subtitle }));
    if (node.badge) {
      row.appendChild(
        el("span", { class: `tree-badge ${node.badge_class || ""}`, text: node.badge })
      );
    }
    row.addEventListener("click", () => {
      if (node.action) runNodeAction(node.action);
      else if (hasChildren) runNodeAction(node.toggle);
    });
    box.appendChild(row);
    if (hasChildren && node.expanded) {
      const kids = el("div", { class: "tree-children" });
      for (const child of children) kids.appendChild(renderTreeNode(child));
      box.appendChild(kids);
    }
    return box;
  }

  function runNodeAction(spec) {
    if (!spec || !spec.action) return;
    postAction(spec.action, spec.value || {});
  }

  function renderRows(block) {    const wrap = el("div", { class: "row-list block" });
    for (const item of block.items || []) {
      const row = el("div", { class: `row-item ${item.active ? "active" : ""}` });
      const main = el("div", { class: "row-main" }, [
        el("div", { class: "row-title", text: item.title || item.text || "" }),
        item.subtitle ? el("div", { class: "row-sub", text: item.subtitle }) : null,
        item.meta ? el("div", { class: "row-meta", text: item.meta }) : null,
      ]);
      row.appendChild(main);
      if (item.action) {
        row.appendChild(actionButton(item.label || "打开", item.action, item.value, item.class || "btn btn-sm"));
      }
      wrap.appendChild(row);
    }
    if (!wrap.children.length) wrap.appendChild(el("div", { class: "empty", text: "暂无数据" }));
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
      if (field.type === "action" || field.input === "button") {
        row.appendChild(
          actionButton(
            field.button_label || field.label || "执行",
            field.action,
            {},
            "btn btn-sm btn-primary"
          )
        );
        if (field.description) row.appendChild(el("div", { class: "hint", text: field.description }));
        form.appendChild(row);
        continue;
      }
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
        if (rawType === "int" || type === "number") input.step = "1";
        if (rawType === "float") input.step = "any";
      }
      input.dataset.fieldType = rawType;
      if (field.readonly) {
        input.readOnly = true;
        input.classList.add("input-readonly");
      }
      row.appendChild(input);
      if (field.secret) {
        const status = el("div", {
          class: `form-status ${field.is_set ? "ok" : ""}`,
          text: field.is_set
            ? `已配置 · ${field.secret_length || "?"} 字符；真实值不会回传前端`
            : "未配置",
        });
        const line = el("div", { class: "form-inline" }, [status]);
        if (field.is_set) {
          line.appendChild(
            actionButton("清除", "settings.clear_secret", { key: fieldName }, "btn btn-sm btn-danger")
          );
        }
        row.appendChild(line);
      }
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
    const key = block.id || block.key || `tabs:${tabs.map((t) => t.title || "").join("\u0001")}`;
    const savedTitle = state.blockTabs[key];
    // A client-side selection wins over the server hint so that periodic
    // refreshes do not yank the reader back to the document it opened with.
    let active = tabs.findIndex((t) => (t.title || "") === savedTitle);
    if (active < 0) active = tabs.findIndex((t) => t.active);
    if (active < 0) active = 0;
    const show = (index, remember = true) => {
      active = index;
      if (remember) state.blockTabs[key] = tabs[index]?.title || "";
      $$(".tab-btn", bar).forEach((b, i) => b.classList.toggle("active", i === active));
      body.replaceChildren(renderBlocks(tabs[index]?.blocks || []));
      if (remember && (body.textContent.includes("$") || body.textContent.includes("\\("))) {
        typeset(body);
      }
    };
    tabs.forEach((tab, i) => {
      const btn = el("button", { class: "tab-btn", type: "button", text: tab.title || `Tab ${i + 1}` });
      btn.addEventListener("click", () => show(i));
      bar.appendChild(btn);
    });
    wrap.appendChild(bar);
    wrap.appendChild(body);
    if (tabs.length) show(active, false);
    return wrap;
  }

  // ------------------------------------------------------------------
  // Video player (class page)
  // ------------------------------------------------------------------
  function playerKey(block) {
    return block.id || "player";
  }

  function attachStream(video, source, startAt) {
    const src = String(source.src || "");
    if (!src) return;
    if (src.includes(".flv") && window.mpegts && window.mpegts.isSupported()) {
      const player = window.mpegts.createPlayer({ type: "flv", isLive: true, url: src });
      player.attachMediaElement(video);
      player.load();
      video._mpegts = player;
      return;
    }
    const isHls = src.includes(".m3u8");
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
        () => { try { video.currentTime = startAt; } catch (_) {} },
        { once: true }
      );
    }
  }

  function renderPlayer(block) {
    const key = playerKey(block);
    const cache = state.playerCache[key] || {};
    const wrap = el("div", { class: "player" });
    const grid = el("div", { class: "player-grid" });
    const videos = [];
    const sources = Array.isArray(block.sources) ? block.sources : [];

    for (const source of sources) {
      const saved = block.live ? {} : cache[source.key] || {};
      const pane = el("div", { class: "player-pane" });
      pane.appendChild(
        el("div", { class: "player-label", text: source.label || source.key || "" })
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
      videos.push(video);
      pane.appendChild(video);
      grid.appendChild(pane);
    }

    if (block.screen_missing) {
      grid.appendChild(
        el("div", { class: "player-pane player-empty", text: "该课程没有屏幕流" })
      );
    }
    wrap.appendChild(grid);

    const bar = el("div", { class: "player-actions" });
    const finished = Boolean(block.finished);
    const live = Boolean(block.live);
    let processing = Boolean(block.processing);

    const stateText = () => {
      if (finished) return "已结束";
      if (live) return processing ? "直播 · 处理中" : "直播 · 已暂停处理";
      return processing ? "处理中" : "已暂停";
    };
    const stateEl = el("span", { class: "player-state", text: stateText() });
    const toggle = el("button", {
      class: "btn btn-sm btn-primary",
      type: "button",
      text: finished ? "已结束" : processing ? "暂停处理" : "继续处理",
    });
    toggle.disabled = finished;

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
    const pauseAll = () => {
      for (const video of videos) {
        try { video.pause(); } catch (_) {}
      }
    };

    if (!finished) {
      toggle.addEventListener("click", async () => {
        const action = processing ? block.pause_action : block.start_action;
        const result = await postAction(action, {});
        if (!result || result.ok === false) return;
        if (!processing && String(result.message || "").includes("准备")) {
          // Streams are still resolving; the server starts processing as soon
          // as they are ready and the page re-renders in the running state.
          stateEl.textContent = "准备中…";
          return;
        }
        processing = !processing;
        toggle.textContent = processing ? "暂停处理" : "继续处理";
        stateEl.textContent = stateText();
        if (processing) playAll();
        else if (!live) pauseAll();
      });
    }

    // Live video keeps playing while processing is paused; recordings only
    // play while processing is running.
    if (!finished && (live || processing)) {
      playAll();
    }

    bar.appendChild(stateEl);
    bar.appendChild(toggle);
    if (block.part && block.parts && block.parts > 1) {
      bar.appendChild(
        el("span", { class: "player-meta", text: `第 ${block.part}/${block.parts} 节` })
      );
    }
    wrap.appendChild(bar);
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

  function destroyPlayers(root) {
    if (!root) return;
    for (const video of $$("video", root)) {
      if (video._hls) {
        try { video._hls.destroy(); } catch (_) {}
        video._hls = null;
      }
      if (video._mpegts) {
        try { video._mpegts.destroy(); } catch (_) {}
        video._mpegts = null;
      }
    }
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
    if (type === "rows") return renderRows(block);
    if (type === "tree") return el("div", { class: "block" }, [renderTree(block)]);
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
    if (type === "player") return renderPlayer(block);
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
    renderMenubar();
    renderSidebar("left");
    renderSidebar("right");
    renderPageTabs();
    const pageChanged = renderPageContent();
    renderStatus();
    if (pageChanged) typeset();
  }

  // ------------------------------------------------------------------
  // Menubar + command palette
  // ------------------------------------------------------------------
  function commands() {
    return state.ui?.commands || [];
  }

  function closeMenus() {
    $$(".menu-wrap.open").forEach((wrap) => {
      wrap.classList.remove("open");
      const drop = $(".menu-drop", wrap);
      if (drop) drop.remove();
      const btn = $(".menu-btn", wrap);
      if (btn) btn.classList.remove("open");
    });
  }

  function runCommand(cmd) {
    closeMenus();
    closePalette();
    if (cmd.page) return openPage(cmd.page);
    if (cmd.action) return postAction(cmd.action, {});
    return null;
  }

  function renderMenubar() {
    const bar = $("#menubar");
    if (!bar) return;
    bar.replaceChildren();
    const groups = new Map();
    for (const cmd of commands()) {
      const name = cmd.category || "其他";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(cmd);
    }
    for (const [name, items] of groups) {
      const wrap = el("div", { class: "menu-wrap" });
      const btn = el("button", { class: "menu-btn", type: "button", text: name });
      btn.addEventListener("click", (event) => {
        event.stopPropagation();
        const wasOpen = wrap.classList.contains("open");
        closeMenus();
        if (wasOpen) return;
        wrap.classList.add("open");
        btn.classList.add("open");
        const drop = el("div", { class: "menu-drop" });
        for (const cmd of items) {
          const item = el("button", { class: "menu-item", type: "button" }, [
            el("span", { text: `${cmd.icon ? cmd.icon + " " : ""}${cmd.title}` }),
            cmd.subtitle ? el("span", { class: "sub", text: cmd.subtitle }) : null,
          ]);
          item.addEventListener("click", (event) => {
            event.stopPropagation();
            runCommand(cmd);
          });
          drop.appendChild(item);
        }
        wrap.appendChild(drop);
      });
      wrap.appendChild(btn);
      bar.appendChild(wrap);
    }
  }

  function paletteMatches(query) {
    const q = String(query || "").trim().toLowerCase();
    const items = commands();
    if (!q) return items;
    return items.filter((cmd) => {
      const hay = `${cmd.title} ${cmd.subtitle || ""} ${cmd.category || ""} ${cmd.keywords || ""}`.toLowerCase();
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
      list.appendChild(el("div", { class: "palette-empty", text: "没有匹配的命令" }));
      return;
    }
    items.forEach((cmd, index) => {
      const row = el("button", {
        class: `palette-item ${index === state.paletteIndex ? "active" : ""}`,
        type: "button",
      }, [
        cmd.category ? el("span", { class: "cat", text: cmd.category }) : null,
        el("span", { text: `${cmd.icon ? cmd.icon + " " : ""}${cmd.title}` }),
        cmd.subtitle ? el("span", { class: "sub", text: cmd.subtitle }) : null,
      ]);
      row.addEventListener("click", () => runCommand(cmd));
      row.addEventListener("mouseenter", () => {
        state.paletteIndex = index;
        $$(".palette-item", list).forEach((node, i) => node.classList.toggle("active", i === index));
      });
      list.appendChild(row);
    });
  }

  function openPalette() {
    const palette = $("#palette");
    if (!palette) return;
    palette.hidden = false;
    const input = $("#palette-input");
    input.value = "";
    state.paletteIndex = 0;
    renderPalette("");
    input.focus();
  }

  function closePalette() {
    const palette = $("#palette");
    if (palette) palette.hidden = true;
    state.paletteIndex = 0;
  }

  function paletteOpen() {
    const palette = $("#palette");
    return palette && !palette.hidden;
  }

  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === "p") {
      event.preventDefault();
      if (paletteOpen()) closePalette();
      else openPalette();
      return;
    }
    if (!paletteOpen()) return;
    const items = state.paletteItems || [];
    if (key === "escape") {
      event.preventDefault();
      closePalette();
    } else if (key === "arrowdown") {
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

  document.addEventListener("click", (event) => {
    if (!event.target.closest("#menubar")) closeMenus();
  });

  function bindPaletteTrigger() {
    const btn = $("#palette-btn");
    if (btn) btn.addEventListener("click", () => openPalette());
    const palette = $("#palette");
    if (palette) {
      palette.addEventListener("click", (event) => {
        if (event.target === palette) closePalette();
      });
    }
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
    // A single tab needs no tab strip: the left sidebar is just the course tree.
    tabsEl.hidden = (tabs || []).length <= 1;
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
      toolboxEl.appendChild(el("div", { class: "empty", text: "这个 tab 还没有工具" }));
      return;
    }
    for (const item of current.toolbox) {
      const panel = el("section", { class: `toolbox-item ${item.title || item.description ? "" : "bare"}` });
      const pageId = item.page || null;
      if (item.title || item.description) {
        const head = el("div", { class: "toolbox-head" }, [
          el("div", { class: "toolbox-title", text: item.title || item.id }),
          item.description ? el("div", { class: "toolbox-desc", text: item.description }) : null,
        ]);
        if (pageId) {
          const open = actionButton("打开页面", "ui.open_page", { page_id: pageId }, "btn btn-sm btn-ghost");
          open.style.float = "right";
          head.appendChild(open);
        }
        panel.appendChild(head);
      }
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

  function renderPageTabs() {
    const bar = $("#page-tabs");
    bar.replaceChildren();
    const openIds = state.ui.open_pages || [];
    const pages = (state.ui.pages || []).filter(
      (p) => openIds.includes(p.id) || p.id === state.ui.active_page
    );
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
    const active = state.ui.active_page || "";
    const blocks = active ? state.ui.page_content?.[active] : null;
    const key = active ? `${active}\u0000${JSON.stringify(blocks || [])}` : "__welcome__";
    const samePage = container.dataset.pageId === active;
    // Page content is polled every few seconds; never re-render or re-run
    // KaTeX when the payload is identical.  This is what made the board and
    // notes pages feel like they were reloading constantly.
    if (samePage && state.pageKey === key) return false;
    const scrollTop = samePage ? container.scrollTop : 0;
    const focused = document.activeElement;
    const editing =
      samePage &&
      focused &&
      container.contains(focused) &&
      ["INPUT", "TEXTAREA", "SELECT"].includes(focused.tagName);
    if (editing) return false; // don't wipe out a form the user is typing in
    destroyPlayers(container);
    container.replaceChildren();
    container.dataset.pageId = active;
    state.pageKey = key;
    if (!active || !blocks) {
      container.appendChild(el("div", { class: "welcome" }, [
        el("div", { class: "welcome-mark", text: "∮" }),
        el("h2", { text: "选择一节课" }),
        el("p", { text: "左侧课程树里展开课程，选择录播或直播。按 Ctrl+P 打开命令面板。" }),
      ]));
      container.scrollTop = 0;
      return true;
    }
    container.appendChild(renderBlocks(blocks));
    container.scrollTop = samePage ? scrollTop : 0;
    return true;
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
    bindPaletteTrigger();
    loadState();
    connectWs();
    setInterval(() => {
      if (Date.now() - state.lastLoad > 2500) loadState({ quiet: true });
    }, 4000);
  }

  boot();
})();
