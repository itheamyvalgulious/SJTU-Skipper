import { test, expect, spyOn } from "bun:test";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { wrapSteeringForModel } from "@oh-my-pi/pi-coding-agent/session/messages";
import { Effort, type AuthStorageData } from "@oh-my-pi/pi-ai";
import { registerOAuthProvider, unregisterOAuthProvider } from "@oh-my-pi/pi-ai/oauth";
import { Runtime } from "./runtime";
import {
  AgentRole,
  IpcMethod,
  RuntimeEvent,
  WorkspaceTool,
  ipcRequestSchema,
  validate,
  runtimeBootstrapSchema,
  type PersistenceSnapshot,
  type RoleDefinition,
} from "./schemas";
import type { RuntimeCallback } from "./extensions";

const registeredRoles: RoleDefinition[] = [
  { id: "note_chunk", title: "Chunk", input: ["text"] },
  { id: "note_writer", title: "Writer", input: ["text"] },
  { id: "note_review", title: "Review", input: ["text"] },
  { id: "ask_instant", title: "Instant", input: ["text"] },
  { id: "ask_research", title: "Research", input: ["text"] },
  { id: "board_recognition", title: "Board", input: ["text", "image"] },
  { id: "default", title: "Default", input: ["text"] },
  { id: "compaction", title: "Compaction", input: ["text"] },
];

const fixedRoles = (defaultSelector: string): Record<string, string> =>
  Object.fromEntries(
    registeredRoles.map(({ id: role }) => [
      role,
      role === AgentRole.DEFAULT ? defaultSelector : "@default",
    ]),
  );

test("first request creates missing runtime configuration", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "runtime-init-"));
  const root = path.join(directory, "missing");
  const runtime = new Runtime(root);
  try {
    await runtime.dispatch(IpcMethod.PING, {}, () => {});
    expect((await stat(path.join(root, "models.yml"))).mode & 0o777).toBe(0o600);
  } finally {
    await runtime.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("default configuration persists once", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "runtime-defaults-"));
  const snapshots: PersistenceSnapshot[] = [];
  const runtime = new Runtime(root, {
    save: (snapshot) => snapshots.push(structuredClone(snapshot)),
  });
  try {
    await runtime.dispatch(IpcMethod.PING, {}, () => {});
    await runtime.dispatch(IpcMethod.INFO, {}, () => {});
    expect(snapshots).toHaveLength(1);
    expect(snapshots[0].settings?.models.providers.deepseek.models[0].id).toBe("deepseek-flash");
    expect(snapshots[0].settings?.roles.default).toBe("deepseek/deepseek-flash:high");
    const catalog = await runtime.dispatch(IpcMethod.CATALOG, {}, () => {});
    for (const role of [AgentRole.COMPACTION]) {
      expect(snapshots[0].settings?.roles[role]).toBe("@default");
      expect(catalog.roles[role]).toEqual(catalog.roles.default);
    }
    expect(catalog.roles.default.thinkingLevel).toBe(Effort.High);
    expect((await stat(path.join(root, "roles.yml"))).mode & 0o777).toBe(0o600);
  } finally {
    await runtime.close();
    await rm(root, { recursive: true, force: true });
  }
});

async function persistenceFixture(credentials: AuthStorageData = {}) {
  const root = await mkdtemp(path.join(tmpdir(), "runtime-persistence-"));
  const snapshots: PersistenceSnapshot[] = [];
  const events: unknown[] = [];
  const bootstrap = validate(
    runtimeBootstrapSchema,
    {
      models: {
        providers: {
          local: {
            api: "openai-completions",
            auth: "none",
            baseUrl: "http://127.0.0.1:1/v1",
            models: [{ id: "proof", input: ["text", "image"], reasoning: true }],
          },
        },
      },
      roles: fixedRoles("local/proof:off"),
      credentials,
    },
    "runtime bootstrap",
  );
  const runtime = new Runtime(
    root,
    {
      models: bootstrap.models,
      roles: bootstrap.roles,
      credentials: bootstrap.credentials,
      save: (snapshot) => snapshots.push(structuredClone(snapshot)),
    },
    undefined,
    registeredRoles,
  );
  const emit = (name: RuntimeEvent, data: Record<string, unknown>) => events.push({ name, data });
  return {
    root,
    runtime,
    snapshots,
    events,
    emit,
    async close() {
      await runtime.close();
      await rm(root, { recursive: true, force: true });
    },
  };
}

test("last provider deletion persists an unconfigured state", async () => {
  const fixture = await persistenceFixture();
  let restored: Runtime | undefined;
  try {
    const before = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    const removed = await fixture.runtime.dispatch(
      IpcMethod.PROVIDER_REMOVE,
      { provider: "local" },
      fixture.emit,
    );
    expect(removed.providers).toEqual([]);
    expect(removed.roles).toEqual({});
    expect(removed.selectors).toEqual(before.selectors);
    // biome-ignore lint/style/noNonNullAssertion: Provider deletion persists settings before dispatch returns.
    const settings = fixture.snapshots.at(-1)!.settings!;
    expect(settings.models.providers).toEqual({});
    await fixture.runtime.close();
    restored = new Runtime(
      fixture.root,
      { ...settings, save: () => {} },
      undefined,
      registeredRoles,
    );
    expect((await restored.dispatch(IpcMethod.CATALOG, {}, fixture.emit)).providers).toEqual([]);
    expect((await restored.dispatch(IpcMethod.INFO, {}, fixture.emit)).roles).toEqual({});
    await expect(
      restored.dispatch(
        IpcMethod.GENERATE,
        {
          role: AgentRole.DEFAULT,
          messages: [{ role: "user", content: "Read" }],
        },
        fixture.emit,
      ),
    ).rejects.toThrow("No providers configured");
    const added = await restored.dispatch(
      IpcMethod.CONFIGURE,
      {
        models: before.definitions.models,
        roles: before.definitions.roles,
      },
      fixture.emit,
    );
    expect(added.providers.map((provider) => provider.id)).toEqual(["local"]);
    expect(added.roles.default.model).toBe("proof");
  } finally {
    await restored?.close();
    await fixture.close();
  }
});

test("object configuration preserves raw definitions and fixed role suffixes", async () => {
  const fixture = await persistenceFixture();
  try {
    const initial = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    const models = initial.definitions.models;
    models.providers.local.headers = { "X-Test": "preserved" };
    models.providers.local.models[0].compat = {
      extraBody: { nested: { enabled: true }, optional: null },
    };
    const roles = {
      ...fixedRoles("local/proof"),
      note_writer: "@default:high",
      note_chunk: "@note_writer",
    };
    validate(
      ipcRequestSchema,
      {
        id: "configure-object",
        method: IpcMethod.CONFIGURE,
        params: { models, roles },
      },
      "object configuration",
    );
    await fixture.runtime.dispatch(IpcMethod.CONFIGURE, { models, roles }, fixture.emit);
    const catalog = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    expect(catalog.definitions).toEqual({ models, roles });
    expect(catalog.definition_schema).toHaveProperty("properties.providers");
    expect(catalog.roles.note_writer.thinkingLevel).toBe(Effort.High);
    expect(catalog.roles.note_chunk).toEqual(catalog.roles.note_writer);
    expect(catalog.selectors.default).toBe("local/proof");
    const rows = structuredClone(models.providers.local.models);
    rows[0].name = "Renamed";
    await fixture.runtime.dispatch(
      IpcMethod.MODELS_SET,
      {
        provider: "local",
        models: rows,
      },
      fixture.emit,
    );
    const edited = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    expect(edited.definitions.models.providers.local.models).toEqual(rows);
    expect(edited.definitions.models.providers.local.headers).toEqual({ "X-Test": "preserved" });
    await fixture.runtime.dispatch(
      IpcMethod.ROLE_SET,
      {
        role: "note_review",
        selector: "@note_writer:low",
      },
      fixture.emit,
    );
    const updated = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    expect(updated.definitions.roles.note_review).toBe("@note_writer:low");
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.CONFIGURE,
        {
          models,
          roles: { ...roles, note_writer: "@note_chunk:high", note_chunk: "@note_writer" },
        },
        fixture.emit,
      ),
    ).rejects.toThrow("Invalid role alias");
    await fixture.runtime.dispatch(
      IpcMethod.CONFIGURE,
      {
        models,
        roles: { ...roles, custom: "@default" },
      },
      fixture.emit,
    );
    const normalized = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    expect(normalized).not.toHaveProperty("selectors.custom");
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.CONFIGURE,
        {
          models,
          roles: { ...roles, note_writer: "@missing" },
        },
        fixture.emit,
      ),
    ).rejects.toThrow("Invalid role alias");
    await fixture.runtime.dispatch(
      IpcMethod.CONFIGURE,
      { models, roles: { default: "local/proof" } },
      fixture.emit,
    );
    const filled = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    expect(filled).toHaveProperty("selectors.note_writer", "@default");
  } finally {
    await fixture.close();
  }
});

test("bootstrap preserves accounts without emitting secrets", async () => {
  const credentials: AuthStorageData = {
    local: { type: "api_key", key: "bootstrap-key" },
    accounts: [
      {
        type: "oauth",
        access: "first-access",
        refresh: "first-refresh",
        expires: 4_000_000_000_000,
        accountId: "first",
      },
      {
        type: "oauth",
        access: "second-access",
        refresh: "second-refresh",
        expires: 4_000_000_000_000,
        accountId: "second",
      },
    ],
  };
  const fixture = await persistenceFixture(credentials);
  try {
    const catalog = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, fixture.emit);
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: The catalog request initializes authentication.
    const auth = fixture.runtime["auth"]!;
    expect(auth.credentials.all()).toEqual(credentials);
    expect(catalog).toHaveProperty("selectors.default", "local/proof:off");
    expect(fixture.snapshots).toEqual([]);
    expect(fixture.events).toEqual([]);
    expect(JSON.stringify(catalog)).not.toContain("bootstrap-key");
    expect(JSON.stringify(catalog)).not.toContain("first-refresh");
  } finally {
    await fixture.close();
  }
});

test("bootstrap rejects null configuration", () => {
  expect(() =>
    validate(runtimeBootstrapSchema, { models: null, credentials: {} }, "runtime bootstrap"),
  ).toThrow();
  expect(() =>
    validate(runtimeBootstrapSchema, { roles: null, credentials: {} }, "runtime bootstrap"),
  ).toThrow();
});

test("only valid configuration edits persist", async () => {
  const fixture = await persistenceFixture();
  try {
    await fixture.runtime.dispatch(IpcMethod.PING, {}, fixture.emit);
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.ROLE_SET,
        {
          role: "note_writer",
          selector: "local/missing:high",
        },
        fixture.emit,
      ),
    ).rejects.toThrow();
    expect(fixture.snapshots).toEqual([]);
    await fixture.runtime.dispatch(
      IpcMethod.ROLE_SET,
      {
        role: "note_writer",
        selector: "local/proof:high",
      },
      fixture.emit,
    );
    expect(fixture.snapshots).toHaveLength(1);
    expect(fixture.snapshots[0].settings?.roles.note_writer).toBe("local/proof:high");
    await fixture.runtime.dispatch(
      IpcMethod.MODELS_SET,
      {
        provider: "local",
        models: [{ id: "proof", input: ["text", "image"], reasoning: true, contextWindow: 64000 }],
      },
      fixture.emit,
    );
    expect(fixture.snapshots).toHaveLength(2);
    expect(fixture.snapshots[1].settings?.models.providers.local.models[0].contextWindow).toBe(
      64000,
    );
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.CONFIGURE,
        {
          models: "providers: {local: {models: []}}",
          roles: "default: local/proof:off",
        },
        fixture.emit,
      ),
    ).rejects.toThrow();
    expect(fixture.snapshots).toHaveLength(2);
  } finally {
    await fixture.close();
  }
});

test("key changes and logout persist credential snapshots", async () => {
  const fixture = await persistenceFixture({ retained: { type: "api_key", key: "retained-key" } });
  try {
    await fixture.runtime.dispatch(
      IpcMethod.CREDENTIALS,
      { provider: "local", key: "saved-key" },
      fixture.emit,
    );
    expect(fixture.snapshots.at(-1)?.credentials).toEqual({
      retained: { type: "api_key", key: "retained-key" },
      local: { type: "api_key", key: "saved-key" },
    });
    await fixture.runtime.dispatch(
      IpcMethod.IMPORT_CREDENTIALS,
      {
        provider: "local",
        credential: {
          type: "oauth",
          access: "import-access",
          refresh: "import-refresh",
          expires: 4_000_000_000_000,
          accountId: "imported",
        },
      },
      fixture.emit,
    );
    expect(fixture.snapshots.at(-1)?.credentials?.local).toEqual({
      type: "oauth",
      access: "import-access",
      refresh: "import-refresh",
      expires: 4_000_000_000_000,
      accountId: "imported",
    });
    await fixture.runtime.dispatch(IpcMethod.LOGOUT, { provider: "local" }, fixture.emit);
    expect(fixture.snapshots.at(-1)?.credentials).toEqual({
      retained: { type: "api_key", key: "retained-key" },
    });
    await fixture.runtime.dispatch(IpcMethod.LOGOUT, { provider: "retained" }, fixture.emit);
    expect(fixture.snapshots.at(-1)?.credentials).toEqual({});
    expect(fixture.events).toEqual([]);
  } finally {
    await fixture.close();
  }
});

test("OAuth refresh persists rotated tokens and sibling accounts", async () => {
  const provider = `runtime-refresh-${crypto.randomUUID()}`;
  registerOAuthProvider({
    id: provider,
    name: "Runtime refresh test",
    async login() {
      throw new Error("Login is not used");
    },
    getApiKey: (credential) => credential.access,
    async refreshToken(credential) {
      return {
        ...credential,
        access: "rotated-access",
        refresh: "rotated-refresh",
        expires: Date.now() + 3600000,
      };
    },
  });
  const fixture = await persistenceFixture({
    [provider]: [
      {
        type: "oauth",
        access: "expired-access",
        refresh: "old-refresh",
        expires: 1,
        accountId: "first",
      },
      {
        type: "oauth",
        access: "sibling-access",
        refresh: "sibling-refresh",
        expires: 4_000_000_000_000,
        accountId: "second",
      },
    ],
  });
  try {
    await fixture.runtime.dispatch(IpcMethod.PING, {}, fixture.emit);
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: The ping request initializes authentication.
    const auth = fixture.runtime["auth"]!;
    const row = auth.credentials
      .list(provider)
      .find((entry) => entry.credential.type === "oauth" && entry.credential.accountId === "first");
    if (!row) throw new Error("Missing first OAuth account");
    await auth.oauth.refresh(row.id);
    expect(fixture.snapshots.at(-1)?.credentials?.[provider]).toEqual([
      expect.objectContaining({
        access: "rotated-access",
        refresh: "rotated-refresh",
        accountId: "first",
      }),
      expect.objectContaining({
        access: "sibling-access",
        refresh: "sibling-refresh",
        accountId: "second",
      }),
    ]);
    expect(fixture.events).toEqual([]);
  } finally {
    unregisterOAuthProvider(provider);
    await fixture.close();
  }
});

test("OAuth refresh persists after runtime close", async () => {
  const provider = `runtime-close-refresh-${crypto.randomUUID()}`;
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  registerOAuthProvider({
    id: provider,
    name: "Runtime close refresh test",
    async login() {
      throw new Error("Login is not used");
    },
    getApiKey: (credential) => credential.access,
    async refreshToken(credential) {
      entered.resolve();
      await release.promise;
      return {
        ...credential,
        access: "closing-access",
        refresh: "closing-refresh",
        expires: Date.now() + 3600000,
      };
    },
  });
  const fixture = await persistenceFixture({
    [provider]: {
      type: "oauth",
      access: "expired-access",
      refresh: "old-refresh",
      expires: 1,
      accountId: "closing-account",
    },
  });
  let access: Promise<unknown> | undefined;
  try {
    await fixture.runtime.dispatch(IpcMethod.PING, {}, fixture.emit);
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: The ping request initializes authentication.
    const auth = fixture.runtime["auth"]!;
    access = auth.oauth.refresh(auth.credentials.list(provider)[0].id);
    await entered.promise;
    await fixture.runtime.close();
    release.resolve();
    await access;
    expect(fixture.snapshots.at(-1)?.credentials?.[provider]).toEqual(
      expect.objectContaining({
        access: "closing-access",
        refresh: "closing-refresh",
        accountId: "closing-account",
      }),
    );
    expect(fixture.events).toEqual([]);
  } finally {
    release.resolve();
    await access;
    unregisterOAuthProvider(provider);
    await fixture.close();
  }
});

test("credential persistence errors fail requests", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "runtime-persistence-error-"));
  const runtime = new Runtime(root, {
    save(snapshot) {
      if (snapshot.credentials) throw new Error("Storage is unavailable");
    },
  });
  try {
    await expect(
      runtime.dispatch(IpcMethod.CREDENTIALS, { provider: "local", key: "secret" }, () => {}),
    ).rejects.toThrow("Credential persistence failed");
    await expect(runtime.dispatch(IpcMethod.PING, {}, () => {})).rejects.toThrow(
      "Credential persistence failed",
    );
  } finally {
    await runtime.close();
    await rm(root, { recursive: true, force: true });
  }
});

test("generation honors disabled thinking", async () => {
  let body: Record<string, unknown> | undefined;
  const server = Bun.serve({
    port: 0,
    async fetch(request) {
      body = (await request.json()) as Record<string, unknown>;
      return new Response(
        'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"Board"},"finish_reason":null}]}\n\ndata: {"id":"proof","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        { headers: { "Content-Type": "text/event-stream" } },
      );
    },
  });
  const root = await mkdtemp(path.join(tmpdir(), "runtime-thinking-"));
  const runtime = new Runtime(root);
  try {
    await runtime.dispatch(
      IpcMethod.CONFIGURE,
      {
        models: JSON.stringify({
          providers: {
            deepseek: {
              api: "openai-completions",
              auth: "none",
              baseUrl: `http://127.0.0.1:${server.port}/v1`,
              models: [{ id: "deepseek-flash", reasoning: true, input: ["text", "image"] }],
            },
          },
        }),
        roles: Bun.YAML.stringify(fixedRoles("deepseek/deepseek-flash:off")),
      },
      () => {},
    );
    const result = await runtime.dispatch(
      IpcMethod.GENERATE,
      {
        messages: [{ role: "user", content: "Read board" }],
      },
      () => {},
    );
    expect(result).toHaveProperty("content", "Board");
    expect(body).toHaveProperty("thinking.type", "disabled");
  } finally {
    await runtime.close();
    server.stop(true);
    await rm(root, { recursive: true, force: true });
  }
});

test("cancel after generation cleanup is idempotent", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "runtime-cancel-"));
  const runtime = new Runtime(root);
  try {
    await expect(
      runtime.dispatch(IpcMethod.CANCEL, { session_id: "completed" }, () => {}),
    ).resolves.toEqual({ ok: true });
  } finally {
    await runtime.close();
    await rm(root, { recursive: true, force: true });
  }
});

interface ModelRequest {
  messages: { role: string; content?: unknown }[];
}
function userTexts(request: ModelRequest) {
  return request.messages
    .filter((message) => message.role === "user")
    .map((message) => {
      if (typeof message.content === "string") return message.content;
      return (message.content as { type: string; text?: string }[])
        .filter((block) => block.type === "text")
        .map((block) => block.text ?? "")
        .join("");
    });
}
function taskWithoutReminder(text: string) {
  return text.replace(/^<system-reminder>\nToday: [^\n]+\n<\/system-reminder>/, "");
}
function nativeSteeringText(text: string) {
  const [message] = wrapSteeringForModel([
    { role: "user", content: text, steering: true, attribution: "user", timestamp: 0 },
  ]);
  if (message?.role !== "user" || typeof message.content !== "string")
    throw new Error("Native steering did not return user text");
  return message.content;
}
function modelText(content: string) {
  return modelDelta({ content }, "stop");
}
function modelDelta(delta: Record<string, unknown>, reason: string) {
  return `data: ${JSON.stringify({ id: "proof", choices: [{ index: 0, delta: { role: "assistant", ...delta }, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ id: "proof", choices: [{ index: 0, delta: {}, finish_reason: reason }] })}\n\ndata: [DONE]\n\n`;
}
function modelTool(name: string, args: Record<string, unknown>, id: string) {
  return modelDelta(
    {
      tool_calls: [
        { index: 0, id, type: "function", function: { name, arguments: JSON.stringify(args) } },
      ],
    },
    "tool_calls",
  );
}
function sse(text: string) {
  return new Response(text, { headers: { "Content-Type": "text/event-stream" } });
}
async function localRuntime(
  respond: (body: ModelRequest) => Response | Promise<Response>,
  callback?: RuntimeCallback,
) {
  const server = Bun.serve({
    port: 0,
    async fetch(request) {
      return respond((await request.json()) as ModelRequest);
    },
  });
  const root = await mkdtemp(path.join(tmpdir(), "runtime-live-"));
  const runtime = new Runtime(root, undefined, callback, registeredRoles);
  await runtime.dispatch(
    IpcMethod.CONFIGURE,
    {
      models: JSON.stringify({
        providers: {
          local: {
            api: "openai-completions",
            auth: "none",
            baseUrl: `http://127.0.0.1:${server.port}/v1`,
            models: [
              {
                id: "proof",
                reasoning: false,
                compat: { supportsDeveloperRole: false },
                input: ["text", "image"],
                contextWindow: 128000,
                maxTokens: 4096,
              },
            ],
          },
        },
      }),
      roles: Bun.YAML.stringify(fixedRoles("local/proof:off")),
    },
    () => {},
  );
  return {
    runtime,
    root,
    async close() {
      await runtime.close();
      server.stop(true);
      await rm(root, { recursive: true, force: true });
    },
  };
}

test("configured instructions remain ordered user messages", async () => {
  const requests: ModelRequest[] = [];
  const fixture = await localRuntime((body) => {
    requests.push(body);
    return sse(modelText("Done without review"));
  });
  try {
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "role-prompt",
        prompt: "CONFIGURABLE_ROLE_INSTRUCTIONS",
      },
      () => {},
    );
    const result = await fixture.runtime.dispatch(
      IpcMethod.RUN,
      {
        session_id: "role-prompt",
        instruction: "/literal task",
        follow_ups: ["first follow-up", "second follow-up"],
      },
      () => {},
    );
    expect(requests).toHaveLength(2);
    expect(userTexts(requests[0]).map(taskWithoutReminder)).toEqual([
      "CONFIGURABLE_ROLE_INSTRUCTIONS\n\n/literal task",
    ]);
    expect(userTexts(requests[1]).map(taskWithoutReminder)).toEqual([
      "CONFIGURABLE_ROLE_INSTRUCTIONS\n\n/literal task",
      "first follow-up",
      "second follow-up",
    ]);
    expect(
      requests[0].messages
        .filter((message) => message.role === "system")
        .some((message) =>
          JSON.stringify(message.content).includes("CONFIGURABLE_ROLE_INSTRUCTIONS"),
        ),
    ).toBe(false);
    expect(result).toHaveProperty("final_message", "Done without review");
    expect(result).toHaveProperty("review_passed", true);
  } finally {
    await fixture.close();
  }
});

const reviewScope = {
  role: "note_review",
  targets: ["note_knowledge.md", "note_gossip.md", "note_notice.md"],
  sources: ["lecture_source.md"],
};

test("writer cannot complete before a passing review", async () => {
  let writerCalls = 0;
  let reviewerCalls = 0;
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("REQUIRED_REVIEW")) {
      ++reviewerCalls;
      return sse(
        modelText(
          JSON.stringify({
            correct: true,
            explanation: "Checked",
            confidence: 1,
            findings: [],
          }),
        ),
      );
    }
    ++writerCalls;
    if (writerCalls === 1) return sse(modelText("Premature completion"));
    if (writerCalls === 2) {
      expect(userTexts(body).at(-1)).toContain("do not have a passing review");
      return sse(modelTool("review_notes", {}, "required-review"));
    }
    return sse(modelText("Reviewed completion"));
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "required",
        files,
        meta: {
          review: { system: "REQUIRED_REVIEW", prompt: "Review", maxRounds: 1, ...reviewScope },
        },
      },
      () => {},
    );
    const result = await fixture.runtime.dispatch(
      IpcMethod.RUN,
      {
        session_id: "required",
        instruction: "Write notes",
      },
      () => {},
    );
    expect(result).toHaveProperty("final_message", "Reviewed completion");
    expect(result).toHaveProperty("review_passed", true);
    expect(reviewerCalls).toBe(1);
  } finally {
    await fixture.close();
  }
});

test.each([false, undefined])("writer cannot finish without approval (%s)", async (correct) => {
  const events: Record<string, unknown>[] = [];
  let writerCalls = 0;
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("REJECT_REVIEW")) {
      if (correct === undefined)
        return Response.json({ error: { message: "Reviewer unavailable" } }, { status: 400 });
      return sse(
        modelText(
          JSON.stringify({ correct, explanation: "Fix notes", confidence: 1, findings: [] }),
        ),
      );
    }
    return ++writerCalls === 1
      ? sse(modelTool("review_notes", {}, "review-rejected"))
      : sse(modelText("Premature completion"));
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "rejected",
        files,
        meta: {
          review: { system: "REJECT_REVIEW", prompt: "Review", maxRounds: 1, ...reviewScope },
        },
      },
      () => {},
    );
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.RUN,
        { session_id: "rejected", instruction: "Write notes" },
        (name, data) => {
          if (name === RuntimeEvent.REVIEWER && typeof data.finished === "boolean")
            events.push(data);
        },
      ),
    ).rejects.toThrow("Review round limit reached without approval");
    expect(events.map((event) => event.finished)).toEqual([false, true]);
    if (correct === undefined) expect(events[1]).not.toHaveProperty("content");
    else expect(JSON.parse(String(events[1]?.content)).correct).toBe(false);
  } finally {
    await fixture.close();
  }
});

test("an accepted review survives a harmless user message", async () => {
  const writerStarted = Promise.withResolvers<void>();
  let reviews = 0;
  let writers = 0;
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("ACCEPTED_REVIEW")) {
      ++reviews;
      return sse(
        modelText(
          JSON.stringify({
            correct: true,
            explanation: "Checked",
            confidence: 1,
            findings: [],
          }),
        ),
      );
    }
    ++writers;
    if (writers === 1) return sse(modelTool("review_notes", {}, "accepted-review"));
    if (writers === 2)
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(
              new TextEncoder().encode(
                'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"Finishing"},"finish_reason":null}]}\n\n',
              ),
            );
            writerStarted.resolve();
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      );
    expect(userTexts(body).at(-1)).toBe(nativeSteeringText("Keep the accepted notes unchanged"));
    return sse(modelText("Finished"));
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "accepted",
        files,
        meta: {
          review: { system: "ACCEPTED_REVIEW", prompt: "Review", maxRounds: 1, ...reviewScope },
        },
      },
      () => {},
    );
    const run = fixture.runtime.dispatch(
      IpcMethod.RUN,
      {
        session_id: "accepted",
        instruction: "Review notes",
      },
      () => {},
    );
    await writerStarted.promise;
    expect(reviews).toBe(1);
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        {
          session_id: "accepted",
          message: "Keep the accepted notes unchanged",
        },
        () => {},
      ),
    ).toEqual({ accepted: true });
    expect(await run).toMatchObject({ final_message: "Finished", review_passed: true });
    expect(reviews).toBe(1);
    expect(writers).toBe(3);
  } finally {
    await fixture.close();
  }
});

test("message admitted during the final review check is delivered", async () => {
  let writers = 0;
  let reviews = 0;
  const delivered: string[] = [];
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("LATE_REVIEW")) {
      ++reviews;
      return sse(
        modelText(
          JSON.stringify({
            correct: true,
            explanation: "Checked",
            confidence: 1,
            findings: [],
          }),
        ),
      );
    }
    ++writers;
    const last = taskWithoutReminder(userTexts(body).at(-1) ?? "");
    if (last === "Late message") {
      delivered.push(last);
      return sse(modelText("Delivered"));
    }
    if (writers === 1) return sse(modelTool("review_notes", {}, "late-review"));
    return sse(modelText("Reviewed"));
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "late",
        files,
        meta: { review: { system: "LATE_REVIEW", prompt: "Review", maxRounds: 1, ...reviewScope } },
      },
      () => {},
    );
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: Session creation above installs the configured review gate.
    const gate = fixture.runtime["sessions"].get("late")!.review!;
    const satisfied = gate.satisfied.bind(gate);
    let injected = false;
    gate.satisfied = async () => {
      const value = await satisfied();
      if (!injected) {
        injected = true;
        expect(
          await fixture.runtime.dispatch(
            IpcMethod.MESSAGE,
            { session_id: "late", message: "Late message" },
            () => {},
          ),
        ).toEqual({ accepted: true });
      }
      return value;
    };
    const result = await fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "late", instruction: "Review notes" },
      () => {},
    );
    expect(injected).toBe(true);
    expect(delivered).toEqual(["Late message"]);
    expect(result).toMatchObject({ final_message: "Delivered", review_passed: true });
    expect(reviews).toBe(1);
    expect(writers).toBeGreaterThan(1);
  } finally {
    await fixture.close();
  }
});

for (const maxAttempts of [1, 2, 3]) {
  for (const exhausted of [false, true]) {
    test(`board loop ${maxAttempts} attempts ${exhausted ? "exhaustion" : "recovery"} keeps failed text private`, async () => {
      let requests = 0;
      const loop = String.raw`\lambda_1 = \frac{1-\lambda\beta}{1-\lambda}\quad \rho=\lambda_1-\gamma^2`;
      const env = await localRuntime(() => {
        ++requests;
        return sse(modelText(exhausted || requests === 1 ? loop.repeat(40) : "Complete board"));
      });
      const events: { name: RuntimeEvent; data: Record<string, unknown> }[] = [];
      try {
        const result = await env.runtime.dispatch(
          IpcMethod.GENERATE,
          {
            messages: [
              {
                role: "user",
                content: [
                  { type: "text", text: "Read board" },
                  { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
                ],
              },
            ],
            loop_max_attempts: maxAttempts,
          },
          (name, data) => events.push({ name, data }),
        );
        expect(requests).toBe(exhausted ? maxAttempts : Math.min(2, maxAttempts));
        const failed = exhausted || maxAttempts === 1;
        expect(result).toHaveProperty("content", failed ? "" : "Complete board");
        if (failed) expect(result).toHaveProperty("error.code", "thinking_loop");
        expect(
          events
            .filter((event) => event.name === RuntimeEvent.LLM_DELTA)
            .map((event) => event.data.text),
        ).toEqual(failed ? [] : ["Complete board"]);
      } finally {
        await env.close();
      }
    });
  }
}

test("cancel during loop backoff prevents another attempt", async () => {
  let requests = 0;
  const env = await localRuntime(() => {
    ++requests;
    return sse(modelText("abcdefghijklmno1234567890".repeat(100)));
  });
  try {
    const result = env.runtime.dispatch(
      IpcMethod.GENERATE,
      {
        session_id: "cancel-loop",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Read board" },
              { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
            ],
          },
        ],
      },
      (name) => {
        if (name === RuntimeEvent.LLM_ATTEMPT)
          void env.runtime.dispatch(IpcMethod.CANCEL, { session_id: "cancel-loop" }, () => {});
      },
    );
    await expect(result).rejects.toThrow();
    expect(requests).toBe(1);
  } finally {
    await env.close();
  }
});

test("ordinary provider errors do not become loop retries", async () => {
  let requests = 0;
  const events: { name: RuntimeEvent; data: Record<string, unknown> }[] = [];
  const env = await localRuntime(() => {
    ++requests;
    return new Response(
      JSON.stringify({
        error: { message: "Invalid board request", type: "invalid_request_error" },
      }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      },
    );
  });
  try {
    await expect(
      env.runtime.dispatch(
        IpcMethod.GENERATE,
        {
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "Read board" },
                { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
              ],
            },
          ],
        },
        (name, data) => events.push({ name, data }),
      ),
    ).rejects.toThrow("Invalid board request");
    expect(requests).toBe(1);
    expect(events.filter((event) => event.name === RuntimeEvent.LLM_DELTA)).toEqual([]);
    expect(
      events
        .filter((event) => event.name === RuntimeEvent.LLM_ATTEMPT)
        .map((event) => event.data.outcome),
    ).toEqual(["error"]);
  } finally {
    await env.close();
  }
});

test("non-board generation preserves streamed deltas", async () => {
  const encoder = new TextEncoder();
  const first = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const env = await localRuntime(
    () =>
      new Response(
        new ReadableStream({
          async start(controller) {
            controller.enqueue(
              encoder.encode(
                'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"First "},"finish_reason":null}]}\n\n',
              ),
            );
            await release.promise;
            controller.enqueue(encoder.encode(modelText("second")));
            controller.close();
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      ),
  );
  const deltas: unknown[] = [];
  const attempts: unknown[] = [];
  try {
    const result = env.runtime.dispatch(
      IpcMethod.GENERATE,
      {
        role: "note_writer",
        messages: [{ role: "user", content: "Write notes" }],
      },
      (name, data) => {
        if (name === RuntimeEvent.LLM_DELTA) {
          deltas.push(data.text);
          first.resolve();
        }
        if (name === RuntimeEvent.LLM_ATTEMPT) attempts.push(data);
      },
    );
    await first.promise;
    expect(deltas).toEqual(["First "]);
    release.resolve();
    expect(await result).toHaveProperty("content", "First second");
    expect(deltas).toEqual(["First ", "second"]);
    expect(attempts).toEqual([]);
  } finally {
    release.resolve();
    await env.close();
  }
});

test("cancel interrupts an active board stream", async () => {
  let requests = 0;
  const started = Promise.withResolvers<void>();
  const originalFetch = globalThis.fetch;
  const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(
    Object.assign(
      async (...args: Parameters<typeof fetch>) => {
        const response = await originalFetch(...args);
        started.resolve();
        return response;
      },
      { preconnect: originalFetch.preconnect },
    ),
  );
  let streamController: ReadableStreamDefaultController<Uint8Array> | undefined;
  const env = await localRuntime(() => {
    ++requests;
    return new Response(
      new ReadableStream({
        start(controller) {
          streamController = controller;
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"Partial board"},"finish_reason":null}]}\n\n',
            ),
          );
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });
  const deltas: unknown[] = [];
  try {
    const result = env.runtime.dispatch(
      IpcMethod.GENERATE,
      {
        session_id: "active-board",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Read board" },
              { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
            ],
          },
        ],
      },
      (name, data) => {
        if (name === RuntimeEvent.LLM_DELTA) deltas.push(data.text);
      },
    );
    const rejected = expect(result).rejects.toThrow();
    await started.promise;
    await env.runtime.dispatch(IpcMethod.CANCEL, { session_id: "active-board" }, () => {});
    await rejected;
    expect(requests).toBe(1);
    expect(deltas).toEqual([]);
  } finally {
    fetchSpy.mockRestore();
    try {
      streamController?.close();
    } catch {
      /* The request can already be cancelled. */
    }
    await env.close();
  }
}, 15000);

test("messages interrupt a stream within one run", async () => {
  const started = Promise.withResolvers<void>();
  const disconnected = Promise.withResolvers<void>();
  const requests: ModelRequest[] = [];
  const fixture = await localRuntime((body) => {
    requests.push(body);
    if (requests.length > 1) return sse(modelText("Finished"));
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Thinking"},"finish_reason":null}]}\n\n',
            ),
          );
          started.resolve();
        },
        cancel() {
          disconnected.resolve();
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });
  try {
    const { runtime } = fixture;
    await runtime.dispatch(IpcMethod.CREATE_SESSION, { id: "live" }, () => {});
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: Session creation above installs the live conversation.
    const conversation = runtime["sessions"].get("live")!;
    const join = spyOn(conversation.session, "joinActiveTurn");
    const abort = spyOn(conversation.session, "abort");
    expect(
      await runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: "live", message: "before run" },
        () => {},
      ),
    ).toHaveProperty("accepted", false);
    const events: string[] = [];
    const run = runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "live", instruction: "Start" },
      (name) => events.push(name),
    );
    await started.promise;
    await expect(
      runtime.dispatch(IpcMethod.RUN, { session_id: "live", instruction: "Duplicate" }, () => {}),
    ).rejects.toThrow("active run");
    const first = runtime.dispatch(
      IpcMethod.MESSAGE,
      { session_id: "live", message: "/literal\nfirst" },
      () => {},
    );
    const second = runtime.dispatch(
      IpcMethod.MESSAGE,
      { session_id: "live", message: "second" },
      () => {},
    );
    expect(await first).toEqual({ accepted: true });
    expect(await second).toEqual({ accepted: true });
    expect(await run).toHaveProperty("final_message", "Finished");
    await disconnected.promise;
    // biome-ignore lint/style/noNonNullAssertion: The completed run and delivered messages require a model request.
    expect(userTexts(requests.at(-1)!).map(taskWithoutReminder)).toEqual([
      "Start",
      nativeSteeringText("/literal\nfirst"),
      nativeSteeringText("second"),
    ]);
    expect(join.mock.calls).toEqual([
      ["/literal\nfirst", undefined, { queue: "steering" }],
      ["second", undefined, { queue: "steering" }],
    ]);
    expect(await Promise.all(join.mock.results.map((result) => result.value))).toEqual([
      true,
      true,
    ]);
    expect(abort).toHaveBeenCalledTimes(1);
    expect(
      conversation.session.messages.filter((message) => message.role === "user"),
    ).toMatchObject([
      { role: "user", content: [{ type: "text", text: "Start" }], attribution: "user" },
      {
        role: "user",
        content: [{ type: "text", text: "/literal\nfirst" }],
        attribution: "user",
        steering: true,
      },
      {
        role: "user",
        content: [{ type: "text", text: "second" }],
        attribution: "user",
        steering: true,
      },
    ]);
    expect(conversation.run).toBeUndefined();
    expect(conversation.session.agent.hasQueuedMessages()).toBe(false);
    expect(events.filter((name) => name === RuntimeEvent.RUN_READY)).toHaveLength(1);
    expect(events.filter((name) => name === RuntimeEvent.MESSAGE)).toHaveLength(1);
    expect(
      await runtime.dispatch(IpcMethod.MESSAGE, { session_id: "live", message: "late" }, () => {}),
    ).toHaveProperty("accepted", false);
  } finally {
    await fixture.close();
  }
});

test("cancel rejects messages without continuation", async () => {
  const started = Promise.withResolvers<void>();
  let calls = 0;
  const fixture = await localRuntime(() => {
    ++calls;
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"Working"},"finish_reason":null}]}\n\n',
            ),
          );
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });
  try {
    await fixture.runtime.dispatch(IpcMethod.CREATE_SESSION, { id: "cancel" }, () => {});
    const run = fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "cancel", instruction: "Start" },
      (name) => {
        if (name === RuntimeEvent.DELTA) started.resolve();
      },
    );
    const outcome = run.then(
      () => undefined,
      (error: unknown) => error,
    );
    await started.promise;
    const cancel = fixture.runtime.dispatch(IpcMethod.CANCEL, { session_id: "cancel" }, () => {});
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: "cancel", message: "Continue" },
        () => {},
      ),
    ).toHaveProperty("accepted", false);
    await cancel;
    expect(String(await outcome)).toContain("cancelled");
    expect(calls).toBe(1);
    await fixture.runtime.dispatch(IpcMethod.CLOSE_SESSION, { session_id: "cancel" }, () => {});
    await fixture.runtime.dispatch(IpcMethod.CLOSE_SESSION, { session_id: "cancel" }, () => {});
  } finally {
    await fixture.close();
  }
});

test("reopen requires readable history and preserves its messages", async () => {
  const requests: ModelRequest[] = [];
  const fixture = await localRuntime((body) => {
    requests.push(body);
    return sse(modelText("Saved answer"));
  });
  try {
    const { runtime, root } = fixture;
    await expect(
      runtime.dispatch(IpcMethod.REOPEN_SESSION, { id: "missing" }, () => {}),
    ).rejects.toThrow("unavailable");
    await Bun.write(path.join(root, "session-broken.path"), path.join(root, "broken.jsonl"));
    await Bun.write(path.join(root, "broken.jsonl"), "not json");
    await expect(
      runtime.dispatch(IpcMethod.REOPEN_SESSION, { id: "broken" }, () => {}),
    ).rejects.toThrow("invalid");
    await runtime.dispatch(IpcMethod.CREATE_SESSION, { id: "history" }, () => {});
    await runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "history", instruction: "Original task" },
      () => {},
    );
    await expect(
      runtime.dispatch(IpcMethod.REOPEN_SESSION, { id: "history" }, () => {}),
    ).rejects.toThrow("already open");
    await runtime.dispatch(IpcMethod.CLOSE_SESSION, { session_id: "history" }, () => {});
    await runtime.dispatch(IpcMethod.REOPEN_SESSION, { id: "history" }, () => {});
    await runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "history", instruction: "Continue task" },
      () => {},
    );
    expect(JSON.stringify(requests.at(-1)?.messages)).toContain("Original task");
    expect(JSON.stringify(requests.at(-1)?.messages)).toContain("Saved answer");
    await runtime.dispatch(IpcMethod.CLOSE_SESSION, { session_id: "history" }, () => {});
    await runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      { id: "history", prompt: "Fresh instructions" },
      () => {},
    );
    await runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "history", instruction: "Fresh task" },
      () => {},
    );
    const fresh = JSON.stringify(requests.at(-1)?.messages);
    expect(fresh).not.toContain("Original task");
    expect(fresh).not.toContain("Saved answer");
    expect(fresh).toContain("Fresh instructions");
    expect(fresh).toContain("Fresh task");
  } finally {
    await fixture.close();
  }
});

test("writer messages preserve acceptance for unchanged files", async () => {
  const reviewerStarted = Promise.withResolvers<void>();
  const correctionStarted = Promise.withResolvers<void>();
  const reviewerRelease = Promise.withResolvers<void>();
  let writerCalls = 0;
  let reviewCalls = 0;
  const requests: ModelRequest[] = [];
  const fixture = await localRuntime(async (body) => {
    requests.push(body);
    if (JSON.stringify(body.messages).includes("REVIEW_PROOF")) {
      if (++reviewCalls === 1) {
        reviewerStarted.resolve();
        await reviewerRelease.promise;
      }
      return sse(
        modelText(
          JSON.stringify({
            correct: true,
            explanation: "Checked",
            confidence: 1,
            findings: [],
          }),
        ),
      );
    }
    ++writerCalls;
    if (writerCalls === 1) return sse(modelTool("review_notes", {}, "review-1"));
    if (writerCalls === 2) correctionStarted.resolve();
    return sse(modelText(writerCalls === 2 ? "Correction applied" : "Final answer"));
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    const events: { name: RuntimeEvent; data: Record<string, unknown> }[] = [];
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "writer",
        files,
        meta: {
          review: {
            system: "REVIEW_PROOF",
            prompt: "Review independently",
            maxRounds: 3,
            ...reviewScope,
          },
        },
      },
      () => {},
    );
    const run = fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "writer", instruction: "Review notes" },
      (name, data) => events.push({ name, data }),
    );
    await reviewerStarted.promise;
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: "writer", message: "Keep the current notes" },
        () => {},
      ),
    ).toEqual({ accepted: true });
    await correctionStarted.promise;
    expect(reviewCalls).toBe(1);
    reviewerRelease.resolve();
    const result = await run;
    expect(result).toHaveProperty("final_message", "Final answer");
    expect(result).toHaveProperty("review_passed", true);
    expect(reviewCalls).toBe(1);
    const finished = events.filter(
      ({ name, data }) => name === RuntimeEvent.REVIEWER && data.finished === true,
    );
    expect(finished.map(({ data }) => data.stale)).toEqual([false]);
    const sourceIds = finished[0].data.block_ids as string[];
    expect(sourceIds.length).toBeGreaterThan(0);
    for (const id of sourceIds) {
      expect(
        events.some(({ name, data }) => {
          const block = data.block as { id?: string; kind?: string } | undefined;
          return (
            name === RuntimeEvent.REVIEWER &&
            data.event === RuntimeEvent.EXECUTION &&
            block?.id === id &&
            block.kind === "text"
          );
        }),
      ).toBe(true);
    }
    const deliveryPrompts = requests.flatMap((request) =>
      request.messages.filter(
        (message) =>
          message.role !== "assistant" &&
          message.role !== "tool" &&
          JSON.stringify(message.content).includes("correct"),
      ),
    );
    const unique = new Set(deliveryPrompts.map((message) => JSON.stringify(message.content)));
    expect(unique.size).toBe(1);
    expect(deliveryPrompts.length).toBeGreaterThan(0);
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: The completed writer run requires the created writer session.
    const native = fixture.runtime["sessions"].get("writer")!.session.messages;
    const deliveries = native.filter(
      (message) =>
        message.role === "developer" && JSON.stringify(message.content).includes("correct"),
    );
    expect(deliveries).toHaveLength(1);
    expect(deliveries[0]).toMatchObject({
      role: "developer",
      synthetic: true,
      attribution: "agent",
    });
    expect(
      native
        .filter((message) => message.role === "user")
        .some((message) => JSON.stringify(message.content).includes("correct")),
    ).toBe(false);
    expect(deliveryPrompts.every((message) => message.role === "user")).toBe(true);
  } finally {
    reviewerRelease.resolve();
    await fixture.close();
  }
});

test("a new prompt waits for an interrupted write to finish", async () => {
  const writing = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  let calls = 0;
  let safe = false;
  let published = false;
  const fixture = await localRuntime(() => {
    ++calls;
    if (calls === 1)
      return sse(modelTool("write", { path: "note.md", content: "Changed" }, "write-proof"));
    expect(safe).toBe(true);
    expect(published).toBe(true);
    return sse(modelText("Saved safely"));
  });
  try {
    const source = path.join(fixture.root, "note.md");
    await Bun.write(source, "Original");
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      { id: "write", tools: [WorkspaceTool.WRITE], files: { "note.md": source } },
      () => {},
    );
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: Session creation above installs the write conversation.
    const conversation = fixture.runtime["sessions"].get("write")!;
    const tool = conversation.session.agent.state.tools.find(
      (candidate) => candidate.name === "write",
    );
    if (!tool) throw new Error("Missing write tool");
    const execute = tool.execute.bind(tool);
    tool.execute = async (...args) => {
      const result = await execute(...args);
      writing.resolve();
      await release.promise;
      safe = true;
      return result;
    };
    const run = fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "write", instruction: "Write the note" },
      (name) => {
        if (name === RuntimeEvent.FILE) published = true;
      },
    );
    await writing.promise;
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: "write", message: "Continue after saving" },
        () => {},
      ),
    ).toEqual({ accepted: true });
    expect(calls).toBe(1);
    release.resolve();
    expect(await run).toHaveProperty("final_message", "Saved safely");
    expect(calls).toBe(2);
  } finally {
    release.resolve();
    await fixture.close();
  }
});

test("completion accepts and runs a boundary message or rejects it", async () => {
  let calls = 0;
  const fixture = await localRuntime(() => sse(modelText(`Answer ${++calls}`)));
  try {
    await fixture.runtime.dispatch(IpcMethod.CREATE_SESSION, { id: "boundary" }, () => {});
    // biome-ignore lint/complexity/useLiteralKeys: Bracket access permits this white-box check of private runtime state.
    // biome-ignore lint/style/noNonNullAssertion: Session creation above installs the boundary conversation.
    const conversation = fixture.runtime["sessions"].get("boundary")!;
    const workspace = conversation.workspace;
    const join = spyOn(conversation.session, "joinActiveTurn");
    const changes = workspace.changes.bind(workspace);
    let injected = false;
    workspace.changes = async () => {
      const result = await changes();
      if (!injected) {
        injected = true;
        expect(
          await fixture.runtime.dispatch(
            IpcMethod.MESSAGE,
            { session_id: "boundary", message: "At completion" },
            () => {},
          ),
        ).toEqual({ accepted: true });
      }
      return result;
    };
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.RUN,
        { session_id: "boundary", instruction: "Start" },
        () => {},
      ),
    ).toHaveProperty("final_message", "Answer 2");
    expect(calls).toBe(2);
    expect(join).toHaveBeenCalledWith("At completion", undefined, { queue: "steering" });
    expect(await join.mock.results[0].value).toBe(false);
    const users = conversation.session.messages.filter((message) => message.role === "user");
    expect(users).toMatchObject([
      { role: "user", content: [{ type: "text", text: "Start" }], attribution: "user" },
      { role: "user", content: [{ type: "text", text: "At completion" }], attribution: "user" },
    ]);
    expect(users[1]).not.toHaveProperty("steering");
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: "boundary", message: "After completion" },
        () => {},
      ),
    ).toHaveProperty("accepted", false);
  } finally {
    await fixture.close();
  }
});

test("reviewer retains history and refreshes notes across rounds", async () => {
  const original = "The derivative is x.";
  const corrected = "The derivative is 2x.";
  const feedback = "Corrected the derivative to 2x. Check the saved correction.";
  const firstReport = JSON.stringify({
    correct: false,
    explanation: "The derivative needs correction.",
    confidence: 1,
    findings: [],
  });
  const finalReport = JSON.stringify({
    correct: true,
    explanation: "The saved derivative is correct.",
    confidence: 1,
    findings: [],
  });
  const reviewerRequests: ModelRequest[] = [];
  let writers = 0;
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("REVIEW_CONTINUITY")) {
      reviewerRequests.push(body);
      const call = reviewerRequests.length;
      if (call === 1 || call === 3)
        return sse(modelTool("read", { path: "note_knowledge.md" }, `read-round-${call}`));
      return sse(modelText(call === 2 ? firstReport : finalReport));
    }
    ++writers;
    if (writers === 1) return sse(modelTool("review_notes", {}, "review-first"));
    if (writers === 2)
      return sse(
        modelTool("write", { path: "note_knowledge.md", content: corrected }, "correct-note"),
      );
    if (writers === 3) return sse(modelTool("review_notes", { message: feedback }, "review-again"));
    return sse(modelText("Reviewed correction"));
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], name === "note_knowledge.md" ? original : "Notes");
    }
    const events: Record<string, unknown>[] = [];
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "continuity",
        tools: [WorkspaceTool.WRITE],
        files,
        meta: {
          review: {
            system: "REVIEW_CONTINUITY",
            prompt: "Review the saved notes",
            maxRounds: 2,
            ...reviewScope,
          },
        },
      },
      () => {},
    );
    const result = await fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "continuity", instruction: "Review and correct the notes" },
      (name, data) => {
        if (name === RuntimeEvent.REVIEWER) events.push(data);
      },
    );
    expect(result).toMatchObject({
      final_message: "Reviewed correction",
      review_passed: true,
    });
    expect(writers).toBe(4);
    expect(reviewerRequests).toHaveLength(4);
    const secondRound = reviewerRequests[2];
    const priorReports = secondRound.messages.filter(
      (message) =>
        message.role === "assistant" &&
        JSON.stringify(message.content).includes("The derivative needs correction."),
    );
    expect(priorReports).toHaveLength(1);
    expect(userTexts(secondRound).at(-1)).toContain(`Writer message:\n${feedback}`);
    const readResult = (request: ModelRequest) =>
      JSON.stringify(request.messages.filter((message) => message.role === "tool").at(-1)?.content);
    expect(readResult(reviewerRequests[1])).toContain(original);
    expect(readResult(reviewerRequests[3])).toContain(corrected);
    expect(readResult(reviewerRequests[3])).not.toContain(original);
    const ready = events.filter((event) => event.event === RuntimeEvent.RUN_READY);
    expect(ready.map((event) => event.reviewer_id)).toEqual([
      "continuity-review",
      "continuity-review",
    ]);
    const finished = events.filter((event) => event.finished === true);
    expect(finished.map((event) => event.stale)).toEqual([false, false]);
    await fixture.runtime.dispatch(IpcMethod.CLOSE_SESSION, { session_id: "continuity" }, () => {});
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.RUN,
        { session_id: "continuity-review", instruction: "Review after close" },
        () => {},
      ),
    ).rejects.toThrow("Unknown session: continuity-review");
  } finally {
    await fixture.close();
  }
});

test("reviewer messages change only the reviewer turn", async () => {
  const reviewerStarted = Promise.withResolvers<void>();
  let reviewers = 0;
  let writers = 0;
  let child = "";
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("REVIEW_TARGET")) {
      if (++reviewers === 1)
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(
                new TextEncoder().encode(
                  'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"Reviewing"},"finish_reason":null}]}\n\n',
                ),
              );
              reviewerStarted.resolve();
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        );
      expect(userTexts(body).at(-1)).toBe(nativeSteeringText("Stop reviewing and finish now"));
      return sse(
        modelText(
          JSON.stringify({
            correct: true,
            explanation: "Checked",
            confidence: 1,
            findings: [],
          }),
        ),
      );
    }
    return sse(
      ++writers === 1 ? modelTool("review_notes", {}, "review-target") : modelText("Reviewed"),
    );
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    const events: Record<string, unknown>[] = [];
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "parent",
        files,
        meta: {
          review: {
            system: "REVIEW_TARGET",
            prompt: "Review",
            maxRounds: 1,
            ...reviewScope,
          },
        },
      },
      () => {},
    );
    const run = fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "parent", instruction: "Review notes" },
      (name, data) => {
        if (name === RuntimeEvent.REVIEWER) {
          events.push(data);
          if (data.event === RuntimeEvent.RUN_READY) child = String(data.reviewer_id);
        }
      },
    );
    await reviewerStarted.promise;
    expect(child).not.toBe("");
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: child, message: "Stop reviewing and finish now" },
        () => {},
      ),
    ).toEqual({ accepted: true });
    expect(await run).toHaveProperty("final_message", "Reviewed");
    expect(writers).toBe(2);
    expect(reviewers).toBe(2);
    expect(events.find((event) => event.finished === true)).toHaveProperty("stale", false);
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: child, message: "After completion" },
        () => {},
      ),
    ).toHaveProperty("accepted", false);
  } finally {
    await fixture.close();
  }
});

test("explicit stop cancels parent and reviewer without delivery", async () => {
  const reviewerStarted = Promise.withResolvers<void>();
  let reviewerId = "";
  const lifecycle: Record<string, unknown>[] = [];
  const fixture = await localRuntime((body) => {
    if (!JSON.stringify(body.messages).includes("REVIEW_STOP"))
      return sse(modelTool("review_notes", {}, "review-stop"));
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'data: {"id":"proof","choices":[{"index":0,"delta":{"role":"assistant","content":"Reviewing"},"finish_reason":null}]}\n\n',
            ),
          );
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "stop-parent",
        files,
        meta: {
          review: {
            system: "REVIEW_STOP",
            prompt: "Review",
            maxRounds: 1,
            ...reviewScope,
          },
        },
      },
      () => {},
    );
    const run = fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "stop-parent", instruction: "Review notes" },
      (name, data) => {
        if (name === RuntimeEvent.REVIEWER) {
          reviewerId = String(data.reviewer_id);
          if (typeof data.finished === "boolean") lifecycle.push(data);
          if (data.event === RuntimeEvent.DELTA) reviewerStarted.resolve();
        }
      },
    );
    const outcome = run.then(
      () => undefined,
      (error: unknown) => error,
    );
    await reviewerStarted.promise;
    await fixture.runtime.dispatch(IpcMethod.CANCEL, { session_id: "stop-parent" }, () => {});
    expect(String(await outcome)).toContain("cancelled");
    expect(lifecycle.map((event) => event.finished)).toEqual([false, true]);
    expect(lifecycle.at(-1)).not.toHaveProperty("content");
    expect(
      await fixture.runtime.dispatch(
        IpcMethod.MESSAGE,
        { session_id: reviewerId, message: "Late" },
        () => {},
      ),
    ).toHaveProperty("accepted", false);
    await fixture.runtime.dispatch(
      IpcMethod.CLOSE_SESSION,
      { session_id: "stop-parent" },
      () => {},
    );
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.RUN,
        { session_id: reviewerId, instruction: "Late" },
        () => {},
      ),
    ).rejects.toThrow("Unknown session");
  } finally {
    await fixture.close();
  }
});

test("reviewer corrects malformed reports in the same round", async () => {
  let reviewerCalls = 0;
  const fixture = await localRuntime((body) => {
    if (JSON.stringify(body.messages).includes("REVIEW_INVALID")) {
      ++reviewerCalls;
      if (reviewerCalls === 1) return sse(modelText("not valid JSON"));
      expect(userTexts(body).at(-1)).toContain("Invalid review report:");
      expect(JSON.stringify(body.messages)).toContain("not valid JSON");
      if (reviewerCalls === 2) return sse(modelText('{"correct":"yes"}'));
      expect(userTexts(body).at(-1)).toContain("correct");
      return sse(
        modelText(
          JSON.stringify({
            correct: true,
            explanation: "Checked",
            confidence: 1,
            findings: [],
          }),
        ),
      );
    }
    return sse(
      body.messages.some((message) => message.role === "tool")
        ? modelText("Complete")
        : modelTool("review_notes", {}, "invalid-review"),
    );
  });
  try {
    const files: Record<string, string> = {};
    for (const name of [
      "lecture_source.md",
      "note_knowledge.md",
      "note_gossip.md",
      "note_notice.md",
    ]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Notes");
    }
    const events: Record<string, unknown>[] = [];
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "invalid",
        files,
        meta: {
          review: {
            system: "REVIEW_INVALID",
            prompt: "Review",
            maxRounds: 1,
            ...reviewScope,
          },
        },
      },
      () => {},
    );
    const result = await fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "invalid", instruction: "Review" },
      (name, data) => {
        if (name === RuntimeEvent.REVIEWER) events.push(data);
      },
    );
    expect(result).toHaveProperty("final_message", "Complete");
    expect(result).toHaveProperty("review_passed", true);
    expect(events.filter((event) => event.finished === false)).toHaveLength(1);
    expect(events.filter((event) => event.finished === true)).toHaveLength(1);
    const child = events.find((event) => event.finished === false)?.reviewer_id;
    await fixture.runtime.dispatch(IpcMethod.CLOSE_SESSION, { session_id: "invalid" }, () => {});
    await expect(
      fixture.runtime.dispatch(IpcMethod.RUN, { session_id: child, instruction: "Late" }, () => {}),
    ).rejects.toThrow("Unknown session");
    expect(reviewerCalls).toBe(3);
  } finally {
    await fixture.close();
  }
});

test("IPC controls require exact message and reopen parameters", () => {
  expect(
    validate(
      ipcRequestSchema,
      { id: "m", method: IpcMethod.MESSAGE, params: { session_id: "writer", message: "a\nb" } },
      "request",
    ),
  ).toHaveProperty("params.message", "a\nb");
  expect(
    validate(
      ipcRequestSchema,
      { id: "r", method: IpcMethod.REOPEN_SESSION, params: { id: "writer", files: {} } },
      "request",
    ),
  ).toHaveProperty("method", IpcMethod.REOPEN_SESSION);
  expect(() =>
    validate(
      ipcRequestSchema,
      {
        id: "m",
        method: IpcMethod.MESSAGE,
        params: { session_id: "writer", message: "a", instruction: "extra" },
      },
      "request",
    ),
  ).toThrow("Invalid request");
});

test("IPC accepts registered role identifiers and open tool names", () => {
  expect(
    ipcRequestSchema.allows({
      id: "valid",
      method: IpcMethod.CREATE_SESSION,
      params: { id: "writer", role: "note_writer", tools: [WorkspaceTool.READ] },
    }),
  ).toBe(true);
  expect(
    ipcRequestSchema.allows({
      id: "role",
      method: IpcMethod.GENERATE,
      params: { role: "unknown_role", messages: [{ role: "user", content: "Read" }] },
    }),
  ).toBe(true);
  expect(
    ipcRequestSchema.allows({
      id: "tool",
      method: IpcMethod.CREATE_SESSION,
      params: { id: "writer", tools: ["unknown_tool"] },
    }),
  ).toBe(true);
});

async function catalogFixture() {
  const server = Bun.serve({
    port: 0,
    fetch(request) {
      const { pathname } = new URL(request.url);
      if (!pathname.endsWith("/models")) return new Response("not found", { status: 404 });
      if (pathname.startsWith("/v2/")) return Response.json({ data: [] });
      return Response.json({
        data: [
          {
            id: "proof",
            limit: { context: 200_000, output: 8_000 },
            modalities: { input: ["text", "image/png"] },
            capabilities: { reasoning: { supported: true } },
          },
          { id: "plain", limit: { context: 64_000, output: 4_000 } },
        ],
      });
    },
  });
  const directory = await mkdtemp(path.join(tmpdir(), "runtime-catalog-"));
  const runtime = new Runtime(
    path.join(directory, "runtime"),
    undefined,
    undefined,
    registeredRoles,
  );
  const base = `http://127.0.0.1:${server.port}`;
  await runtime.dispatch(
    IpcMethod.CONFIGURE,
    {
      models: JSON.stringify({
        providers: {
          local: {
            api: "openai-completions",
            auth: "none",
            baseUrl: `${base}/v1`,
            models: [
              {
                id: "proof",
                input: ["text", "image"],
                contextWindow: 8_000,
                maxTokens: 1_000,
                compat: { supportsDeveloperRole: false },
              },
            ],
          },
          second: {
            api: "openai-completions",
            auth: "none",
            baseUrl: `${base}/v1`,
            models: [
              { id: "plain", input: ["text", "image"], contextWindow: 4_000, maxTokens: 500 },
            ],
          },
        },
      }),
      roles: Bun.YAML.stringify({
        ...fixedRoles("local/proof:off"),
        note_writer: "second/plain:high",
      }),
    },
    () => {},
  );
  return {
    runtime,
    base,
    async close() {
      await runtime.close();
      server.stop(true);
      await rm(directory, { recursive: true, force: true });
    },
  };
}

test("catalog reports structured providers, models and selectors", async () => {
  const fixture = await catalogFixture();
  try {
    const catalog = await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, () => {});
    expect(catalog.providers.map((provider) => provider.id)).toEqual(["local", "second"]);
    expect(catalog.providers[0].models).toEqual([
      {
        id: "proof",
        contextWindow: 8_000,
        maxTokens: 1_000,
        input: ["text", "image"],
        compat: { supportsDeveloperRole: false },
      },
    ]);
    expect(catalog.selectors.default).toBe("local/proof:off");
    expect(catalog.selectors.note_writer).toBe("second/plain:high");
    expect(catalog.roles.default).toEqual({
      provider: "local",
      model: "proof",
      thinkingLevel: "off",
    });
  } finally {
    await fixture.close();
  }
});

test("models_set replaces complete model definitions", async () => {
  const fixture = await catalogFixture();
  try {
    const catalog = await fixture.runtime.dispatch(
      IpcMethod.MODELS_SET,
      {
        provider: "local",
        models: [
          {
            id: "proof",
            contextWindow: 200_000,
            maxTokens: 8_000,
            input: ["text", "image"],
            reasoning: true,
          },
        ],
      },
      () => {},
    );
    expect(catalog.providers[0].models).toEqual([
      {
        id: "proof",
        contextWindow: 200_000,
        maxTokens: 8_000,
        input: ["text", "image"],
        reasoning: true,
      },
    ]);
    expect(catalog.definitions.models.providers.local.models[0]).toEqual(
      catalog.providers[0].models[0],
    );
    expect(catalog.definitions.models.providers.local.models[0].compat).toBeUndefined();
  } finally {
    await fixture.close();
  }
});

test("discover reads the endpoint model list", async () => {
  const fixture = await catalogFixture();
  try {
    const result = await fixture.runtime.dispatch(
      IpcMethod.DISCOVER,
      { provider: "local" },
      () => {},
    );
    expect(result.base_url).toBe(`${fixture.base}/v1`);
    expect(result.models).toEqual([
      {
        id: "plain",
        contextWindow: 64_000,
        maxTokens: 4_000,
        input: ["text"],
        reasoning: false,
      },
      {
        id: "proof",
        contextWindow: 200_000,
        maxTokens: 8_000,
        input: ["text", "image"],
        reasoning: true,
      },
    ]);
  } finally {
    await fixture.close();
  }
});

test("provider_set creates a provider from the endpoint and provider_remove remaps roles", async () => {
  const fixture = await catalogFixture();
  try {
    const added = await fixture.runtime.dispatch(
      IpcMethod.PROVIDER_SET,
      {
        id: "third",
        base_url: `${fixture.base}/v1/`,
        api: "openai-completions",
        auth: "none",
      },
      () => {},
    );
    const third = added.providers.find((provider) => provider.id === "third");
    if (!third) throw new Error("Missing third provider");
    expect(third.models.map((model) => model.id)).toEqual(["plain", "proof"]);
    expect(third.baseUrl).toBe(`${fixture.base}/v1`);

    await expect(
      fixture.runtime.dispatch(
        IpcMethod.PROVIDER_SET,
        { id: "empty", base_url: `${fixture.base}/v2`, api: "openai-completions", auth: "none" },
        () => {},
      ),
    ).rejects.toThrow("端点没有返回任何模型");

    const removed = await fixture.runtime.dispatch(
      IpcMethod.PROVIDER_REMOVE,
      { provider: "second" },
      () => {},
    );
    expect(removed.providers.map((provider) => provider.id)).toEqual(["local", "third"]);
    expect(removed.selectors.note_writer).toBe("@default");
  } finally {
    await fixture.close();
  }
});

test("provider_set remaps roles left dangling by the last provider removal", async () => {
  const fixture = await catalogFixture();
  try {
    await fixture.runtime.dispatch(
      IpcMethod.PROVIDER_REMOVE,
      { provider: "second" },
      () => {},
    );
    await fixture.runtime.dispatch(IpcMethod.PROVIDER_REMOVE, { provider: "local" }, () => {});
    const added = await fixture.runtime.dispatch(
      IpcMethod.PROVIDER_SET,
      {
        id: "third",
        base_url: `${fixture.base}/v1`,
        api: "openai-completions",
        auth: "none",
      },
      () => {},
    );
    expect(added.providers.map((provider) => provider.id)).toEqual(["third"]);
    // proof satisfies the image-capable board_recognition role, so it wins.
    expect(added.selectors.default).toBe("third/proof");
    expect(added.roles.default).toMatchObject({ provider: "third", model: "proof" });
    for (const [role, selector] of Object.entries(added.selectors)) {
      if (role === AgentRole.DEFAULT) continue;
      expect(selector).toBe("@default");
      expect(added.roles[role as string]).toEqual(added.roles.default);
    }
    // An unresolvable selector passed explicitly still fails.
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.ROLE_SET,
        { role: "note_writer", selector: "third/missing" },
        () => {},
      ),
    ).rejects.toThrow();
  } finally {
    await fixture.close();
  }
});

test("role_set rewrites one selector and keeps the rest", async () => {
  const fixture = await catalogFixture();
  try {
    const catalog = await fixture.runtime.dispatch(
      IpcMethod.ROLE_SET,
      { role: "note_writer", selector: "local/proof:high" },
      () => {},
    );
    expect(catalog.selectors.note_writer).toBe("local/proof:high");
    expect(catalog.selectors.default).toBe("local/proof:off");
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.ROLE_SET,
        { role: "note_writer", selector: "local/missing:high" },
        () => {},
      ),
    ).rejects.toThrow();
    expect(
      (await fixture.runtime.dispatch(IpcMethod.CATALOG, {}, () => {})).selectors.note_writer,
    ).toBe("local/proof:high");
  } finally {
    await fixture.close();
  }
});

test("review extensions enforce tools and ordered hooks", async () => {
  let writers = 0;
  let reviewers = 0;
  const hooks: string[] = [];
  const report = JSON.stringify({
    correct: true,
    explanation: "Checked",
    confidence: 1,
    findings: [
      {
        title: "Optional",
        body: "Wording only",
        priority: 3,
        confidence: 1,
        file_path: "appendix.md",
        line_start: 1,
        line_end: 1,
      },
    ],
  });
  const fixture = await localRuntime(
    (body) => {
      const review = JSON.stringify(body.messages).includes("EXTENSION_REVIEW");
      if (!review) {
        expect(JSON.stringify(body)).not.toContain("audit_document");
        return sse(
          ++writers === 1 ? modelTool("review_notes", {}, "extension-review") : modelText("Done"),
        );
      }
      expect(JSON.stringify(body.messages)).toContain("Review extension criterion");
      ++reviewers;
      if (reviewers === 1) return sse(modelText(report));
      if (reviewers === 2) {
        expect(userTexts(body).at(-1)).toContain("Run audit_document before completion");
        return sse(modelTool("audit_document", { path: "appendix.md" }, "audit"));
      }
      expect(JSON.stringify(body)).toContain("AUDIT_OK");
      return sse(modelText(report));
    },
    async (request) => {
      if (request.kind === "tool") {
        expect(request.role).toBe("note_review");
        expect(request.name).toBe("audit_document");
        const workspace = request.arguments.workspace as Record<string, string>;
        expect(await Bun.file(workspace["appendix.md"]).text()).toBe("Appendix");
        return { content: [{ type: "text", text: "AUDIT_OK" }], details: {} };
      }
      hooks.push(request.name);
      if (request.name === "alpha/require-audit") {
        const calls = request.arguments.tool_calls as { name: string }[];
        return calls.some((call) => call.name === "audit_document")
          ? null
          : { continue: "Run audit_document before completion" };
      }
      return null;
    },
  );
  try {
    const files: Record<string, string> = {};
    for (const name of [...reviewScope.sources, ...reviewScope.targets, "appendix.md"]) {
      files[name] = path.join(fixture.root, name);
      await Bun.write(files[name], "Appendix");
    }
    await fixture.runtime.dispatch(
      IpcMethod.CREATE_SESSION,
      {
        id: "extension-parent",
        files,
        meta: {
          review: {
            system: "EXTENSION_REVIEW",
            prompt: "Review",
            maxRounds: 1,
            ...reviewScope,
            targets: [...reviewScope.targets, "appendix.md"],
          },
        },
        extensions: {
          reviewer: {
            prompts: ["Review extension criterion"],
            tools: [
              {
                name: "audit_document",
                description: "Audit a document.",
                parameters: {
                  type: "object",
                  properties: { path: { type: "string" } },
                  required: ["path"],
                  additionalProperties: false,
                },
              },
            ],
            completion_hooks: [
              { owner: "zeta", id: "last", priority: 10 },
              { owner: "alpha", id: "require-audit", priority: 0 },
              { owner: "alpha", id: "after", priority: 1 },
            ],
          },
        },
      },
      () => {},
    );
    const result = await fixture.runtime.dispatch(
      IpcMethod.RUN,
      { session_id: "extension-parent", instruction: "Write" },
      () => {},
    );
    expect(result).toHaveProperty("final_message", "Done");
    expect(reviewers).toBe(3);
    expect(hooks).toEqual([
      "alpha/require-audit",
      "alpha/after",
      "zeta/last",
      "alpha/require-audit",
      "alpha/after",
      "zeta/last",
    ]);
  } finally {
    await fixture.close();
  }
});

test("session rejects tools without a declaration", async () => {
  const fixture = await localRuntime(() => sse(modelText("Unused")));
  try {
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.CREATE_SESSION,
        {
          id: "undeclared",
          tools: ["unknown_tool"],
        },
        () => {},
      ),
    ).rejects.toThrow("Unknown tool: unknown_tool");
  } finally {
    await fixture.close();
  }
});

for (const role of ["note_chunk", "ask_research", "note_writer"]) {
  test(`${role} completion hooks require a tool without review`, async () => {
    let turns = 0;
    const hooks: string[] = [];
    const fixture = await localRuntime(
      (body) => {
        ++turns;
        if (turns === 1) return sse(modelText("Premature answer"));
        if (turns === 2) {
          expect(userTexts(body).at(-1)).toContain("Run output_check before completion");
          return sse(modelTool("output_check", {}, "output-check"));
        }
        expect(JSON.stringify(body)).toContain("OUTPUT_OK");
        return sse(modelText("Checked answer"));
      },
      async (request) => {
        expect(request.role).toBe(role);
        if (request.kind === "tool")
          return { content: [{ type: "text", text: "OUTPUT_OK" }], details: {} };
        expect(request.kind).toBe("completion_hook");
        hooks.push(request.name);
        const calls = request.arguments.tool_calls as { name: string }[];
        return request.name === "a/require-check" &&
          !calls.some((call) => call.name === "output_check")
          ? { continue: "Run output_check before completion" }
          : null;
      },
    );
    try {
      await fixture.runtime.dispatch(
        IpcMethod.CREATE_SESSION,
        {
          id: "role-parent",
          role,
          tools: [],
          extensions: {
            tools: [
              {
                name: "output_check",
                description: "Check the output.",
                parameters: { type: "object", properties: {}, additionalProperties: false },
              },
            ],
            completion_hooks: [
              { owner: "z", id: "after", priority: 0 },
              { owner: "a", id: "require-check", priority: 0 },
            ],
          },
        },
        () => {},
      );
      const result = await fixture.runtime.dispatch(
        IpcMethod.RUN,
        { session_id: "role-parent", instruction: "Answer" },
        () => {},
      );
      expect(result).toHaveProperty("final_message", "Checked answer");
      expect(turns).toBe(3);
      expect(hooks).toEqual(["a/require-check", "z/after", "a/require-check", "z/after"]);
    } finally {
      await fixture.close();
    }
  });
}

test("registered roles prune stale settings and preserve custom assignments", async () => {
  const fixture = await persistenceFixture();
  try {
    await fixture.runtime.dispatch(IpcMethod.PING, {}, fixture.emit);
    const roles: RoleDefinition[] = [{ id: "plugin.custom", title: "Custom", input: ["text"] }];
    const catalog = await fixture.runtime.dispatch(
      IpcMethod.ROLES_SYNC,
      { roles, selectors: { default: "local/proof:off", retired: { invalid: true } } },
      fixture.emit,
    );
    expect(catalog).toHaveProperty("selectors", {
      default: "local/proof:off",
      compaction: "@default",
      "plugin.custom": "@default",
    });
    await fixture.runtime.dispatch(
      IpcMethod.ROLE_SET,
      { role: "plugin.custom", selector: "local/proof:high" },
      fixture.emit,
    );
    await fixture.runtime.dispatch(IpcMethod.ROLES_SYNC, { roles }, fixture.emit);
    // biome-ignore lint/style/noNonNullAssertion: Role synchronization persists settings before dispatch returns.
    const selected = fixture.snapshots.at(-1)!.settings!.roles;
    expect(selected["plugin.custom"]).toBe("local/proof:high");
    const result = await fixture.runtime.dispatch(
      IpcMethod.ROLES_SYNC,
      { roles: [] },
      fixture.emit,
    );
    expect(result).toHaveProperty("selectors", {
      default: "local/proof:off",
      compaction: "@default",
    });
    await expect(
      fixture.runtime.dispatch(
        IpcMethod.ROLE_SET,
        { role: "plugin.custom", selector: "@default" },
        fixture.emit,
      ),
    ).rejects.toThrow("Unknown role");
  } finally {
    await fixture.close();
  }
});
