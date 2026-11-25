import { Effort, type, type Type } from "@oh-my-pi/pi-ai";
import type { AuthStorageData } from "@oh-my-pi/pi-ai";

export enum AgentRole {
  DEFAULT = "default",
  COMPACTION = "compaction",
}

export enum IpcMethod {
  PING = "ping",
  INFO = "info",
  CONFIGURE = "configure",
  CATALOG = "catalog",
  DISCOVER = "discover",
  MODELS_SET = "models_set",
  PROVIDER_SET = "provider_set",
  PROVIDER_REMOVE = "provider_remove",
  ROLE_SET = "role_set",
  ROLES_SYNC = "roles_sync",
  GENERATE = "generate",
  CREATE_SESSION = "create_session",
  REOPEN_SESSION = "reopen_session",
  SESSION_TRANSCRIPT = "session_transcript",
  RUN = "run",
  MESSAGE = "message",
  CANCEL = "cancel",
  CLOSE_SESSION = "close_session",
  COMPACT = "compact",
  LOGOUT = "logout",
  CREDENTIALS = "credentials",
  IMPORT_CREDENTIALS = "import_credentials",
}

export enum IpcResponseType {
  ERROR = "error",
  EVENT = "event",
  RESULT = "result",
  PERSISTENCE = "persistence",
}

export enum RuntimeEvent {
  EXECUTION = "agent.execution",
  REVIEWER = "agent.reviewer",
  FILE = "agent.file",
  DELTA = "agent.delta",
  REASONING = "agent.reasoning",
  TOOL_CALL = "agent.tool_call",
  TOOL_RESULT = "agent.tool_result",
  RUN_READY = "agent.run.ready",
  MESSAGE = "agent.message",
  LLM_ATTEMPT = "llm.attempt",
  LLM_DELTA = "llm.delta",
  LLM_REASONING = "llm.reasoning",
}

export enum WorkspaceTool {
  READ = "read",
  GREP = "grep",
  GLOB = "glob",
  EDIT = "edit",
  WRITE = "write",
  REVIEW_NOTES = "review_notes",
}

export enum ReasoningMode {
  OFF = "off",
  AUTO = "auto",
}

export enum ImageDetail {
  AUTO = "auto",
  LOW = "low",
  HIGH = "high",
}

export enum ProviderSource {
  CONFIG = "config",
  PROJECT = "project",
}

export type Emit = (name: RuntimeEvent, data: Record<string, unknown>) => void;

export const agentRoleSchema = type("string").atLeastLength(1);
export const workspaceToolSchema = type.enumerated(...Object.values(WorkspaceTool));

const nonemptyString = type("string").atLeastLength(1);
const finiteNumber = type("number").narrow(Number.isFinite);
const sessionId = nonemptyString.matching(/^[A-Za-z0-9_-]+$/);
const stringMap = type({ "[string]": "string" });
const imageDataUrl = nonemptyString
  .matching(
    /^data:image\/[A-Za-z0-9.+-]+;base64,(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/,
  )
  .narrow((value) => value.slice(value.indexOf(",") + 1).length > 0);

export const textPartSchema = type({
  "+": "reject",
  type: "'text'",
  text: "string",
});
export const imagePartSchema = type({
  "+": "reject",
  type: "'image_url'",
  image_url: {
    "+": "reject",
    url: imageDataUrl,
    "detail?": type.enumerated(...Object.values(ImageDetail)),
  },
});
const messageMetadata = {
  "name?": "string | null",
  "tool_calls?": type("never").array(),
  "tool_call_id?": "null",
} as const;
const systemMessageSchema = type({
  "+": "reject",
  ...messageMetadata,
  role: "'system'",
  content: type("string").or(textPartSchema.array()),
});
const conversationMessageSchema = type({
  "+": "reject",
  ...messageMetadata,
  role: "'user' | 'assistant' | 'developer'",
  content: type("string").or(textPartSchema.or(imagePartSchema).array()),
});

export const generateInputSchema = type({
  "+": "reject",
  messages: systemMessageSchema.or(conversationMessageSchema).array().atLeastLength(1),
  "role?": agentRoleSchema,
  "session_id?": sessionId.or("null"),
  "system_prompt?": "string",
  "max_tokens?": type("number.integer > 0").or("null"),
  "temperature?": finiteNumber.atLeast(0).or("null"),
  "loop_max_attempts?": "number.integer > 0",
  "reasoning?": type
    .enumerated(Effort.Minimal, Effort.Low, Effort.Medium, Effort.High, Effort.XHigh, Effort.Max)
    .or(type.enumerated(...Object.values(ReasoningMode))),
});
export type GenerateInput = typeof generateInputSchema.infer;

const apiKeyCredentialSchema = type({
  "+": "reject",
  type: "'api_key'",
  key: nonemptyString,
  "source?": "'login'",
});
const oauthCredentialSchema = type({
  "+": "reject",
  type: "'oauth'",
  access: nonemptyString,
  refresh: nonemptyString,
  expires: finiteNumber.atLeast(0),
  "enterpriseUrl?": "string",
  "projectId?": "string",
  "email?": "string",
  "accountId?": "string",
  "apiEndpoint?": "string",
  "orgId?": "string",
  "org_id?": "string",
  "orgName?": "string",
  "region?": "string",
  "inferenceRegion?": "'global' | 'eu' | 'us'",
  "activeOrganizationId?": "string",
  "authorizedAt?": finiteNumber.atLeast(0),
});
const storedCredentialSchema = apiKeyCredentialSchema.or(oauthCredentialSchema);
function normalizeCredential(credential: typeof storedCredentialSchema.infer) {
  if (credential.type === "api_key") return credential;
  const { org_id, ...oauth } = credential;
  return {
    ...oauth,
    ...((oauth.orgId ?? org_id) === undefined ? {} : { orgId: oauth.orgId ?? org_id }),
  };
}
export const credentialSchema = storedCredentialSchema.pipe(normalizeCredential);
export const credentialsSchema = type({
  "[string]": storedCredentialSchema.or(storedCredentialSchema.array()),
}).pipe((providers) =>
  Object.fromEntries(
    Object.entries(providers).map(([provider, credential]) => [
      provider,
      Array.isArray(credential)
        ? credential.map(normalizeCredential)
        : normalizeCredential(credential),
    ]),
  ),
);
export type Credential = typeof credentialSchema.infer;

// The SDK validates native model fields. Preserve them for its configuration parser.
const modelSchema = type({ id: nonemptyString, "[string]": "unknown" });
const providerSchema = type({
  "api?": "string",
  "auth?": "'apiKey' | 'oauth' | 'none'",
  "apiKey?": "string",
  "baseUrl?": "string",
  models: modelSchema.array().atLeastLength(1),
  "[string]": "unknown",
});
export const modelsConfigurationSchema = type({
  providers: type({ "[string]": providerSchema }),
  "[string]": "unknown",
});
export type ModelsConfiguration = typeof modelsConfigurationSchema.infer;
export const rolesConfigurationSchema = type({ "[string]": "unknown" });
export const roleDefinitionSchema = type({
  id: agentRoleSchema,
  title: nonemptyString,
  input: type.enumerated("text", "image").array(),
});
export const roleDefinitionsSchema = roleDefinitionSchema.array();
export type RoleDefinition = typeof roleDefinitionSchema.infer;

export const runtimeBootstrapSchema = type({
  "models?": modelsConfigurationSchema,
  "roles?": rolesConfigurationSchema,
  "registered_roles?": roleDefinitionsSchema,
  credentials: credentialsSchema,
});
export interface PersistenceSnapshot {
  settings?: { models: ModelsConfiguration; roles: Record<string, string> };
  credentials?: AuthStorageData;
}
export interface RuntimePersistence {
  models?: ModelsConfiguration;
  roles?: Record<string, unknown>;
  credentials?: AuthStorageData;
  save: (snapshot: PersistenceSnapshot) => void;
}

export const openCodeTokenResponseSchema = type({
  access_token: nonemptyString,
  "refresh_token?": nonemptyString,
  expires_in: finiteNumber.moreThan(0),
  "org_id?": "string",
  "token_type?": "string",
});
const emptyParams = type({ "+": "reject" }).default(() => ({}));
const configurationParams = type({
  "+": "reject",
  models: nonemptyString.or(modelsConfigurationSchema),
  roles: nonemptyString.or(rolesConfigurationSchema),
});

const modelsSetParams = type({
  "+": "reject",
  provider: nonemptyString,
  models: modelSchema.array(),
});
const providerSetParams = type({
  "+": "reject",
  id: nonemptyString,
  "base_url?": "string",
  "api?": nonemptyString,
  "auth?": type.enumerated("apiKey", "oauth", "none"),
});
const roleSetParams = type({
  "+": "reject",
  role: agentRoleSchema,
  selector: nonemptyString,
});
const extensionFields = {
  "prompts?": "string[]",
  "tools?": type({
    name: nonemptyString,
    description: "string",
    parameters: { "[string]": "unknown" },
    "label?": "string",
  }).array(),
  "completion_hooks?": type({
    owner: nonemptyString,
    id: nonemptyString,
    priority: "number.integer",
  }).array(),
} as const;
export const extensionsSchema = type({
  ...extensionFields,
  "reviewer?": type(extensionFields),
});
export type SessionExtensions = typeof extensionsSchema.infer;

export const createSessionInputSchema = type({
  "+": "reject",
  id: sessionId,
  "role?": agentRoleSchema,
  "extensions?": extensionsSchema,
  "prompt?": "string",
  "files?": stringMap,
  "readonly?": "string[]",
  "tools?": "string[]",
  "compaction?": { "[string]": "unknown" },
  "meta?": { "[string]": "unknown" },
});
const sessionParams = type({ "+": "reject", session_id: sessionId });
const providerParams = type({ "+": "reject", provider: nonemptyString });

export const ipcRequestHeaderSchema = type({ id: nonemptyString });
export const ipcRequestSchema = type({
  "+": "reject",
  id: nonemptyString,
  method: type.enumerated(IpcMethod.PING, IpcMethod.INFO),
  params: emptyParams,
})
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.ROLES_SYNC),
    params: type({ roles: roleDefinitionsSchema, "selectors?": rolesConfigurationSchema }),
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.CONFIGURE),
    params: configurationParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.GENERATE),
    params: generateInputSchema,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.CREATE_SESSION, IpcMethod.REOPEN_SESSION),
    params: createSessionInputSchema,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.RUN),
    params: type({
      "+": "reject",
      session_id: sessionId,
      instruction: "string",
      "follow_ups?": "string[]",
    }),
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.MESSAGE),
    params: type({ "+": "reject", session_id: sessionId, message: "string" }),
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.CANCEL, IpcMethod.CLOSE_SESSION),
    params: sessionParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.SESSION_TRANSCRIPT),
    params: type({ "+": "reject", session_id: "string" }),
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.LOGOUT),
    params: providerParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.CREDENTIALS),
    params: providerParams.merge({ key: nonemptyString }),
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.IMPORT_CREDENTIALS),
    params: providerParams.merge({ credential: credentialSchema }),
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.CATALOG),
    params: emptyParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.PROVIDER_REMOVE, IpcMethod.DISCOVER),
    params: providerParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.MODELS_SET),
    params: modelsSetParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.PROVIDER_SET),
    params: providerSetParams,
  })
  .or({
    "+": "reject",
    id: nonemptyString,
    method: type.enumerated(IpcMethod.ROLE_SET),
    params: roleSetParams,
  });
export type IpcRequest = typeof ipcRequestSchema.infer;

export function validate<T>(schema: Type<T>, value: unknown, label: string): T {
  const result = schema(value);
  // Schema diagnostics can include credentials or message contents.
  if (result instanceof type.errors) throw new Error(`Invalid ${label}`);
  return result;
}
