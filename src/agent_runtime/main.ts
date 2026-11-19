import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline";
import type { IpcRequest } from "./schemas";

const dataDirectory = process.argv[2];
if (!dataDirectory) throw new Error("The runtime data directory is required");
const root = path.resolve(dataDirectory);
await mkdir(root, { recursive: true, mode: 0o700 });
for (const key of Object.keys(process.env)) {
  if (key.startsWith("PI_") || key.startsWith("OMP_")) delete process.env[key];
}
process.env.PI_CODING_AGENT_DIR = root;
// SDK imports read environment settings. Load them only after environment isolation.
const [
  { Runtime },
  { IpcResponseType, ipcRequestHeaderSchema, ipcRequestSchema, runtimeBootstrapSchema, validate },
] = await Promise.all([import("./runtime"), import("./schemas")]);
const bootstrap = validate(
  runtimeBootstrapSchema,
  JSON.parse(await readFile(path.join(root, "bootstrap.json"), "utf8")),
  "runtime bootstrap",
);
const send = (message: unknown) => process.stdout.write(`${JSON.stringify(message)}\n`);
const runtime = new Runtime(root, {
  models: bootstrap.models ?? undefined,
  roles: bootstrap.roles ?? undefined,
  credentials: bootstrap.credentials,
  save: (snapshot) => {
    send({ type: IpcResponseType.PERSISTENCE, ...snapshot });
  },
});
const input = createInterface({ input: process.stdin });
const requests = new Set<Promise<void>>();
for await (const line of input) {
  let request: IpcRequest;
  let requestId: string | undefined;
  try {
    const input: unknown = JSON.parse(line);
    if (ipcRequestHeaderSchema.allows(input)) requestId = input.id;
    const parsed = ipcRequestSchema(input);
    if ("summary" in parsed) throw new Error("Invalid IPC request");
    request = parsed;
  } catch {
    send({ id: requestId, type: IpcResponseType.ERROR, error: "Invalid IPC request" });
    continue;
  }
  const operation = (async () => {
    try {
      const result = await runtime.dispatch(request.method, request.params, (name, data) =>
        send({ id: request.id, type: IpcResponseType.EVENT, name, data }),
      );
      send({ id: request.id, type: IpcResponseType.RESULT, result });
    } catch (error) {
      send({
        id: request.id,
        type: IpcResponseType.ERROR,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  })();
  requests.add(operation);
  operation.finally(() => requests.delete(operation));
}
await runtime.close();
await Promise.allSettled(requests);
