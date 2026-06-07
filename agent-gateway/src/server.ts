import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { Readable } from "node:stream";
import type { AgentRuntime, ChatIn, GatewayConfig, NdjsonEvent, ToolDefinition } from "./types.js";
import { appendChatMessage } from "./redis/chatHistory.js";
import { connectRedis, type RedisLike } from "./redis/redisClient.js";
import { requireAuthUser } from "./redis/auth.js";
import { PiEventAdapter } from "./stream/eventAdapter.js";
import { ndjson } from "./utils/json.js";
import { registerOpenAIRoutes } from "./routes/openai.js";

type GatewayDeps = {
  config: GatewayConfig;
  runtime: AgentRuntime;
  localTools: ToolDefinition[];
  mcpTools: ToolDefinition[];
  chatRedis?: RedisLike;
  authRedis?: RedisLike;
};

function isAgentRoute(method: string, path: string): boolean {
  if (method === "POST" && path === "/api/chat/send") return true;
  if (method === "POST" && path === "/v1/chat/completions") return true;
  if (method === "GET" && path === "/v1/models") return true;
  if (method === "GET" && (path === "/api/ping" || path === "/api/tools" || path === "/api/health")) return true;
  return false;
}

async function proxyToPython(request: FastifyRequest, reply: FastifyReply, baseUrl: string): Promise<void> {
  const target = new URL(request.url, baseUrl);
  const headers = new Headers();
  for (const [key, value] of Object.entries(request.headers)) {
    if (value === undefined || key.toLowerCase() === "host") continue;
    headers.set(key, Array.isArray(value) ? value.join(", ") : String(value));
  }
  const body = ["GET", "HEAD"].includes(request.method) ? undefined : JSON.stringify(request.body || {});
  const response = await fetch(target, { method: request.method, headers, body });
  reply.code(response.status);
  response.headers.forEach((value, key) => {
    if (!["content-encoding", "transfer-encoding"].includes(key.toLowerCase())) reply.header(key, value);
  });
  return reply.send(Readable.fromWeb(response.body as Parameters<typeof Readable.fromWeb>[0]));
}

async function* chatStream(
  runtime: AgentRuntime,
  config: GatewayConfig,
  chatRedis: RedisLike,
  userId: string,
  sessionId: string,
  message: string,
  enabledTools: Set<string>
): AsyncGenerator<string> {
  const adapter = new PiEventAdapter();
  let finalText = "";
  try {
    for await (const event of runtime.run({ userId, sessionId, message, enabledTools })) {
      if (event.kind === "message_update") finalText += event.delta || event.text || "";
      const outgoing = adapter.adapt(event);
      for (const item of outgoing) yield ndjson(item);
    }
    if (finalText.trim()) await appendChatMessage(chatRedis, userId, sessionId, "assistant", finalText, config.chatHistoryMaxLen);
  } catch (error) {
    yield ndjson({ type: "error", error: error instanceof Error ? error.message : String(error) } satisfies NdjsonEvent);
    yield ndjson({ type: "done" });
  }
}

export async function createGatewayServer(deps: GatewayDeps): Promise<FastifyInstance> {
  const app = Fastify({ logger: true });
  const chatRedis = deps.chatRedis || await connectRedis(deps.config.chatRedisUrl);
  const authRedis = deps.authRedis || await connectRedis(deps.config.authRedisUrl);

  app.addHook("onClose", async () => {
    await chatRedis.quit?.();
    await authRedis.quit?.();
  });

  app.get("/api/ping", async () => ({ status: "ok", agent_ready: true, gateway: "pi" }));
  app.get("/v1/models", async () => ({
    object: "list",
    data: [
      {
        id: deps.config.model.model || "unknown",
        object: "model",
        created: 0,
        owned_by: "robotagent-pi-gateway"
      }
    ]
  }));
  app.get("/api/health", async () => {
    const redis = { chat: false, auth: false };
    try { await chatRedis.ping(); redis.chat = true; } catch {}
    try { await authRedis.ping(); redis.auth = true; } catch {}
    return {
      status: "healthy",
      agent_ready: true,
      gateway: "pi",
      redis,
      config: {
        llm_model: deps.config.model.model,
        llm_url: deps.config.model.baseUrl
      }
    };
  });

  app.get("/api/tools", async () => ({
    local_tools: deps.localTools.map((tool) => ({
      name: tool.name,
      brief: tool.description.split("\n")[0] || "无描述",
      description: tool.description,
      parameters: Object.keys((tool.parameters.properties as Record<string, unknown>) || {})
    })),
    mcp_tools: deps.mcpTools.filter((tool) => tool.source === "simulator").map((tool) => ({
      name: tool.name,
      source: "simulator",
      description: tool.description
    })),
    total: deps.localTools.length + deps.mcpTools.length
  }));

  app.post("/api/chat/send", async (request, reply) => {
    const currentUser = await requireAuthUser(authRedis, request.headers.authorization, deps.config.authSessionTtlSeconds);
    const payload = (request.body || {}) as ChatIn;
    const userMessage = payload.message || "";
    const sessionId = payload.session_id || "default_session";
    const enabledTools = new Set((payload.enabled_tools || []).map((tool) => String(tool).trim()).filter(Boolean));
    await appendChatMessage(chatRedis, currentUser.uid, sessionId, "user", userMessage, deps.config.chatHistoryMaxLen);

    reply.header("Content-Type", "application/x-ndjson");
    reply.header("Cache-Control", "no-transform");
    reply.header("X-Accel-Buffering", "no");
    return reply.send(chatStream(deps.runtime, deps.config, chatRedis, currentUser.uid, sessionId, userMessage, enabledTools));
  });

  registerOpenAIRoutes(app, deps.config, deps.runtime);

  app.all("*", async (request, reply) => {
    const path = request.routeOptions.url || request.url.split("?")[0] || "";
    if (isAgentRoute(request.method, path)) return reply.callNotFound();
    return proxyToPython(request, reply, deps.config.pythonLegacyBaseUrl);
  });

  return app;
}
