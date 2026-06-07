import crypto from "node:crypto";
import type { FastifyInstance } from "fastify";
import type { AgentRuntime, GatewayConfig, ModelConfig, PiRuntimeEvent } from "../types.js";
import { resolveModelForRequest } from "../config/loadConfig.js";

function lastUserMessage(messages: unknown): string {
  if (!Array.isArray(messages)) return "";
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const msg = messages[index];
    if (!msg || typeof msg !== "object") continue;
    const row = msg as Record<string, unknown>;
    if (String(row.role || "").toLowerCase() === "user" && row.content) return String(row.content);
  }
  return "";
}

function enabledTools(payload: Record<string, unknown>): Set<string> {
  const result = new Set<string>();
  const tools = Array.isArray(payload.tools) ? payload.tools : [];
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") continue;
    const fn = (tool as Record<string, unknown>).function;
    if (fn && typeof fn === "object" && (fn as Record<string, unknown>).name) {
      result.add(String((fn as Record<string, unknown>).name));
    }
  }
  return result;
}

async function collectText(runtime: AgentRuntime, message: string, modelOverride: ModelConfig, tools: Set<string>): Promise<string> {
  let text = "";
  for await (const event of runtime.run({
    userId: "openai",
    sessionId: crypto.createHash("md5").update(message).digest("hex").slice(0, 12),
    message,
    enabledTools: tools,
    modelOverride
  })) {
    if (event.kind === "message_update") text += event.delta || event.text || "";
    if (event.kind === "agent_end" && event.text && !text) text = event.text;
  }
  return text;
}

function chunk(model: string, content: string, finishReason: string | null = null): Record<string, unknown> {
  return {
    id: `chatcmpl-${crypto.randomBytes(4).toString("hex")}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      delta: content ? { content } : {},
      finish_reason: finishReason
    }]
  };
}

export function registerOpenAIRoutes(app: FastifyInstance, config: GatewayConfig, runtime: AgentRuntime): void {
  app.post("/v1/chat/completions", async (request, reply) => {
    const payload = (request.body || {}) as Record<string, unknown>;
    const modelCfg = resolveModelForRequest({ llm: config.model.model, model_url: config.model.baseUrl, api_key: config.model.apiKey }, payload);
    const userMessage = lastUserMessage(payload.messages);
    if (!userMessage) {
      reply.code(400);
      return { error: { message: "No user message found", type: "invalid_request_error", code: "invalid_request" } };
    }

    const tools = enabledTools(payload);
    if (payload.stream) {
      reply.header("Content-Type", "application/x-ndjson");
      reply.header("Cache-Control", "no-transform");
      reply.header("X-Accel-Buffering", "no");
      async function* stream() {
        for await (const event of runtime.run({
          userId: "openai",
          sessionId: crypto.createHash("md5").update(JSON.stringify(payload.messages || [])).digest("hex").slice(0, 12),
          message: userMessage,
          enabledTools: tools,
          modelOverride: modelCfg
        })) {
          const content = event.kind === "message_update" ? event.delta || event.text || "" : "";
          if (content) yield `${JSON.stringify(chunk(modelCfg.model, content))}\n`;
        }
        yield `${JSON.stringify(chunk(modelCfg.model, "", "stop"))}\n`;
      }
      return reply.send(stream());
    }

    const content = await collectText(runtime, userMessage, modelCfg, tools);
    return {
      id: `chatcmpl-${crypto.randomBytes(4).toString("hex")}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: modelCfg.model,
      choices: [{
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop"
      }]
    };
  });
}
