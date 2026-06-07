import fs from "node:fs/promises";
import path from "node:path";
import type { AgentRunRequest, AgentRuntime, GatewayConfig, PiRuntimeEvent, ToolDefinition } from "../types.js";
import { toPiModelAuthConfig } from "../config/piModel.js";
import { getOrCreatePiSessionId } from "../redis/chatHistory.js";
import type { RedisLike } from "../redis/redisClient.js";

type PiModule = Record<string, unknown>;

async function loadPiSdk(): Promise<PiModule | null> {
  try {
    return await import("@earendil-works/pi-coding-agent") as PiModule;
  } catch {
    return null;
  }
}

function selectTools(tools: ToolDefinition[], enabledTools: Set<string>): ToolDefinition[] {
  const searchEnabled = ["search", "academic_search", "web_search"].some((name) => enabledTools.has(name));
  return tools.filter((tool) => {
    if (tool.source === "simulator") return true;
    if (tool.name === "current_time") return true;
    if (["search", "academic_search", "web_search"].includes(tool.name)) return searchEnabled || enabledTools.has(tool.name);
    return enabledTools.has(tool.name);
  });
}

function toPiTool(tool: ToolDefinition): Record<string, unknown> {
  return {
    name: tool.name,
    label: tool.name,
    description: tool.description,
    promptSnippet: tool.description,
    parameters: tool.parameters,
    execute: async (_toolCallId: string, params: Record<string, unknown>) => {
      const content = await tool.execute(params || {});
      return {
        content: [{ type: "text", text: content }],
        details: { source: tool.source || "main" }
      };
    }
  };
}

export class OfflineAgentRuntime implements AgentRuntime {
  constructor(private readonly tools: ToolDefinition[]) {}

  async *run(request: AgentRunRequest): AsyncIterable<PiRuntimeEvent> {
    const selected = selectTools(this.tools, request.enabledTools);
    yield { kind: "thinking", delta: "Analyzing request and available tools.", source: "main" };
    yield { kind: "thinking", done: true, truncated: false, source: "main" };
    if (request.enabledTools.has("search")) {
      yield { kind: "tool_execution_start", toolName: "search", source: "main" };
      yield { kind: "tool_execution_end", toolName: "search", output: { results: [] }, source: "main" };
    }
    if (selected.some((tool) => tool.source === "simulator") && /(仿真|simulate|simulation|机器人|robot)/i.test(request.message)) {
      yield { kind: "tool_execution_start", toolName: "initialize_simulation", source: "simulator" };
      yield { kind: "tool_execution_end", toolName: "initialize_simulation", output: { status: "offline" }, source: "simulator" };
    }
    const text = process.env.PI_OFFLINE_RESPONSE || `Pi gateway offline mode: ${request.message}`;
    for (const chunk of text.match(/.{1,24}/gs) || [text]) {
      yield { kind: "message_update", delta: chunk, source: "main" };
    }
    yield { kind: "agent_end", text };
  }
}

export class PiAgentRuntime implements AgentRuntime {
  private piPromise: Promise<PiModule | null>;

  constructor(
    private readonly config: GatewayConfig,
    private readonly redis: RedisLike,
    private readonly systemPrompt: string,
    private readonly tools: ToolDefinition[]
  ) {
    this.piPromise = loadPiSdk();
  }

  async *run(request: AgentRunRequest): AsyncIterable<PiRuntimeEvent> {
    if (this.config.piOffline) {
      yield* new OfflineAgentRuntime(this.tools).run(request);
      return;
    }

    const pi = await this.piPromise;
    const createAgentSession = pi?.createAgentSession as
      | ((args: Record<string, unknown>) => Promise<{ session: Record<string, unknown> }>)
      | undefined;
    const SessionManager = pi?.SessionManager as { create?: (cwd: string, sessionDir?: string) => unknown } | undefined;

    if (!createAgentSession) {
      yield* new OfflineAgentRuntime(this.tools).run(request);
      return;
    }

    await fs.mkdir(this.config.piAgentDir, { recursive: true });
    await fs.mkdir(this.config.piSessionDir, { recursive: true });
    const piSessionId = await getOrCreatePiSessionId(this.redis, request.userId, request.sessionId);
    const sessionPath = path.join(this.config.piSessionDir, piSessionId);
    await fs.mkdir(sessionPath, { recursive: true });

    const selectedTools = selectTools(this.tools, request.enabledTools).map(toPiTool);
    const sessionManager = SessionManager?.create?.(this.config.rootDir, sessionPath);
    const { session } = await createAgentSession({
      cwd: this.config.rootDir,
      agentDir: this.config.piAgentDir,
      sessionManager,
      noTools: "all",
      tools: selectedTools.map((tool) => String(tool.name)),
      customTools: selectedTools,
      sessionStartEvent: { type: "session_start", reason: "resume" }
    });

    if (typeof session.setActiveToolsByName === "function") {
      session.setActiveToolsByName(selectedTools.map((tool) => String(tool.name)));
    }

    const queue: PiRuntimeEvent[] = [];
    let done = false;
    let failure: Error | undefined;
    let notify: (() => void) | undefined;
    const wake = () => {
      notify?.();
      notify = undefined;
    };

    const unsubscribe = typeof session.subscribe === "function"
      ? session.subscribe((event: Record<string, unknown>) => {
          const normalized = normalizePiSdkEvent(event);
          queue.push(normalized);
          if (normalized.kind === "agent_end" || normalized.kind === "error") done = true;
          wake();
        })
      : undefined;

    const prompt = session.prompt as ((text: string, options?: Record<string, unknown>) => Promise<void>) | undefined;
    if (!prompt) {
      yield { kind: "error", error: "Pi session does not expose prompt()" };
      return;
    }

    prompt.call(session, String(`${this.systemPrompt}\n\n${request.message}`), { expandPromptTemplates: false, source: "rpc" })
      .then(() => {
        done = true;
        wake();
      })
      .catch((error: Error) => {
        failure = error;
        done = true;
        wake();
      });

    try {
      while (!done || queue.length) {
        if (!queue.length) await new Promise<void>((resolve) => { notify = resolve; });
        while (queue.length) yield queue.shift()!;
      }
      if (failure) yield { kind: "error", error: failure.message };
    } finally {
      unsubscribe?.();
      if (typeof session.dispose === "function") session.dispose();
    }
  }
}

function normalizePiSdkEvent(event: Record<string, unknown>): PiRuntimeEvent {
  const type = String(event.type || event.kind || "");
  if (type === "message_update") {
    const assistantEvent = event.assistantMessageEvent && typeof event.assistantMessageEvent === "object"
      ? event.assistantMessageEvent as Record<string, unknown>
      : {};
    return {
      kind: "message_update",
      delta: assistantEvent.delta ? String(assistantEvent.delta) : undefined,
      text: assistantEvent.text || event.text || event.content ? String(assistantEvent.text || event.text || event.content) : undefined,
      source: event.source ? String(event.source) : undefined
    };
  }
  if (type === "tool_execution_start") {
    return { kind: "tool_execution_start", toolName: String(event.toolName || ""), input: event.args };
  }
  if (type === "tool_execution_update") {
    return { kind: "tool_execution_update", toolName: String(event.toolName || ""), output: event.partialResult };
  }
  if (type === "tool_execution_end") {
    return { kind: "tool_execution_end", toolName: String(event.toolName || ""), output: event.result, error: event.isError ? String(event.result || "Tool execution failed") : undefined };
  }
  if (type.includes("thinking")) {
    return {
      kind: "thinking",
      delta: event.delta ? String(event.delta) : undefined,
      text: event.text ? String(event.text) : undefined,
      done: Boolean(event.done),
      truncated: Boolean(event.truncated),
      source: event.source ? String(event.source) : undefined
    };
  }
  if (type.includes("tool") && type.includes("start")) {
    return { kind: "tool_execution_start", toolName: String(event.toolName || event.name || ""), input: event.input, source: event.source ? String(event.source) : undefined };
  }
  if (type.includes("tool") && type.includes("end")) {
    return { kind: "tool_execution_end", toolName: String(event.toolName || event.name || ""), output: event.output || event.result, error: event.error ? String(event.error) : undefined, source: event.source ? String(event.source) : undefined };
  }
  if (type.includes("usage")) return { kind: "usage", usage: (event.usage || {}) as Record<string, unknown> };
  if (type.includes("end")) return { kind: "agent_end", text: event.text ? String(event.text) : undefined };
  if (type.includes("error")) return { kind: "error", error: String(event.error || event.message || "Pi runtime error") };
  return {
    kind: "message_update",
    delta: event.delta ? String(event.delta) : undefined,
    text: event.text || event.content ? String(event.text || event.content) : undefined,
    source: event.source ? String(event.source) : undefined
  };
}
