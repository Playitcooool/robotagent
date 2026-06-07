export type ChatIn = {
  message: string;
  session_id?: string | null;
  enabled_tools?: string[] | null;
};

export type GatewayConfig = {
  rootDir: string;
  port: number;
  pythonLegacyBaseUrl: string;
  piAgentDir: string;
  piSessionDir: string;
  piOffline: boolean;
  redisUrl: string;
  chatRedisUrl: string;
  authRedisUrl: string;
  authSessionTtlSeconds: number;
  chatHistoryMaxLen: number;
  model: ModelConfig;
  tavilyApiKey: string;
  mcp: {
    urls: Record<string, string>;
  };
};

export type ModelConfig = {
  model: string;
  baseUrl: string;
  apiKey: string;
};

export type AuthUser = {
  token: string;
  uid: string;
  username?: string;
};

export type ChatHistoryMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
  session_id: string;
  created_at: number;
};

export type NdjsonEvent =
  | { type: "delta"; text: string; source?: string }
  | { type: "thinking"; text: string; source?: string }
  | { type: "thinking_done"; truncated?: boolean; source?: string }
  | { type: "status"; text: string; source?: string; status_kind?: string }
  | {
      type: "planning";
      plan?: PlanningStep[];
      steps?: PlanningStep[];
      updated_at?: number;
      updatedAt?: number;
      status_text?: string;
      active_source?: string;
      is_active?: boolean;
    }
  | { type: "web_search_results"; results: SearchReference[]; source?: string }
  | { type: "rag_results"; results: SearchReference[]; source?: string }
  | { type: "usage"; usage: Record<string, unknown>; source?: string }
  | { type: "error"; error: string; source?: string }
  | { type: "done" };

export type PlanningStep = {
  id: string;
  title?: string;
  step?: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  source?: string;
};

export type SearchReference = {
  title: string;
  url: string;
  snippet?: string;
  authors?: string;
  year?: string;
  source?: string;
};

export type ToolDefinition = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (args: Record<string, unknown>) => Promise<string>;
  source?: "main" | "simulator" | "analysis";
};

export type PiRuntimeEvent =
  | { kind: "message_update"; text?: string; delta?: string; source?: string }
  | { kind: "thinking"; text?: string; delta?: string; done?: boolean; truncated?: boolean; source?: string }
  | { kind: "tool_execution_start"; toolName: string; input?: unknown; source?: string }
  | { kind: "tool_execution_update"; toolName: string; output?: unknown; source?: string }
  | { kind: "tool_execution_end"; toolName: string; output?: unknown; error?: string; source?: string }
  | { kind: "usage"; usage: Record<string, unknown>; source?: string }
  | { kind: "agent_end"; text?: string; usage?: Record<string, unknown> }
  | { kind: "error"; error: string };

export type AgentRunRequest = {
  userId: string;
  sessionId: string;
  message: string;
  enabledTools: Set<string>;
  modelOverride?: ModelConfig;
};

export interface AgentRuntime {
  run(request: AgentRunRequest): AsyncIterable<PiRuntimeEvent>;
}
