import fs from "node:fs";
import path from "node:path";
import { parse } from "yaml";
import type { GatewayConfig, ModelConfig } from "../types.js";

function env(name: string): string | undefined {
  const value = process.env[name];
  return value && value.trim() ? value.trim() : undefined;
}

function envBool(name: string, fallback = false): boolean {
  const value = env(name);
  if (value === undefined) return fallback;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

function envInt(name: string, fallback: number): number {
  const value = env(name);
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function resolveOpenAICompatibleModel(rawConfig: Record<string, unknown>, role = "main"): ModelConfig {
  const roleUpper = role.toUpperCase();
  const modelKey = role === "main" ? "llm" : `${role}_llm`;
  const urlKey = role === "main" ? "model_url" : `${role}_model_url`;
  const apiKeyKey = role === "main" ? "api_key" : `${role}_api_key`;

  const model =
    env(`${roleUpper}_LLM`) ||
    env(`${roleUpper}_MODEL`) ||
    env("OPENAI_COMPATIBLE_MODEL") ||
    env("OPENAI_MODEL") ||
    env("LLM") ||
    String(rawConfig[modelKey] || rawConfig.llm || "");

  const baseUrl =
    env(`${roleUpper}_MODEL_URL`) ||
    env(`${roleUpper}_BASE_URL`) ||
    env("OPENAI_COMPATIBLE_BASE_URL") ||
    env("OPENAI_BASE_URL") ||
    env("MODEL_URL") ||
    String(rawConfig[urlKey] || rawConfig.model_url || "");

  const apiKey =
    env(`${roleUpper}_API_KEY`) ||
    env("OPENAI_COMPATIBLE_API_KEY") ||
    env("OPENAI_API_KEY") ||
    env("API_KEY") ||
    String(rawConfig[apiKeyKey] || rawConfig.api_key || "no_need");

  return { model, baseUrl, apiKey };
}

export function resolveModelForRequest(rawConfig: Record<string, unknown>, payload?: Record<string, unknown>): ModelConfig {
  const base = resolveOpenAICompatibleModel(rawConfig, "main");
  if (!payload) return base;
  return {
    model: String(payload.model || payload.llm || base.model),
    baseUrl: String(payload.base_url || payload.model_url || base.baseUrl),
    apiKey: String(payload.api_key || base.apiKey || "no_need")
  };
}

function resolveMcpUrls(rawConfig: Record<string, unknown>): Record<string, string> {
  const mcp = (rawConfig.mcp && typeof rawConfig.mcp === "object" ? rawConfig.mcp : {}) as Record<string, unknown>;
  const servers = mcp.servers;
  if (servers && typeof servers === "object" && !Array.isArray(servers)) {
    const resolved: Record<string, string> = {};
    for (const [name, urlValue] of Object.entries(servers as Record<string, unknown>)) {
      if (!urlValue) continue;
      const raw = String(urlValue).trim();
      resolved[name] = raw.endsWith("/mcp") ? raw : `${raw.replace(/\/+$/, "")}/mcp`;
    }
    if (Object.keys(resolved).length) return resolved;
  }

  const base = String(mcp.ip || "http://127.0.0.1").replace(/\/+$/, "");
  const port = String(mcp.port || "18001");
  const urls: Record<string, string> = { pybullet: `${base}:${port}/mcp` };
  if (mcp.gazebo_port) urls.gazebo = `${base}:${String(mcp.gazebo_port)}/mcp`;
  return urls;
}

export function loadGatewayConfig(rootDir = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../../..")): GatewayConfig {
  const configPath = path.join(rootDir, "config", "config.yml");
  const rawConfig = fs.existsSync(configPath) ? parse(fs.readFileSync(configPath, "utf8")) || {} : {};
  const agentDir = env("PI_AGENT_DIR") || path.join(rootDir, ".runtime", "pi-agent");
  const sessionDir = env("PI_SESSION_DIR") || path.join(rootDir, ".runtime", "pi-sessions");

  return {
    rootDir,
    port: envInt("PI_GATEWAY_PORT", 8000),
    pythonLegacyBaseUrl: env("PYTHON_LEGACY_BASE_URL") || "http://127.0.0.1:8001",
    piAgentDir: agentDir,
    piSessionDir: sessionDir,
    piOffline: envBool("PI_OFFLINE", false),
    redisUrl: env("REDIS_URL") || "redis://127.0.0.1:6379/0",
    chatRedisUrl: env("CHAT_REDIS_URL") || "redis://127.0.0.1:6379/1",
    authRedisUrl: env("AUTH_REDIS_URL") || "redis://127.0.0.1:6379/2",
    authSessionTtlSeconds: envInt("AUTH_SESSION_TTL_SECONDS", 30 * 24 * 3600),
    chatHistoryMaxLen: envInt("CHAT_HISTORY_MAX_LEN", 200),
    model: resolveOpenAICompatibleModel(rawConfig, "main"),
    tavilyApiKey: env("TAVILY_API_KEY") || String(rawConfig.tavily && typeof rawConfig.tavily === "object" ? (rawConfig.tavily as Record<string, unknown>).api_key || "" : ""),
    mcp: { urls: resolveMcpUrls(rawConfig) }
  };
}
