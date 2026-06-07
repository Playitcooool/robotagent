import { describe, expect, it } from "vitest";
import { createLocalTools } from "../src/tools/localTools.js";
import type { GatewayConfig } from "../src/types.js";

const config: GatewayConfig = {
  rootDir: process.cwd().replace(/\/agent-gateway$/, ""),
  port: 8000,
  pythonLegacyBaseUrl: "http://127.0.0.1:8001",
  piAgentDir: ".runtime/pi-agent",
  piSessionDir: ".runtime/pi-sessions",
  piOffline: true,
  redisUrl: "redis://127.0.0.1:6379/0",
  chatRedisUrl: "redis://127.0.0.1:6379/1",
  authRedisUrl: "redis://127.0.0.1:6379/2",
  authSessionTtlSeconds: 60,
  chatHistoryMaxLen: 200,
  model: { model: "m", baseUrl: "http://x/v1", apiKey: "k" },
  tavilyApiKey: "",
  mcp: { urls: {} }
};

describe("local tools", () => {
  it("exposes expected production tool names", () => {
    const names = createLocalTools(config).map((tool) => tool.name).sort();
    expect(names).toEqual([
      "academic_search",
      "current_time",
      "format_json",
      "http_get",
      "list_workspace_files",
      "read_workspace_file",
      "search",
      "search_workspace_text",
      "web_search"
    ]);
  });

  it("keeps workspace file reads inside repo root", async () => {
    const read = createLocalTools(config).find((tool) => tool.name === "read_workspace_file");
    await expect(read?.execute({ path: "../AGENTS.md" })).rejects.toThrow(/outside repository/);
  });
});
