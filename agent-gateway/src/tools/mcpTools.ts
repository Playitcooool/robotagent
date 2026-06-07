import type { GatewayConfig, ToolDefinition } from "../types.js";

const CORE_TOOL_NAMES = new Set([
  "initialize_simulation",
  "run_pybullet_navigation_task",
  "run_gazebo_navigation_task",
  "create_object",
  "load_urdf",
  "load_robot",
  "get_robot_state",
  "drive_robot",
  "follow_waypoints",
  "simulate_lidar",
  "get_contacts",
  "set_joint_positions",
  "move_end_effector",
  "grasp_object",
  "release_object",
  "set_object_position",
  "step_simulation",
  "get_object_state",
  "delete_object",
  "get_simulation_info",
  "check_simulation_state",
  "cleanup_simulation_tool"
]);

type McpToolInfo = {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  server: string;
  url: string;
};

async function mcpRpc(url: string, method: string, params: Record<string, unknown> = {}): Promise<unknown> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream"
    },
    body: JSON.stringify({ jsonrpc: "2.0", id: Date.now(), method, params })
  });
  if (!response.ok) throw new Error(`MCP ${method} failed: ${response.status}`);
  const text = await response.text();
  const jsonLine = text
    .split(/\r?\n/)
    .map((line) => line.replace(/^data:\s*/, "").trim())
    .find((line) => line.startsWith("{"));
  const parsed = JSON.parse(jsonLine || text);
  if (parsed.error) throw new Error(JSON.stringify(parsed.error));
  return parsed.result;
}

async function listServerTools(server: string, url: string): Promise<McpToolInfo[]> {
  const result = await mcpRpc(url, "tools/list");
  const tools = result && typeof result === "object" && Array.isArray((result as Record<string, unknown>).tools)
    ? ((result as Record<string, unknown>).tools as Record<string, unknown>[])
    : [];
  return tools.map((tool) => ({
    name: String(tool.name || ""),
    description: tool.description ? String(tool.description) : "",
    inputSchema: tool.inputSchema && typeof tool.inputSchema === "object" ? (tool.inputSchema as Record<string, unknown>) : {},
    server,
    url
  })).filter((tool) => tool.name);
}

async function callServerTool(tool: McpToolInfo, args: Record<string, unknown>): Promise<string> {
  const result = await mcpRpc(tool.url, "tools/call", { name: tool.name, arguments: args });
  return typeof result === "string" ? result : JSON.stringify(result, null, 2);
}

export async function createMcpTools(config: GatewayConfig): Promise<ToolDefinition[]> {
  const allTools: McpToolInfo[] = [];
  await Promise.all(Object.entries(config.mcp.urls).map(async ([server, url]) => {
    try {
      allTools.push(...await listServerTools(server, url));
    } catch (error) {
      console.warn(`[mcp] ${server} unavailable:`, error);
    }
  }));

  const core = allTools.filter((tool) => CORE_TOOL_NAMES.has(tool.name));
  const extended = allTools.filter((tool) => !CORE_TOOL_NAMES.has(tool.name));
  const extendedByName = new Map(extended.map((tool) => [tool.name, tool]));

  return [
    ...core.map((tool): ToolDefinition => ({
      name: tool.name,
      description: tool.description || "",
      parameters: tool.inputSchema || { type: "object", properties: {} },
      source: "simulator",
      execute: (args) => callServerTool(tool, args)
    })),
    {
      name: "list_available_tools",
      description: "List all extended simulation tools available for on-demand use.",
      parameters: { type: "object", properties: {} },
      source: "simulator",
      execute: async () => extended.length ? extended.map((tool) => `- ${tool.name}: ${(tool.description || "").slice(0, 80)}`).join("\n") : "No extended tools available."
    },
    {
      name: "call_extended_tool",
      description: "Call an extended simulation tool by name with JSON arguments.",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string" },
          args_json: { type: "string", default: "{}" }
        },
        required: ["name"]
      },
      source: "simulator",
      execute: async (args) => {
        const name = String(args.name || "");
        const tool = extendedByName.get(name);
        if (!tool) return JSON.stringify({ error: `Tool '${name}' not found. Use list_available_tools to see available tools.` });
        const parsedArgs = args.args_json ? JSON.parse(String(args.args_json)) : {};
        return callServerTool(tool, parsedArgs);
      }
    }
  ];
}
