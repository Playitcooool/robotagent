import { loadGatewayConfig } from "./config/loadConfig.js";
import { createGatewayServer } from "./server.js";
import { createLocalTools } from "./tools/localTools.js";
import { createMcpTools } from "./tools/mcpTools.js";
import { connectRedis } from "./redis/redisClient.js";
import { PiAgentRuntime } from "./pi/piRuntime.js";
import { loadMainSystemPrompt } from "./prompts/systemPrompt.js";

async function main(): Promise<void> {
  const config = loadGatewayConfig();
  const localTools = createLocalTools(config);
  const mcpTools = await createMcpTools(config);
  const chatRedis = await connectRedis(config.chatRedisUrl);
  const authRedis = await connectRedis(config.authRedisUrl);
  const runtime = new PiAgentRuntime(config, chatRedis, loadMainSystemPrompt(config.rootDir), [...localTools, ...mcpTools]);
  const server = await createGatewayServer({ config, runtime, localTools, mcpTools, chatRedis, authRedis });
  await server.listen({ port: config.port, host: "0.0.0.0" });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
