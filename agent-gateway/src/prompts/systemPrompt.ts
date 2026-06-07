import fs from "node:fs";
import path from "node:path";

export function loadRobotContext(rootDir: string, maxChars = 1200): string {
  const contextPath = path.join(rootDir, "robot_context.md");
  if (!fs.existsSync(contextPath)) return "";
  const content = fs.readFileSync(contextPath, "utf8");
  const body = content.startsWith("---") ? content.replace(/^---\n[\s\S]*?\n---\n/, "") : content;
  return body.length > maxChars ? `${body.slice(0, maxChars).trimEnd()}\n\n[...context truncated for faster prefill...]` : body;
}

export function loadMainSystemPrompt(rootDir: string): string {
  const promptPath = path.join(rootDir, "prompts", "MainAgentPrompt.py");
  const content = fs.existsSync(promptPath) ? fs.readFileSync(promptPath, "utf8") : "";
  const match = content.match(/SYSTEM_PROMPT\s*=\s*"""([\s\S]*?)"""/);
  const base = match ? match[1].trim() : "你是机器人任务编排代理，直接操作仿真工具完成任务。";
  const context = loadRobotContext(rootDir);
  return context ? `${base}\n\n## 项目级 Context\n\n${context}\n` : base;
}
