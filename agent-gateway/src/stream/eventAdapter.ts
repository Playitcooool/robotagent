import type { NdjsonEvent, PiRuntimeEvent, PlanningStep, SearchReference } from "../types.js";
import { truncate } from "../utils/json.js";

function normalizeSource(source: string | undefined, toolName?: string): string {
  if (source) return source;
  if (toolName && !["current_time", "search", "web_search", "academic_search", "format_json", "http_get", "list_workspace_files", "read_workspace_file", "search_workspace_text"].includes(toolName)) {
    return "simulator";
  }
  return "main";
}

function parseObject(value: unknown): Record<string, unknown> | null {
  if (!value) return null;
  if (typeof value === "object") return value as Record<string, unknown>;
  if (typeof value !== "string") return null;
  try {
    return JSON.parse(value) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function extractSearchRefs(value: unknown): SearchReference[] {
  const parsed = parseObject(value);
  const results = parsed && Array.isArray(parsed.results) ? parsed.results : [];
  return results.slice(0, 8).flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const row = item as Record<string, unknown>;
    const title = String(row.title || "Search Result");
    const url = String(row.url || "");
    if (!url) return [];
    return [{
      title: truncate(title, 120),
      url,
      snippet: truncate(String(row.snippet || row.abstract || ""), 220),
      authors: row.authors ? String(row.authors) : undefined,
      year: row.year ? String(row.year) : undefined,
      source: row.source ? String(row.source) : undefined
    }];
  });
}

function planningPayload(steps: PlanningStep[], statusText: string, activeSource: string, isActive: boolean): NdjsonEvent {
  return {
    type: "planning",
    plan: steps,
    steps,
    updated_at: Date.now() / 1000,
    updatedAt: Date.now(),
    status_text: statusText,
    active_source: activeSource,
    is_active: isActive
  };
}

export class PiEventAdapter {
  private steps: PlanningStep[] = [];

  adapt(event: PiRuntimeEvent): NdjsonEvent[] {
    if (event.kind === "message_update") {
      const text = event.delta || event.text || "";
      return text ? [{ type: "delta", text, source: normalizeSource(event.source) }] : [];
    }
    if (event.kind === "thinking") {
      if (event.done) return [{ type: "thinking_done", truncated: event.truncated, source: normalizeSource(event.source) }];
      const text = event.delta || event.text || "";
      return text ? [{ type: "thinking", text, source: normalizeSource(event.source) }] : [];
    }
    if (event.kind === "tool_execution_start") {
      const source = normalizeSource(event.source, event.toolName);
      const out: NdjsonEvent[] = [{ type: "status", text: `调用工具：${event.toolName}`, source, status_kind: event.toolName.includes("search") ? "search" : undefined }];
      if (source === "simulator") {
        const existing = this.steps.find((step) => step.step === event.toolName);
        if (existing) existing.status = "in_progress";
        else this.steps.push({ id: String(this.steps.length + 1), step: event.toolName, title: event.toolName, status: "in_progress", source });
        out.push(planningPayload(this.steps, `执行 ${event.toolName}`, source, true));
      }
      return out;
    }
    if (event.kind === "tool_execution_update") {
      return [{ type: "status", text: `${event.toolName} 更新中`, source: normalizeSource(event.source, event.toolName) }];
    }
    if (event.kind === "tool_execution_end") {
      const source = normalizeSource(event.source, event.toolName);
      const out: NdjsonEvent[] = [];
      if (event.error) out.push({ type: "error", error: event.error, source });
      else out.push({ type: "status", text: `${event.toolName} 完成`, source });
      if (["search", "web_search", "academic_search"].includes(event.toolName)) {
        out.push({ type: "web_search_results", results: extractSearchRefs(event.output), source: "main" });
      }
      if (event.toolName === "search_workspace_text" || event.toolName.toLowerCase().includes("rag")) {
        out.push({ type: "rag_results", results: extractSearchRefs(event.output), source: "main" });
      }
      if (source === "simulator") {
        const step = this.steps.find((item) => item.step === event.toolName);
        if (step) step.status = event.error ? "failed" : "completed";
        out.push(planningPayload(this.steps, event.error ? "执行失败" : "执行中", source, !event.error));
      }
      return out;
    }
    if (event.kind === "usage") return [{ type: "usage", usage: event.usage, source: normalizeSource(event.source) }];
    if (event.kind === "agent_end") {
      const out: NdjsonEvent[] = [];
      if (this.steps.length) {
        this.steps = this.steps.map((step) => step.status === "in_progress" ? { ...step, status: "completed" } : step);
        out.push(planningPayload(this.steps, "执行完成", "simulator", false));
      }
      if (event.usage) out.push({ type: "usage", usage: event.usage });
      out.push({ type: "done" });
      return out;
    }
    if (event.kind === "error") return [{ type: "error", error: event.error }];
    return [];
  }
}
