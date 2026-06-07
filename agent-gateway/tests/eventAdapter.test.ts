import { describe, expect, it } from "vitest";
import { PiEventAdapter } from "../src/stream/eventAdapter.js";

describe("PiEventAdapter", () => {
  it("maps message and thinking events", () => {
    const adapter = new PiEventAdapter();
    expect(adapter.adapt({ kind: "thinking", delta: "plan" })).toEqual([{ type: "thinking", text: "plan", source: "main" }]);
    expect(adapter.adapt({ kind: "thinking", done: true, truncated: true })).toEqual([{ type: "thinking_done", truncated: true, source: "main" }]);
    expect(adapter.adapt({ kind: "message_update", delta: "hello" })).toEqual([{ type: "delta", text: "hello", source: "main" }]);
  });

  it("maps simulator tools to status and planning", () => {
    const adapter = new PiEventAdapter();
    const start = adapter.adapt({ kind: "tool_execution_start", toolName: "initialize_simulation" });
    expect(start[0]).toMatchObject({ type: "status", source: "simulator" });
    expect(start[1]).toMatchObject({ type: "planning", active_source: "simulator", is_active: true });
    const end = adapter.adapt({ kind: "tool_execution_end", toolName: "initialize_simulation", output: "{}" });
    expect(end.at(-1)).toMatchObject({ type: "planning" });
  });

  it("extracts search references", () => {
    const adapter = new PiEventAdapter();
    const events = adapter.adapt({
      kind: "tool_execution_end",
      toolName: "search",
      output: JSON.stringify({ results: [{ title: "T", url: "https://example.com", snippet: "S" }] })
    });
    expect(events).toContainEqual({ type: "web_search_results", results: [{ title: "T", url: "https://example.com", snippet: "S", authors: undefined, year: undefined, source: undefined }], source: "main" });
  });
});
