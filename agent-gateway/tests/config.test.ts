import { describe, expect, it } from "vitest";
import { resolveModelForRequest, resolveOpenAICompatibleModel } from "../src/config/loadConfig.js";
import { toPiModelAuthConfig } from "../src/config/piModel.js";

describe("model config", () => {
  it("resolves main model from config", () => {
    const model = resolveOpenAICompatibleModel({ llm: "m", model_url: "http://x/v1", api_key: "k" });
    expect(model).toEqual({ model: "m", baseUrl: "http://x/v1", apiKey: "k" });
  });

  it("allows per-request override", () => {
    const model = resolveModelForRequest(
      { llm: "base", model_url: "http://base/v1", api_key: "base-key" },
      { model: "override", base_url: "http://override/v1", api_key: "override-key" }
    );
    expect(model).toEqual({ model: "override", baseUrl: "http://override/v1", apiKey: "override-key" });
  });

  it("converts to Pi OpenAI-compatible config", () => {
    expect(toPiModelAuthConfig({ model: "m", baseUrl: "http://x/v1", apiKey: "" })).toEqual({
      provider: "openai-compatible",
      model: "m",
      baseURL: "http://x/v1",
      apiKey: "no_need"
    });
  });
});
