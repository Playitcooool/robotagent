import type { ModelConfig } from "../types.js";

export type PiModelAuthConfig = {
  provider: "openai-compatible";
  model: string;
  baseURL: string;
  apiKey: string;
};

export function toPiModelAuthConfig(model: ModelConfig): PiModelAuthConfig {
  return {
    provider: "openai-compatible",
    model: model.model,
    baseURL: model.baseUrl,
    apiKey: model.apiKey || "no_need"
  };
}
