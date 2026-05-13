import os
from typing import Any


def _first_env(names: list[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def resolve_openai_compatible_model(config: dict[str, Any], role: str = "main") -> dict[str, str]:
    """Resolve ChatOpenAI settings from config with environment overrides.

    Role-specific environment variables take precedence. Generic OpenAI-compatible
    variables provide a convenient way to switch every agent at once.

    Supports both local (e.g. MLX/LM Studio) and remote (e.g. DeepSeek, OpenAI) APIs.
    """
    role_upper = role.upper()
    model_key = "llm" if role == "main" else f"{role}_llm"
    url_key = "model_url" if role == "main" else f"{role}_model_url"
    api_key_key = "api_key" if role == "main" else f"{role}_api_key"

    fallback_model = str(config.get(model_key) or config.get("llm") or "").strip()
    fallback_url = str(config.get(url_key) or config.get("model_url") or "").strip()
    fallback_api_key = str(config.get(api_key_key) or config.get("api_key") or "no_need").strip()

    model = _first_env([
        f"{role_upper}_LLM",
        f"{role_upper}_MODEL",
        "OPENAI_COMPATIBLE_MODEL",
        "OPENAI_MODEL",
        "LLM",
    ]) or fallback_model
    base_url = _first_env([
        f"{role_upper}_MODEL_URL",
        f"{role_upper}_BASE_URL",
        "OPENAI_COMPATIBLE_BASE_URL",
        "OPENAI_BASE_URL",
        "MODEL_URL",
    ]) or fallback_url
    api_key = _first_env([
        f"{role_upper}_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_API_KEY",
        "API_KEY",
    ]) or fallback_api_key

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def resolve_model_for_request(config: dict[str, Any], payload: dict | None = None) -> dict[str, str]:
    """Resolve model config for a specific request.

    If payload contains 'model', 'base_url', 'api_key' fields, use those (overrides config).
    Otherwise fall back to 'main' role config from config.yml.
    Supports both local and remote OpenAI-compatible APIs.
    """
    if payload:
        model = payload.get("model") or payload.get("llm")
        base_url = payload.get("base_url") or payload.get("model_url")
        api_key = payload.get("api_key")
        if model or base_url:
            main_config = resolve_openai_compatible_model(config, "main")
            return {
                "model": str(model or "").strip() or main_config["model"],
                "base_url": str(base_url or "").strip() or main_config["base_url"],
                "api_key": str(api_key or "no_need").strip() or "no_need",
            }
    return resolve_openai_compatible_model(config, "main")


def create_chat_model(model_config: dict[str, str]) -> "ChatOpenAI":
    """Create a ChatOpenAI instance from model config dict.

    Works with both local (LM Studio, MLX) and remote (DeepSeek, OpenAI) APIs.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=model_config.get("base_url", ""),
        model=model_config.get("model", ""),
        api_key=model_config.get("api_key", "no_need"),
        streaming=True,
    )