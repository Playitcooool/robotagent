import argparse
import json
import time
import urllib.request
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def http_post_json(url: str, payload: dict, headers: dict):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=300)


def http_post_auth(base_url: str, endpoint: str, payload: dict):
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"Content-Type": "application/json"}
    with http_post_json(url, payload, headers) as resp:
        return json.loads(resp.read().decode("utf-8"))


def login_or_register(base_url: str, username: str, password: str, register: bool):
    if register:
        return http_post_auth(base_url, "/api/auth/register", {"username": username, "password": password})
    return http_post_auth(base_url, "/api/auth/login", {"username": username, "password": password})


def parse_tool_name(status_text: str):
    prefix = "正在执行工具："
    if status_text.startswith(prefix):
        return status_text[len(prefix) :].strip()
    if "联网搜索中" in status_text:
        return "web_search"
    return ""


def run_one_prompt(base_url: str, token: str, prompt: str, session_id: str, enabled_tools: list[str]):
    url = f"{base_url.rstrip('/')}/api/chat/send"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {"message": prompt, "session_id": session_id, "enabled_tools": enabled_tools}
    send_ts = time.time()
    events = []
    text_by_source = {"main": "", "analysis": "", "simulator": ""}
    tool_names = []
    references = []
    usage = {}

    with http_post_json(url, payload, headers) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj["timestamp"] = time.time()
            events.append(obj)

            etype = str(obj.get("type", "")).lower()
            if etype == "delta":
                source = str(obj.get("source", "main") or "main").lower()
                text_by_source[source] = text_by_source.get(source, "") + str(obj.get("text", ""))
            elif etype == "status":
                tool_name = parse_tool_name(str(obj.get("text", "")).strip())
                if tool_name:
                    tool_names.append(tool_name)
            elif etype == "web_search_results":
                for item in obj.get("results") or []:
                    title = str(item.get("title") or item.get("url") or "").strip()
                    url = str(item.get("url") or "").strip()
                    if title and url:
                        references.append(f"[{title}]({url})")
            elif etype == "usage":
                usage = obj.get("usage") or usage

    final_answer = text_by_source.get("main") or ""
    if not final_answer:
        # fallback to the longest source if main has no output
        final_answer = max(text_by_source.values(), key=len, default="")

    return {
        "send_ts": send_ts,
        "answer": final_answer.strip(),
        "references": references,
        "tool_names": tool_names,
        "events": events,
        "token_usage": usage,
        "sources": text_by_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate predictions by calling backend /api/chat/send.")
    parser.add_argument("--input", required=True, help="JSONL with id/prompt")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--auth-token", default="")
    parser.add_argument("--username", default="exp_runner")
    parser.add_argument("--password", default="exp_runner_12345")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--enable-web-search", action="store_true")
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]

    token = args.auth_token
    if not token:
        auth = login_or_register(args.api_base, args.username, args.password, args.register)
        token = auth.get("token", "")
        if not token:
            raise RuntimeError("Failed to obtain auth token from backend.")

    enabled_tools = ["web_search"] if args.enable_web_search else []

    outputs = []
    for idx, row in enumerate(rows):
        prompt = f"{args.prompt_prefix}{row.get('prompt', '')}{args.prompt_suffix}"
        session_id = f"exp_{int(time.time() * 1000)}_{idx}"
        result = run_one_prompt(args.api_base, token, prompt, session_id, enabled_tools)
        outputs.append(
            {
                "id": row.get("id", f"row-{idx}"),
                "prompt": row.get("prompt", ""),
                "answer": result["answer"],
                "references": result["references"],
                "tool_names": result["tool_names"],
                "events": result["events"],
                "token_usage": result["token_usage"],
                "sources": result["sources"],
            }
        )

    dump_jsonl(Path(args.out), outputs)


if __name__ == "__main__":
    main()
