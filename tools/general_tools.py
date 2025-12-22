from langchain_core.tools import tool
from langchain_core.embeddings import embeddings
from langchain_core.documents import Document

import requests
import os
import json
import re
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
from typing import List, Tuple, Optional


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    query = embeddings.embed_query(query)
    retrieved_docs = client.query_points(query=query, 
                                         collection_name='markdown_chunks',
                                         limit=3)

    documents = []
    serialized_parts = []

    # Qdrant's response may come in different shapes (tuple, dict, or object).
    # Normalize to an iterable of scored points called `points`.
    points = None
    if isinstance(retrieved_docs, dict):
        points = retrieved_docs.get("points") or retrieved_docs.get("result") or []
    elif isinstance(retrieved_docs, (tuple, list)) and len(retrieved_docs) >= 2 and isinstance(retrieved_docs[1], list):
        # e.g. ('points', [ScoredPoint(...), ...])
        points = retrieved_docs[1]
    elif hasattr(retrieved_docs, "points"):
        points = getattr(retrieved_docs, "points")
    elif hasattr(retrieved_docs, "result"):
        points = getattr(retrieved_docs, "result")
    else:
        # Fallback: assume it's already an iterable of points
        points = retrieved_docs

    for doc in points:
        # ScoredPoint objects expose `.payload`; sometimes doc can be a tuple/indexed structure.
        if hasattr(doc, "payload"):
            metadata = doc.payload or {}
        elif isinstance(doc, (tuple, list)) and len(doc) >= 2 and hasattr(doc[1], "payload"):
            metadata = doc[1].payload or {}
        elif isinstance(doc, dict):
            metadata = doc.get("payload") or {}
        else:
            metadata = {}

        text = ""
        if isinstance(metadata, dict):
            text = metadata.get("text", "")
        elif isinstance(metadata, str):
            text = metadata

        if not text:
            print("[WARN] No text in metadata:", metadata)
            continue

        documents.append(Document(page_content=text, metadata=metadata))
        serialized_parts.append(f"Source: {metadata}\nContent: {text}")

    serialized = "\n\n".join(serialized_parts)

    if not serialized:
        print("[WARN] No chunks retrieved for query:", query)

    return serialized, documents


# ---------------------------
# General-purpose agent tools
# ---------------------------

@tool(response_format="content")
def http_get(url: str, max_chars: int = 2000, timeout: float = 5.0):
    """Perform a safe HTTP GET request and return status, headers, and a body snippet.

    Only http/https URLs are allowed. Returns a dict-like string.
    """
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return "Error: only http/https URLs are allowed."

    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return f"Error fetching URL: {e}"

    headers = dict(resp.headers)
    body = resp.text or ""
    snippet = body[:max_chars]
    truncated = len(body) > max_chars

    result = {
        "url": url,
        "status_code": resp.status_code,
        "headers": {k: v for k, v in list(headers.items())[:20]},
        "body_snippet": snippet,
        "truncated": truncated,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(response_format="content")
def read_workspace_file(path: str, max_chars: int = 10000):
    """Read a text file under the repository workspace safely.

    The file path must be inside the repository root (parent of `tools/`). Binary data is handled safely and returned as replaced text.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = os.path.abspath(path)

    # if the user provided a relative path, resolve it relative to repo root
    if not os.path.isabs(path):
        candidate = os.path.abspath(os.path.join(repo_root, path))

    try:
        common = os.path.commonpath([repo_root, candidate])
    except Exception:
        return "Error resolving paths."

    if common != repo_root:
        return "Error: access denied. Path outside repository root."

    if not os.path.exists(candidate):
        return "Error: file does not exist."

    if os.path.isdir(candidate):
        return "Error: path is a directory."

    try:
        with open(candidate, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    truncated = len(content) > max_chars
    snippet = content[:max_chars]

    return json.dumps({"path": os.path.relpath(candidate, repo_root), "content": snippet, "truncated": truncated}, ensure_ascii=False, indent=2)


@tool(response_format="content")
def format_json(json_str: str):
    """Validate and pretty-format JSON input."""
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        return f"Error parsing JSON: {e}"

    return json.dumps(parsed, ensure_ascii=False, indent=2)


@tool(response_format="content")
def extract_emails(text: str, max_results: int = 100):
    """Extract email addresses from text and return a de-duplicated list."""
    if not text:
        return "[]"

    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    found = re.findall(pattern, text)
    unique = list(dict.fromkeys(found))[:max_results]
    return json.dumps(unique, ensure_ascii=False)


@tool(response_format="content")
def summarize_text(text: str, max_sentences: int = 5):
    """Return a very lightweight extractive summary: first N sentences.

    This is intentionally simple — for high-quality summaries integrate an LLM summarizer.
    """
    if not text:
        return ""

    # Split into sentences with a simple regex that handles common punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        summary = " ".join(sentences)
        return summary

    summary = " ".join(sentences[:max_sentences])
    return summary + ("\n\n[Note: truncated to first %d sentences]" % max_sentences)


@tool(response_format="content")
def current_time(tz: Optional[str] = "UTC"):
    """Return current timestamp in ISO format for a timezone (default UTC)."""
    try:
        if tz and ZoneInfo is not None:
            dt = datetime.now(tz=ZoneInfo(tz))
        elif tz and tz.upper() in ("UTC", "Z"):
            dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now()
    except Exception:
        dt = datetime.now()

    return dt.isoformat()
