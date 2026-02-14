from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from langchain_core.tools import tool

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    TFIDF_AVAILABLE = True
except Exception:
    TFIDF_AVAILABLE = False

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


REPO_ROOT = Path(__file__).resolve().parent.parent

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "markdown_chunks")
QDRANT_EMBED_MODEL = os.environ.get(
    "QDRANT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_qdrant_client = None
_embedder = None


def _normalize_qdrant_response(retrieved):
    if isinstance(retrieved, dict):
        return retrieved.get("points") or retrieved.get("result") or []
    if isinstance(retrieved, (tuple, list)) and len(retrieved) >= 2 and isinstance(
        retrieved[1], list
    ):
        return retrieved[1]
    if hasattr(retrieved, "points"):
        return getattr(retrieved, "points")
    if hasattr(retrieved, "result"):
        return getattr(retrieved, "result")
    return retrieved or []


def _extract_payload(point):
    if hasattr(point, "payload"):
        payload = point.payload or {}
    elif isinstance(point, dict):
        payload = point.get("payload") or {}
    elif isinstance(point, (tuple, list)) and len(point) >= 2 and hasattr(
        point[1], "payload"
    ):
        payload = point[1].payload or {}
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {"text": str(payload)}

    text = (
        payload.get("text")
        or payload.get("content")
        or payload.get("chunk")
        or payload.get("page_content")
        or ""
    )
    if not isinstance(text, str):
        text = str(text)

    doc_id = None
    if hasattr(point, "id"):
        doc_id = getattr(point, "id")
    elif isinstance(point, dict):
        doc_id = point.get("id")

    score = None
    if hasattr(point, "score"):
        score = getattr(point, "score")
    elif isinstance(point, dict):
        score = point.get("score")

    return {
        "id": doc_id,
        "text": text,
        "metadata": payload,
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
    }


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    if QdrantClient is None:
        raise RuntimeError("qdrant_client not installed")
    _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("langchain_huggingface not installed")
    _embedder = HuggingFaceEmbeddings(model_name=QDRANT_EMBED_MODEL)
    return _embedder


def _vector_retrieve(query: str, top_k: int):
    try:
        client = _get_qdrant_client()
        vec = _get_embedder().embed_query(query)
        resp = client.query_points(
            query=vec,
            collection_name=QDRANT_COLLECTION,
            limit=top_k,
            with_payload=True,
        )
        points = _normalize_qdrant_response(resp)
        hits = []
        for point in points:
            item = _extract_payload(point)
            if item["text"]:
                item["source"] = "vector"
                hits.append(item)
        return hits
    except Exception:
        return []


def _load_corpus(limit_per_scroll: int = 512):
    try:
        client = _get_qdrant_client()
    except Exception:
        return []

    all_docs = []
    offset = None
    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                with_payload=True,
                with_vectors=False,
                limit=limit_per_scroll,
                offset=offset,
            )
            if not points:
                break
            for point in points:
                item = _extract_payload(point)
                if item["text"]:
                    all_docs.append(item)
            offset = next_offset
            if offset is None:
                break
    except Exception:
        return []
    return all_docs


def _keyword_retrieve(query: str, corpus, top_k: int):
    terms = [t.lower() for t in re.findall(r"\w+", query)]
    if not terms:
        return []

    scored = []
    for doc in corpus:
        text = doc["text"].lower()
        score = sum(text.count(t) for t in terms)
        if score > 0:
            item = dict(doc)
            item["score"] = float(score)
            item["source"] = "keyword"
            scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _tfidf_retrieve(query: str, corpus, top_k: int):
    if not corpus:
        return []
    if not TFIDF_AVAILABLE:
        return _keyword_retrieve(query, corpus, top_k)

    try:
        texts = [d["text"] for d in corpus]
        vec = TfidfVectorizer(stop_words="english")
        x = vec.fit_transform(texts)
        q = vec.transform([query])
        sims = cosine_similarity(q, x).flatten()

        idx = sims.argsort()[::-1][:top_k]
        hits = []
        for i in idx:
            score = float(sims[i])
            if score <= 0:
                continue
            item = dict(corpus[int(i)])
            item["score"] = score
            item["source"] = "tfidf"
            hits.append(item)
        return hits
    except Exception:
        return _keyword_retrieve(query, corpus, top_k)


def _rrf_merge(*ranked_lists, top_k: int):
    # Reciprocal Rank Fusion for robust multi-route retrieval.
    c = 60
    scores = {}
    items = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = hit.get("id") or hit.get("text", "")[:240]
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
            if key not in items:
                items[key] = dict(hit)
            srcs = items[key].get("sources", set())
            if not isinstance(srcs, set):
                srcs = set(srcs)
            srcs.add(hit.get("source", "unknown"))
            items[key]["sources"] = srcs

    merged = []
    for key, score in scores.items():
        item = dict(items[key])
        item["rrf_score"] = score
        item["sources"] = sorted(item.get("sources", []))
        merged.append(item)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k]


def _resolve_repo_path(path: str) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (REPO_ROOT / raw)
    candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT)
    except Exception as e:
        raise ValueError("access denied: path outside repository root") from e

    return candidate


@tool(response_format="content")
def list_workspace_files(glob_pattern: str = "**/*", max_results: int = 200) -> str:
    """List files under the workspace with an optional glob pattern."""
    safe_limit = max(1, min(max_results, 2000))
    files = []

    for p in REPO_ROOT.glob(glob_pattern):
        if p.is_file():
            files.append(str(p.relative_to(REPO_ROOT)))
            if len(files) >= safe_limit:
                break

    return json.dumps(
        {
            "root": str(REPO_ROOT),
            "glob_pattern": glob_pattern,
            "count": len(files),
            "files": files,
            "truncated": len(files) >= safe_limit,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool(response_format="content")
def read_workspace_file(path: str, max_chars: int = 12000) -> str:
    """Read a text file under the repository workspace safely."""
    if not path:
        return "Error: path is required."

    try:
        candidate = _resolve_repo_path(path)
    except Exception as e:
        return f"Error: {e}"

    if not candidate.exists():
        return "Error: file does not exist."
    if candidate.is_dir():
        return "Error: path is a directory."

    try:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    safe_max = max(1, max_chars)
    snippet = content[:safe_max]
    truncated = len(content) > safe_max

    return json.dumps(
        {
            "path": str(candidate.relative_to(REPO_ROOT)),
            "content": snippet,
            "truncated": truncated,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool(response_format="content")
def search_workspace_text(
    pattern: str,
    glob_pattern: str = "**/*",
    max_results: int = 100,
    max_line_chars: int = 300,
) -> str:
    """Search for plain text in workspace files and return matched lines."""
    if not pattern:
        return "Error: pattern is required."

    safe_limit = max(1, min(max_results, 1000))
    safe_line_chars = max(20, max_line_chars)
    matches = []

    for p in REPO_ROOT.glob(glob_pattern):
        if not p.is_file():
            continue
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    if pattern in line:
                        matches.append(
                            {
                                "path": str(p.relative_to(REPO_ROOT)),
                                "line": line_no,
                                "text": line.rstrip("\n")[:safe_line_chars],
                            }
                        )
                        if len(matches) >= safe_limit:
                            return json.dumps(
                                {
                                    "pattern": pattern,
                                    "count": len(matches),
                                    "matches": matches,
                                    "truncated": True,
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
        except Exception:
            continue

    return json.dumps(
        {
            "pattern": pattern,
            "count": len(matches),
            "matches": matches,
            "truncated": False,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool(response_format="content")
def http_get(url: str, max_chars: int = 3000, timeout: float = 8.0) -> str:
    """Perform a safe HTTP GET request and return status, headers, and body snippet."""
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return "Error: only http/https URLs are allowed."

    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return f"Error fetching URL: {e}"

    body = resp.text or ""
    safe_max = max(1, max_chars)
    snippet = body[:safe_max]

    result = {
        "url": url,
        "status_code": resp.status_code,
        "headers": {k: v for k, v in list(dict(resp.headers).items())[:20]},
        "body_snippet": snippet,
        "truncated": len(body) > safe_max,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool(response_format="content")
def format_json(json_str: str, sort_keys: bool = False) -> str:
    """Validate and pretty-format JSON input."""
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        return f"Error parsing JSON: {e}"

    return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=sort_keys)


@tool(response_format="content")
def current_time(tz: Optional[str] = "UTC") -> str:
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


@tool(response_format="content")
def qdrant_retrieve_context(
    query: str,
    top_k: int = 8,
    route_k: int = 8,
    include_content: bool = True,
) -> str:
    """Agentic RAG retrieval from Qdrant with multi-route recall and fusion ranking."""
    if not query or not query.strip():
        return "Error: query is required."

    safe_top_k = max(1, min(top_k, 20))
    safe_route_k = max(safe_top_k, min(route_k, 50))

    vector_hits = _vector_retrieve(query, safe_route_k)
    corpus = _load_corpus()
    tfidf_hits = _tfidf_retrieve(query, corpus, safe_route_k) if corpus else []
    keyword_hits = _keyword_retrieve(query, corpus, safe_route_k) if corpus else []
    merged = _rrf_merge(vector_hits, tfidf_hits, keyword_hits, top_k=safe_top_k)

    results = []
    for item in merged:
        payload = {
            "id": item.get("id"),
            "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
            "routes": item.get("sources", []),
            "metadata": item.get("metadata", {}),
        }
        if include_content:
            payload["content"] = item.get("text", "")
        results.append(payload)

    return json.dumps(
        {
            "query": query,
            "collection": QDRANT_COLLECTION,
            "qdrant": {"host": QDRANT_HOST, "port": QDRANT_PORT},
            "embed_model": QDRANT_EMBED_MODEL,
            "route_hits": {
                "vector": len(vector_hits),
                "tfidf": len(tfidf_hits),
                "keyword": len(keyword_hits),
            },
            "returned": len(results),
            "results": results,
            "warnings": (
                []
                if results
                else [
                    "no documents retrieved; check Qdrant service, collection name, or embedding model"
                ]
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


__all__ = [
    "current_time",
    "format_json",
    "http_get",
    "list_workspace_files",
    "qdrant_retrieve_context",
    "read_workspace_file",
    "search_workspace_text",
]
