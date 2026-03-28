from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.tools import tool

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

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

TAVILY_API_KEY = os.environ.get(
    "TAVILY_API_KEY",
    "",
)
# Load from config.yml if not set in environment
if not TAVILY_API_KEY:
    try:
        import yaml
        _cfg_path = REPO_ROOT / "config" / "config.yml"
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.load(_f.read(), Loader=yaml.FullLoader) or {}
        TAVILY_API_KEY = _cfg.get("tavily", {}).get("api_key", "")
    except Exception:
        pass
_tavily_client = None

_qdrant_client = None
_embedder = None
_corpus_cache: List[Dict[str, Any]] = []
_corpus_cache_at: float = 0.0
_tfidf_vectorizer = None
_tfidf_matrix = None

CORPUS_CACHE_TTL_SECONDS = int(os.environ.get("RAG_CORPUS_CACHE_TTL_SECONDS", "600"))

ROBOTICS_TERMS = [
    "robotics",
    "manipulation",
    "motion planning",
    "control",
    "trajectory optimization",
    "simulation",
]
SIM_TERMS = [
    "pybullet",
    "simulation task",
    "action sequence",
    "constraint",
    "reward design",
]
STOPWORDS_EN = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "what",
    "how",
    "why",
    "is",
    "are",
    "be",
    "this",
    "that",
    "please",
    "about",
    "from",
}
STOPWORDS_ZH = {
    "什么",
    "怎么",
    "如何",
    "以及",
    "关于",
    "问题",
    "一下",
    "哪些",
    "还有",
    "请问",
}


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


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    en = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text.lower())
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    mixed = en + zh
    return [t for t in mixed if t]


def _keyword_tokens(query: str, max_terms: int = 10) -> List[str]:
    terms = _tokenize(query)
    filtered = []
    for t in terms:
        if t in STOPWORDS_EN or t in STOPWORDS_ZH:
            continue
        filtered.append(t)
    if not filtered:
        return terms[:max_terms]
    return filtered[:max_terms]


def _query_profile(query: str) -> Dict[str, Any]:
    q = (query or "").lower()
    is_robotics = any(
        k in q
        for k in (
            "robot",
            "机器人",
            "manipulation",
            "grasp",
            "抓取",
            "motion planning",
            "路径规划",
            "control",
            "控制",
        )
    )
    is_sim = any(
        k in q
        for k in (
            "simulate",
            "simulation",
            "仿真",
            "pybullet",
            "simulator",
            "执行",
            "任务规划",
            "trajectory",
            "轨迹",
        )
    )
    is_cn = bool(re.search(r"[\u4e00-\u9fff]", query or ""))
    return {
        "robotics_related": is_robotics,
        "simulation_related": is_sim,
        "contains_chinese": is_cn,
    }


def _rewrite_queries(query: str, max_variants: int = 5) -> List[Dict[str, Any]]:
    base = " ".join((query or "").strip().split())
    if not base:
        return []

    profile = _query_profile(base)
    keywords = _keyword_tokens(base, max_terms=8)
    variants: List[Dict[str, Any]] = [
        {"name": "original", "query": base, "weight": 1.25}
    ]

    if keywords:
        variants.append(
            {
                "name": "keyword_focus",
                "query": " ".join(keywords),
                "weight": 1.1,
            }
        )

    if profile["robotics_related"] or len(keywords) < 5:
        variants.append(
            {
                "name": "robotics_expand",
                "query": f"{base} {' '.join(ROBOTICS_TERMS)}",
                "weight": 1.0,
            }
        )

    if profile["simulation_related"]:
        variants.append(
            {
                "name": "simulation_expand",
                "query": f"{base} {' '.join(SIM_TERMS)}",
                "weight": 1.0,
            }
        )

    if profile["contains_chinese"]:
        variants.append(
            {
                "name": "bilingual_expand",
                "query": f"{base} robot manipulation planning control simulation",
                "weight": 0.95,
            }
        )

    dedup = []
    seen = set()
    for v in variants:
        q = v["query"].strip()
        if not q or q in seen:
            continue
        seen.add(q)
        dedup.append(v)
        if len(dedup) >= max_variants:
            break
    return dedup


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


def _load_corpus_from_qdrant(limit_per_scroll: int = 512):
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
    terms = _keyword_tokens(query, max_terms=12)
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


def _build_tfidf_index(corpus: List[Dict[str, Any]]):
    if not TFIDF_AVAILABLE or not corpus:
        return None, None
    texts = [d["text"] for d in corpus]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(texts)
    return vec, matrix


def _tfidf_retrieve(query: str, corpus, top_k: int, vec=None, matrix=None):
    if not corpus:
        return []
    if not TFIDF_AVAILABLE:
        return _keyword_retrieve(query, corpus, top_k)

    try:
        if vec is None or matrix is None:
            vec, matrix = _build_tfidf_index(corpus)
        if vec is None or matrix is None:
            return _keyword_retrieve(query, corpus, top_k)
        q = vec.transform([query])
        sims = cosine_similarity(q, matrix).flatten()

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


def _doc_source(hit: Dict[str, Any]) -> str:
    md = hit.get("metadata") or {}
    source = md.get("source") or md.get("file_name") or md.get("doc_id") or "unknown"
    return str(source)


def _clean_label(text: str, max_len: int = 96) -> str:
    t = " ".join((text or "").strip().split())
    if not t:
        return "Reference"
    if len(t) > max_len:
        return t[: max_len - 3] + "..."
    return t


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith(("http://", "https://")):
        return u
    return ""


def _build_reference_from_item(item: Dict[str, Any]) -> Dict[str, str]:
    md = item.get("metadata") or {}
    title = _clean_label(str(md.get("title") or ""))
    paper_id = str(md.get("paper_id") or "")
    source_name = _clean_label(str(md.get("source") or ""))
    label = title if title and title != "Reference" else (paper_id or source_name or "Reference")

    arxiv_url = _normalize_url(str(md.get("arxiv_url") or ""))
    pdf_url = _normalize_url(str(md.get("pdf_url") or ""))
    if not arxiv_url:
        source_raw = str(md.get("source") or "")
        source_match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", source_raw)
        if source_match:
            arxiv_url = f"https://arxiv.org/abs/{source_match.group(1)}"
    url = arxiv_url or pdf_url
    markdown = f"[{label}]({url})" if url else label
    return {"label": label, "url": url, "markdown": markdown}


def _weighted_rrf_merge(
    retrieval_groups: List[Dict[str, Any]],
    top_k: int,
    max_per_source: int = 3,
):
    c = 60
    scores = {}
    items = {}

    for group in retrieval_groups:
        ranked = group["hits"]
        route_weight = float(group.get("route_weight", 1.0))
        query_weight = float(group.get("query_weight", 1.0))
        query_variant = group.get("query_variant", "unknown")
        for rank, hit in enumerate(ranked, start=1):
            key = hit.get("id") or hit.get("text", "")[:240]
            if not key:
                continue
            contribution = (route_weight * query_weight) / (c + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            if key not in items:
                items[key] = dict(hit)
            route_set = items[key].get("routes", set())
            if not isinstance(route_set, set):
                route_set = set(route_set)
            route_set.add(hit.get("source", "unknown"))
            items[key]["routes"] = route_set

            variant_set = items[key].get("query_variants", set())
            if not isinstance(variant_set, set):
                variant_set = set(variant_set)
            variant_set.add(query_variant)
            items[key]["query_variants"] = variant_set

    merged = []
    for key, score in scores.items():
        item = dict(items[key])
        item["rrf_score"] = score
        item["routes"] = sorted(item.get("routes", []))
        item["query_variants"] = sorted(item.get("query_variants", []))
        item["doc_source"] = _doc_source(item)
        merged.append(item)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    if max_per_source < 1:
        return merged[:top_k]

    diversified = []
    source_counts: Dict[str, int] = {}
    overflow = []
    for item in merged:
        src = item.get("doc_source", "unknown")
        if source_counts.get(src, 0) >= max_per_source:
            overflow.append(item)
            continue
        source_counts[src] = source_counts.get(src, 0) + 1
        diversified.append(item)
        if len(diversified) >= top_k:
            return diversified

    for item in overflow:
        diversified.append(item)
        if len(diversified) >= top_k:
            break
    return diversified


def _get_corpus_cached(refresh: bool = False):
    global _corpus_cache, _corpus_cache_at, _tfidf_vectorizer, _tfidf_matrix
    now = time.time()
    cache_valid = (
        bool(_corpus_cache)
        and not refresh
        and (now - _corpus_cache_at) < max(30, CORPUS_CACHE_TTL_SECONDS)
    )
    if cache_valid:
        return _corpus_cache, _tfidf_vectorizer, _tfidf_matrix, True

    corpus = _load_corpus_from_qdrant()
    vec, matrix = _build_tfidf_index(corpus) if corpus else (None, None)
    _corpus_cache = corpus
    _corpus_cache_at = now
    _tfidf_vectorizer = vec
    _tfidf_matrix = matrix
    return _corpus_cache, _tfidf_vectorizer, _tfidf_matrix, False


def _simulation_hints(query: str, merged_results: List[Dict[str, Any]]) -> List[str]:
    q = (query or "").lower()
    merged_text = " ".join((m.get("text") or "")[:240] for m in merged_results).lower()
    text = f"{q} {merged_text}"
    hints = []

    if any(k in text for k in ["grasp", "抓取", "pick", "manipulation"]):
        hints.append("仿真可先固定抓取目标姿态与接触参数，再搜索可行抓取位姿。")
    if any(k in text for k in ["trajectory", "path", "轨迹", "路径规划", "motion planning"]):
        hints.append("执行前应分离全局路径与局部控制，优先验证轨迹平滑性和碰撞约束。")
    if any(k in text for k in ["control", "控制", "policy", "rl", "强化学习"]):
        hints.append("建议记录控制频率、动作裁剪范围和奖励项权重，便于复现实验。")
    if any(k in text for k in ["sim2real", "domain randomization", "迁移"]):
        hints.append("可加入质量、摩擦、传感噪声随机化，以提升策略迁移稳定性。")
    if any(k in text for k in ["multi-agent", "coordination", "协作"]):
        hints.append("多体任务建议先做单体基线，再逐步增加协作约束与通信机制。")

    return hints[:5]


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


def _get_tavily_client():
    """Lazy init Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY not set in environment")
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def _web_search_impl(query: str, max_results: int = 5, timeout: float = 8.0) -> str:
    """Core web search implementation (no decorator)."""
    q = " ".join((query or "").split())
    if not q:
        return json.dumps({"query": "", "results": [], "error": "query is required"}, ensure_ascii=False, indent=2)
    limit = max(1, min(max_results, 10))

    try:
        client = _get_tavily_client()
        results_raw = client.search(
            query=q,
            max_results=limit,
            timeout_seconds=timeout,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception as e:
        return json.dumps(
            {"query": q, "results": [], "error": f"Tavily search failed: {e}"},
            ensure_ascii=False,
            indent=2,
        )

    results = []
    for item in (results_raw.get("results") or [])[:limit]:
        results.append({
            "title": item.get("title", "Unknown"),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "source": "tavily",
        })

    # Build citations
    citations = []
    for i, r in enumerate(results, 1):
        citations.append(f"[{i}] {r.get('title', 'Unknown')}")
        citations.append(f"    URL: {r.get('url', 'N/A')}")
        citations.append(f"    来源: {r.get('source', 'tavily')}")

    answer = results_raw.get("answer", "")

    payload = {
        "query": q,
        "engine": "tavily",
        "returned": len(results),
        "results": results,
        "answer": answer,
        "citations": "\n".join(citations),
    }
    if not results:
        payload["warning"] = "no results from Tavily"
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(response_format="content")
def web_search(query: str, max_results: int = 5, timeout: float = 8.0) -> str:
    """
    Search the web for recent/general information using Tavily.

    IMPORTANT: Every factual claim in your answer must cite a source from results
    using [number] notation, e.g. [1], [2]. Always include a reference list at the end.
    """
    return _web_search_impl(query=query, max_results=max_results, timeout=timeout)


@tool(response_format="content")
def academic_search(query: str, max_results: int = 5, timeout: float = 15.0) -> str:
    """
    Search academic papers from OpenAlex and arXiv.

    IMPORTANT: Every factual claim in your answer must cite a source from results
    using [number] notation, e.g. [1], [2]. Always include a reference list at the end.
    """
    return _academic_search_impl(query=query, max_results=max_results, timeout=timeout)


def _academic_search_impl(query: str, max_results: int = 5, timeout: float = 15.0) -> str:
    """Core academic search implementation (no decorator)."""
    q = " ".join((query or "").split())
    if not q:
        return "Error: query is required."

    limit = max(1, min(max_results, 10))
    all_results = []

    # 1. Search OpenAlex API (published papers)
    openalex_endpoint = "https://api.openalex.org/works"
    openalex_params = {
        "search": q,
        "per_page": limit,
        "sort": "relevance_score:desc",
        "filter": "type:paper",
    }

    try:
        resp = requests.get(openalex_endpoint, params=openalex_params, timeout=timeout)
        resp.raise_for_status()
        openalex_data = resp.json()

        for work in openalex_data.get("results", []):
            # Extract authors
            authors = work.get("authorships", [])[:3]
            author_names = ", ".join([a.get("author", {}).get("display_name", "Unknown") for a in authors])

            # Get publication year
            year = work.get("publication_year", "N/A")

            # Get venue
            venue_data = work.get("host_venue", {})
            venue = venue_data.get("display_name", "") or venue_data.get("publisher", "")

            # Get DOI and URL
            doi = work.get("doi", "")
            url = work.get("doi", "")
            if not url:
                url = work.get("id", "")

            # Get abstract
            abstract = work.get("abstract", "") or ""
            if abstract:
                abstract = abstract[:500] + ("..." if len(abstract) > 500 else "")

            # Get citation count
            cited_by_count = work.get("cited_by_count", 0)

            # Try to get PDF URL from open access
            pdf_url = ""
            oa = work.get("open_access", {})
            if oa.get("is_oa", False):
                pdf_url = oa.get("pdf_url", "")

            # Check if also on arXiv
            arxiv_id = ""
            ids = work.get("ids", {})
            if "arxiv" in ids:
                arxiv_id = ids.get("arxiv", "")
                if not pdf_url:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            all_results.append({
                "type": "paper",
                "title": work.get("title", "Unknown Title"),
                "authors": author_names,
                "year": year,
                "venue": venue,
                "abstract": abstract,
                "url": url,
                "pdf_url": pdf_url,
                "citations": cited_by_count,
                "arxiv_id": arxiv_id,
                "source": "openalex",
            })
    except Exception as e:
        all_results.append({
            "type": "error",
            "source": "openalex",
            "error": f"OpenAlex search failed: {e}"
        })

    # 2. Search arXiv API as supplement
    if len(all_results) < limit:
        arxiv_limit = limit - len(all_results)
        arxiv_endpoint = "http://export.arxiv.org/api/query"
        arxiv_params = {
            "search_query": f"all:{q}",
            "start": 0,
            "max_results": arxiv_limit,
            "sortBy": "relevance",
        }

        try:
            resp = requests.get(arxiv_endpoint, params=arxiv_params, timeout=timeout)
            resp.raise_for_status()

            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)

            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns)[:arxiv_limit]:
                title = entry.find('atom:title', ns).text or "Unknown"
                summary = entry.find('atom:summary', ns).text or ""
                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                author_names = ", ".join(authors[:3])

                # Get arXiv ID and PDF
                arxiv_id = ""
                pdf_url = ""
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href', '')
                        arxiv_id = pdf_url.split('/')[-1].replace('.pdf', '')
                        break

                # Extract year from published date
                published = entry.find('atom:published', ns).text or ""
                year = published[:4] if published else "N/A"

                # Check if already added (avoid duplicates)
                is_dup = any(r.get('title', '').lower() == title.lower() for r in all_results)

                if not is_dup:
                    all_results.append({
                        "type": "paper",
                        "title": title.strip(),
                        "authors": author_names,
                        "year": year,
                        "venue": "arXiv",
                        "abstract": summary[:500] + ("..." if len(summary) > 500 else ""),
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "pdf_url": pdf_url,
                        "citations": 0,
                        "arxiv_id": arxiv_id,
                        "source": "arxiv",
                    })
        except Exception as e:
            all_results.append({
                "type": "error",
                "source": "arxiv",
                "error": f"arXiv search failed: {e}"
            })

    # Build citations for traceability
    citations = []
    for i, r in enumerate(all_results, 1):
        if r.get("type") == "error":
            continue
        title = r.get("title", "Unknown")[:80]
        year = r.get("year", "N/A")
        citations.append(f"[{i}] {title} ({year})")
        citations.append(f"    作者: {r.get('authors', 'Unknown')}")
        if r.get("url"):
            citations.append(f"    URL: {r.get('url')}")
        if r.get("pdf_url"):
            citations.append(f"    PDF: {r.get('pdf_url')}")
        citations.append(f"    来源: {r.get('source', 'unknown')}")

    payload = {
        "query": q,
        "returned": len([r for r in all_results if r.get("type") != "error"]),
        "results": all_results,
        "citations": "\n".join(citations) if citations else "No results found",
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


# Academic keywords - queries containing these are routed to academic_search
_ACADEMIC_KEYWORDS = {
    "paper", "arxiv", "publication", "conference", "journal", "doi",
    "cite", "citation", "research", "thesis", "dissertation",
    "author", "author:", "by ", "icra", "iros", "rss", "coRL",
    "ieee", "acm", "springer", "elsevier", "arxiv.org",
    "proposed method", "we propose", "this paper", "our work",
    "methodology", "experiments show", "we show",
    "year:", "vol.", "volume", "proceedings",
}

_WEB_KEYWORDS = {
    "news", "event", "release", "announce", "2024", "2025", "2026",
    "latest", "recent", "today", "yesterday", "breaking",
    "price", "buy", "amazon", "shop", "product",
    "weather", "stock", "currency", "exchange rate",
}


def _should_use_academic(query: str) -> bool:
    """Route query to academic or web search based on keywords."""
    q = query.lower()
    academic_score = sum(1 for kw in _ACADEMIC_KEYWORDS if kw in q)
    web_score = sum(1 for kw in _WEB_KEYWORDS if kw in q)
    # Explicit academic markers
    if any(m in q for m in ["arxiv:", "doi:", "paper on", "paper about", "论文", "发表", "学术"]):
        return True
    return academic_score > web_score


@tool(response_format="content")
def search(query: str, max_results: int = 5, timeout: float = 15.0) -> str:
    """
    Unified search tool that automatically routes queries to the most appropriate engine:
    - Academic search (OpenAlex + arXiv) for research papers, publications, methodologies
    - Web search (Tavily) for news, products, current events, general knowledge

    IMPORTANT: Every factual claim in your answer must cite a source from results
    using [number] notation, e.g. [1], [2]. Always include a reference list at the end.

    Returns structured results with title, authors, year, URL, snippet, and source.
    """
    q = " ".join((query or "").split())
    if not q:
        return json.dumps({"query": "", "results": [], "error": "query is required"}, ensure_ascii=False, indent=2)

    limit = max(1, min(max_results, 10))

    if _should_use_academic(q):
        engine = "academic (openalex + arxiv)"
        raw = _academic_search_impl(query=q, max_results=limit, timeout=timeout)
        try:
            data = json.loads(raw)
            results = []
            for item in data.get("results", []):
                if item.get("type") == "error":
                    continue
                results.append({
                    "type": item.get("type", "paper"),
                    "title": item.get("title", "Unknown"),
                    "authors": item.get("authors", ""),
                    "year": item.get("year", ""),
                    "venue": item.get("venue", ""),
                    "abstract": item.get("abstract", ""),
                    "url": item.get("url", ""),
                    "pdf_url": item.get("pdf_url", ""),
                    "citations": item.get("citations", 0),
                    "arxiv_id": item.get("arxiv_id", ""),
                    "source": item.get("source", "academic"),
                })
        except Exception:
            results = []
    else:
        engine = "web (tavily)"
        raw = _web_search_impl(query=q, max_results=limit, timeout=timeout)
        try:
            data = json.loads(raw)
            if data.get("error"):
                return json.dumps(
                    {"query": q, "results": [], "error": data["error"], "engine": engine},
                    ensure_ascii=False, indent=2
                )
            results = []
            for item in data.get("results", []):
                results.append({
                    "type": "web",
                    "title": item.get("title", "Unknown"),
                    "authors": "",
                    "year": "",
                    "venue": item.get("source", ""),
                    "abstract": item.get("snippet", ""),
                    "url": item.get("url", ""),
                    "pdf_url": "",
                    "citations": 0,
                    "arxiv_id": "",
                    "source": "tavily",
                })
        except Exception:
            results = []

    # Build citations
    citations = []
    for i, r in enumerate(results[:limit], 1):
        title = r.get("title", "Unknown")[:80]
        year = r.get("year", "N/A")
        src = r.get("source", "unknown")
        if r.get("type") == "paper":
            citations.append(f"[{i}] {title} ({year})")
            if r.get("authors"):
                citations.append(f"    作者: {r.get('authors')}")
        else:
            citations.append(f"[{i}] {title}")
        if r.get("url"):
            citations.append(f"    URL: {r.get('url')}")
        if r.get("pdf_url"):
            citations.append(f"    PDF: {r.get('pdf_url')}")
        citations.append(f"    来源: {src}")

    payload = {
        "query": q,
        "engine": engine,
        "returned": len(results),
        "results": results[:limit],
        "citations": "\n".join(citations) if citations else "No results found",
    }
    if not results:
        payload["warning"] = "no results found"

    return json.dumps(payload, ensure_ascii=False, indent=2)


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


__all__ = [
    "academic_search",
    "current_time",
    "format_json",
    "http_get",
    "list_workspace_files",
    # "qdrant_retrieve_context",  # disabled: HuggingFace network timeout
    "read_workspace_file",
    "search",
    "search_workspace_text",
    "web_search",
]
