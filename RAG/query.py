import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import re

# 连接 Qdrant
client = QdrantClient(host="localhost", port=6333)

# 与写入时一致的 embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 从已有 collection 创建向量库对象
qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="markdown_chunks",
    embedding=embeddings
)

# support modules for keyword/tfidf retrieval
from collections import defaultdict
import math
import heapq

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    TFIDF_AVAILABLE = True
except Exception:
    TFIDF_AVAILABLE = False


@tool(response_format="content_and_artifact")
def retrieve_context(query: str, top_k: int = 6):
    """Multi-strategy retrieval: vector, keyword, and TF-IDF (if available).

    Returns serialized text with source tags and a list of `Document` objects.
    """

    def _normalize_qdrant_response(retrieved):
        points = None
        if isinstance(retrieved, dict):
            points = retrieved.get("points") or retrieved.get("result") or []
        elif isinstance(retrieved, (tuple, list)) and len(retrieved) >= 2 and isinstance(retrieved[1], list):
            points = retrieved[1]
        elif hasattr(retrieved, "points"):
            points = getattr(retrieved, "points")
        elif hasattr(retrieved, "result"):
            points = getattr(retrieved, "result")
        else:
            points = retrieved
        return points

    def _extract_payload_text(point):
        if hasattr(point, "payload"):
            payload = point.payload or {}
        elif isinstance(point, (tuple, list)) and len(point) >= 2 and hasattr(point[1], "payload"):
            payload = point[1].payload or {}
        elif isinstance(point, dict):
            payload = point.get("payload") or {}
        else:
            payload = {}

        text = ""
        if isinstance(payload, dict):
            text = payload.get("text", "")
        elif isinstance(payload, str):
            text = payload
        return payload, text

    def _vector_retrieve(q, limit=top_k):
        try:
            vec = embeddings.embed_query(q)
            retrieved = client.query_points(query=vec, collection_name="markdown_chunks", limit=limit)
        except Exception as e:
            print("[WARN] vector retrieval failed:", e)
            return []

        points = _normalize_qdrant_response(retrieved)
        results = []
        for doc in points:
            payload, text = _extract_payload_text(doc)
            if not text:
                continue
            # try to get an id for dedup purposes
            doc_id = None
            if hasattr(doc, "id"):
                doc_id = getattr(doc, "id")
            elif isinstance(doc, dict) and "id" in doc:
                doc_id = doc.get("id")
            results.append({"id": doc_id, "text": text, "metadata": dict(payload), "score": getattr(doc, "score", None) or 0.0, "source": "vector"})
        return results

    def _load_all_points(limit_per_scroll=500):
        # Read all payloads from collection with scroll; fallback to small scan if that fails
        all_points = []
        offset = 0
        try:
            while True:
                resp = client.scroll(collection_name="markdown_chunks", with_payload=True, limit=limit_per_scroll, offset=offset)
                pts = _normalize_qdrant_response(resp)
                if not pts:
                    break
                for p in pts:
                    payload, text = _extract_payload_text(p)
                    if text:
                        pid = None
                        if isinstance(p, dict) and "id" in p:
                            pid = p.get("id")
                        elif hasattr(p, "id"):
                            pid = getattr(p, "id")
                        all_points.append({"id": pid, "text": text, "metadata": dict(payload)})
                offset += len(pts)
                if len(pts) < limit_per_scroll:
                    break
        except Exception as e:
            print("[WARN] scroll failed or not supported, attempting small fallback:", e)
            # fallback: try a small query that returns many points
            try:
                resp = client.query_points(query=[0.0], collection_name="markdown_chunks", limit=1000)
                pts = _normalize_qdrant_response(resp)
                for p in pts:
                    payload, text = _extract_payload_text(p)
                    if text:
                        pid = None
                        if isinstance(p, dict) and "id" in p:
                            pid = p.get("id")
                        elif hasattr(p, "id"):
                            pid = getattr(p, "id")
                        all_points.append({"id": pid, "text": text, "metadata": dict(payload)})
            except Exception as e2:
                print("[WARN] fallback retrieval failed:", e2)
        return all_points

    def _keyword_retrieve(q, docs, limit=top_k):
        # simple case-insensitive substring matching with frequency scoring
        q_terms = [t.lower() for t in re.findall(r"\w+", q)]
        if not q_terms:
            return []
        scores = []
        for i, d in enumerate(docs):
            text_lower = d["text"].lower()
            score = sum(text_lower.count(t) for t in q_terms)
            if score > 0:
                scores.append((score, i, d))
        scores.sort(reverse=True)
        return [{"id": d["id"], "text": d["text"], "metadata": d.get("metadata", {}), "score": s, "source": "keyword"} for s, _, d in scores[:limit]]

    def _tfidf_retrieve(q, docs, limit=top_k):
        if not TFIDF_AVAILABLE:
            # fallback to keyword
            return _keyword_retrieve(q, docs, limit)
        texts = [d["text"] for d in docs]
        try:
            vect = TfidfVectorizer(stop_words="english")
            X = vect.fit_transform(texts)
            q_vec = vect.transform([q])
            sims = cosine_similarity(q_vec, X).flatten()
            top_idx = sims.argsort()[::-1][:limit]
            results = []
            for idx in top_idx:
                score = float(sims[idx])
                if score <= 0:
                    continue
                d = docs[int(idx)]
                results.append({"id": d["id"], "text": d["text"], "metadata": d.get("metadata", {}), "score": score, "source": "tfidf"})
            return results
        except Exception as e:
            print("[WARN] TF-IDF retrieval failed:", e)
            return _keyword_retrieve(q, docs, limit)

    # perform retrievals
    vector_hits = _vector_retrieve(query, limit=top_k)
    # load corpus once for keyword/tfidf
    corpus = _load_all_points()
    tfidf_hits = _tfidf_retrieve(query, corpus, limit=top_k)
    keyword_hits = _keyword_retrieve(query, corpus, limit=top_k)

    # merge with deduplication, prefer vector then tfidf then keyword
    seen = set()
    merged = []
    for source_hits in (vector_hits, tfidf_hits, keyword_hits):
        for h in source_hits:
            key = h.get("id") or (h.get("text")[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
            if len(merged) >= top_k:
                break
        if len(merged) >= top_k:
            break

    documents = []
    serialized_parts = []
    for m in merged:
        text = m.get("text", "")
        metadata = dict(m.get("metadata", {}))
        metadata.setdefault("_retrieval_source", m.get("source"))
        if not text:
            continue
        documents.append(Document(page_content=text, metadata=metadata))
        serialized_parts.append(f"Source: {metadata}\nRetrievalSource: {m.get('source')}\nScore: {m.get('score')}\nContent: {text}")

    serialized = "\n\n".join(serialized_parts)
    if not serialized:
        print("[WARN] No chunks retrieved for query:", query)

    return serialized, documents




tools = [retrieve_context]

sys_prompt = (
    "You have access to a tool that retrieves paper from vector database. "
    "Use the tool to help answer user queries."
)

llm = ChatOllama(
    model='qwen3:1.7b',
    base_url='http://localhost:11434',
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt= sys_prompt
)

query = (
    "<AlignSurvey: A Comprehensive Benchmark for Human Preferences Alignment in Social Surveys>这篇论文的研究方法是什么，请详细说说"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()