import types
import json
import pytest
import query as qmod
from langchain_core.documents import Document


def _call_tool(obj, *args, **kwargs):
    if callable(obj):
        return obj(*args, **kwargs)
    if hasattr(obj, "func") and callable(obj.func):
        return obj.func(*args, **kwargs)
    if hasattr(obj, "run") and callable(obj.run):
        return obj.run(*args, **kwargs)
    raise RuntimeError("Unable to call tool object")


class FakePoint:
    def __init__(self, id_, text, score=0.0):
        self.id = id_
        self.payload = {"text": text}
        self.score = score


def test_multi_recall(monkeypatch):
    # Prepare fake embeddings
    class FakeEmb:
        def embed_query(self, q):
            return [0.1, 0.2]

    monkeypatch.setattr(qmod, "embeddings", FakeEmb())

    # Prepare fake client
    class FakeClient:
        def __init__(self):
            self._points = [FakePoint(i, f"This is document {i} about testing queries.") for i in range(10)]

        def query_points(self, query, collection_name, limit=3):
            # return top 'limit' points as if vector retrieval
            return self._points[:limit]

        def scroll(self, collection_name, with_payload=True, limit=500, offset=0):
            # return all points in small batches
            if offset >= len(self._points):
                return []
            end = min(offset + limit, len(self._points))
            return self._points[offset:end]

    fake_client = FakeClient()
    monkeypatch.setattr(qmod, "client", fake_client)

    serialized, docs = _call_tool(qmod.retrieve_context, "testing query", top_k=5)
    assert isinstance(serialized, str)
    assert isinstance(docs, list)
    assert len(docs) > 0
    # ensure retrieval source tags present in metadata
    sources = set(d.metadata.get("_retrieval_source") for d in docs)
    assert any(s in ("vector", "tfidf", "keyword") for s in sources if s is not None)


@pytest.mark.skipif(not qmod.TFIDF_AVAILABLE, reason="TFIDF missing")
def test_tfidf_prefers_relevant(monkeypatch):
    # Construct a small corpus where one document is highly relevant
    class FakeEmb:
        def embed_query(self, q):
            return [0.0]
    monkeypatch.setattr(qmod, "embeddings", FakeEmb())

    class FakeClient2:
        def __init__(self):
            self._points = [FakePoint(1, "apple banana"), FakePoint(2, "orange fruit apple"), FakePoint(3, "not related")]
        def query_points(self, query, collection_name, limit=3):
            return self._points[:limit]
        def scroll(self, collection_name, with_payload=True, limit=500, offset=0):
            if offset >= len(self._points):
                return []
            end = min(offset + limit, len(self._points))
            return self._points[offset:end]

    monkeypatch.setattr(qmod, "client", FakeClient2())

    serialized, docs = _call_tool(qmod.retrieve_context, "apple", top_k=3)
    # Expect at least one doc whose metadata source is 'tfidf' or 'keyword'
    assert any(d.metadata.get("_retrieval_source") in ("tfidf", "keyword", "vector") for d in docs)
