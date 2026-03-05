import argparse
import json
import re
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def parse_args():
    parser = argparse.ArgumentParser(description="Write markdown chunks into Qdrant.")
    parser.add_argument("--chunk-dir", default="RAG/chunks")
    parser.add_argument("--pdf-dir", default="RAG/arxiv_pdfs_filtered")
    parser.add_argument("--collection", default="markdown_chunks")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def _normalize_arxiv_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", (raw or "").strip())


def _infer_paper_id_from_chunk_name(chunk_name: str) -> str:
    base = re.sub(r"_chunk_\d+\.md$", "", chunk_name)
    return _normalize_arxiv_id(base)


def _load_metadata_index(pdf_dir: Path) -> dict:
    index = {}
    for meta_file in sorted(pdf_dir.glob("metadata_*.jsonl")):
        try:
            with meta_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    pid = _normalize_arxiv_id(
                        record.get("base_id") or record.get("id") or ""
                    )
                    if not pid:
                        continue
                    index[pid] = record
        except Exception:
            continue
    return index


def _build_payload(chunk_file: Path, content: str, meta_index: dict) -> dict:
    paper_id = _infer_paper_id_from_chunk_name(chunk_file.name)
    meta = meta_index.get(paper_id, {})
    title = str(meta.get("title") or paper_id or chunk_file.stem)
    arxiv_url = meta.get("arxiv_url") or (
        f"https://arxiv.org/abs/{paper_id}" if paper_id else ""
    )
    pdf_url = str(meta.get("pdf_url") or "")
    published = str(meta.get("published") or "")
    year = ""
    if published:
        year = published[:4]

    return {
        "source": chunk_file.name,
        "text": content,
        "n_tokens": len(content.split()),
        "paper_id": paper_id,
        "title": title,
        "arxiv_url": arxiv_url,
        "pdf_url": pdf_url,
        "published": published,
        "year": year,
    }


def main():
    args = parse_args()
    chunk_dir = Path(args.chunk_dir)
    pdf_dir = Path(args.pdf_dir)
    if not chunk_dir.exists():
        raise FileNotFoundError(f"chunk dir not found: {chunk_dir}")

    meta_index = _load_metadata_index(pdf_dir)
    print(f"[Meta] loaded papers={len(meta_index)} from {pdf_dir}")

    texts = []
    payloads = []
    chunk_files = sorted(chunk_dir.glob("*.md"))
    for chunk_file in chunk_files:
        content = chunk_file.read_text(encoding="utf-8")
        texts.append(content)
        payloads.append(_build_payload(chunk_file, content, meta_index))

    if not texts:
        print("[Qdrant] no chunks found, exit")
        return

    print(f"[Qdrant] chunks={len(texts)}")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)

    client = QdrantClient(host=args.host, port=args.port)
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(
            size=vectors.shape[1],
            distance=Distance.COSINE,
        ),
    )

    batch = []
    for idx, payload in enumerate(payloads):
        batch.append(
            PointStruct(
                id=idx,
                vector=vectors[idx].tolist(),
                payload=payload,
            )
        )
        if len(batch) >= args.batch_size:
            client.upsert(collection_name=args.collection, points=batch)
            batch = []
    if batch:
        client.upsert(collection_name=args.collection, points=batch)

    print(
        f"[Done] upserted={len(payloads)} collection={args.collection} host={args.host}:{args.port}"
    )


if __name__ == "__main__":
    main()
