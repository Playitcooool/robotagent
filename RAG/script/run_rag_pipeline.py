import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pdfs(pdf_dir: Path, batch_dir: Path, batch_size: int) -> int:
    batch_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    print(f"[Split] total_pdfs={len(pdfs)}, batch_size={batch_size}")

    batch_count = 0
    for i in range(0, len(pdfs), batch_size):
        batch_pdfs = pdfs[i : i + batch_size]
        batch_path = batch_dir / f"batch_{batch_count:03d}"
        if batch_path.exists():
            shutil.rmtree(batch_path)
        batch_path.mkdir(parents=True, exist_ok=True)
        for pdf in batch_pdfs:
            shutil.copy2(pdf, batch_path / pdf.name)
        batch_count += 1
    print(f"[Split] created_batches={batch_count}")
    return batch_count


def run_mineru(
    batch_dir: Path,
    output_dir: Path,
    log_dir: Path,
    max_retry: int = 3,
    sleep_between: int = 2,
    backend: str = "pipeline",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    batches = sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")])
    if not batches:
        print("[MinerU] no batches found, skip")
        return

    env = {
        **dict(os.environ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }

    for batch in batches:
        log_file = log_dir / f"{batch.name}.log"
        done_file = log_dir / f"{batch.name}.done"
        retry_file = log_dir / f"{batch.name}.retry"
        if done_file.exists():
            print(f"[MinerU] skip done {batch.name}")
            continue

        retry = int(retry_file.read_text()) if retry_file.exists() else 0
        while retry < max_retry:
            print(f"[MinerU] run {batch.name} retry={retry}")
            with log_file.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    ["mineru", "-p", str(batch), "-o", str(output_dir), "--backend", backend],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
            if proc.returncode == 0:
                done_file.touch()
                if retry_file.exists():
                    retry_file.unlink()
                print(f"[MinerU] ok {batch.name}")
                break
            retry += 1
            retry_file.write_text(str(retry), encoding="utf-8")
            print(f"[MinerU] fail {batch.name}, retry={retry}")
            time.sleep(sleep_between)


def extract_markdown(source_dir: Path, dest_dir: Path) -> int:
    if not source_dir.exists():
        print(f"[Extract] source not found: {source_dir}")
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.endswith(".md"):
                continue
            src = Path(root) / file
            dst = dest_dir / file
            idx = 1
            while dst.exists():
                dst = dest_dir / f"{src.stem}_{idx}{src.suffix}"
                idx += 1
            shutil.copy2(src, dst)
            copied += 1
    print(f"[Extract] copied_markdown={copied} -> {dest_dir}")
    return copied


def cut_chunks(
    md_dir: Path,
    chunk_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    manifest_name: str = "chunks_manifest.jsonl",
) -> int:
    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk-overlap must be smaller than chunk-size")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    manifest_path = chunk_dir / manifest_name
    seen_hashes = set()
    files_count = 0
    chunk_count = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for file_path in sorted(md_dir.glob("*.md")):
            files_count += 1
            text = file_path.read_text(encoding="utf-8")
            chunks = splitter.split_text(text)
            for idx, raw_chunk in enumerate(chunks):
                chunk = raw_chunk.strip()
                if len(chunk) < min_chars:
                    skipped += 1
                    continue
                sha1 = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                if sha1 in seen_hashes:
                    continue
                seen_hashes.add(sha1)
                out = chunk_dir / f"{file_path.stem}_chunk_{idx:04d}.md"
                out.write_text(chunk, encoding="utf-8")
                manifest.write(
                    json.dumps(
                        {
                            "source_file": str(file_path),
                            "chunk_file": str(out),
                            "chunk_index": idx,
                            "num_chars": len(chunk),
                            "sha1": sha1,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunk_count += 1
    print(
        f"[Chunk] files={files_count}, chunks={chunk_count}, skip_short={skipped}, manifest={manifest_path}"
    )
    return chunk_count


def write_to_qdrant(
    chunk_dir: Path,
    collection: str,
    host: str,
    port: int,
    batch_size: int,
    embedding_model: str,
):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from langchain_huggingface import HuggingFaceEmbeddings

    texts = []
    payloads = []
    for p in sorted(chunk_dir.glob("*.md")):
        content = p.read_text(encoding="utf-8")
        texts.append(content)
        payloads.append({"source": p.name, "n_tokens": len(content.split()), "text": content})

    if not texts:
        print("[Qdrant] no chunks found, skip")
        return

    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    vectors = embedder.embed_documents(texts)
    client = QdrantClient(host=host, port=port)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )

    points = []
    for i, vec in enumerate(vectors):
        points.append(PointStruct(id=i, vector=vec, payload=payloads[i]))
        if len(points) >= batch_size:
            client.upsert(collection_name=collection, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection, points=points)
    print(f"[Qdrant] upserted={len(texts)} collection={collection}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified RAG pipeline entry.")
    parser.add_argument("--max-results", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--years-back", type=int, default=5)
    parser.add_argument("--per-year-overfetch", type=int, default=3)
    parser.add_argument("--min-quality-score", type=float, default=2.0)

    parser.add_argument("--pdf-dir", default="RAG/arxiv_pdfs_filtered")
    parser.add_argument("--batch-dir", default="RAG/batches")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output-dir", default="RAG/output")
    parser.add_argument("--log-dir", default="RAG/logs")
    parser.add_argument("--md-dir", default="RAG/extracted_md")
    parser.add_argument("--chunk-dir", default="RAG/chunks")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=30)

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-mineru", action="store_true")
    parser.add_argument("--to-qdrant", action="store_true")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--qdrant-collection", default="markdown_chunks")
    parser.add_argument("--qdrant-batch-size", type=int, default=256)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    md_dir = Path(args.md_dir)
    chunk_dir = Path(args.chunk_dir)

    if not args.skip_download:
        print("[Pipeline] step1 download")
        # Lazy import: allow --help and non-download stages without arxiv package.
        from download_arxiv_pdfs import download_arxiv_pdfs

        download_arxiv_pdfs(
            max_results=args.max_results,
            save_dir=str(pdf_dir),
            workers=args.workers,
            years_back=args.years_back,
            per_year_overfetch=args.per_year_overfetch,
            min_quality_score=args.min_quality_score,
        )
    else:
        print("[Pipeline] skip download")

    print("[Pipeline] step2 split")
    split_pdfs(pdf_dir=pdf_dir, batch_dir=batch_dir, batch_size=args.batch_size)

    if not args.skip_mineru:
        print("[Pipeline] step3 mineru")
        run_mineru(batch_dir=batch_dir, output_dir=output_dir, log_dir=log_dir)
    else:
        print("[Pipeline] skip mineru")

    print("[Pipeline] step4 extract markdown")
    extract_markdown(source_dir=output_dir, dest_dir=md_dir)

    print("[Pipeline] step5 chunk")
    cut_chunks(
        md_dir=md_dir,
        chunk_dir=chunk_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
    )

    if args.to_qdrant:
        print("[Pipeline] step6 qdrant")
        write_to_qdrant(
            chunk_dir=chunk_dir,
            collection=args.qdrant_collection,
            host=args.qdrant_host,
            port=args.qdrant_port,
            batch_size=args.qdrant_batch_size,
            embedding_model=args.embedding_model,
        )
    else:
        print("[Pipeline] skip qdrant (use --to-qdrant to enable)")

    print("[Pipeline] done")


if __name__ == "__main__":
    main()
