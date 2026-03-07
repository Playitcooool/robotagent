import argparse
import hashlib
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_markdown_separators():
    # Prefer keeping markdown section/code boundaries intact.
    return [
        "\n```",
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n",
        " ",
        "",
    ]


def build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    mode: str = "token",
    token_model: str = "cl100k_base",
) -> RecursiveCharacterTextSplitter:
    separators = build_markdown_separators()
    if mode == "token":
        # Token-aware splitter (requires tiktoken in environment).
        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=token_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        except Exception:
            # Fallback to char splitter when tiktoken is unavailable.
            pass
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def iter_md_files(md_dir: Path):
    # Stable order for reproducible chunk ids across runs.
    return sorted(md_dir.glob("*.md"))


def main():
    parser = argparse.ArgumentParser(description="Split markdown files into RAG chunks.")
    parser.add_argument("--md-dir", default="RAG/extracted_md")
    parser.add_argument("--chunk-dir", default="RAG/chunks")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--mode",
        choices=["token", "char"],
        default="token",
        help="Split by token (preferred) or by char.",
    )
    parser.add_argument(
        "--token-model",
        default="cl100k_base",
        help="tiktoken encoding name when --mode token.",
    )
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument(
        "--manifest-name",
        default="chunks_manifest.jsonl",
        help="JSONL manifest saved under chunk-dir.",
    )
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    chunk_dir = Path(args.chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must be smaller than chunk-size")

    splitter = build_splitter(
        args.chunk_size,
        args.chunk_overlap,
        mode=args.mode,
        token_model=args.token_model,
    )
    manifest_path = chunk_dir / args.manifest_name

    files_count = 0
    chunk_count = 0
    skip_short = 0
    seen_hashes = set()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for file_path in iter_md_files(md_dir):
            files_count += 1
            text = file_path.read_text(encoding="utf-8")
            chunks = splitter.split_text(text)
            base_name = file_path.stem

            for i, raw_chunk in enumerate(chunks):
                chunk = raw_chunk.strip()
                if len(chunk) < args.min_chars:
                    skip_short += 1
                    continue

                sha1 = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                if sha1 in seen_hashes:
                    continue
                seen_hashes.add(sha1)

                chunk_file = chunk_dir / f"{base_name}_chunk_{i:04d}.md"
                chunk_file.write_text(chunk, encoding="utf-8")
                manifest.write(
                    json.dumps(
                        {
                            "source_file": str(file_path),
                            "chunk_file": str(chunk_file),
                            "chunk_index": i,
                            "num_chars": len(chunk),
                            "sha1": sha1,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunk_count += 1

    print(
        f"[Done] mode={args.mode}, files={files_count}, chunks={chunk_count}, skip_short={skip_short}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
