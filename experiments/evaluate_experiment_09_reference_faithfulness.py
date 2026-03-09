import argparse
from pathlib import Path

from experiment_utils import (
    add_judge_args,
    call_judge,
    dump_json,
    dump_jsonl,
    get_judge_config,
    is_valid_url,
    judge_enabled,
    load_jsonl,
    maybe_mean,
    parse_markdown_links,
    ratio,
)


def normalize_reference_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def collect_tool_urls(row):
    urls = set()
    for key in ("tool_references", "rag_references", "web_references", "references"):
        value = row.get(key)
        for item in normalize_reference_list(value):
            for _, url in parse_markdown_links(item):
                urls.add(url)
            if is_valid_url(item):
                urls.add(item)
    tool_payload = row.get("tool_outputs") or {}
    if isinstance(tool_payload, dict):
        for value in tool_payload.values():
            for item in normalize_reference_list(value):
                for _, url in parse_markdown_links(item):
                    urls.add(url)
                if is_valid_url(item):
                    urls.add(item)
    return urls


def extract_answer_urls(row):
    answer = str(row.get("answer") or row.get("final_answer") or "")
    refs = normalize_reference_list(row.get("references"))
    links = parse_markdown_links(answer)
    for ref in refs:
        links.extend(parse_markdown_links(ref))
        if is_valid_url(ref):
            links.append((ref, ref))
    return answer, links


def build_judge_prompt(row, answer_links):
    return (
        "Evaluate whether the references faithfully support the answer.\n"
        "Return strict JSON with keys: faithfulness, support, verdict.\n"
        "Scores are integers 1-5.\n\n"
        f"Prompt:\n{row.get('prompt', '')}\n\n"
        f"Answer:\n{row.get('answer', row.get('final_answer', ''))}\n\n"
        f"Answer references:\n{answer_links}\n\n"
        f"Tool references:\n{row.get('tool_references') or row.get('rag_references') or row.get('web_references') or row.get('references')}\n"
    )


def evaluate_row(row):
    answer, answer_links = extract_answer_urls(row)
    answer_urls = [url for _, url in answer_links]
    tool_urls = collect_tool_urls(row)
    needs_references = bool(row.get("requires_references", True))
    has_references = bool(answer_links)
    valid_urls = [url for url in answer_urls if is_valid_url(url)]
    fabricated = [url for url in answer_urls if url not in tool_urls and tool_urls]
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt", ""),
        "needs_references": needs_references,
        "reference_presence": has_references,
        "reference_count": len(answer_urls),
        "valid_reference_url_rate": ratio(len(valid_urls), len(answer_urls)),
        "fabricated_reference_count": len(fabricated),
        "fabricated_references": fabricated,
        "missing_reference": needs_references and not has_references,
        "tool_reference_count": len(tool_urls),
        "answer_reference_urls": answer_urls,
    }


def summarize(rows):
    need_rows = [row for row in rows if row["needs_references"]]
    return {
        "count": len(rows),
        "need_reference_count": len(need_rows),
        "reference_presence_rate": ratio(sum(1 for row in need_rows if row["reference_presence"]), len(need_rows)),
        "reference_valid_url_rate": maybe_mean(row["valid_reference_url_rate"] for row in rows),
        "citation_fabrication_rate": ratio(sum(1 for row in rows if row["fabricated_reference_count"] > 0), len(rows)),
        "missing_reference_rate": ratio(sum(1 for row in rows if row["missing_reference"]), len(rows)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate reference faithfulness.")
    parser.add_argument("--input", required=True, help="JSONL with answer and tool references")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--with-judge", action="store_true")
    add_judge_args(parser)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    results = [evaluate_row(row) for row in rows]
    summary = summarize(results)

    if args.with_judge:
        if not judge_enabled(args):
            raise ValueError("Judge requested but base-url/model are missing.")
        base_url, api_key, model, timeout = get_judge_config(args)
        judgments = []
        for row, result in zip(rows, results):
            prompt = build_judge_prompt(row, result["answer_reference_urls"])
            judgment = call_judge(base_url, api_key, model, prompt, timeout=timeout)
            result["judgment"] = judgment
            judgments.append(judgment)
        summary["judge_faithfulness"] = maybe_mean(j.get("faithfulness", 0) for j in judgments)
        summary["judge_support"] = maybe_mean(j.get("support", 0) for j in judgments)

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", results)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
