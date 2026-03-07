import argparse
import re
from pathlib import Path

from experiment_utils import (
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


RAW_URL_RE = re.compile(r"(?<!\]\()https?://[^\s)]+")


def normalize_rows(rows):
    by_id = {}
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            by_id[rid] = row
    return by_id


def extract_reference_stats(row):
    answer = str(row.get("answer") or "")
    refs = row.get("references") or []
    links = parse_markdown_links(answer)
    for ref in refs if isinstance(refs, list) else [refs]:
        links.extend(parse_markdown_links(str(ref)))
    raw_urls = RAW_URL_RE.findall(answer)
    valid_markdown_urls = [url for _, url in links if is_valid_url(url)]
    return {
        "reference_count": len(links),
        "reference_coverage": 1 if links else 0,
        "raw_url_violation": 1 if raw_urls else 0,
        "valid_markdown_url_rate": ratio(len(valid_markdown_urls), len(links)),
    }


def build_pairwise_prompt(row_a, row_b):
    return (
        "Compare answer A and answer B for the same knowledge question.\n"
        "Return strict JSON with keys: winner,factuality_a,factuality_b,grounding_a,grounding_b,overall_a,overall_b,reason.\n"
        "winner must be one of A,B,TIE. Scores are 1-5 integers.\n\n"
        f"Prompt:\n{row_a.get('prompt', '')}\n\n"
        f"Answer A:\n{row_a.get('answer', '')}\n\n"
        f"Answer B:\n{row_b.get('answer', '')}\n"
    )


def summarize(details, key_prefix):
    return {
        f"{key_prefix}_reference_coverage_rate": ratio(
            sum(row[f"{key_prefix}_reference_coverage"] for row in details),
            len(details),
        ),
        f"{key_prefix}_raw_url_violation_rate": ratio(
            sum(row[f"{key_prefix}_raw_url_violation"] for row in details),
            len(details),
        ),
        f"{key_prefix}_valid_markdown_url_rate": maybe_mean(
            row[f"{key_prefix}_valid_markdown_url_rate"] for row in details
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate agentic RAG ablation.")
    parser.add_argument("--system-a", required=True)
    parser.add_argument("--system-b", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    rows_a = normalize_rows(load_jsonl(Path(args.system_a)))
    rows_b = normalize_rows(load_jsonl(Path(args.system_b)))
    common_ids = sorted(set(rows_a) & set(rows_b))

    details = []
    judgments = []
    if args.with_judge and not judge_enabled(args):
        raise ValueError("Judge requested but base-url/model are missing.")
    base_url, api_key, model = get_judge_config(args)

    for rid in common_ids:
        row_a = rows_a[rid]
        row_b = rows_b[rid]
        stats_a = extract_reference_stats(row_a)
        stats_b = extract_reference_stats(row_b)
        item = {"id": rid, **{f"a_{k}": v for k, v in stats_a.items()}, **{f"b_{k}": v for k, v in stats_b.items()}}
        if args.with_judge:
            judgment = call_judge(base_url, api_key, model, build_pairwise_prompt(row_a, row_b))
            item["judgment"] = judgment
            judgments.append(judgment)
        details.append(item)

    summary = {"count": len(details), **summarize(details, "a"), **summarize(details, "b")}
    if judgments:
        wins = {"A": 0, "B": 0, "TIE": 0}
        for judgment in judgments:
            winner = str(judgment.get("winner", "TIE")).upper()
            wins[winner] = wins.get(winner, 0) + 1
        summary.update(
            {
                "wins": wins,
                "avg_factuality_a": maybe_mean(j.get("factuality_a", 0) for j in judgments),
                "avg_factuality_b": maybe_mean(j.get("factuality_b", 0) for j in judgments),
                "avg_grounding_a": maybe_mean(j.get("grounding_a", 0) for j in judgments),
                "avg_grounding_b": maybe_mean(j.get("grounding_b", 0) for j in judgments),
                "avg_overall_a": maybe_mean(j.get("overall_a", 0) for j in judgments),
                "avg_overall_b": maybe_mean(j.get("overall_b", 0) for j in judgments),
            }
        )

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", details)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

