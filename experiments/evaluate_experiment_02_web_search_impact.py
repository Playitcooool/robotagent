import argparse
from pathlib import Path

from experiment_utils import (
    add_judge_args,
    call_judge,
    dump_json,
    dump_jsonl,
    get_judge_config,
    judge_enabled,
    load_jsonl,
    maybe_mean,
    parse_markdown_links,
    ratio,
)


def normalize_rows(rows):
    return {str(row.get("id")): row for row in rows if str(row.get("id", "")).strip()}


def web_search_triggered(row):
    if bool(row.get("web_search_used")):
        return True
    tools = row.get("tool_names") or row.get("tools") or []
    return any(str(tool).strip().lower() == "web_search" for tool in tools)


def source_coverage(row):
    answer = str(row.get("answer") or "")
    refs = row.get("references") or []
    link_count = len(parse_markdown_links(answer))
    if isinstance(refs, list):
        for item in refs:
            link_count += len(parse_markdown_links(str(item)))
    return 1 if link_count > 0 else 0


def build_pairwise_prompt(row_a, row_b):
    return (
        "Compare answer A and answer B for the same time-sensitive question.\n"
        "Return strict JSON with keys: winner,factuality_a,factuality_b,grounding_a,grounding_b,overall_a,overall_b,reason.\n"
        "winner must be one of A,B,TIE. Scores are 1-5 integers.\n\n"
        f"Prompt:\n{row_a.get('prompt', '')}\n\n"
        f"Answer A:\n{row_a.get('answer', '')}\n\n"
        f"Answer B:\n{row_b.get('answer', '')}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate web search impact.")
    parser.add_argument("--system-a", required=True)
    parser.add_argument("--system-b", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--with-judge", action="store_true")
    add_judge_args(parser)
    args = parser.parse_args()

    rows_a = normalize_rows(load_jsonl(Path(args.system_a)))
    rows_b = normalize_rows(load_jsonl(Path(args.system_b)))
    common_ids = sorted(set(rows_a) & set(rows_b))

    details = []
    judgments = []
    if args.with_judge and not judge_enabled(args):
        raise ValueError("Judge requested but base-url/model are missing.")
    base_url, api_key, model, timeout = get_judge_config(args)

    for rid in common_ids:
        row_a = rows_a[rid]
        row_b = rows_b[rid]
        item = {
            "id": rid,
            "a_web_search_triggered": int(web_search_triggered(row_a)),
            "b_web_search_triggered": int(web_search_triggered(row_b)),
            "a_source_coverage": source_coverage(row_a),
            "b_source_coverage": source_coverage(row_b),
        }
        if args.with_judge:
            judgment = call_judge(base_url, api_key, model, build_pairwise_prompt(row_a, row_b), timeout=timeout)
            item["judgment"] = judgment
            judgments.append(judgment)
        details.append(item)

    summary = {
        "count": len(details),
        "a_search_trigger_rate": ratio(sum(row["a_web_search_triggered"] for row in details), len(details)),
        "b_search_trigger_rate": ratio(sum(row["b_web_search_triggered"] for row in details), len(details)),
        "a_source_coverage_rate": ratio(sum(row["a_source_coverage"] for row in details), len(details)),
        "b_source_coverage_rate": ratio(sum(row["b_source_coverage"] for row in details), len(details)),
    }
    if judgments:
        wins = {"A": 0, "B": 0, "TIE": 0}
        for judgment in judgments:
            wins[str(judgment.get("winner", "TIE")).upper()] = wins.get(
                str(judgment.get("winner", "TIE")).upper(),
                0,
            ) + 1
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
