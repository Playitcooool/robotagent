import argparse
from pathlib import Path

from experiment_utils import dump_json, dump_jsonl, load_jsonl, maybe_mean


def extract_total_tokens(row):
    if isinstance(row.get("total_tokens"), (int, float)):
        return float(row["total_tokens"])
    usage = row.get("token_usage") or {}
    if isinstance(usage, dict):
        if isinstance(usage.get("total_tokens"), (int, float)):
            return float(usage["total_tokens"])
        total = 0.0
        for value in usage.values():
            if isinstance(value, dict):
                total += float(value.get("total_tokens", 0) or 0)
        if total > 0:
            return total
    return 0.0


def extract_agent_breakdown(row):
    usage = row.get("token_usage") or {}
    breakdown = {}
    if not isinstance(usage, dict):
        return breakdown
    for key, value in usage.items():
        if isinstance(value, dict):
            breakdown[str(key)] = float(value.get("total_tokens", 0) or 0)
    return breakdown


def extract_quality(row):
    if isinstance(row.get("overall"), (int, float)):
        return float(row["overall"])
    judgment = row.get("judgment") or {}
    if isinstance(judgment, dict) and isinstance(judgment.get("overall"), (int, float)):
        return float(judgment["overall"])
    return 0.0


def evaluate_rows(rows):
    details = []
    for row in rows:
        total_tokens = extract_total_tokens(row)
        overall = extract_quality(row)
        quality_per_1k = overall / (total_tokens / 1000.0) if total_tokens > 0 else 0.0
        details.append(
            {
                "id": row.get("id"),
                "prompt": row.get("prompt", ""),
                "total_tokens": total_tokens,
                "overall": overall,
                "quality_per_1k_tokens": quality_per_1k,
                "agent_breakdown": extract_agent_breakdown(row),
            }
        )
    summary = {
        "count": len(details),
        "avg_total_tokens": maybe_mean(row["total_tokens"] for row in details),
        "avg_overall": maybe_mean(row["overall"] for row in details),
        "avg_quality_per_1k_tokens": maybe_mean(row["quality_per_1k_tokens"] for row in details),
        "avg_main_tokens": maybe_mean(row["agent_breakdown"].get("main", 0) for row in details),
        "avg_simulator_tokens": maybe_mean(row["agent_breakdown"].get("simulator", 0) for row in details),
        "avg_analysis_tokens": maybe_mean(row["agent_breakdown"].get("analysis", 0) for row in details),
    }
    return details, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate token efficiency tradeoff.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    details, summary = evaluate_rows(rows)

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", details)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

