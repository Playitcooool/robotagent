import argparse
import re
from pathlib import Path

from experiment_utils import (
    add_judge_args,
    call_judge,
    dump_json,
    dump_jsonl,
    get_judge_config,
    judge_enabled,
    load_jsonl,
    looks_like_done,
    maybe_mean,
    ratio,
)


PATH_RE = re.compile(r"(/[A-Za-z0-9._/\-]+)")


def extract_tool_outputs(row):
    outputs = row.get("tool_outputs")
    if isinstance(outputs, list):
        return [str(item) for item in outputs]
    if isinstance(outputs, dict):
        return [str(value) for value in outputs.values()]
    return []


def answer_paths(answer: str):
    return PATH_RE.findall(answer or "")


def evaluate_row(row):
    answer = str(row.get("answer") or row.get("final_answer") or "")
    tool_outputs = extract_tool_outputs(row)
    tool_output_text = "\n".join(tool_outputs)
    frame = row.get("frame") or row.get("latest_frame") or {}
    has_frame = bool(frame) or bool(row.get("has_frame"))
    success = looks_like_done(answer) or str(row.get("status", "")).lower() == "ok"
    mentioned_paths = answer_paths(answer)
    hallucinated_paths = [path for path in mentioned_paths if path not in tool_output_text]
    numeric_consistent = 1
    if row.get("result_text") and tool_output_text:
        numeric_consistent = 1 if str(row["result_text"]) in tool_output_text else 0
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt", ""),
        "task_completed": success,
        "has_frame": has_frame,
        "numeric_consistency": numeric_consistent,
        "hallucinated_path_count": len(hallucinated_paths),
        "hallucinated_paths": hallucinated_paths,
    }


def build_judge_prompt(row):
    return (
        "Evaluate simulation task quality.\n"
        "Return strict JSON with keys: overall,consistency,readability,verdict.\n"
        "Scores are integers 1-5.\n\n"
        f"Prompt:\n{row.get('prompt', '')}\n\n"
        f"Answer:\n{row.get('answer', row.get('final_answer', ''))}\n\n"
        f"Tool outputs:\n{row.get('tool_outputs')}\n\n"
        f"Frame/meta:\n{row.get('frame') or row.get('latest_frame')}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation task quality.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--with-judge", action="store_true")
    add_judge_args(parser)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    details = [evaluate_row(row) for row in rows]
    summary = {
        "count": len(details),
        "task_completion_rate": ratio(sum(1 for row in details if row["task_completed"]), len(details)),
        "frame_coverage_rate": ratio(sum(1 for row in details if row["has_frame"]), len(details)),
        "numeric_consistency_rate": ratio(sum(row["numeric_consistency"] for row in details), len(details)),
        "hallucination_rate": ratio(sum(1 for row in details if row["hallucinated_path_count"] > 0), len(details)),
    }

    if args.with_judge:
        if not judge_enabled(args):
            raise ValueError("Judge requested but base-url/model are missing.")
        base_url, api_key, model, timeout = get_judge_config(args)
        judgments = []
        for row, detail in zip(rows, details):
            judgment = call_judge(base_url, api_key, model, build_judge_prompt(row), timeout=timeout)
            detail["judgment"] = judgment
            judgments.append(judgment)
        summary["judge_overall"] = maybe_mean(j.get("overall", 0) for j in judgments)
        summary["judge_consistency"] = maybe_mean(j.get("consistency", 0) for j in judgments)
        summary["judge_readability"] = maybe_mean(j.get("readability", 0) for j in judgments)

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", details)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
