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
    looks_like_done,
    maybe_mean,
    ratio,
)


INIT_KEYS = ("initialize", "init")
BUILD_KEYS = ("spawn", "load", "create", "add")
ACTION_KEYS = ("push", "move", "control", "step", "run")
CHECK_KEYS = ("check", "state", "status")
PYBULLET_KEYS = ("pybullet", "simulation", "simulator")


def extract_tool_calls(row):
    if isinstance(row.get("tool_calls"), list):
        return row["tool_calls"]
    events = row.get("events") or []
    calls = []
    for event in events:
        if str(event.get("type", "")).lower() not in {"tool", "status"}:
            continue
        name = str(event.get("name") or event.get("tool") or "").strip()
        if not name:
            continue
        calls.append({"name": name, "payload": event})
    return calls


def classify_tool(name: str) -> str:
    lowered = (name or "").strip().lower()
    if any(key in lowered for key in INIT_KEYS):
        return "init"
    if any(key in lowered for key in BUILD_KEYS):
        return "build"
    if any(key in lowered for key in ACTION_KEYS):
        return "action"
    if any(key in lowered for key in CHECK_KEYS):
        return "check"
    return "other"


def is_pybullet_related(name: str) -> bool:
    lowered = (name or "").strip().lower()
    return any(key in lowered for key in PYBULLET_KEYS)


def build_judge_prompt(row, tool_names):
    return (
        "Evaluate whether the simulation answer is reliable and consistent with the tool sequence.\n"
        "Return strict JSON with keys: overall,consistency,verdict.\n"
        "Scores are integers 1-5.\n\n"
        f"Prompt:\n{row.get('prompt', '')}\n\n"
        f"Tool sequence:\n{tool_names}\n\n"
        f"Final answer:\n{row.get('answer', row.get('final_answer', ''))}\n"
    )


def evaluate_row(row):
    calls = extract_tool_calls(row)
    tool_names = [str(call.get("name") or "") for call in calls]
    pybullet_names = [name for name in tool_names if is_pybullet_related(name)]
    categories = [classify_tool(name) for name in pybullet_names]
    first_action_idx = next((idx for idx, item in enumerate(categories) if item == "action"), -1)
    init_before_action = first_action_idx == -1 or "init" in categories[:first_action_idx]
    has_check_after_action = (
        first_action_idx == -1 or "check" in categories[first_action_idx + 1 :]
    )
    order_violation = bool(first_action_idx != -1 and not init_before_action)
    answer = str(row.get("answer") or row.get("final_answer") or "")
    success = looks_like_done(answer) or str(row.get("status", "")).lower() == "ok"
    retry_count = int(row.get("retry_count") or 0)
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt", ""),
        "tool_names": tool_names,
        "pybullet_tool_names": pybullet_names,
        "init_before_action": init_before_action,
        "has_check_after_action": has_check_after_action,
        "tool_order_violation": order_violation,
        "simulation_success": success,
        "retry_count": retry_count,
    }


def summarize(rows):
    total = len(rows)
    actionable = [row for row in rows if row["pybullet_tool_names"]]
    return {
        "count": total,
        "pybullet_trace_count": len(actionable),
        "simulation_success_rate": ratio(sum(1 for row in rows if row["simulation_success"]), total),
        "init_before_action_rate": ratio(sum(1 for row in actionable if row["init_before_action"]), len(actionable)),
        "check_after_action_rate": ratio(sum(1 for row in actionable if row["has_check_after_action"]), len(actionable)),
        "tool_order_violation_rate": ratio(sum(1 for row in actionable if row["tool_order_violation"]), len(actionable)),
        "avg_retry_count": maybe_mean(row["retry_count"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyBullet skill impact.")
    parser.add_argument("--input", required=True, help="JSONL with tool traces and answers")
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
            prompt = build_judge_prompt(row, result["tool_names"])
            judgment = call_judge(base_url, api_key, model, prompt, timeout=timeout)
            result["judgment"] = judgment
            judgments.append(judgment)
        summary["judge_overall"] = maybe_mean(j.get("overall", 0) for j in judgments)
        summary["judge_consistency"] = maybe_mean(j.get("consistency", 0) for j in judgments)

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", results)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
