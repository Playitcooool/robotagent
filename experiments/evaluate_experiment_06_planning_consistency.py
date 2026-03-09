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


def extract_planning_events(row):
    if isinstance(row.get("planning_events"), list):
        return row["planning_events"]
    if isinstance(row.get("planning"), list):
        return row["planning"]
    events = row.get("events") or []
    return [event for event in events if str(event.get("type", "")).lower() == "planning"]


def extract_steps(event):
    raw = event.get("steps")
    if not isinstance(raw, list):
        raw = event.get("plan") if isinstance(event.get("plan"), list) else []
    steps = []
    for idx, item in enumerate(raw):
        step = str(item.get("step") or item.get("text") or item.get("title") or "").strip()
        if not step:
            continue
        status = str(item.get("status") or item.get("state") or "pending").strip().lower()
        status = status.replace("-", "_")
        steps.append(
            {
                "id": str(item.get("id") or idx + 1),
                "step": step,
                "status": status,
            }
        )
    return steps


def build_judge_prompt(row, final_steps, update_count):
    return (
        "Evaluate planning quality for a multi-step agent task.\n"
        "Return strict JSON with keys: conciseness_score,execution_alignment_score,verdict.\n"
        "Scores are integers 1-5.\n\n"
        f"Prompt:\n{row.get('prompt', '')}\n\n"
        f"Planning updates: {update_count}\n"
        f"Final plan:\n{final_steps}\n\n"
        f"Final answer:\n{row.get('answer', row.get('final_answer', ''))}\n"
    )


def evaluate_row(row):
    planning_events = extract_planning_events(row)
    update_count = len(planning_events)
    final_steps = extract_steps(planning_events[-1]) if planning_events else []
    has_plan = bool(final_steps)
    has_in_progress_final = any(step["status"] == "in_progress" for step in final_steps)
    all_completed = bool(final_steps) and all(step["status"] == "completed" for step in final_steps)
    answer = str(row.get("answer") or row.get("final_answer") or "")
    answer_done = looks_like_done(answer)
    stalled_plan = update_count <= 1 and has_in_progress_final
    status_mismatch = answer_done and has_in_progress_final
    overplanned = len(final_steps) > 5
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt", ""),
        "plan_presence": has_plan,
        "plan_update_count": update_count,
        "final_step_count": len(final_steps),
        "all_completed": all_completed,
        "has_in_progress_final": has_in_progress_final,
        "stalled_plan": stalled_plan,
        "status_mismatch": status_mismatch,
        "overplanned": overplanned,
        "final_steps": final_steps,
    }


def summarize(rows):
    total = len(rows)
    multi_step = [row for row in rows if row["final_step_count"] >= 2 or row["plan_update_count"] >= 1]
    return {
        "count": total,
        "multi_step_count": len(multi_step),
        "plan_presence_rate": ratio(sum(1 for row in multi_step if row["plan_presence"]), len(multi_step)),
        "avg_plan_update_count": maybe_mean(row["plan_update_count"] for row in rows),
        "completion_alignment_rate": ratio(
            sum(1 for row in rows if not row["status_mismatch"]),
            total,
        ),
        "stalled_plan_rate": ratio(sum(1 for row in rows if row["stalled_plan"]), total),
        "overplanned_rate": ratio(sum(1 for row in rows if row["overplanned"]), total),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate planning consistency.")
    parser.add_argument("--input", required=True, help="JSONL with planning events and final answer")
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
            prompt = build_judge_prompt(row, result["final_steps"], result["plan_update_count"])
            judgment = call_judge(base_url, api_key, model, prompt, timeout=timeout)
            result["judgment"] = judgment
            judgments.append(judgment)
        summary["judge_conciseness_score"] = maybe_mean(j.get("conciseness_score", 0) for j in judgments)
        summary["judge_execution_alignment_score"] = maybe_mean(
            j.get("execution_alignment_score", 0) for j in judgments
        )

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", results)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
