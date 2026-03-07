import argparse
from pathlib import Path

from experiment_utils import (
    call_judge,
    dump_json,
    dump_jsonl,
    get_judge_config,
    judge_enabled,
    load_jsonl,
    maybe_mean,
    ratio,
)


SIM_KEYS = ("仿真", "模拟", "pybullet", "gazebo", "抓取", "机械臂", "push", "simulation")
ANALYSIS_KEYS = ("分析", "统计", "指标", "图表", "可视化", "analysis")
TIME_KEYS = ("最新", "最近", "today", "latest", "recent")


def expected_route(prompt: str) -> str:
    lowered = (prompt or "").lower()
    if any(key in lowered for key in SIM_KEYS):
        return "simulator"
    if any(key in lowered for key in ANALYSIS_KEYS):
        return "analysis"
    if any(key in lowered for key in TIME_KEYS):
        return "web_search"
    return "main"


def actual_routes(row):
    routes = set()
    for source in row.get("subagent_sources") or row.get("sources") or []:
        routes.add(str(source).strip().lower())
    for tool in row.get("tool_names") or row.get("tools") or []:
        name = str(tool).strip().lower()
        if name == "web_search":
            routes.add("web_search")
        if name == "qdrant_retrieve_context":
            routes.add("rag")
    if not routes:
        routes.add("main")
    return routes


def build_judge_prompt(row, expected, actual):
    return (
        "Judge whether the routing/tool selection is appropriate.\n"
        "Return strict JSON with keys: correctness,overcall,missed_call,verdict.\n"
        "Scores correctness are 1-5 integers. overcall/missed_call are 0 or 1.\n\n"
        f"Prompt:\n{row.get('prompt', '')}\n\n"
        f"Expected route:\n{expected}\n\n"
        f"Actual route/tools:\n{sorted(actual)}\n\n"
        f"Final answer:\n{row.get('answer', row.get('final_answer', ''))}\n"
    )


def evaluate_row(row):
    expected = expected_route(str(row.get("prompt", "")))
    actual = actual_routes(row)
    correct = expected in actual or (expected == "main" and actual == {"main"})
    overcall = expected == "main" and actual != {"main"}
    missed_call = expected != "main" and expected not in actual
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt", ""),
        "expected_route": expected,
        "actual_routes": sorted(actual),
        "route_correct": correct,
        "overcall": overcall,
        "missed_call": missed_call,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate routing and tool selection.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    details = [evaluate_row(row) for row in rows]

    summary = {
        "count": len(details),
        "route_correct_rate": ratio(sum(1 for row in details if row["route_correct"]), len(details)),
        "overcall_rate": ratio(sum(1 for row in details if row["overcall"]), len(details)),
        "missed_call_rate": ratio(sum(1 for row in details if row["missed_call"]), len(details)),
    }

    if args.with_judge:
        if not judge_enabled(args):
            raise ValueError("Judge requested but base-url/model are missing.")
        base_url, api_key, model = get_judge_config(args)
        judgments = []
        for row, detail in zip(rows, details):
            prompt = build_judge_prompt(row, detail["expected_route"], detail["actual_routes"])
            judgment = call_judge(base_url, api_key, model, prompt)
            detail["judgment"] = judgment
            judgments.append(judgment)
        summary["judge_correctness"] = maybe_mean(j.get("correctness", 0) for j in judgments)
        summary["judge_overcall_rate"] = ratio(sum(int(j.get("overcall", 0)) for j in judgments), len(judgments))
        summary["judge_missed_call_rate"] = ratio(sum(int(j.get("missed_call", 0)) for j in judgments), len(judgments))

    out_dir = Path(args.out_dir)
    dump_jsonl(out_dir / "details.jsonl", details)
    dump_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

