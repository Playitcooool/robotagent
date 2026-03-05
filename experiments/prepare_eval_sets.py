import argparse
import json
import re
from pathlib import Path


SIM_KEYS = [
    "仿真",
    "模拟",
    "机器人臂",
    "抓取",
    "push",
    "pybullet",
    "gazebo",
    "simulation",
]
ANALYSIS_KEYS = [
    "分析",
    "统计",
    "可视化",
    "plot",
    "csv",
    "指标",
]
TIME_KEYS = [
    "最新",
    "今天",
    "最近",
    "recent",
    "latest",
    "today",
]


def _classify(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(k in p for k in [x.lower() for x in SIM_KEYS]):
        return "simulation"
    if any(k in p for k in [x.lower() for x in ANALYSIS_KEYS]):
        return "analysis"
    return "knowledge"


def _is_time_sensitive(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k in p for k in [x.lower() for x in TIME_KEYS])


def _extract_final_assistant(messages) -> str:
    for msg in reversed(messages or []):
        if str(msg.get("role", "")).lower() != "assistant":
            continue
        text = str(msg.get("content", "")).strip()
        if text:
            return text
    return ""


def _clean_prompt(text: str) -> str:
    # Keep raw prompt content, but normalize whitespace.
    return re.sub(r"\s+", " ", (text or "").strip())


def build_sets(trajectories_path: Path):
    all_rows = []
    with trajectories_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            messages = item.get("messages") or []
            user_prompt = ""
            for m in messages:
                if str(m.get("role", "")).lower() == "user":
                    user_prompt = _clean_prompt(str(m.get("content", "")))
                    if user_prompt:
                        break
            if not user_prompt:
                continue
            row = {
                "id": f"traj-{idx}",
                "prompt": user_prompt,
                "category": _classify(user_prompt),
                "time_sensitive": _is_time_sensitive(user_prompt),
                "reference_answer": _extract_final_assistant(messages),
            }
            all_rows.append(row)
    return all_rows


def dump_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare eval sets from existing trajectories.")
    parser.add_argument("--trajectories", default="trajectories.jsonl")
    parser.add_argument("--out-dir", default="experiments/data")
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    out_dir = Path(args.out_dir)
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectories not found: {traj_path}")

    all_rows = build_sets(traj_path)
    knowledge = [r for r in all_rows if r["category"] == "knowledge"]
    simulation = [r for r in all_rows if r["category"] == "simulation"]
    analysis = [r for r in all_rows if r["category"] == "analysis"]
    timesensitive = [r for r in all_rows if r["time_sensitive"]]

    dump_jsonl(out_dir / "eval_all.jsonl", all_rows)
    dump_jsonl(out_dir / "eval_knowledge.jsonl", knowledge)
    dump_jsonl(out_dir / "eval_simulation.jsonl", simulation)
    dump_jsonl(out_dir / "eval_analysis.jsonl", analysis)
    dump_jsonl(out_dir / "eval_timesensitive.jsonl", timesensitive)

    summary = {
        "all": len(all_rows),
        "knowledge": len(knowledge),
        "simulation": len(simulation),
        "analysis": len(analysis),
        "timesensitive": len(timesensitive),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
