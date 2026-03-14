import argparse
import json
import re
from pathlib import Path
from collections import Counter


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
# 常见的子代理名称
SUBAGENT_KEYS = [
    "simulation",
    "analysis",
    "data-analyzer",
    "general-purpose",
    "planner",
    "researcher",
    "coder",
]


def _classify(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(k in p for k in [k.lower() for k in SIM_KEYS]):
        return "simulation"
    if any(k in p for k in [k.lower() for k in ANALYSIS_KEYS]):
        return "analysis"
    return "knowledge"


def _is_time_sensitive(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k in p for k in [k.lower() for k in TIME_KEYS])


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


def _extract_tool_names(messages) -> list[str]:
    """从消息中提取工具调用名称"""
    tools = []
    for msg in messages:
        if str(msg.get("role", "")).lower() == "tool":
            name = msg.get("name", "")
            if name:
                tools.append(name)
    return tools


def _extract_subagent_usage(messages) -> list[str]:
    """检测使用了哪些子代理"""
    subagents = set()
    for msg in messages:
        # 从 tool 消息的 name 字段检测
        if str(msg.get("role", "")).lower() == "tool":
            name = str(msg.get("name", "")).lower()
            if name:
                for sa in SUBAGENT_KEYS:
                    if sa in name:
                        subagents.add(sa)

        # 从 assistant 消息内容检测子代理调用
        if str(msg.get("role", "")).lower() == "assistant":
            content = str(msg.get("content", "")).lower()
            for sa in SUBAGENT_KEYS:
                if sa in content:
                    subagents.add(sa)

    return sorted(list(subagents))


def _has_references(text: str) -> bool:
    """检测回答中是否包含引用（markdown 链接）"""
    if not text:
        return False
    return bool(re.search(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text))


def _extract_reference_urls(text: str) -> list[str]:
    """提取回答中的引用 URL"""
    if not text:
        return []
    return [url for _, url in re.findall(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text)]


def _detect_rag_usage(messages) -> bool:
    """检测是否使用了 RAG"""
    for msg in messages:
        content = str(msg.get("content", "")).lower()
        if "rag" in content or "知识库" in content or "vector" in content:
            return True
        name = str(msg.get("name", "")).lower()
        if "rag" in name or "knowledge" in name:
            return True
    return False


def _detect_web_search_usage(messages) -> bool:
    """检测是否使用了联网搜索"""
    for msg in messages:
        name = str(msg.get("name", "")).lower()
        if "web_search" in name or "search" in name:
            return True
    return False


def _detect_error(messages) -> bool:
    """检测是否有错误发生"""
    for msg in messages:
        content = str(msg.get("content", "")).lower()
        error_patterns = ["error", "failed", "exception", "错误", "失败", "无法", "cannot", "unable"]
        if any(p in content for p in error_patterns):
            if not re.search(r"(no error|没有错误|without error)", content):
                return True
    return False


def _check_task_success(messages) -> bool:
    """检测任务是否成功完成"""
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() != "assistant":
            continue
        content = str(msg.get("content", "")).lower()
        success_patterns = ["完成", "成功", "completed", "success", "done", "finished", "已解决"]
        if any(p in content for p in success_patterns):
            return True
    return False


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

            # 提取用户 prompt
            user_prompt = ""
            for m in messages:
                if str(m.get("role", "")).lower() == "user":
                    user_prompt = _clean_prompt(str(m.get("content", "")))
                    if user_prompt:
                        break
            if not user_prompt:
                continue

            # 提取参考回答
            reference_answer = _extract_final_assistant(messages)

            # 提取工具使用信息
            tool_names = _extract_tool_names(messages)
            tool_counter = Counter(tool_names)

            # 提取子代理使用情况
            subagent_usage = _extract_subagent_usage(messages)

            # 检测引用
            has_references = _has_references(reference_answer)
            reference_urls = _extract_reference_urls(reference_answer)

            # 检测 RAG 和搜索使用
            rag_used = _detect_rag_usage(messages)
            web_search_used = _detect_web_search_usage(messages)

            # 检测错误和任务状态
            has_error = _detect_error(messages)
            task_success = _check_task_success(messages)

            # 计算对话轮次
            turn_count = sum(1 for m in messages if str(m.get("role", "")).lower() == "assistant")

            row = {
                "id": f"traj-{idx}",
                "prompt": user_prompt,
                "category": _classify(user_prompt),
                "time_sensitive": _is_time_sensitive(user_prompt),
                "reference_answer": reference_answer,
                # 新增的丰富信息
                "tool_names": tool_names,
                "tool_count": len(tool_names),
                "tool_usage": dict(tool_counter),
                "subagent_usage": subagent_usage,
                "subagent_count": len(subagent_usage),
                "has_references": has_references,
                "reference_urls": reference_urls,
                "rag_used": rag_used,
                "web_search_used": web_search_used,
                "has_error": has_error,
                "task_success": task_success,
                "turn_count": turn_count,
                "response_length": len(reference_answer),
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

    # 统计新增的信息
    tool_stats = Counter()
    subagent_stats = Counter()
    for row in all_rows:
        for t in row.get("tool_names", []):
            tool_stats[t] += 1
        for s in row.get("subagent_usage", []):
            subagent_stats[s] += 1

    summary = {
        "all": len(all_rows),
        "knowledge": len(knowledge),
        "simulation": len(simulation),
        "analysis": len(analysis),
        "timesensitive": len(timesensitive),
        "stats": {
            "with_references": sum(1 for r in all_rows if r.get("has_references")),
            "with_rag": sum(1 for r in all_rows if r.get("rag_used")),
            "with_web_search": sum(1 for r in all_rows if r.get("web_search_used")),
            "with_errors": sum(1 for r in all_rows if r.get("has_error")),
            "task_success_rate": sum(1 for r in all_rows if r.get("task_success")) / max(1, len(all_rows)),
            "avg_tool_count": sum(r.get("tool_count", 0) for r in all_rows) / max(1, len(all_rows)),
            "avg_turn_count": sum(r.get("turn_count", 0) for r in all_rows) / max(1, len(all_rows)),
            "top_tools": dict(tool_stats.most_common(10)),
            "top_subagents": dict(subagent_stats.most_common(10)),
        }
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()