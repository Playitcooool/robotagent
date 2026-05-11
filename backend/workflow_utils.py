import os


def restore_env_var(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


def heuristic_simulator_intent(user_message: str, last_assistant_message: str = "") -> dict:
    """Deterministic fallback for short confirmations and common sim requests."""
    user_text = (user_message or "").strip().lower()
    assistant_text = (last_assistant_message or "").strip().lower()
    combined = f"{user_text}\n{assistant_text}"

    sim_keywords = (
        "仿真",
        "模拟",
        "物理",
        "机器人",
        "机械臂",
        "抓取",
        "放置",
        "轨迹",
        "路径规划",
        "运动规划",
        "pybullet",
        "gazebo",
        "simulator",
        "simulate",
        "simulation",
        "trajectory",
        "motion planning",
    )
    confirm_keywords = (
        "确认",
        "确认执行",
        "执行",
        "开始",
        "开始执行",
        "运行",
        "好的",
        "可以",
        "同意",
        "ok",
        "yes",
        "go",
        "run",
        "execute",
    )
    assistant_asked_confirmation = any(
        phrase in assistant_text
        for phrase in (
            "确认执行",
            "确认问题",
            "请回复",
            "同意这个",
            "我将立即调用 simulator",
            "调用 simulator",
            "运行仿真",
        )
    )
    sim_context = any(keyword in combined for keyword in sim_keywords)
    confirmed = any(keyword in user_text for keyword in confirm_keywords) and (
        assistant_asked_confirmation or any(keyword in assistant_text for keyword in sim_keywords)
    )
    return {
        "simulator_required": bool(sim_context or confirmed),
        "execution_confirmed": bool(confirmed),
    }
