import os
import re


def restore_env_var(name: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous_value


def extract_chat_history_text(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("text") or payload.get("content") or "").strip()


_SIMULATOR_STRONG_RE = re.compile(
    r"(仿真|物理环境|抓取|夹爪|放置|轨迹|末端执行器|pybullet|gazebo|mcp|urdf|"
    r"pick[\s-]*and[\s-]*place|simulat(?:e|ion|or)|grasp|pick|place|trajectory|end[\s-]*effector)",
    re.IGNORECASE,
)
_ROBOT_SUBJECT_RE = re.compile(
    r"(机械臂|机器人|robot|robotic)",
    re.IGNORECASE,
)
_ROBOT_ACTION_RE = re.compile(
    r"(执行|运行|启动|验证|控制|规划|任务|移动|抓|放|execute|run|start|verify|control|plan|task|move)",
    re.IGNORECASE,
)
_PENDING_CONFIRM_RE = re.compile(
    r"("
    r"确认|确定|可以|开始|执行|运行|启动|按默认|默认参数|就这样|按这个|按该方案|"
    r"confirm|confirmed|start|run|execute|go ahead|proceed|use defaults?"
    r")",
    re.IGNORECASE,
)
_PENDING_MODIFY_RE = re.compile(
    r"("
    r"改一下|修改|调整|换成|不要.*默认|改为|换一个|"
    r"change|modify|adjust|instead"
    r")",
    re.IGNORECASE,
)
_PENDING_REJECT_RE = re.compile(
    r"(取消|不要执行|先不执行|停止|算了|拒绝|cancel|stop|do not execute|don't execute|reject)",
    re.IGNORECASE,
)


def text_mentions_simulator_task(text: str) -> bool:
    value = str(text or "")
    return bool(
        _SIMULATOR_STRONG_RE.search(value)
        or (_ROBOT_SUBJECT_RE.search(value) and _ROBOT_ACTION_RE.search(value))
    )


def infer_pending_action_response(user_message: str) -> str:
    text = str(user_message or "").strip()
    if not text:
        return "unclear"
    if _PENDING_REJECT_RE.search(text):
        return "rejected"
    if _PENDING_MODIFY_RE.search(text) and not _PENDING_CONFIRM_RE.search(text):
        return "modify"
    if _PENDING_CONFIRM_RE.search(text):
        return "confirmed"
    return "unclear"


def text_claims_simulator_execution(text: str) -> bool:
    value = str(text or "")
    return any(
        phrase in value
        for phrase in (
            "已委托",
            "已经委托",
            "已调用",
            "已经调用",
            "已启动",
            "启动 PyBullet",
            "启动 Gazebo",
            "正在等待",
            "已将任务委托",
            "已转交",
            "正在执行",
            "执行完成",
        )
    )


def text_waits_for_simulator_result(text: str) -> bool:
    value = str(text or "")
    return any(
        phrase in value
        for phrase in (
            "等待结果",
            "等待结果返回",
            "正在执行",
            "subagent 已启动",
            "正在等待",
            "等待 simulator 返回结果",
        )
    )


def normalize_intent_result(
    result: dict,
    pending_action: str = "",
    *,
    user_message: str = "",
    recent_context: str = "",
) -> dict:
    pending_response = str(
        result.get("pending_action_response") or "unclear"
    ).strip().lower()
    if pending_response not in {"confirmed", "modify", "rejected", "unclear"}:
        pending_response = "unclear"
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    guard_detected_simulator = text_mentions_simulator_task(user_message)
    context_mentions_simulator = text_mentions_simulator_task(recent_context)
    simulator_required = bool(result.get("simulator_required")) or guard_detected_simulator
    execution_confirmed = False
    if pending_action == "simulator":
        if pending_response == "unclear":
            pending_response = infer_pending_action_response(user_message)
        if pending_response == "unclear":
            # User continued conversation with pending simulator action;
            # default to confirmed unless clearly modifying/rejecting.
            execution_confirmed = True
            simulator_required = True
        elif pending_response == "confirmed":
            execution_confirmed = True
            simulator_required = True
        elif pending_response == "modify":
            simulator_required = True
        elif pending_response == "rejected":
            simulator_required = False
        elif context_mentions_simulator and guard_detected_simulator:
            simulator_required = True
    elif pending_response == "confirmed":
        pending_response = "unclear"

    return {
        "simulator_required": simulator_required,
        "execution_confirmed": execution_confirmed,
        "pending_action_response": pending_response,
        "confidence": confidence,
    }
