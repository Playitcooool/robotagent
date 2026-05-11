import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import workflow_utils


def test_workflow_utils_do_not_expose_keyword_intent_fallbacks():
    assert not hasattr(workflow_utils, "heuristic_simulator_intent")
    assert not hasattr(workflow_utils, "is_short_sim_confirmation")


@pytest.mark.parametrize(
    "previous",
    [None, "", "0", "custom"],
)
def test_restore_rag_disabled_environment(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv("RAG_DISABLED", raising=False)
    else:
        monkeypatch.setenv("RAG_DISABLED", previous)

    monkeypatch.setenv("RAG_DISABLED", "1")
    workflow_utils.restore_env_var("RAG_DISABLED", previous)

    assert workflow_utils.os.environ.get("RAG_DISABLED") == previous


def test_extract_chat_history_text_accepts_text_and_content_fields():
    assert workflow_utils.extract_chat_history_text({"text": "saved text"}) == "saved text"
    assert workflow_utils.extract_chat_history_text({"content": "saved content"}) == "saved content"


def test_intent_normalization_requires_pending_simulator_for_confirmation():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": False,
            "execution_confirmed": True,
            "pending_action_response": "confirmed",
            "confidence": 0.99,
        },
        pending_action="",
        user_message="执行一个机械臂抓取放置任务",
    )

    assert result["simulator_required"] is True
    assert result["execution_confirmed"] is False
    assert result["pending_action_response"] == "unclear"


def test_intent_normalization_confirms_only_saved_simulator_action():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": False,
            "execution_confirmed": False,
            "pending_action_response": "confirmed",
            "confidence": 0.9,
        },
        pending_action="simulator",
    )

    assert result["simulator_required"] is True
    assert result["execution_confirmed"] is True


def test_intent_normalization_does_not_auto_confirm_pending_simulator_task():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": True,
            "execution_confirmed": True,
            "pending_action_response": "unclear",
            "confidence": 0.95,
        },
        pending_action="simulator",
    )

    assert result["simulator_required"] is True
    assert result["execution_confirmed"] is False


def test_simulator_guard_overrides_false_negative_intent_for_robot_task():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": False,
            "execution_confirmed": False,
            "pending_action_response": "unclear",
            "confidence": 0.95,
        },
        pending_action="",
        user_message="请执行一个机械臂抓取放置任务",
    )

    assert result["simulator_required"] is True
    assert result["execution_confirmed"] is False


def test_simulator_guard_does_not_route_plain_robot_questions():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": False,
            "execution_confirmed": False,
            "pending_action_response": "unclear",
            "confidence": 0.95,
        },
        pending_action="",
        user_message="机器人是什么？",
    )

    assert result["simulator_required"] is False
    assert result["execution_confirmed"] is False


def test_future_execution_planning_text_is_not_a_simulator_execution_claim():
    text = "我理解您想执行机械臂抓取放置任务，需要先获取关键参数并确认计划后再执行。"

    assert workflow_utils.text_claims_simulator_execution(text) is False


def test_active_execution_text_is_a_simulator_execution_claim():
    assert workflow_utils.text_claims_simulator_execution("正在执行 PyBullet 机械臂任务") is True
    assert workflow_utils.text_claims_simulator_execution("已调用 simulator 执行仿真") is True


def test_waiting_for_simulator_result_text_is_detected():
    assert (
        workflow_utils.text_waits_for_simulator_result(
            "仿真器正在执行抓取放置任务，让我等待 simulator 返回结果。"
        )
        is True
    )
    assert workflow_utils.text_waits_for_simulator_result("仿真执行已完成") is False


def test_pending_simulator_confirmation_can_be_inferred_from_user_message():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": False,
            "execution_confirmed": False,
            "pending_action_response": "unclear",
            "confidence": 0.95,
        },
        pending_action="simulator",
        user_message="按默认参数执行",
    )

    assert result["simulator_required"] is True
    assert result["execution_confirmed"] is True
    assert result["pending_action_response"] == "confirmed"


def test_pending_simulator_start_execution_confirms_even_after_intent_fallback():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": True,
            "execution_confirmed": False,
            "pending_action_response": "unclear",
            "confidence": 0.0,
        },
        pending_action="simulator",
        user_message="开始执行",
        recent_context="assistant: ## 机械臂抓取放置任务计划（PyBullet）",
    )

    assert result["simulator_required"] is True
    assert result["execution_confirmed"] is True
    assert result["pending_action_response"] == "confirmed"


def test_pending_simulator_rejection_clears_simulator_requirement():
    result = workflow_utils.normalize_intent_result(
        {
            "simulator_required": True,
            "execution_confirmed": False,
            "pending_action_response": "unclear",
            "confidence": 0.95,
        },
        pending_action="simulator",
        user_message="取消，不要执行",
    )

    assert result["simulator_required"] is False
    assert result["execution_confirmed"] is False
    assert result["pending_action_response"] == "rejected"
