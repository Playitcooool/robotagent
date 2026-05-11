import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import workflow_utils


def test_heuristic_detects_short_execution_confirmation_after_sim_plan():
    result = workflow_utils.heuristic_simulator_intent(
        "确认执行",
        "我会先初始化 PyBullet 仿真，然后运行双球碰撞。请回复确认执行。",
    )

    assert result == {"simulator_required": True, "execution_confirmed": True}


def test_heuristic_does_not_treat_plain_confirmation_as_sim_without_context():
    result = workflow_utils.heuristic_simulator_intent("好的", "这是一个普通问答。")

    assert result == {"simulator_required": False, "execution_confirmed": False}


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
