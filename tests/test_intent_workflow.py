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
