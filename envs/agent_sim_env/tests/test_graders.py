import sqlite3
from pathlib import Path

from agent_sim_env.server.graders.final_state import FinalStateCodeGrader
from agent_sim_env.server.graders.policy import PolicyGrader
from agent_sim_env.server.graders.text_message import TextMessageGrader
from agent_sim_env.server.graders.tool_call import ToolCallGrader


def test_tool_call_grader_awards_partial_name_and_argument_scores() -> None:
    grader = ToolCallGrader()

    assert (
        grader.grade(
            actual={"tool_name": "lookup", "arguments": {"id": "1"}},
            expected={"tool_name": "lookup", "arguments": {"id": "1"}},
        )
        == 1.0
    )
    assert (
        grader.grade(
            actual={"tool_name": "other", "arguments": {"id": "1"}},
            expected={"tool_name": "lookup", "arguments": {"id": "1"}},
        )
        == 0.0
    )
    assert (
        grader.grade(
            actual={"tool_name": "lookup", "arguments": {"id": "wrong"}},
            expected={"tool_name": "lookup", "arguments": {"id": "1"}},
        )
        == 0.5
    )


def test_text_message_grader_uses_rouge_l() -> None:
    grader = TextMessageGrader()

    assert grader.grade("Please share your email.", "Please share your email.") == 1.0
    assert 0 < grader.grade("Please share email.", "Please share your email.") < 1
    assert grader.grade("anything", "") == 1.0


def test_final_state_code_grader_reads_initial_and_final_databases(
    tmp_path: Path,
) -> None:
    initial_db = tmp_path / "initial.db"
    final_db = tmp_path / "final.db"
    for database, status in ((initial_db, "pending"), (final_db, "cancelled")):
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE orders (status TEXT NOT NULL)")
            connection.execute("INSERT INTO orders VALUES (?)", (status,))

    code = """
def grade(initial_db_path, final_db_path, task):
    import sqlite3
    with sqlite3.connect(final_db_path) as connection:
        status = connection.execute("SELECT status FROM orders").fetchone()[0]
    return {"reward": 1.0 if status == "cancelled" else 0.0, "reason": status}
"""

    result = FinalStateCodeGrader().grade(
        code=code,
        initial_db_path=initial_db,
        final_db_path=final_db,
        task={"id": "task"},
    )

    assert result.reward == 1.0
    assert result.reason == "cancelled"


def test_policy_grader_normalizes_generated_policy_reward() -> None:
    code = """
def grade(system_prompt, action, messages, latest_tool_result):
    return {"reward": 2, "reason": "too high"}
"""

    result = PolicyGrader(code).grade(
        system_prompt="policy",
        action={"type": "text_message", "content": "hello"},
        messages=[],
        latest_tool_result=None,
    )

    assert result.reward == 1.0
    assert result.reason == "too high"
