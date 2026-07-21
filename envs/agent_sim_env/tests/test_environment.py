import json
import sqlite3
from pathlib import Path

from agent_sim_env.models import TextMessage, ToolCallAction
from agent_sim_env.server.environment import AgentSimEnvironment


FINAL_GRADER = """
def grade(initial_db_path, final_db_path, task):
    import sqlite3
    with sqlite3.connect(final_db_path) as connection:
        completed = connection.execute("SELECT completed FROM jobs WHERE id = '1'").fetchone()[0]
    return {"reward": float(completed), "reason": "job completion"}
"""


def _artifacts(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with sqlite3.connect(data_dir / "store.db") as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, completed INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO jobs VALUES ('1', 0)")
    task = {
        "id": "task-1",
        "conversation_id": "conversation-1",
        "system_prompt": "Only complete known jobs.",
        "initial_messages": [
            {"role": "system", "content": "Only complete known jobs."},
            {"role": "user", "content": "Please complete job 1."},
        ],
        "tool_definitions": [],
        "expert_actions": [
            {
                "kind": "tool_call",
                "tool_name": "complete_job",
                "arguments": {"id": "1"},
            },
            {"kind": "text_message", "content": "The job is complete."},
        ],
        "final_state_code_grader": FINAL_GRADER,
    }
    (data_dir / "tasks.json").write_text(
        json.dumps({"tasks": [task]}), encoding="utf-8"
    )
    return data_dir


class FakeTools:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = db_path

    def call(self, name: str, arguments: dict[str, str]) -> dict[str, bool]:
        if name != "complete_job" or arguments.get("id") != "1":
            return {"success": False}
        with sqlite3.connect(self._db_path) as connection:
            connection.execute("UPDATE jobs SET completed = 1 WHERE id = '1'")
        return {"success": True}


def test_terminal_reward_combines_generated_final_state_and_step_graders(
    tmp_path: Path,
) -> None:
    env = AgentSimEnvironment(data_dir=_artifacts(tmp_path), tools_factory=FakeTools)
    env.reset(task_id="task-1")

    tool_observation = env.step(
        ToolCallAction(tool_name="complete_job", arguments={"id": "1"})
    )
    final_observation = env.step(TextMessage(content="The job is complete."))

    assert tool_observation.reward == 0.3
    assert final_observation.done
    assert final_observation.reward == 0.8


def test_out_of_order_action_is_matched_with_a_penalty(tmp_path: Path) -> None:
    env = AgentSimEnvironment(data_dir=_artifacts(tmp_path), tools_factory=FakeTools)
    env.reset(task_id="task-1")

    observation = env.step(TextMessage(content="The job is complete."))

    assert observation.reward == 0.15
    assert not observation.done


def test_reset_does_not_leak_expert_actions_or_code(tmp_path: Path) -> None:
    env = AgentSimEnvironment(data_dir=_artifacts(tmp_path), tools_factory=FakeTools)

    observation = env.reset(task_id="task-1")

    serialized = observation.model_dump_json()
    assert "expert_actions" not in serialized
    assert "final_state_code_grader" not in serialized
