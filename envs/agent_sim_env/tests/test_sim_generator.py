import json
import sqlite3
from pathlib import Path

from agent_sim_env.scripts.sim_generator import SimGenerator


class FakeCompletions:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = iter(responses)

    def create(self, **_: object) -> object:
        payload = next(self._responses)
        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": json.dumps(payload)}
                            )()
                        },
                    )
                ]
            },
        )()


class FakeAzureOpenAI:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


def _trace() -> dict[str, object]:
    return {
        "conversation_id": "conversation-1",
        "traces": [
            {
                "ConversationId": "conversation-1",
                "TraceId": "trace-1",
                "InputMessages": [
                    {"role": "system", "parts": [{"content": "Follow the policy."}]},
                    {"role": "user", "parts": [{"content": "Find my order."}]},
                ],
                "OutputMessages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "find_order",
                                "arguments": '{"order_id":"A1"}',
                            }
                        ],
                    },
                ],
                "ToolDefinitions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "find_order",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"],
                            },
                        },
                    }
                ],
            },
            {
                "ConversationId": "conversation-1",
                "TraceId": "trace-2",
                "InputMessages": [
                    {
                        "role": "system",
                        "parts": [{"content": "Follow the policy."}],
                    },
                    {
                        "role": "user",
                        "parts": [{"content": "Find my order."}],
                    },
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "id": "call-1",
                                "response": '{"order_id":"A1","status":"pending"}',
                            }
                        ],
                    },
                ],
                "OutputMessages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {"type": "text", "content": "Your order is pending."}
                        ],
                    }
                ],
                "ToolDefinitions": [],
            },
        ],
    }


def test_generator_emits_one_task_semantic_store_and_tools(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(json.dumps(_trace()) + "\n", encoding="utf-8")
    responses = [
        {
            "schema_sql": "CREATE TABLE orders (order_id TEXT PRIMARY KEY, status TEXT NOT NULL);",
            "seed_rows": [
                {"table": "orders", "values": {"order_id": "A1", "status": "pending"}}
            ],
        },
        {
            "tools_code": """
import sqlite3

class Tools:
    def __init__(self, db_path):
        self._db_path = db_path

    def call(self, name, arguments):
        if name != "find_order":
            return {"error": "Unknown tool"}
        with sqlite3.connect(self._db_path) as connection:
            row = connection.execute("SELECT order_id, status FROM orders WHERE order_id = ?", (arguments["order_id"],)).fetchone()
        return {"order_id": row[0], "status": row[1]} if row else {"error": "Order not found"}
""",
        },
        {
            "code": """
def grade(initial_db_path, final_db_path, task):
    return {"reward": 1.0, "reason": "verified"}
""",
        },
        {
            "code": """
def grade(system_prompt, action, messages, latest_tool_result):
    return {"reward": 1.0, "reason": "policy followed"}
""",
        },
    ]

    output = tmp_path / "output"
    SimGenerator(
        traces_dir=tmp_path,
        output_dir=output,
        llm=FakeAzureOpenAI(responses),
        deployment="test-deployment",
    ).generate()

    tasks = json.loads((output / "data" / "tasks.json").read_text(encoding="utf-8"))[
        "tasks"
    ]
    assert len(tasks) == 1
    assert tasks[0]["expert_actions"] == [
        {
            "kind": "tool_call",
            "tool_name": "find_order",
            "arguments": {"order_id": "A1"},
        },
        {"kind": "text_message", "content": "Your order is pending."},
    ]
    assert "final_state_code_grader" in tasks[0]
    with sqlite3.connect(output / "data" / "store.db") as connection:
        assert (
            connection.execute(
                "SELECT status FROM orders WHERE order_id = 'A1'"
            ).fetchone()[0]
            == "pending"
        )
    assert (output / "server" / "tools.py").exists()
    assert (output / "server" / "graders" / "policy.py").exists()
