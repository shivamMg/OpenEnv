"""Generate Agent Sim tasks, semantic SQLite state, tools, and code graders.

The generator intentionally keeps Azure OpenAI behind an injected client so tests
can use a deterministic fake and serving generated artifacts needs no credentials.
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


class SimGenerator:
    """Generate a complete Agent Sim artifact set from trace JSONL files."""

    def __init__(
        self,
        *,
        traces_dir: Path,
        output_dir: Path,
        llm: Any,
        deployment: str,
    ) -> None:
        self._traces_dir = traces_dir
        self._output_dir = output_dir
        self._llm = llm
        self._deployment = deployment

    def generate(self) -> None:
        """Generate and atomically publish tasks, store, tools, and policy code."""
        conversations = self._load_conversations()
        if not conversations:
            raise ValueError(f"No conversation traces found in {self._traces_dir}")
        tasks, examples = self._build_tasks(conversations)
        schema = self._ask_json(
            "Generate SQLite DDL and normalized seed rows that semantically model these "
            "tool definitions and observed tool responses. Do not create a generic tool "
            "call replay table. Return JSON with schema_sql and seed_rows.",
            {"tasks": tasks, "examples": examples},
        )
        tools = self._ask_json(
            "Generate Python source for a semantic SQLite tool dispatcher. It must define "
            "a Tools class whose __init__(db_path) stores the SQLite database path and whose "
            "call(name, arguments) method uses parameterized SQLite queries, performs mutations "
            "transactionally, and returns JSON-compatible values. Return JSON with tools_code only.",
            {"tool_definitions": self._tool_definitions(tasks), "schema": schema},
        )
        for task in tasks:
            response = self._ask_json(
                "Generate deterministic Python code defining grade(initial_db_path, "
                "final_db_path, task) that returns {'reward': float, 'reason': str}. "
                "Grade the final database state needed to solve this task.",
                {"task": task},
            )
            task["final_state_code_grader"] = self._required_string(response, "code")
        policy = self._ask_json(
            "Generate Python code defining grade(system_prompt, action, messages, "
            "latest_tool_result) that returns {'reward': float, 'reason': str}. It must "
            "enforce the shared system policies without accessing expert actions. Return "
            "JSON with code only.",
            {"system_prompts": sorted({task["system_prompt"] for task in tasks})},
        )
        self._publish(tasks, examples, schema, tools, policy)

    def _load_conversations(self) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in sorted(self._traces_dir.glob("*.jsonl")):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    wrapper = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed JSON in {path}:{line_number}"
                    ) from error
                for trace in wrapper.get("traces", []):
                    conversation_id = trace.get("ConversationId") or wrapper.get(
                        "conversation_id"
                    )
                    if not isinstance(conversation_id, str) or not conversation_id:
                        raise ValueError(
                            f"Trace in {path}:{line_number} has no conversation ID"
                        )
                    groups[conversation_id].append(trace)
        return groups

    def _build_tasks(
        self, conversations: dict[str, list[dict[str, Any]]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tasks: list[dict[str, Any]] = []
        examples: list[dict[str, Any]] = []
        for index, (conversation_id, traces) in enumerate(
            sorted(conversations.items()), 1
        ):
            traces.sort(
                key=lambda trace: (trace.get("TimeStamp", ""), trace.get("TraceId", ""))
            )
            snapshots_by_input_count: dict[int, dict[str, Any]] = {}
            for trace in traces:
                snapshots_by_input_count.setdefault(
                    len(trace.get("InputMessages") or []), trace
                )
            canonical_traces = [
                snapshots_by_input_count[count]
                for count in sorted(snapshots_by_input_count)
            ]
            first = next(
                (trace for trace in canonical_traces if trace.get("InputMessages")),
                canonical_traces[0],
            )
            responses = self._responses_by_call_id(traces)
            actions: list[dict[str, Any]] = []
            for trace in canonical_traces:
                action = self._action_from_trace(trace)
                if action is None:
                    continue
                if action["kind"] == "tool_call":
                    result = responses.get(action.pop("call_id"))
                    if result is None:
                        raise ValueError(
                            f"Missing response for {action['tool_name']} in {conversation_id}"
                        )
                    examples.append({"tool_call": action.copy(), "response": result})
                actions.append(action)
            if not actions:
                raise ValueError(
                    f"Conversation {conversation_id} has no assistant actions"
                )
            input_messages = first.get("InputMessages") or []
            initial_messages = [
                {"role": message["role"], "content": self._message_content(message)}
                for message in input_messages
                if message.get("role") in {"system", "user"}
            ]
            system_prompt = next(
                (
                    message["content"]
                    for message in initial_messages
                    if message["role"] == "system"
                ),
                "",
            )
            tasks.append(
                {
                    "id": f"trace_{index:04d}",
                    "conversation_id": conversation_id,
                    "source_trace_ids": [
                        trace.get("TraceId") for trace in canonical_traces
                    ],
                    "system_prompt": system_prompt,
                    "initial_messages": initial_messages,
                    "tool_definitions": next(
                        (
                            trace.get("ToolDefinitions")
                            for trace in canonical_traces
                            if trace.get("ToolDefinitions")
                        ),
                        [],
                    ),
                    "expert_actions": actions,
                }
            )
        return tasks, examples

    def _publish(
        self,
        tasks: list[dict[str, Any]],
        examples: list[dict[str, Any]],
        schema: dict[str, Any],
        tools: dict[str, Any],
        policy: dict[str, Any],
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-sim-") as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            graders_dir = root / "server" / "graders"
            data_dir.mkdir(parents=True)
            graders_dir.mkdir(parents=True)
            self._create_store(data_dir / "store.db", schema)
            (data_dir / "tasks.json").write_text(
                json.dumps({"schema_version": 1, "tasks": tasks}, indent=2) + "\n",
                encoding="utf-8",
            )
            tools_code = self._required_string(tools, "tools_code")
            self._validate_python(tools_code, "Tools")
            (root / "server" / "tools.py").write_text(tools_code, encoding="utf-8")
            policy_code = self._required_string(policy, "code")
            self._validate_python(policy_code, "grade")
            (graders_dir / "policy.py").write_text(
                self._render_policy_module(policy_code), encoding="utf-8"
            )
            self._validate_artifacts(root, tasks, examples)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            for relative in ("data", "server/tools.py", "server/graders/policy.py"):
                source = root / relative
                target = self._output_dir / relative
                if source.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(source, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)

    def _create_store(self, path: Path, schema: dict[str, Any]) -> None:
        schema_sql = self._required_string(schema, "schema_sql")
        rows = schema.get("seed_rows")
        if not isinstance(rows, list):
            raise ValueError("schema generation must return a seed_rows list")
        connection = sqlite3.connect(path)
        try:
            connection.executescript(schema_sql)
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("seed row must be an object")
                table, values = row.get("table"), row.get("values")
                if (
                    not isinstance(table, str)
                    or not isinstance(values, dict)
                    or not values
                ):
                    raise ValueError("seed row requires table and non-empty values")
                columns = list(values)
                placeholders = ", ".join("?" for _ in columns)
                connection.execute(
                    f"INSERT INTO {self._quoted_identifier(table)} "
                    f"({', '.join(self._quoted_identifier(column) for column in columns)}) "
                    f"VALUES ({placeholders})",
                    tuple(values[column] for column in columns),
                )
            connection.commit()
        finally:
            connection.close()

    def _validate_artifacts(
        self,
        root: Path,
        tasks: list[dict[str, Any]],
        examples: list[dict[str, Any]],
    ) -> None:
        tools_path = root / "server" / "tools.py"
        namespace: dict[str, Any] = {}
        exec(tools_path.read_text(encoding="utf-8"), namespace)  # noqa: S102 - generated validation
        if not callable(namespace.get("Tools")):
            raise ValueError("generated tools.py must define a Tools class")
        for example in examples:
            temporary_store = root / "validation.db"
            self._copy_database(root / "data" / "store.db", temporary_store)
            try:
                call = example["tool_call"]
                actual = self._call_generated_tool(
                    tools_path,
                    temporary_store,
                    call["tool_name"],
                    call["arguments"],
                )
            finally:
                temporary_store.unlink(missing_ok=True)
            if actual != example["response"]:
                raise ValueError(
                    "generated tools do not reproduce the observed response for "
                    f"{call['tool_name']}: expected {example['response']!r}, got {actual!r}"
                )
        for task in tasks:
            self._validate_python(task["final_state_code_grader"], "grade")

    @staticmethod
    def _copy_database(source: Path, target: Path) -> None:
        source_connection = sqlite3.connect(source)
        target_connection = sqlite3.connect(target)
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
            source_connection.close()

    @staticmethod
    def _call_generated_tool(
        tools_path: Path, db_path: Path, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        script = """
import importlib.util
import json
import sys

payload = json.loads(sys.stdin.read())
spec = importlib.util.spec_from_file_location("generated_tools", payload["tools_path"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tools = module.Tools(payload["db_path"])
result = tools.call(payload["tool_name"], payload["arguments"])
print(json.dumps(result))
"""
        payload = {
            "tools_path": str(tools_path),
            "db_path": str(db_path),
            "tool_name": tool_name,
            "arguments": arguments,
        }
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise ValueError(
                f"generated tool failed validation: {completed.stderr[:500]}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("generated tool returned non-JSON output") from error

    def _ask_json(self, instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
        completion = self._llm.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        try:
            value = json.loads(content or "")
        except json.JSONDecodeError as error:
            raise ValueError("LLM did not return a JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("LLM did not return a JSON object")
        return value

    @staticmethod
    def _required_string(value: dict[str, Any], key: str) -> str:
        result = value.get(key)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"LLM output is missing non-empty {key!r}")
        return result

    @staticmethod
    def _validate_python(code: str, required_name: str) -> None:
        namespace: dict[str, Any] = {}
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - validation only
        if not callable(namespace.get(required_name)):
            raise ValueError(f"generated code must define {required_name}")

    @staticmethod
    def _render_policy_module(code: str) -> str:
        return f'''"""Generated global policy grader. Do not edit manually."""

from typing import Any

from .runner import GradeResult, execute_code


GENERATED_POLICY_CODE = {code!r}


class PolicyGrader:
    """Run the generated global policy checker."""

    def grade(
        self,
        *,
        system_prompt: str,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        latest_tool_result: Any,
    ) -> GradeResult:
        return execute_code(
            code=GENERATED_POLICY_CODE,
            function_name="grade",
            arguments={{
                "system_prompt": system_prompt,
                "action": action,
                "messages": messages,
                "latest_tool_result": latest_tool_result,
            }},
        )
'''

    @staticmethod
    def _quoted_identifier(value: str) -> str:
        if not value.replace("_", "").isalnum():
            raise ValueError(f"Unsafe SQL identifier {value!r}")
        return f'"{value}"'

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        return "\n".join(
            part.get("content", "")
            for part in message.get("parts", [])
            if isinstance(part, dict) and isinstance(part.get("content"), str)
        )

    @staticmethod
    def _action_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
        messages = trace.get("OutputMessages") or []
        if not messages:
            return None
        part = (messages[0].get("parts") or [{}])[0]
        if part.get("type") == "tool_call":
            try:
                arguments = json.loads(part.get("arguments", "{}"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed tool arguments for {part.get('name')!r}"
                ) from error
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            return {
                "kind": "tool_call",
                "tool_name": str(part.get("name", "")),
                "arguments": arguments,
                "call_id": part.get("id"),
            }
        content = part.get("content")
        return (
            {"kind": "text_message", "content": content}
            if isinstance(content, str)
            else None
        )

    @staticmethod
    def _responses_by_call_id(traces: list[dict[str, Any]]) -> dict[str, Any]:
        responses: dict[str, Any] = {}
        for trace in traces:
            for message in trace.get("InputMessages") or []:
                if message.get("role") != "tool":
                    continue
                for part in message.get("parts") or []:
                    if part.get("type") != "tool_call_response" or not part.get("id"):
                        continue
                    raw_response = part.get("response")
                    try:
                        responses[part["id"]] = json.loads(raw_response)
                    except (TypeError, json.JSONDecodeError):
                        responses[part["id"]] = raw_response
        return responses

    @staticmethod
    def _tool_definitions(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [definition for task in tasks for definition in task["tool_definitions"]]


def main() -> None:
    """Generate artifacts with Azure OpenAI credentials from the environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=Path("traces"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    arguments = parser.parse_args()
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    SimGenerator(
        traces_dir=arguments.traces_dir,
        output_dir=arguments.output_dir,
        llm=client,
        deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    ).generate()


if __name__ == "__main__":
    main()
