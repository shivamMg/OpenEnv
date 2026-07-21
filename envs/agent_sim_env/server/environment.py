"""Runtime for generated, trace-derived customer-service simulations."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        AgentSimAction,
        AgentSimObservation,
        AgentSimState,
        TextMessage,
        ToolCallAction,
    )
    from .graders import (
        FinalStateCodeGrader,
        PolicyGrader,
        TextMessageGrader,
        ToolCallGrader,
    )
except ImportError:  # pragma: no cover
    from models import (
        AgentSimAction,
        AgentSimObservation,
        AgentSimState,
        TextMessage,
        ToolCallAction,
    )
    from server.graders import (
        FinalStateCodeGrader,
        PolicyGrader,
        TextMessageGrader,
        ToolCallGrader,
    )


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MAX_STEPS = 25
OUT_OF_ORDER_PENALTY = 0.5
TOOL_MATCH_THRESHOLD = 0.8
TEXT_MATCH_THRESHOLD = 0.6
WEIGHTS = {"final_state": 0.6, "tool_call": 0.2, "text_message": 0.1, "policy": 0.1}

ToolDispatcher = Callable[[str | Path, str, dict[str, Any]], Any]


class AgentSimEnvironment(
    Environment[AgentSimAction, AgentSimObservation, AgentSimState]
):
    """Execute generated tools against an isolated SQLite store and grade actions."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *,
        data_dir: Path = DATA_DIR,
        tool_dispatcher: ToolDispatcher | None = None,
        policy_grader: PolicyGrader | None = None,
    ) -> None:
        self._data_dir = data_dir
        tasks_path = data_dir / "tasks.json"
        store_path = data_dir / "store.db"
        if not tasks_path.exists() or not store_path.exists():
            raise FileNotFoundError(
                "Generated Agent Sim artifacts are missing. Run scripts/sim_generator.py first."
            )
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        self._tasks = payload["tasks"]
        self._store_path = store_path
        self._tool_dispatcher = tool_dispatcher or self._load_tool_dispatcher()
        self._policy_grader = policy_grader or PolicyGrader()
        self._tool_grader = ToolCallGrader()
        self._text_grader = TextMessageGrader()
        self._final_state_grader = FinalStateCodeGrader()
        self._session_dir: Path | None = None
        self._initial_db_path: Path | None = None
        self._live_db_path: Path | None = None
        self._task: dict[str, Any] | None = None
        self._messages: list[dict[str, Any]] = []
        self._latest_tool_result: Any = None
        self._matched_action_indices: set[int] = set()
        self._state = AgentSimState(episode_id=str(uuid4()))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentSimObservation:
        """Start a fresh episode with isolated baseline and initial DB snapshots."""
        self.close()
        task_id = kwargs.get("task_id")
        if task_id is None:
            self._task = self._tasks[(seed or 0) % len(self._tasks)]
        else:
            self._task = next(
                (task for task in self._tasks if task["id"] == task_id), None
            )
            if self._task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
        self._session_dir = Path(tempfile.mkdtemp(prefix="agent-sim-"))
        self._initial_db_path = self._session_dir / "initial.db"
        self._live_db_path = self._session_dir / "live.db"
        self._copy_database(self._store_path, self._initial_db_path)
        self._copy_database(self._store_path, self._live_db_path)
        self._messages = copy.deepcopy(self._task["initial_messages"])
        self._latest_tool_result = None
        self._matched_action_indices = set()
        self._state = AgentSimState(
            episode_id=episode_id or str(uuid4()), task_id=self._task["id"]
        )
        return self._observation()

    def step(
        self, action: AgentSimAction, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> AgentSimObservation:
        """Apply one action, return dense online reward, and terminally grade state."""
        if self._task is None or self._live_db_path is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.completed:
            return self._observation(done=True, reward=0.0)

        self._state.step_count += 1
        self._append_action(action)
        candidate_index, raw_score, out_of_order = self._grade_against_experts(action)
        shape_score = raw_score * (OUT_OF_ORDER_PENALTY if out_of_order else 1.0)
        component = (
            "tool_call" if isinstance(action, ToolCallAction) else "text_message"
        )
        threshold = (
            TOOL_MATCH_THRESHOLD if component == "tool_call" else TEXT_MATCH_THRESHOLD
        )
        if candidate_index is not None and raw_score >= threshold:
            self._matched_action_indices.add(candidate_index)

        if isinstance(action, ToolCallAction):
            self._latest_tool_result = self._tool_dispatcher(
                str(self._live_db_path), action.tool_name, action.arguments
            )
            self._messages.append(
                {
                    "role": "tool",
                    "name": action.tool_name,
                    "content": json.dumps(self._latest_tool_result),
                }
            )

        policy_result = self._policy_grader.grade(
            system_prompt=self._task["system_prompt"],
            action=action.model_dump(),
            messages=copy.deepcopy(self._messages),
            latest_tool_result=copy.deepcopy(self._latest_tool_result),
        )
        expert_actions = self._task["expert_actions"]
        self._state.completed = (
            len(self._matched_action_indices) == len(expert_actions)
            or self._state.step_count >= MAX_STEPS
        )
        final_score = self._final_score() if self._state.completed else 0.0
        reward = round(
            WEIGHTS[component] * shape_score
            + WEIGHTS["policy"] * policy_result.reward
            + WEIGHTS["final_state"] * final_score,
            6,
        )
        self._state.last_reward = reward
        return self._observation(done=self._state.completed, reward=reward)

    @property
    def state(self) -> AgentSimState:
        """Return public non-leaking runtime state."""
        return self._state

    def close(self) -> None:
        """Remove temporary episode database artifacts."""
        if self._session_dir is not None:
            shutil.rmtree(self._session_dir, ignore_errors=True)
        self._session_dir = self._initial_db_path = self._live_db_path = None

    def _grade_against_experts(
        self, action: TextMessage | ToolCallAction
    ) -> tuple[int | None, float, bool]:
        assert self._task is not None
        expert_actions = self._task["expert_actions"]
        current_index = next(
            (
                index
                for index in range(len(expert_actions))
                if index not in self._matched_action_indices
            ),
            None,
        )
        if current_index is None:
            return None, 0.0, False
        current_score = self._action_score(action, expert_actions[current_index])
        if current_score > 0:
            return current_index, current_score, False

        candidates = [
            (index, self._action_score(action, expected))
            for index, expected in enumerate(
                expert_actions[current_index + 1 :], current_index + 1
            )
            if index not in self._matched_action_indices
            and self._same_kind(action, expected)
        ]
        if not candidates:
            return current_index, 0.0, False
        best_index, best_score = max(
            candidates, key=lambda candidate: (candidate[1], -candidate[0])
        )
        return best_index, best_score, best_score > 0

    def _action_score(
        self, action: TextMessage | ToolCallAction, expected: dict[str, Any]
    ) -> float:
        if isinstance(action, ToolCallAction):
            if expected.get("kind") != "tool_call":
                return 0.0
            return self._tool_grader.grade(
                actual={"tool_name": action.tool_name, "arguments": action.arguments},
                expected={
                    "tool_name": expected["tool_name"],
                    "arguments": expected["arguments"],
                },
            )
        if expected.get("kind") != "text_message":
            return 0.0
        return self._text_grader.grade(action.content, expected.get("content", ""))

    @staticmethod
    def _same_kind(
        action: TextMessage | ToolCallAction, expected: dict[str, Any]
    ) -> bool:
        return (
            isinstance(action, ToolCallAction) and expected.get("kind") == "tool_call"
        ) or (
            isinstance(action, TextMessage) and expected.get("kind") == "text_message"
        )

    def _append_action(self, action: TextMessage | ToolCallAction) -> None:
        if isinstance(action, TextMessage):
            self._messages.append({"role": "assistant", "content": action.content})
        else:
            self._messages.append(
                {
                    "role": "assistant",
                    "tool_call": {
                        "name": action.tool_name,
                        "arguments": action.arguments,
                    },
                }
            )

    def _final_score(self) -> float:
        assert (
            self._task is not None
            and self._initial_db_path is not None
            and self._live_db_path is not None
        )
        final_path = self._session_dir / "final.db"  # type: ignore[operator]
        self._copy_database(self._live_db_path, final_path)
        return self._final_state_grader.grade(
            code=self._task["final_state_code_grader"],
            initial_db_path=self._initial_db_path,
            final_db_path=final_path,
            task={
                "id": self._task["id"],
                "conversation_id": self._task["conversation_id"],
            },
        ).reward

    def _observation(
        self, *, done: bool = False, reward: float | None = None
    ) -> AgentSimObservation:
        assert self._task is not None
        return AgentSimObservation(
            system_prompt=self._task["system_prompt"],
            tool_definitions=self._task["tool_definitions"],
            messages=copy.deepcopy(self._messages),
            task_id=self._task["id"],
            latest_tool_result=copy.deepcopy(self._latest_tool_result),
            remaining_steps=max(0, MAX_STEPS - self._state.step_count),
            done=done,
            reward=reward,
        )

    def _load_tool_dispatcher(self) -> ToolDispatcher:
        module = importlib.import_module("agent_sim_env.server.tools")
        dispatcher = getattr(module, "call_tool", None)
        if not callable(dispatcher):
            raise RuntimeError("Generated server/tools.py must define call_tool()")
        return dispatcher

    @staticmethod
    def _copy_database(source: Path, target: Path) -> None:
        source_connection = sqlite3.connect(source)
        target_connection = sqlite3.connect(target)
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
            source_connection.close()
