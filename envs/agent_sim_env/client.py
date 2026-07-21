"""Client for the trace-derived retail agent simulator."""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import AgentSimAction, AgentSimObservation, AgentSimState


class AgentSimEnv(EnvClient[AgentSimAction, AgentSimObservation, AgentSimState]):
    """Persistent WebSocket client for assistant-message and tool-call actions."""

    def _step_payload(self, action: AgentSimAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[AgentSimObservation]:
        data = payload.get("observation", {})
        observation = AgentSimObservation(
            system_prompt=data.get("system_prompt", ""),
            tool_definitions=data.get("tool_definitions", []),
            messages=data.get("messages", []),
            task_id=data.get("task_id", ""),
            latest_tool_result=data.get("latest_tool_result"),
            remaining_steps=data.get("remaining_steps", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> AgentSimState:
        return AgentSimState(**payload)
