"""Typed contracts for the trace-derived retail-agent simulator."""

from typing import Annotated, Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, TypeAdapter


class TextMessage(Action):
    """An assistant's natural-language response."""

    type: Literal["text_message"] = "text_message"
    content: str = Field(min_length=1)


class ToolCallAction(Action):
    """An OpenAI-function-style tool invocation."""

    type: Literal["tool_call"] = "tool_call"
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


_ActionUnion = Annotated[TextMessage | ToolCallAction, Field(discriminator="type")]
_ACTION_ADAPTER = TypeAdapter(_ActionUnion)


class AgentSimAction(Action):
    """Schema facade that deserializes to one of the two supported agent actions."""

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> TextMessage | ToolCallAction:  # type: ignore[override]
        return _ACTION_ADAPTER.validate_python(obj, **kwargs)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return _ACTION_ADAPTER.json_schema(**kwargs)


class AgentSimObservation(Observation):
    """Non-leaking conversational state supplied to the training agent."""

    system_prompt: str
    tool_definitions: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    task_id: str
    latest_tool_result: Any = None
    remaining_steps: int


class AgentSimState(State):
    """Server-side episode progress without canonical-answer leakage."""

    task_id: str = ""
    expected_action_index: int = 0
    completed: bool = False
    last_reward: float = 0.0
