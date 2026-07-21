"""Policy-code grading for agent actions.

`SimGenerator` replaces `GENERATED_POLICY_CODE` while preserving this stable
runtime wrapper.
"""

from typing import Any

from .runner import execute_code, GradeResult


GENERATED_POLICY_CODE = """
def grade(system_prompt, action, messages, latest_tool_result):
    return {"reward": 1.0, "reason": "No generated policy is installed."}
"""


class PolicyGrader:
    """Execute the generated global policy grader at every environment step."""

    def __init__(self, code: str = GENERATED_POLICY_CODE) -> None:
        self._code = code

    def grade(
        self,
        *,
        system_prompt: str,
        action: dict[str, Any],
        messages: list[dict[str, Any]],
        latest_tool_result: Any,
    ) -> GradeResult:
        """Return the generated policy score without exposing expert actions."""
        return execute_code(
            code=self._code,
            function_name="grade",
            arguments={
                "system_prompt": system_prompt,
                "action": action,
                "messages": messages,
                "latest_tool_result": latest_tool_result,
            },
        )
