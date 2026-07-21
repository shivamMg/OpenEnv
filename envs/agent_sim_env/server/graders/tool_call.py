"""Online tool-call similarity grading."""

import json
from collections import Counter
from typing import Any


class ToolCallGrader:
    """Grade one tool call using name and recursive expected-argument matches."""

    def grade(self, *, actual: dict[str, Any], expected: dict[str, Any]) -> float:
        """Return the reference tool-call score in the inclusive range `[0, 1]`."""
        return grade_tool_calls([actual], [expected])


def grade_tool_calls(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> float:
    """Score tool calls using the documented name-then-arguments algorithm."""
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0

    actual_names = [call.get("tool_name", "") for call in actual]
    expected_names = [call.get("tool_name", "") for call in expected]
    matched_names = sum((Counter(actual_names) & Counter(expected_names)).values())
    if not matched_names:
        return 0.0
    name_score = matched_names / max(len(actual), len(expected))
    if name_score < 1.0:
        return round(0.5 * name_score, 2)

    remaining = list(range(len(expected)))
    scores: list[float] = []
    for actual_call, name in zip(actual, actual_names):
        for expected_index in remaining:
            if name != expected_names[expected_index]:
                continue
            scores.append(
                _compare_args(
                    _parse_arguments(actual_call.get("arguments", {})),
                    _parse_arguments(expected[expected_index].get("arguments", {})),
                )
            )
            remaining.remove(expected_index)
            break
    return round(0.5 + 0.5 * (sum(scores) / len(scores)), 2)


def _parse_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def _compare_args(actual: Any, expected: Any) -> float:
    matched, total = _count_leaves(actual, expected)
    return matched / total if total else 1.0


def _count_leaves(actual: Any, expected: Any) -> tuple[int, int]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0, _leaf_count(expected)
        matched = total = 0
        for key, expected_value in expected.items():
            if key in actual:
                found, count = _count_leaves(actual[key], expected_value)
            else:
                found, count = 0, _leaf_count(expected_value)
            matched += found
            total += count
        return matched, total
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0, _leaf_count(expected)
        matched = total = 0
        for index, expected_value in enumerate(expected):
            if index < len(actual):
                found, count = _count_leaves(actual[index], expected_value)
            else:
                found, count = 0, _leaf_count(expected_value)
            matched += found
            total += count
        return matched, total
    return (1, 1) if actual == expected else (0, 1)


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(item) for item in value.values()) if value else 1
    if isinstance(value, list):
        return sum(_leaf_count(item) for item in value) if value else 1
    return 1
