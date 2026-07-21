"""Isolated execution for generated Agent Sim grading code.

This runner is containment for generator-produced code, not a security boundary for
hostile Python. It uses a separate interpreter, a strict JSON protocol, a wall
clock timeout, and POSIX resource limits when available.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_S = 5.0
_RUNNER_PATH = Path(__file__).with_name("_runner_process.py")


@dataclass(frozen=True)
class GradeResult:
    """A normalized reward emitted by generated grader code."""

    reward: float
    reason: str


def execute_code(
    *,
    code: str,
    function_name: str,
    arguments: dict[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> GradeResult:
    """Execute one generated grading function in a separate Python process."""
    payload = {"code": code, "function_name": function_name, "arguments": arguments}
    try:
        completed = subprocess.run(
            [sys.executable, str(_RUNNER_PATH)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return GradeResult(reward=0.0, reason="grader execution failed or timed out")

    if completed.returncode or not completed.stdout.strip():
        return GradeResult(reward=0.0, reason="grader subprocess failed")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return GradeResult(reward=0.0, reason="grader returned invalid JSON")
    if not isinstance(result, dict) or result.get("status") != "ok":
        return GradeResult(
            reward=0.0, reason=str(result.get("reason", "grader failed"))
        )

    raw_reward = result.get("reward")
    if not isinstance(raw_reward, int | float) or isinstance(raw_reward, bool):
        return GradeResult(reward=0.0, reason="grader returned a non-numeric reward")
    return GradeResult(
        reward=max(0.0, min(1.0, float(raw_reward))),
        reason=str(result.get("reason", "")),
    )


def readonly_database(path: Path) -> str:
    """Return a SQLite URI that opens a database without write permission."""
    return f"file:{path.resolve().as_posix()}?mode=ro"


def temporary_readonly_permissions(paths: list[Path]) -> dict[Path, int]:
    """Apply best-effort read-only file modes while a child process runs."""
    modes: dict[Path, int] = {}
    for path in paths:
        try:
            modes[path] = os.stat(path).st_mode
            os.chmod(path, 0o444)
        except OSError:
            continue
    return modes


def restore_permissions(modes: dict[Path, int]) -> None:
    """Restore file modes changed by `temporary_readonly_permissions`."""
    for path, mode in modes.items():
        try:
            os.chmod(path, mode)
        except OSError:
            continue
