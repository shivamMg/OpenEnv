"""Terminal final-state code grading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runner import (
    execute_code,
    GradeResult,
    restore_permissions,
    temporary_readonly_permissions,
)


class FinalStateCodeGrader:
    """Run a task's generated `grade` function against database snapshots."""

    def grade(
        self,
        *,
        code: str,
        initial_db_path: Path,
        final_db_path: Path,
        task: dict[str, Any],
    ) -> GradeResult:
        """Return the normalized terminal score from isolated generated code."""
        modes = temporary_readonly_permissions([initial_db_path, final_db_path])
        try:
            return execute_code(
                code=code,
                function_name="grade",
                arguments={
                    "initial_db_path": str(initial_db_path),
                    "final_db_path": str(final_db_path),
                    "task": task,
                },
            )
        finally:
            restore_permissions(modes)
