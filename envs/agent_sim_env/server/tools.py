"""Generated SQLite-backed tool dispatcher placeholder.

Run `python -m agent_sim_env.scripts.sim_generator` with Azure credentials to
replace this module with tool definitions specialized to the current trace set.
"""

from __future__ import annotations

from typing import Any


def call_tool(
    db_path: str, tool_name: str, arguments: dict[str, Any]
) -> dict[str, str]:
    """Return a deterministic error until generated tools are installed."""
    del db_path, arguments
    return {"error": f"No generated implementation for tool: {tool_name}"}
