"""Generated SQLite-backed tool dispatcher placeholder.

Run `python -m agent_sim_env.scripts.sim_generator` with Azure credentials to
replace this module with tool definitions specialized to the current trace set.
"""

from typing import Any


class Tools:
    """Dispatch generated tools against one episode's SQLite database."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, str]:
        """Return a deterministic error until generated tools are installed."""
        del arguments
        return {"error": f"No generated implementation for tool: {name}"}
