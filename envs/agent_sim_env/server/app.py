"""FastAPI application for agent_sim_env."""

from openenv.core.env_server.http_server import create_app

try:
    from ..models import AgentSimAction, AgentSimObservation
    from .environment import AgentSimEnvironment
except ImportError:  # pragma: no cover
    from models import AgentSimAction, AgentSimObservation
    from server.environment import AgentSimEnvironment


app = create_app(
    AgentSimEnvironment,
    AgentSimAction,
    AgentSimObservation,
    env_name="agent_sim_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
