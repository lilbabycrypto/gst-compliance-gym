from typing import Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, State


class GSTComplianceEnvironment(MCPEnvironment):
    """Indian GST compliance auditing environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        mcp = FastMCP("gst_compliance_gym")
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        return Observation(done=False, reward=0.0, metadata={"status": "ready"})

    def _step_impl(self, action, timeout_s=None, **kwargs) -> Observation:
        return Observation(
            done=False, reward=0.0, metadata={"error": "Unknown action"}
        )

    def step(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
