"""What every in-process Contender provides (Door A of the Eval Suite).

An LLM does not implement this: it drives the AttemptSession through the
MCP server's tools instead (Door B). See ADR 0007.
"""

from typing import Protocol

from doom_player.session import Observation


class Contender(Protocol):
    name: str  # the fixed name it has on the Scoreboard
    tics_per_action: int  # how long each decision is held, 1 to 35

    def reset(self, seed: int, buttons: list[str]) -> None:
        """Prepare for a new Attempt. `buttons` names each slot of an action."""

    def act(self, observation: Observation) -> list[bool]:
        """Choose which buttons to hold for the next `tics_per_action` tics."""
