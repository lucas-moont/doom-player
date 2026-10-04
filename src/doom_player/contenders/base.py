"""What an in-process Map Contender provides to the Eval Suite.

Two kinds, both refereed by the AttemptSession:
- `Contender` (Door A): the suite's loop asks it for every decision.
- `WholeAttemptContender`: a learned Driver that plays each Attempt whole
  through the Map env it trained in, so it sees exactly its training view
  (ADR 0011).

An LLM implements neither: it drives the AttemptSession through the MCP
server's tools instead (Door B). See ADR 0007.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from doom_player.session import Observation

if TYPE_CHECKING:
    from doom_player.eval import EvalSpec  # eval imports this module, so only for type checkers


class Contender(Protocol):
    name: str  # the fixed name it has on the Scoreboard
    tics_per_action: int  # how long each decision is held, 1 to 35
    training: dict | None  # what its Training Run cost (`train.training_record`); None if untrained

    def reset(self, seed: int, buttons: list[str]) -> None:
        """Prepare for a new Attempt. `buttons` names each slot of an action."""

    def act(self, observation: Observation) -> list[bool]:
        """Choose which buttons to hold for the next `tics_per_action` tics."""


@runtime_checkable
class WholeAttemptContender(Protocol):
    name: str
    training: dict | None

    def play_attempt(self, spec: "EvalSpec", seed: int, on_frame: Callable | None = None) -> dict:
        """Play one Attempt of `spec` on game `seed`; return its AttemptRecord as a dict.

        `on_frame` gets every tic's screen, for video.
        """
