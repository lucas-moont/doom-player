"""The Doom MCP server: Door B of the Eval Suite.

An MCP client (Claude Code, or any other) connects and plays one Attempt
through three tools: `look`, `act` and `automap`. The game runs in ViZDoom's
synchronous mode, so it waits while the client thinks.

The server plays the same AttemptSession as the Eval Suite's own loop, so the
rules and the record are identical. The seed comes from the command line,
never from the client. Every tool returns Human-equivalent Observations only:
images of the screen or automap, numbers the HUD shows, and events a player
would notice. Progress is measured but never returned.

Run with `uv run doom-mcp` (stdio). With `--spec`, the record goes to
`results/` like any Door A Attempt; without it, the Attempt is a practice run
and its record goes to `runs/mcp/` (gitignored).
"""

import argparse
import io
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from PIL import Image as PILImage

from doom_player.eval import RESULTS_DIR, SPECS, append_record, attempts_path
from doom_player.paths import REPO_ROOT
from doom_player.session import MAX_TICS_PER_ACTION, AttemptSession

PRACTICE_DIR = REPO_ROOT / "runs" / "mcp"
DEFAULT_TICS = 8


def png(frame) -> Image:
    buffer = io.BytesIO()
    PILImage.fromarray(frame).save(buffer, format="PNG")
    return Image(data=buffer.getvalue(), format="png")


class Game:
    """One Attempt behind the tools; the tools never touch the session directly."""

    def __init__(self, session: AttemptSession, record_path: Path, spec: str | None):
        self.session = session
        self.record_path = record_path
        self.spec = spec
        self.lock = threading.Lock()  # tool calls may arrive on different threads
        self.last_frame = None
        self.last_hud = None
        self.saved = False

    @property
    def over(self) -> bool:
        # Once saved, the game is closed and must not be asked anything.
        return self.saved or self.session.finished

    def status(self, events: list[str]) -> dict:
        status = {"hud": self.last_hud, "events": events, "tics_left": 0 if self.over else self.session.observe().tics_left}
        if self.over:
            r = self.session.record
            status["attempt_over"] = "exit reached" if r.cleared else "died" if r.died else "time up"
        return status

    def look(self) -> list:
        with self.lock:
            if not self.over:
                obs = self.session.observe()
                self.last_frame, self.last_hud = obs.screen, obs.hud
            return [png(self.last_frame), self.status([])]

    def automap(self) -> list:
        with self.lock:
            if self.over:
                return ["The Attempt is over; there is no automap to show."]
            return [png(self.session.observe().automap)]

    def act(self, buttons: list[str], tics: int) -> list:
        with self.lock:
            if self.over:
                return [self.status(["the Attempt is already over"])]
            before = self.session.observe().hud
            self.session.press(buttons, tics)
            events = []
            if self.session.finished:
                self._save()
            else:
                obs = self.session.observe()
                self.last_frame, self.last_hud = obs.screen, obs.hud
                events = hud_events(before, obs.hud)
            return [png(self.last_frame), self.status(events)]

    def _save(self) -> None:
        if not self.saved:
            record = self.session.record.to_dict()
            append_record(self.record_path, {"spec": self.spec, **record} if self.spec else record)
            self.session.close()
            self.saved = True


def hud_events(before: dict, after: dict) -> list[str]:
    """What a player notices on the status bar between two moments."""
    events = []
    if after["HEALTH"] < before["HEALTH"]:
        events.append(f"took {before['HEALTH'] - after['HEALTH']} damage")
    if after["HEALTH"] > before["HEALTH"]:
        events.append(f"gained {after['HEALTH'] - before['HEALTH']} health")
    if after["ARMOR"] > before["ARMOR"]:
        events.append(f"gained {after['ARMOR'] - before['ARMOR']} armor")
    for ammo in ("BULLETS", "SHELLS", "ROCKETS", "CELLS"):
        if after[ammo] > before[ammo]:
            events.append(f"picked up {after[ammo] - before[ammo]} {ammo.lower()}")
    if after["SELECTED_WEAPON"] != before["SELECTED_WEAPON"]:
        events.append(f"switched to weapon {after['SELECTED_WEAPON']}")
    return events


def build_server(game: Game) -> MCPServer:
    buttons = ", ".join(game.session.buttons)
    server = MCPServer(
        name="doom",
        instructions=(
            f"You are playing Doom (1993), Map {game.session.map}, Difficulty {game.session.difficulty}. "
            "Reach the exit alive. The game is paused between your tool calls. "
            f"You have {game.session.tic_limit} tics (35 tics = 1 second of game time). "
            f"Buttons: {buttons}."
        ),
    )

    @server.tool(structured_output=False)
    def look() -> list:
        """See the screen (with the status bar) and read the HUD numbers. Uses no game time."""
        return game.look()

    @server.tool(structured_output=False)
    def act(buttons: list[str], tics: int = DEFAULT_TICS) -> list:
        """Hold these buttons for `tics` tics (1 to 35; 35 tics = 1 second), then see the result.

        Pass an empty list to wait. Turning: TURN_LEFT / TURN_RIGHT. Moving:
        MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT. Fire: ATTACK.
        Open doors and press switches: USE. Returns the new screen, the HUD,
        and events such as damage taken or items picked up.
        """
        # ToolError's message reaches the model, so it can correct itself.
        if not 1 <= tics <= MAX_TICS_PER_ACTION:
            raise ToolError(f"tics must be between 1 and {MAX_TICS_PER_ACTION}")
        try:
            return game.act(buttons, tics)
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool(structured_output=False)
    def automap() -> list:
        """See the automap, the map a player opens with Tab. Uses no game time."""
        return game.automap()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contender", default="mcp-practice", help="name on the Scoreboard")
    parser.add_argument("--spec", choices=sorted(SPECS), help="evaluate under this spec; needs --seed from it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.spec:
        spec = SPECS[args.spec]
        if args.seed not in spec.seeds:
            raise SystemExit(f"seed {args.seed} is not in spec {spec.name}: {spec.seeds}")
        session = AttemptSession(args.contender, spec.map, spec.difficulty, args.seed, spec.tic_limit)
        record_path = attempts_path(RESULTS_DIR, args.contender, spec)
    else:
        session = AttemptSession(args.contender, seed=args.seed)
        record_path = PRACTICE_DIR / f"{args.contender}.jsonl"

    build_server(Game(session, record_path, args.spec)).run()


if __name__ == "__main__":
    main()
