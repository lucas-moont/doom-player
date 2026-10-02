"""The Harness rungs: what surrounds the LLM while it plays.

A Harness is the decision loop, the memory and the tools around an LLM
(CONTEXT.md). Here the decision loop is the Claude Code CLI itself; a rung
chooses which MCP tools the server offers and what the system prompt says.
The prompt holds only what a game manual would tell a new player: the goal,
the controls, the clock. It never describes the Map.
"""

from dataclasses import dataclass

MANUAL = """\
You are playing the original Doom (1993), Map E1M1 "Hangar", on Difficulty 3
("Hurt Me Plenty"). Your goal: reach the Map's exit alive. The exit is a
switch you press with USE, usually in a small room at the end of the level.

You play through tools. The game is paused between your tool calls, so take
the time you need to think. The game clock only moves when you call `act`.
You have {tic_limit} tics of game time (35 tics = 1 second), about {minutes}
minutes. When the time runs out or you die, the Attempt is over.

Controls for `act(buttons, tics)`:
- MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT (strafe)
- TURN_LEFT, TURN_RIGHT: turning starts slow, then speeds up. Held for 10,
  20 or 35 tics you turn about 10, 45 or 100 degrees; together with SPEED,
  about 20, 90 or 195 degrees (measured in this game)
- ATTACK fires the current weapon; USE opens doors and presses switches
  (stand close and face them); SPEED makes you run
- SELECT_WEAPON1..7, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
Buttons can be combined, for example ["MOVE_FORWARD", "TURN_LEFT"].
`tics` is how long the buttons are held, from 1 to 35.

The screen shows the status bar at the bottom: ammo, health, arms, your face,
armor, keys. Enemies shoot back; kill them or avoid them.
{rung_tools}
Keep playing until a tool result says the Attempt is over. Do not stop early
to summarise or ask questions: nobody will answer until the Attempt ends.
"""

AUTOMAP = """
`automap` shows the map a player sees by pressing Tab: the walls you have
seen so far, with you as an arrow. It uses no game time.
"""

NOTES = """
You have a notebook. `write_note(text)` saves a note, `read_notes()` shows
them all. Use it to remember what you learned: places visited, doors that
were locked, where enemies were. It uses no game time, and it survives even
if your earlier messages are summarised away.
"""

TASK = "The Attempt has started. Begin by calling `look`."
RESUME = "The Attempt is not over yet. Keep playing: call `look`, then `act`."


@dataclass(frozen=True)
class Harness:
    name: str
    server_flags: tuple[str, ...]
    tools: tuple[str, ...]
    extra: str = ""

    def system_prompt(self, tic_limit: int) -> str:
        minutes = round(tic_limit / 35 / 60, 1)
        return MANUAL.format(tic_limit=tic_limit, minutes=minutes, rung_tools=self.extra)

    @property
    def allowed_tools(self) -> list[str]:
        return [f"mcp__doom__{tool}" for tool in self.tools]


HARNESSES = {
    "h0": Harness("h0", ("--no-automap",), ("look", "act")),
    "h1": Harness("h1", (), ("look", "act", "automap"), AUTOMAP),
    "h2": Harness("h2", ("--notes",), ("look", "act", "automap", "write_note", "read_notes"), AUTOMAP + NOTES),
}
