# One ruler, two doors: Contenders play through an AttemptSession

A random agent and a neural network are called by the game loop once per decision, while an LLM sits on the other side of the MCP server and calls the game itself, one tool at a time. Forcing both shapes into one interface would make either the LLM or the network awkward. So the Eval Suite's core is an `AttemptSession` (`src/doom_player/session.py`): it owns the Map, Difficulty, seed and tic limit, builds the only observations a Contender may see, measures Progress privately, and ends in an `AttemptRecord`. In-process Contenders enter through Door A, a `Contender` protocol with `reset(seed, buttons)` and `act(observation) -> buttons` (`src/doom_player/contenders/base.py`), driven by the Eval Suite's loop. The MCP server is Door B: it wraps the same session, and the LLM drives it through tools. Settled with the owner on 2026-10-01.

## Considered Options

- **Everything through MCP**: one uniform door, but a neural network would pay tool-call overhead on millions of training steps from M3 on.

## Consequences

- Rules live in one place. A Contender cannot choose its seed, see past the tic limit, or reach Privileged Information, whichever door it uses.
- The session talks to `DoomGame` directly, not through the Gymnasium wrapper, because MCP actions last a variable number of tics. Training from M3 on still uses Gymnasium; its observations must stay equal to `AttemptSession.observe`.
