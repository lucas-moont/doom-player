# Progress is the share of the walking distance to the exit an Attempt closed

Clear Rate is 0 for every weak Contender, which hides the differences between them, so the Scoreboard needs a partial-progress number next to it. We chose the walking distance to the exit, computed from the Map's own geometry: walls are read from the WAD with `omgifol`, laid on an 8-unit grid, and a Dijkstra pass from the exit line gives each reachable cell its distance (`src/doom_player/progress.py`). For one Attempt, Progress = 1 - (closest distance reached) / (distance at spawn), clipped to 0..1, and 1 for a Clear. It reads the player's position, which is Privileged Information, and it only measures; no Contender ever sees it (ADR 0002). Settled with the owner on 2026-10-01.

## Considered Options

- **Hand-placed checkpoints along a route**: simpler to explain, but every new Map needs a new set by hand, and the choice of route is a judgement call baked into the ruler.

## Consequences

- Any Map with an exit line gets a Progress number without manual work, which M5 and M9 need.
- Simplifications, each one a known way the number can be off:
  - Walls are one-sided lines, impassable lines, and two-sided lines whose floors differ by more than 24 units. Areas reachable only by dropping down a ledge (on `E1M1`: the courtyard west of the start, the zigzag over the nukage) count as unreachable; while the player stands there, Progress stays at its best value so far.
  - Doors are treated as open, and locked doors as unlocked. From M5 on, a Map with keys needs the route through the key to be part of the distance, or Progress will reward walking up to a locked door.
  - Lifts, crushers and teleporters are ignored.
- On `E1M1` the spawn is 4761 units from the exit on foot, against 2184 in a straight line. A random agent that crosses the start room to its east door scores about 0.2 without leaving the room. Read Progress as "share of the route", not "share of the Map".
