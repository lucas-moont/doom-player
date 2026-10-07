# Progress on Maps with keys follows the route through the keys

ADR 0008's Progress treated every door as open and every lift as a wall. On `E1M2` that broke twice: the lifts cut the exit off from the spawn, so Progress was NaN, and once lifts were passable, the open red door made Progress pay up to 20% for reaching a door the player cannot pass without the red key, and nothing for the detour to fetch it. From M5, Progress has two rules, and each Eval Spec names the one it is scored with. `doors-open` is ADR 0008's rule with lifts and remotely opened doors made passable (line actions 62, 88, 103, 46 only, a list kept closed so `E1M1`'s field stays exactly what its results were measured with; a test pins it). `keyed` is the walking distance still to cover through the keys the player still needs: with keys K held, the shorter of going straight to the exit through the doors K opens, or to a key not yet held and on from there (`progress.KeyedRoute`). Each leg weighs its walking length, so no weights or list of required keys are chosen by hand, and on a Map without keys `keyed` equals `doors-open`. Which keys the player holds is read from ViZDoom's list of objects, since ViZDoom has no game variable for keys: a picked-up key leaves the list. Like the position, it is Privileged Information, read only to measure. A Training Run's reward may use a different rule from the one its Eval Spec scores with: M5's baseline trains on `doors-open`, as M4 did, and is scored on `keyed`. Proposed in M5 and accepted by the owner on 2026-10-07.

## Considered Options

- **Doors-open everywhere**: one rule, but on `E1M2` a policy standing at the locked red door would score as a fifth of the way to a Clear.
- **Hand-placed key checkpoints with chosen weights**: simple to read, but a judgement call per Map, which ADR 0008 rejected for the same reason.
- **Key held from the player being near the key**: no extra engine data, but a policy moves 4 tics per decision and can step over the pickup box between two readings.

## Consequences

- `e1m1-v1` keeps `doors-open`, and every `E1M1` number keeps its meaning. A record without a rule, written before M5, counts as `doors-open`; the Eval Suite refuses to mix rules under one Eval Spec name.
- ADR 0008's simplifications still hold for both rules, except two: lifts are no longer walls, and under `keyed` locked doors are locked. Crushers, teleporters and drop-down areas are still ignored.
- A lift or a remote door with an action outside the list is still a wall. A new Map may need the list extended, which must not change `E1M1`'s pinned field.
