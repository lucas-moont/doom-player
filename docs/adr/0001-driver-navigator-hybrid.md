# Driver and Navigator: the LLM stays out of the per-tic loop

Doom advances 35 tics per second and published LLM-only attempts took up to a minute per decision and Cleared nothing, while small learned policies decide in milliseconds but lose their way on Maps with keys and backtracking. We therefore target a hybrid: a fast learned Driver presses buttons every tic, and a slow Navigator (an LLM, or a classical algorithm over the automap) chooses Subgoals. When an LLM is involved, ViZDoom runs in synchronous mode so the game waits for it.

## Considered Options

- **LLM only**: kept as a Contender for comparison (M2), not as the target architecture.
- **RL only**: kept as a Contender (M3-M6). Evidence exists for single Maps; none for a full Doom Episode.

## Consequences

The Driver must accept a Subgoal as input, which shapes its design from M4 on.
