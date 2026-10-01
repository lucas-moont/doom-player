# M1 - Ruler and socket

## Goal

Two pieces every later milestone depends on: the Eval Suite (the ruler that measures any Contender the same way) and the Doom MCP server (the socket any LLM client plugs into). Both are validated with a random agent, since no real Contender exists yet.

## Concepts the owner learns

- Why evaluation needs fixed seeds, fixed Maps, and many Attempts
- The difference between `terminated` and `truncated`
- What MCP is: tools, their schemas, and how a client calls them
- Tool design: what to return to an LLM, and how much
- Synchronous versus asynchronous game modes

## Design constraints already decided

- Contenders receive Human-equivalent Observations only (`adr/0002`).
- The MCP server wraps ViZDoom in synchronous mode, so the game waits for the caller (`adr/0001`).
- The Eval Suite is resumable: an evaluation interrupted by a subscription limit continues from the last finished Attempt (`adr/0003`).

## Known facts about the environment

From `research/2026-09-28-fact-check-tooling.md`:

- An Attempt ends at the Map exit or on death. ViZDoom does not advance to the next Map.
- Timeout is 126000 tics (60 minutes of game time) and is reported as `truncated`.
- Default reward is 1 for reaching the exit alive, 0 otherwise.
- The default config exposes 19 buttons at 320x240.

## Design questions to settle in this milestone

Settle each with the owner, then record it in `CONTEXT.md` or an ADR as appropriate.

1. **Contender interface.** What single interface do a random agent, an LLM behind MCP, and a neural network all satisfy, so the Eval Suite treats them alike?
2. **Eval Suite parameters.** Number of seeds, Difficulty, and Attempt time limit for the standard `E1M1` evaluation. LLM Attempts are slow, so the seed count must be affordable for the slowest Contender.
3. **Progress metric.** Clear Rate will be 0 for weak Contenders, which hides differences between them. Choose a secondary metric that shows partial progress (for example, furthest point reached along the route to the exit). It may use Privileged Information, since it measures and does not feed the Contender.
4. **MCP tool set.** Which tools exist, what each returns, and how many tics one action lasts.
5. **What the automap tool returns**: image, text rendering, or both.

### Answers, settled with the owner on 2026-10-01

| # | Answer | Recorded in |
|---|---|---|
| 1 | One ruler, two doors: an `AttemptSession` owns the rules and the record; in-process Contenders implement `reset`/`act` (Door A), the MCP server drives the same session for an LLM (Door B) | `adr/0007-contender-interface.md` |
| 2 | Spec `e1m1-v1`: `E1M1`, Difficulty 3, seeds 0-4, 6300 tics (3 minutes of game time) per Attempt, reported as `truncated` | `STANDARD_E1M1` in `src/doom_player/eval.py` |
| 3 | Progress: share of the walking distance from spawn to exit closed at the Attempt's best moment, computed around the walls read from the WAD | `adr/0008-progress-metric.md` |
| 4 | Tools `look`, `act(buttons, tics)` with 1 to 35 tics (default 8), and `automap`; `act` returns the new screen, HUD numbers and events | M1 Results |
| 5 | The automap tool returns the image only | M1 Results |

## Open facts to test

| # | Fact | Outcome |
|---|---|---|
| 1 | The `mcp` SDK 2.x returns images as MCP image content | Untested (PR 2) |
| 2 | `omgifol` 0.5.1 reads `E1M1` from The Ultimate Doom WAD on Python 3.12 | **Confirmed 2026-10-01**: 470 vertexes, 486 linedefs; player 1 start `(1056, -3616)` matches ViZDoom's `POSITION_X/Y` at spawn; exit switch is linedef 326, action 11 |
| 3 | Claude Code shows MCP image results to the model when the server runs in WSL and the client on Windows | Untested (PR 2) |
| 4 | `get_game_variable(POSITION_X)` works with position left out of the observation | **Confirmed 2026-10-01**: position read while `available_game_variables` holds only HUD numbers; `depth_buffer` and `labels_buffer` are `None` |
| 5 | Turning the audio buffer off keeps seeded Attempts repeatable | **Confirmed 2026-10-01**: seed 0, 6300 tics, two runs identical (1575 actions, same final position); the Eval Suite's repeat test passes |

## Completion criteria

- [x] One documented command evaluates a Contender on `E1M1` and writes one Scoreboard row
- [x] Evaluating the random agent twice with the same seeds produces identical metrics
- [x] An evaluation stopped midway resumes without repeating finished Attempts
- [x] Each Scoreboard row records: Contender, Map, Difficulty, seeds, Clear Rate, progress metric, observation class, cost (tokens and wall-clock time)
- [ ] An MCP client (Claude Code) connects to the server, reads the screen, and moves the player
- [ ] Every MCP tool returns Human-equivalent Observations only, checked tool by tool
- [x] The random agent's row is on the Scoreboard in `README.md`
- [ ] All five design questions are answered and recorded
- [ ] `docs/learn/M1-ruler-and-socket.md` exists
- [ ] The M2 brief is reviewed against what was built and corrected

## Results

Filled in when the milestone is done.
