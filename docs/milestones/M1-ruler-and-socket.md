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

## Completion criteria

- [ ] One documented command evaluates a Contender on `E1M1` and writes one Scoreboard row
- [ ] Evaluating the random agent twice with the same seeds produces identical metrics
- [ ] An evaluation stopped midway resumes without repeating finished Attempts
- [ ] Each Scoreboard row records: Contender, Map, Difficulty, seeds, Clear Rate, progress metric, observation class, cost (tokens and wall-clock time)
- [ ] An MCP client (Claude Code) connects to the server, reads the screen, and moves the player
- [ ] Every MCP tool returns Human-equivalent Observations only, checked tool by tool
- [ ] The random agent's row is on the Scoreboard in `README.md`
- [ ] All five design questions are answered and recorded
- [ ] `docs/learn/M1-ruler-and-socket.md` exists
- [ ] The M2 brief is reviewed against what was built and corrected

## Results

Filled in when the milestone is done.
