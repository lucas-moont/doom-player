# M2 - LLM plays E1M1

## Goal

Measure how far an LLM gets on `E1M1` when it plays through the MCP server with a Harness around it, and publish the result as the first post.

Either outcome is a result. Published attempts with 2024-2025 models Cleared nothing; current models may do better. The post reports what was measured.

## Concepts the owner learns

- What a Harness is: decision loop, memory, tools
- Why LLMs struggle with spatial tasks: object permanence, getting stuck, latency
- Prompt design for an acting agent versus a chatting one
- Cost accounting: tokens and time per Attempt
- Failure analysis: turning recordings into categories of failure

## Prior evidence

From `research/2026-09-28-state-of-the-art-doom-agents.md`:

- GPT-4 on `E1M1`: no Attempt Cleared the Map; best case reached the final room and died; about one minute per step.
- VideoGameBench: 0% on Doom II for every model, including with the game paused.
- Documented failure modes: no object permanence, getting stuck in corners.
- Claude and Gemini progressed in Pokémon once given memory and tools, which suggests the Harness matters as much as the model.

## What M1 built for this milestone

Reviewed against the M1 code on 2026-10-01.

- **The ruler:** Eval Spec `e1m1-v1` (`E1M1`, Difficulty 3, seeds 0-4, 6300 tics). Each Attempt is recorded with Clear and Progress, and Scoreboard rows are generated with `uv run doom-scoreboard`. The random agent's row is the floor: Clear Rate 0%, Progress 20%.
- **The socket:** `uv run doom-mcp --contender <name> --spec e1m1-v1 --seed N` serves one Attempt through `look`, `act(buttons, tics)` and `automap`. When the Attempt ends, its record goes to the same `results/attempts/<name>/e1m1-v1.jsonl` as Door A. At 8 tics per action, an Attempt allows up to about 790 decisions.
- **Gaps M2 must close:**
  1. **An LLM launcher.** `run_eval` drives Door A only. M2 needs a loop that, for each missing seed, starts a headless CLI session (for example `claude -p` with an MCP config naming that seed's `doom-mcp` command) and waits for the record. Resuming works per seed; an Attempt cut off by a subscription limit restarts from its spawn.
  2. **Tokens.** `AttemptRecord.tokens` exists and is `null`. Fill it from the CLI's usage report (for example `--output-format json`), and record the model name and version alongside it.
  3. **Video and reasoning trace.** The MCP server does not record video yet. Add frame capture to `Game` and save the CLI transcript next to each record.
  4. **Harness features as switches.** The minimum Harness is screen and actions only, so the server needs a way to leave the `automap` tool out (for example `--no-automap`). Each later feature is a flag, measured on its own.
  5. **Failure analysis.** `progress.render` draws the Attempt's path over the distance field. It is a debugging visualisation, allowed by ADR 0002, and useful for sorting failures into categories.

## Steps

1. Run the LLM with the minimum Harness: screen and actions, no memory. Evaluate. This is the floor.
2. Add one Harness feature at a time (notes memory, automap access, a route summary), evaluating after each, so every feature's effect is measured separately.
3. Record every Attempt as video together with the LLM's reasoning trace.
4. Watch the failed Attempts and sort the failures into named categories with counts.
5. Draft the post material: one Scoreboard excerpt, one short video, one chart of failure categories.

## Constraints

- LLM access is through the CLI subscription. When the limit is reached, pause and resume after reset.
- The LLM receives Human-equivalent Observations only.
- The game runs in synchronous mode.

## Completion criteria

- [ ] Scoreboard rows exist for the minimum Harness and for each added feature, all from the standard Eval Suite
- [ ] Each row reports tokens and wall-clock time per Attempt
- [ ] Every failed Attempt is assigned to a failure category, and the counts are tabulated
- [ ] A video of the best Attempt exists
- [ ] The results section states the model name and version used
- [ ] `docs/learn/M2-llm-only.md` exists
- [ ] Post material is drafted in `docs/posts/01-llm-plays-doom.md` for the owner to edit and publish
- [ ] The M3 brief is written

## Results

Filled in when the milestone is done.
