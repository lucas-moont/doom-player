# M2 - LLM plays E1M1: study guide

## 1. What was built

A launcher that lets Claude Opus 5.5 play `E1M1` through the Doom MCP server, isolated from everything except the game, under the same Eval Suite rules as the random agent. Three Harness rungs were measured, giving the model one more tool each time: H0 (screen and actions) Cleared 1 Attempt in 5, H1 (plus the automap) 4 in 5, H2 (plus a notebook) 5 in 5. Every result was checked by replaying the Attempt from the model's own transcript, and turned into the draft of Post 1.

## 2. Concepts

**Harness.** A climbing harness doesn't make you stronger. It decides what you can safely attempt. Around an LLM, the Harness is the decision loop, the memory and the tools it may call (`CONTEXT.md`). Here the loop is the Claude Code CLI itself, and the rungs change only the tools and the instructions. Source: https://www.anthropic.com/engineering/building-effective-agents

**Ablation, one rung at a time.** To learn which ingredient made the cake rise, you bake it again with one ingredient added and change nothing else. Each Harness rung adds one tool to the same model, Map, seeds and rules, so the change in Clear Rate belongs to that tool. Source: https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)

**Headless agent and isolation.** An exam taken in a room with only a pencil. `claude -p` runs the agent with no human typing. `--tools ""` takes away files, shell and web. `--strict-mcp-config` allows only the Doom server. A separate config folder leaves the owner's notes and settings outside the room. Otherwise the model could read the map file or the Progress code, which would be Privileged Information. Source: https://code.claude.com/docs/en/headless

**Tokens and the cache.** A token is a piece of text (or of an image) the model reads or writes. Each turn, the model rereads the whole conversation; parts it has already seen come from a cache, like rereading a page you bookmarked. That is why an H0 Attempt "costs" 24 M tokens: 24.3 M of them are cache rereads of a growing history. Source: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

**Deterministic replay.** A chess game written in notation can be played again move by move and must end in the same position. ViZDoom with a fixed seed is deterministic, so the `act` calls in a transcript replay the Attempt exactly, with no model and no tokens. The replay checks every record, and it also draws the path. Source: https://vizdoom.farama.org/api/python/doom_game/#vizdoom.DoomGame.set_seed

**Failure analysis.** A doctor groups symptoms into diagnoses before deciding what to treat. Watching every failed Attempt and naming its failure ("lost in loops", "out of time on the right route", "killed in combat") shows what to build next: the loops vanish with the automap, so a map is worth more than a stronger model here. Source: https://hamel.dev/blog/posts/evals/#looking-at-your-traces

## 3. Reading order

1. `src/doom_player/harness.py` (76 lines): read it all. `MANUAL` is the whole of what the model is told; note what it does *not* say (no route). `HARNESSES` defines the three rungs.
2. `src/doom_player/llm_eval.py`:
   - lines 127-149, `claude_command`: every isolation flag, one per line
   - lines 152-239, `play_llm_attempt`: start the game server, run the CLI, resume if it stops early, add tokens to the record. Note why the server is started here and not by the CLI (the comment near `--http`)
   - lines 60-102, `CliUsage.read_stream`: how tokens are read from the stream
   - lines 242-266, `run_llm_eval`: compare it line by line with `eval.run_eval` from M1. Skip `main` on first read
3. `src/doom_player/mcp_server.py`:
   - lines 182-235, `build_server`: the rungs as switches (`automap_tool`, `notes_tools`)
   - lines 114-128: the notebook
   - lines 146-163, `VideoWriter`, and lines 238-283, `main`: the `--http` mode, and why `SIGTERM` is caught
4. `src/doom_player/session.py`, lines 132-172: `act` with `on_frame` (tic-by-tic stepping for video) and `_clock` (why ViZDoom's own clock could not be trusted at the exit).
5. `src/doom_player/replay.py` (91 lines): read it all. `check` is how every number in the post was verified.
6. `docs/milestones/M2-llm-only.md`: Pilot, Failure analysis and Results sections.

Safe to skip: `tests/fake_claude.py` and `tests/test_llm_eval.py` (read them after `llm_eval.py`, as an example of testing an agent without paying for one), `tests/test_replay.py`, the new tests in `tests/test_mcp_tools.py` and `tests/test_session.py`, `src/doom_player/progress.py` (the rule for closed doors, and path lines in `render`), `src/doom_player/eval.py` (one line, nested values in W&B tables), `docs/posts/make_figures.py` and `docs/posts/media/` (the post's figures), `results/` (the records), `pyproject.toml` and `uv.lock` (`doom-llm-eval`, `doom-replay`, `matplotlib`), `docs/adr/0009-llm-contenders-headless-cli.md`, and the edits to `README.md`, `CONTEXT.md`, `docs/ROADMAP.md` and `docs/milestones/M3-rl-on-scenarios.md`.

## 4. Follow one Attempt

`uv run doom-llm-eval --harness h1`, seed 1, the fastest Clear.

1. `main` picks `HARNESSES["h1"]` and the spec `e1m1-v1`. `run_llm_eval` reads `results/attempts/opus-5.5-h1/e1m1-v1.jsonl`, checks its rules with `check_rules`, and finds seed 1 missing.
2. `play_llm_attempt` makes a temporary folder, picks a free port, and starts `python -m doom_player.mcp_server --seed 1 --http PORT --video ... --record-out ...`. The server builds an `AttemptSession` (the M1 referee) and waits on `127.0.0.1:PORT`.
3. It writes an MCP config pointing at that port, and runs `claude_command`: the system prompt is `MANUAL` plus the automap paragraph, and the allowed tools are `look`, `act`, `automap`.
4. The model calls `look`; the server's `Game.look` returns a PNG and the HUD. It calls `act(["MOVE_FORWARD"], 20)`; `Game.act` calls `session.press(..., on_frame=VideoWriter.add)`, which steps 20 tics one at a time, writing each frame to the video, then measures Progress once.
5. After 94 actions, `USE` at the exit switch ends the episode. `session.act` sets `_clock` to 1507 and calls `_finish`: `cleared=True`. `Game._save` writes the record to `--record-out` and closes the video.
6. The model reads "attempt_over: exit reached" and stops. The CLI prints its final `result` event; `CliUsage.read_stream` adds up 2.6 M tokens.
7. `record_out` exists, so the loop ends; the server is stopped. The record gains `tokens`, `model`, `transcript` and `video`, and `append_record` stores it.
8. Later, `uv run doom-replay results/attempts/opus-5.5-h1/e1m1-v1.jsonl` reads the 94 accepted `act` calls back from the transcript, replays them, and confirms: same actions, same 1507 tics, exit reward +1.

## 5. Try it

These cost no tokens: they reuse the transcripts already recorded.

1. Run `uv run doom-replay results/attempts/opus-5.5-h0/e1m1-v1.jsonl --seed 0` and open `runs/paths/opus-5.5-h0-seed0.png`. Predict: will the path cross into the big room east of the start? Check.
2. In `replay.py`, drop the last accepted `act` from H1 seed 1 before replaying. Predict: `cleared`, `tics` and `progress` of the result.
3. Count the `act` calls per transcript and compare with `tics`: which rung uses the longest actions on average? Predict before you count.
4. Read the system prompt with `HARNESSES["h0"].system_prompt(6300)`. Find one sentence you think helps the model most, and one you would add. Predict what each would change, without running it.
5. `CliUsage` sums four kinds of tokens. From the H0 records, compute the share that is cache reads. Predict it first.

## 6. Rebuild it

In `doom-player-study/`, write `m2_replay.py` by hand, without importing anything from this repository.

- **Build:** a function that reads a CLI stream-json transcript and returns the list of `(buttons, tics)` the game accepted (skip calls whose result is an error), and a function that replays them on a ViZDoom `DoomGame` set up like `doom.cfg` (Difficulty 3, `E1M1`, the record's seed) and returns how the game ended and the player's last position.
- **Given:** one transcript and its record from `results/` (copy them over), the ViZDoom docs for `make_action` and `get_game_variable`, and the stream-json event shapes you can see by opening the transcript.
- **Check:** your replay of H1 seed 1 ends with the episode finished, the player not dead, and the total reward 1.0; your replay of H0 seed 0 ends by timeout after 6300 tics.

## 7. Check yourself

1. H0 used 24 M tokens per Attempt and H1 only 5 M, yet H1 gives the model *more* to look at. Why is the cheaper rung the one with more tools?
2. Why does the launcher start the game server itself instead of letting the CLI start it from `.mcp.json`, as in M1?
3. A record says `cleared: true`. Name two independent checks in this repository that would catch it if that were false.
4. H1 Cleared 4/5 and H2 5/5. Can you say the notebook helped? What would you need to say so?
5. The 2024 GPT-4 study Cleared nothing and this one Cleared 10 of 15. List three differences in setup that someone could fairly point to before crediting the newer model.
