# M1 - Ruler and socket: study guide

## 1. What was built

A ruler and a socket. The ruler, the Eval Suite, plays every Contender on the same Map with the same seeds, Difficulty and time limit, and writes one Scoreboard row with its Clear Rate and its Progress, the share of the route to the exit it covered. The socket, the Doom MCP server, lets any LLM client play the same game through three tools, under the same rules and with the same record at the end.

## 2. Concepts

**Fixed seeds and many Attempts.** A race run on different tracks in different weather proves nothing about who is faster. An evaluation fixes the track (the Map), the weather (the seed, which decides every random event in the game) and the rules (Difficulty, time limit), then runs several races, because one race can be luck. Here: Eval Spec `e1m1-v1`, seeds 0 to 4. Source: https://gymnasium.farama.org/api/env/#gymnasium.Env.reset (the `seed` argument)

**`terminated` versus `truncated`.** A match can end because the game says so (a goal, a knockout) or because the clock runs out. `terminated` is the first: the player died or reached the exit. `truncated` is the second: the Attempt hit the 6300-tic limit. Mixing them up teaches a learner that "the world ends at 3 minutes", which is false. Source: https://farama.org/Gymnasium-Terminated-Truncated-Step-API

**MCP: tools and schemas.** A restaurant menu. Each tool is a dish with a name, a description, and the ingredients you may choose (its schema: `buttons` is a list of words, `tics` a whole number). The client (Claude Code) reads the menu and orders; the server (the kitchen) cooks and sends back a plate, here an image plus a small text. MCP is the standard menu format, so any client that reads it can order from any server. Source: https://modelcontextprotocol.io/docs/concepts/tools

**Tool design: what to return.** A co-pilot reads you the speedometer, not the engine's wiring diagram. Each tool returns what a player would see or notice (screen, HUD numbers, "took 10 damage") and nothing more: no position, no Progress, no seed. The less a tool returns, the cheaper each call is in tokens, and the more honest the result. Source: https://www.anthropic.com/engineering/writing-tools-for-agents

**Synchronous versus asynchronous.** A chess game by mail versus a live football match. In ViZDoom's synchronous `PLAYER` mode, the game waits for each action, like chess by mail, so a slow LLM loses no game time while thinking. In asynchronous mode, the game runs at 35 tics per second whether you act or not. Source: https://vizdoom.farama.org/api/python/enums/#vizdoom.Mode

**Path distance (Dijkstra).** Pour water at the exit on a floor plan: it spreads one tile at a time, around the walls, and each tile notes how far the water travelled to reach it. That is Dijkstra's algorithm on a grid, and the note on each tile is its walking distance to the exit. Progress compares the best note the player stood on with the note at the spawn. Source: https://www.redblobgames.com/pathfinding/a-star/introduction.html (the "Dijkstra's Algorithm" section)

## 3. Reading order

1. `src/doom_player/session.py`: the referee.
   - lines 26-37: the HUD numbers a Contender may see
   - lines 76-117: building the game; note `set_mode(PLAYER)`, `set_episode_timeout`, `set_seed`
   - lines 122-152: `observe`, `act`, `press`, the only ways in and out
   - lines 158-172: `_measure` (position, privately) and `_finish` (how `terminated`, `truncated`, `died` and `cleared` are decided)
2. `src/doom_player/contenders/base.py` and `random.py`: Door A's whole contract, in about 20 lines.
3. `src/doom_player/eval.py`:
   - lines 28-40: `EvalSpec` and `STANDARD_E1M1`
   - lines 60-97: `play` (Door A's loop) and `run_eval` (the resume logic: read what is done, skip it)
   - lines 100-108: `check_rules`, which refuses to mix Attempts played under different rules
   - lines 111-139: the Scoreboard row. Skip `log_to_wandb` and `_git_commit` on first read
4. `src/doom_player/progress.py`:
   - lines 81-93: which lines count as walls
   - lines 105-158: the distance field. Read the "Sources" comment and the Dijkstra loop; skip `_rasterize` and `render`
   - lines 64-78: `ProgressMeter`, which keeps the best moment
5. `src/doom_player/mcp_server.py`:
   - lines 119-158: `build_server`, the menu: three tools and their descriptions
   - lines 42-100: `Game`, which runs each order. Note what `status` returns, and what it leaves out
   - lines 102-116: `hud_events`, events computed only from what the status bar shows
6. `tests/test_mcp_tools.py`: how "Human-equivalent only" is checked, tool by tool.
7. `docs/adr/0007-contender-interface.md` and `docs/adr/0008-progress-metric.md`: why, and what the shortcuts cost.

Safe to skip: `src/doom_player/scoreboard.py` (formats a Markdown table), `src/doom_player/contenders/__init__.py` (a name-to-class lookup), `tests/test_session.py`, `tests/test_progress.py` and `tests/test_eval.py` (read them after the code, as examples of each claim), `results/` (the records themselves), `.mcp.json` (tells Claude Code to start the server through `wsl`), `docs/learn/img/` (debug pictures of the distance field), `pyproject.toml` and `uv.lock` (four new commands, three new packages: `omgifol`, `pillow`, `mcp`), and the edits to `README.md`, `CONTEXT.md`, the M1 and M2 briefs and `ROADMAP.md`.

## 4. Follow one Attempt

Door A first: `uv run doom-eval --contender random`.

1. `eval.main` picks `STANDARD_E1M1` and builds a `RandomContender`.
2. `run_eval` reads `results/attempts/random/e1m1-v1.jsonl`, `check_rules` makes sure any records there were played under the same rules, and it collects the seeds already done; on a fresh run, none.
3. For seed 0 it builds an `AttemptSession`. `__post_init__` builds the distance field for `E1M1` (`progress.distance_field`, about 1.6 s the first time, then cached), then `_build_game` starts ViZDoom with Difficulty 3, seed 0 and a 6300-tic limit, and `_measure` records the spawn position.
4. `play` calls `contender.reset(0, buttons)`, then loops: `session.observe()` builds the `Observation` (screen, automap, HUD), `contender.act` flips 19 coins, and `session.act` holds those buttons for 4 tics and calls `_measure` again.
5. At tic 6300 ViZDoom ends the episode. `session.act` sees `finished` and calls `_finish`: `truncated=True`, `died=False`, `cleared=False`, `progress=0.1895`.
6. `append_record` writes that record as one line. If the process died now, a rerun would skip seed 0.
7. After seed 4, `scoreboard_row` averages the five records, `write_row` replaces the random agent's row in `results/scoreboard.jsonl`, and `log_to_wandb` sends the table.

Door B, the same Attempt through MCP: `act(["MOVE_FORWARD"], 20)` arrives at the server's `act` tool, which checks `tics`, then `Game.act` reads the HUD, calls `session.press` (which turns names into the 19 button states and calls the same `session.act`), compares the HUD before and after with `hud_events`, and returns a PNG of the screen plus the status. When the Attempt ends, `Game._save` appends the same kind of record, and the client is told only "time up", "died" or "exit reached".

## 5. Try it

Predict first, then run. Use a scratch results folder so the real Scoreboard stays clean: in Python, `run_eval(RandomContender(), spec, Path("/tmp/try"))`.

1. Make an `EvalSpec` with `tic_limit=700`. Predict: higher or lower mean Progress than the standard 20%? Why might it not change much?
2. Change `RandomContender.tics_per_action` from 4 to 1. Predict: same Progress? (The agent decides four times as often, but each decision is shorter.)
3. Run the same spec at Difficulty 5. Predict: does the random agent die before the time limit? Look at `died` in the records.
4. Run `uv run doom-eval --contender random` and press Ctrl+C after "seed 1". Predict what the file holds, then run it again and count the lines.
5. In `progress.py`, set `MAX_STEP = 1000`, run `render(distance_field.__wrapped__("E1M1"), out="field.png")`, and compare with `docs/learn/img/e1m1-distance-field.png`. Predict which areas change colour. Set it back.

## 6. Rebuild it

In `doom-player-study/`, write `m1_ruler.py` by hand, without importing anything from this repository.

- **Build:** a small Eval Suite for Freedoom's `E1M1` through the Gymnasium wrapper from M0. A function `evaluate(policy, seeds, max_steps)` runs one Attempt per seed and appends one JSON line per Attempt (seed, steps, total reward, terminated, truncated) to a file, skipping seeds already in the file. Then a function that turns the file into one summary: number of Attempts, share terminated, mean steps.
- **Given:** M0's `doom-random` loop as a starting point, Python's `json` module, and the reproducibility link above. A "policy" is any function from observation to action.
- **Check:** evaluating twice gives identical lines; deleting the last line and re-running adds exactly that line back; a policy that always presses only `ATTACK` gives different numbers from the random one, and you can say why.

## 7. Check yourself

1. The random agent's Progress is about 20% but it never leaves the start room. How can both be true, and why is that acceptable for this metric?
2. Why does the MCP server take the seed from the command line instead of letting the LLM pass it to a tool?
3. An LLM spends 40 seconds thinking before each `act`. How much game time passes during that thinking, and which ViZDoom setting guarantees it?
4. `act` reports "took 10 damage" but never "an imp is 300 units to your left". Both facts exist inside the engine. What rule decides which one may be returned, and where is that rule checked?
5. You change `STANDARD_E1M1`'s tic limit from 6300 to 10500 but keep the name `e1m1-v1`, then re-run the random agent. What happens, why, and what should you have done instead?
