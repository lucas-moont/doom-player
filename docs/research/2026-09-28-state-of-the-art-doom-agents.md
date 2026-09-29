# State of the art: agents that play Doom (research from 2026-09-28)

> Research done against primary sources (official docs, repositories, papers). Every claim has its source next to it.
> Anything that could not be confirmed in a primary source is marked as **(not verified)**.
> Opinions are marked as **(opinion)**.
> Translated from the original Portuguese research notes. The milestone table in this document is the researcher's draft; the authoritative plan is `docs/ROADMAP.md`.

## Executive summary

1. **The right environment is ViZDoom.** Version 1.3.1 was released on 2026-09-20, installs with `pip install vizdoom` on Windows (`win_amd64` wheel, Python 3.10 to 3.14) and already ships with Gymnasium wrappers [R1][R2][R3].
2. **Since 1.3.0 (Feb/2026) the original maps are ready-made Gymnasium environments**, for example `VizdoomDoomE1M1-S1-v0`, with a default reward of 1 on reaching the exit and 0 otherwise [R4][R2].
3. **License:** ViZDoom ships with Freedoom (maps different from the originals). For the 1993 maps you need a purchased `doom.wad` (Steam/GOG) [R4][R5][R6].
4. **Nobody has published, as far as this research found, an agent that beats the whole of Doom.** Classic RL shone in deathmatch and short scenarios [R7][R8][R9][R10].
5. **The closest to "clearing a level":** the 2018 competition (generated, easy maps) [R11] and a hobby repository with PPO+RND that clears E1M2 and MAP01 in ~6M steps, ~8h on an RTX 3080 [R12].
6. **Pure LLMs/VLMs do poorly in Doom.** GPT-4 did not complete E1M1 in any attempt and took ~1 minute per step [R13]. In VideoGameBench, Doom II stayed at 0% for all models [R14].
7. **A small, specialized model beats a large LLM in real time:** 1.3M parameters, 31 ms per decision, 178 frags against 13 for all the LLMs combined [R15].
8. **The central problem is sparse reward + long horizon** (keys, doors, mazes). That is where curiosity/RND, memory and Go-Explore come in [R16][R17][R18][R19].
9. **Sample Factory is the fastest framework for ViZDoom, but it does not support Windows** [R20]. On native Windows: Stable-Baselines3 or CleanRL; for Sample Factory, WSL2.
10. **World models:** GameNGen simulated Doom with diffusion, but trained on 128 TPUs [R21]. DIAMOND trained a playable CS:GO in 12 days on an RTX 4090 [R22]. On a personal PC only a reduced version is possible.
11. **The harness matters as much as the model:** Claude and Gemini made progress in Pokémon with memory, tools and function calls [R23][R24]; PokéAgent shows gaps between LLM, RL and humans [R25].
12. **Architecture with the best chance:** hybrid. A fast RL policy for fine control + a planning/memory layer (automap, LLM) for long-term goals.
13. **Market (opinion, with data):** "Agentic AI" grew more than 280% in US job postings from 2024 to 2025 [R26]; AI engineering leads LinkedIn's 2026 skills on the rise [R27].

---

## 1. Environments

### ViZDoom (recommended)

| Item | Finding | Source |
|---|---|---|
| Current version | 1.3.1, published on 2026-09-20 | [R2][R3] |
| Python | `>=3.10, <3.15` | [R3] |
| Windows | `win_amd64` wheel on PyPI; `pip install vizdoom` | [R1][R3] |
| Windows caveat | "the Windows version is not as well-tested as Linux and macOS versions"; for long experiments they recommend Docker or WSL | [R1] |
| Gymnasium | "Gymnasium environments are installed along with ViZDoom and are available on all platforms" | [R1] |
| Engine | Based on ZDoom | [R1] |
| License | ViZDoom code under MIT; ZDoom has varied licenses | [R1] |
| Speed | Up to 7000 fps on a single CPU thread (README claim) | [R1] |

**Recent release history** (GitHub API) [R2]:
- 1.3.0.dev1 (2025-05-29): Python 3.13, removal of the old OpenAI Gym wrapper.
- 1.3.0.dev2 (2025-06-29): Freedoom 0.13, truncation support, MultiBinary buttons.
- 1.3.0.dev3 (2025-10-22): notifications buffer, audio buffer improvements, label categories for semantic segmentation.
- 1.3.0rc1 (2026-02-08): **original Doom levels**, Python 3.14.
- 1.3.0 (2026-02-11): "Mature Farama Release".
- 1.3.1 (2026-09-20): improvements to `state.objects`, fixes, s390x support.

**Ready-made scenarios** [R28]: Basic, BasicAudio, BasicNotifications, DeadlyCorridor, Deathmatch, DefendCenter, DefendLine, HealthGathering, HealthGatheringSupreme, MyWayHome, PredictPosition, TakeCover. IDs in the format `VizdoomBasic-v1`, with `-MultiBinary-v1` variants.

**Original maps** [R4]:
- Name pattern: `Vizdoom<Game><Map>-S<X>-v0`, with `<Game>` in `Freedoom`, `Freedoom2`, `Doom`, `Doom2` and `<X>` from 1 to 5 (skill).
- Doom 1: E1M1–E4M9. Doom 2: MAP01–MAP32. Freedoom 1 and 2 likewise.
- Default reward: "1 is assigned for reaching the end of the level, and 0 otherwise". Customizable (`set_living_reward`, `set_death_penalty`, `set_kill_reward`, `set_item_reward` etc.).
- Actions: ATTACK, SPEED, STRAFE, USE, movement, turning, weapon selection.
- Observation: full HUD, default resolution 320x240, automap and audio available.
- Configs: `doom.cfg`, `doom2.cfg`, `freedoom1.cfg`, `freedoom2.cfg`.
- **Not documented**: the episode timeout and whether there is automatic progression from one map to the next (see the open questions section).

**API features useful for the project** [R29]:
- `save()` / `load()`: save and load the game state (ZDoom savegame). This is the piece that Go-Explore needs.
- `new_episode(recording_file_path)` and `replay_episode()`: recording and replay to a `.lmp` file.
- Buffers: depth, labels (segmentation), automap, audio, notifications, object and sector information.
- `set_doom_map`, `set_doom_skill`, `set_seed`, synchronous modes (`PLAYER`, `SPECTATOR`) and asynchronous ones.

**Caveat about replays** [R30]: "Replay files are known to have wonky issues at times". An issue reports wrong rewards when replaying in `Mode.PLAYER`; the recommendation is to use `SPECTATOR` [R31].

### WADs and license

- ViZDoom cannot distribute the original assets; it uses `freedoom2.wad` by default [R30].
- For the original maps: place `doom.wad` / `doom2.wad` (lowercase names) in the package directory or the working directory; purchase via Steam or GOG [R4].
- Any base WAD can be used via `set_doom_game_path` [R1].
- Freedoom: Phase 1 has 4 episodes of 9 maps (36), Phase 2 has 32 maps. **The maps are original to the project, not copies of the Doom maps** [R5].
- The shareware `doom1.wad` (episode 1 only) is cited as free by the doomgeneric README [R6]. Whether ViZDoom accepts the shareware `doom1.wad` directly: **(not verified)**.
- Exact Freedoom license (3-clause BSD): **(not verified in this research; the README points to the `COPYING` file)** [R5].

### Alternatives

| Project | What it is | When to use | Source |
|---|---|---|---|
| doomgeneric | Minimalist port in C; you only need to implement 5 functions (`DG_Init`, `DG_DrawFrame`, `DG_SleepMs`, `DG_GetTicksMs`, `DG_GetKey`). GPL-2.0. | Embedding the engine in your own harness | [R6] |
| cyDoomgeneric | Python binding for doomgeneric, used in the GPT-4 paper | LLM agent without ViZDoom | [R13] |
| doom-mcp | MCP server in Rust with doomgeneric via FFI. Tools: `doom_start`, `doom_action`, `doom_screenshot`. Supports Windows x64. Ships with Freedoom. | Design reference for exposing Doom via MCP | [R32] |
| Chocolate Doom | Port faithful to DOS, GPL-2.0, compatible with vanilla `.lmp` demos | Replaying vanilla human demos | [R33] |
| VideoGameBench | Runs Doom/Doom II via JS-DOS + Playwright, models via LiteLLM, `--lite` mode pauses the game | Ready-made VLM benchmark | [R34] |
| EnvPool | Vectorized C++ engine with single-player ViZDoom; wheels for Windows | High throughput | [R35] **(Windows support seen only in a search result; check before depending on it)** |

---

## 2. Classic RL in Doom

| Work | Method | What it actually achieved | Source |
|---|---|---|---|
| ViZDoom (Kempka et al., 2016) | DQN with experience replay | Two scenarios: basic shooting and maze navigation | [R7] |
| Arnold (Lample & Chaplot, 2016) | DRQN + game features, separate networks for navigation and action | Deathmatch. 2nd place in 2016 in both tracks; Arnold4 won Track 2 in 2017 (275 frags) | [R8][R10][R36] |
| F1 (2016) | A3C with curriculum learning | Won Track 1 in 2016 with 559 frags | [R10] |
| IntelAct / DFP (Dosovitskiy & Koltun, 2016) | Direct Future Prediction: supervised learning predicting future measurements | Won Track 2 (Full Deathmatch) in 2016, 297 frags, on never-seen maps | [R9][R10] |
| Marvin (2017) | A3C pre-trained with human replays | Won Track 1 in 2017 (248 frags) | [R10] |
| Sample Factory / APPO (Petrenko et al., 2020) | Asynchronous PPO, self-play, PBT | Battle 59.37, Battle2 36.40, Deathmatch-Bots 85.66, Duel-Bots 55.39; more than 10^5 fps on a single machine | [R20][R37][R38] |
| 2018 Competition, Track 1 | Various | Completing **randomly generated, easy** maps, with monsters, acid and doors. Winner TSAIL: 25.34 min summed over 10 maps | [R11] |
| vizdoom_ppo_rnd (hobby) | PPO + RND + LSTM | "able to learn to beat each level within approximately 6M global steps" on **E1M2 (Doom)** and **MAP01 (Doom 2)**; ~8h on an RTX 3080 12GB with 20 environments | [R12] |

**Documented limits** [R10]:
- The competition bots "are still weaker than humans".
- Weaknesses: poor strafing and dodging, they do not chase enemies out of sight, many suicides.
- The paper describes the 2018 single-player task as a relatively easy problem compared to deathmatch. Note: this applies to generated, easy maps, not to the original maps.

**Has anyone already beaten Doom with RL?** No primary source was found showing an agent that completes a whole episode (9 maps in sequence) or the entire game. The results are per map, one agent trained per map, with limited generalization: "More testing is required to get the agent to generalize across further levels" [R12].

**Transformers in ViZDoom**: a 2025 study compared DTQN (online) and Decision Transformer (offline) and concluded that traditional methods were better in both cases [R39].

---

## 3. Exploration and long-term navigation

The problem: the default reward in the original maps only appears at the exit [R4]. It is like looking for the exit of a building in the dark with nobody saying "hot" or "cold".

| Technique | Idea | Evidence in Doom | Source |
|---|---|---|---|
| ICM (Pathak et al., 2017) | Curiosity = error in predicting the consequence of one's own action, in a learned feature space | Tested in VizDoom (sparse MyWayHome) and Super Mario | [R16] |
| RND (Burda et al., 2018) | Bonus = error in predicting the output of a fixed random network | State of the art in Montezuma's Revenge at the time; used in the repository that clears E1M2 | [R17][R12] |
| Episodic Curiosity (Savinov et al., 2018) | Episodic memory + reachability | In ViZDoom it reaches the goal at least 2x faster than ICM | [R18] |
| Go-Explore (Ecoffet et al., 2019) | Remember states, return to a promising state, explore from there, then robustify with imitation learning | No published application in Doom was found. It needs restorable states, which ViZDoom offers with `save()`/`load()` | [R19][R29] |
| SPTM (Savinov et al., 2018) | Topological memory: graph of places + network that recognizes where it is | Navigation in Doom-style 3D mazes after 5 min of video | [R40] |
| Neural Map (Parisotto & Salakhutdinov, 2017) | Learned 2D spatial memory | Alternative to LSTM for partial observability | [R41] |
| SLAM-augmented DQN (Bhatti et al., 2016) | DQN + object detection + 3D reconstruction | Evaluated in Doom; learns better policies than pixels alone | [R42] |
| Auxiliary tasks (Mirowski et al., 2016) | Predicting depth as an extra training signal | Navigation in 3D mazes (DMLab, not Doom) | [R43] |

Legitimate shortcuts available in ViZDoom: automap buffer, depth buffer, labels buffer, player position and sector information [R29]. Using this is "privileged information": great for learning and for reward shaping, but it needs to be declared honestly in the portfolio.

Benchmarks derived from ViZDoom, useful for testing generalization: LevDoom (generalization), COOM (continual RL), HASARD (safe RL) [R44].

---

## 4. Imitation learning / offline RL

| Work | What it shows | Source |
|---|---|---|
| Behavioural Cloning in VizDoom (Spick et al., Sony, 2024) | BC from pixels only in Doom 2; agents at the level of the average player in the dataset, more "human" than RL, but weaker than RL | [R45] |
| Marvin (2017) | A3C pre-trained with human replays won Track 1 | [R10] |
| Playing DOOM with 1.3M Parameters (2026) | 31,000 human demonstrations, 1.3M-parameter model, 178 frags in 10 episodes on `defend_the_center` | [R15] |
| VPT (Baker et al., OpenAI, 2022) | Inverse Dynamics Model labels unlabeled videos; BC at scale; with fine-tuning it crafts diamond tools in Minecraft | [R46] |
| Counter-Strike BC (Pearce & Zhu, 2021) | Large-scale BC in a modern FPS | [R47] |
| Decision Transformer (Chen et al., 2021) | RL as sequence modeling conditioned on the desired return | [R48] |
| Multi-Game Decision Transformers (2022) | One offline model plays up to 46 Atari games close to human level | [R49] |
| Do We Need Transformers to Play FPS? (2025) | In ViZDoom, DT lost to traditional offline methods | [R39] |

**How to collect demos in the project**: play in `SPECTATOR` mode and record with `new_episode(recording_file_path)`; afterwards `replay_episode()` returns states, variables and rewards [R29][R31]. There is the official example `record_episodes.py` [R50].

**Community vanilla `.lmp` demos (speedruns)**: Chocolate Doom is compatible with DOS demos [R33]. Whether ViZDoom (ZDoom-based) replays vanilla demos correctly: **(not verified; the ZDoom demo format has historically differed from vanilla)**.

---

## 5. World models

| Work | What it is | Cost | Feasible on a personal PC? | Source |
|---|---|---|---|---|
| DreamerV3 (Hafner et al., Nature 2025) | Learns a model of the world and trains the policy "in imagination"; fixed hyperparameters; first to collect diamonds in Minecraft from scratch | Code in JAX, tested on Linux and Mac; configurable model sizes | Yes with a small model, via WSL2. Native Windows **(not verified)** | [R51][R52] |
| GameNGen (Google, ICLR 2025) | Doom simulated by diffusion (adapted Stable Diffusion v1.4), 20 fps | Training on 128 TPU-v5e, 700k steps. Data from a PPO agent (Stable Baselines 3, ViZDoom, 50M steps) | Not at the original scale. Memory of only ~3 seconds (64 frames) | [R21] |
| Unofficial GameNGen | Open reimplementation | Author used an NVIDIA A100 80GB | Only by scaling down a lot | [R53] |
| DIAMOND (Alonso et al., NeurIPS 2024) | RL agent trained inside a diffusion world model; 1.46 human normalized score on Atari 100k | CS:GO model: 381M parameters, 12 days on an RTX 4090, runs at 10 Hz on an RTX 3090 | It is the most realistic reference for consumer hardware | [R22] |
| Genie 3 (DeepMind, 2025-08-05) | Interactive worlds from text, 720p, 24 fps, consistency for a few minutes | Closed model | No. Serves as context | [R54] |

Important point for the project: GameNGen **does not play** Doom, it **simulates** Doom. And to train it needed an ordinary PPO agent generating data [R21]. In other words, the world model milestone depends on the earlier RL milestones.

---

## 6. LLM/VLM agents in Doom and other games

### In Doom

| Work | Setup | Result | Source |
|---|---|---|---|
| Will GPT-4 Run DOOM? (de Wynter, 2024) | E1M1, "Hurt Me Plenty", 10 runs per prompt, cyDoomgeneric. Prompts: naïve, walkthrough, planner, k-levels | **No run completed the map.** Best case: entered the final room and died. Failures: no object permanence, gets stuck in corners. Each step took about one minute | [R13] |
| VideoGameBench (2025) | 23 games; Doom in the dev set, Doom II in the test set; ReAct agent with a text scratchpad | Doom II: 0% for all models, including in Lite mode (game paused). Best model overall: 0.48% of the benchmark | [R14][R34] |
| Playing DOOM with 1.3M Parameters (2026) | ASCII + depth input; LLMs: Nemotron-120B, Qwen3.5-27B, GPT-4o-mini | LLMs totaled 13 frags; the small model got 178 and was the only one to engage enemies | [R15] |
| doom-mcp (2026) | MCP + doomgeneric; returns structured state (HP, ammo, visible enemies with direction and distance, doors) + PNG thumbnail | README claims "about 5-10 kills per session on E1M1 at medium difficulty". Does not claim to complete a map | [R32] |
| Claude Sonnet plays DOOM (Andrea Ricci, 2026) | Frames via WebSocket, depth map, ASCII automap | Qualitative description: navigates, opens doors, shoots. No metrics. **(secondary source: Hackaday)** | [R55] |
| Game-TARS (2025) | Generalist model pre-trained on more than 500B tokens, keyboard/mouse actions | Claims to outperform GPT-5, Gemini-2.5-Pro and Claude-4-Sonnet on FPS benchmarks. Which FPS benchmark: **(not verified)** | [R56] |

### In other games (harness lessons)

- **Claude Plays Pokémon**: Claude was given "basic memory, screen pixel input, and function calls to press buttons". Claude 3.7 Sonnet earned 3 badges; Claude 3.0 Sonnet could not get out of the first house [R23].
- **Gemini 2.5 Pro completed Pokémon Blue** with an agent harness built by an independent developer. Harness details in the Gemini 2.5 technical report: **(not verified; the arXiv excerpt accessed did not contain the section)** [R24].
- **PokéAgent Challenge (NeurIPS 2025)**: battle and speedrun tracks, more than 100 teams; "considerable gaps between generalist (LLM), specialist (RL), and elite human performance" [R25].
- **SIMA 2 (DeepMind, 2025)**: generalist agent in 3D worlds with Gemini at its core; uses Gemini to generate tasks and rewards for self-improvement [R57].
- **NitroGen (NVIDIA, 2025/2026)**: open vision-action model, 40,000 hours of gameplay from more than 1,000 games; weights on Hugging Face [R58].

### LLM + RL hybrids

- LLM as a strategic controller that chooses among RL skills, with RL doing the reactive execution [R59].
- Two levels: a high-level policy sets subgoals, a low-level controller executes [R60].
- VLM as an action advisor for online RL [R61].

None of these was evaluated in Doom in the sources read. Applying them to Doom would be the project's own contribution.

### Latency and cost

- Doom runs at 35 tics per second [R20]. An LLM at ~1 min per step [R13] does not play in real time.
- VideoGameBench: "by the time they return an action to perform, the game state has already substantially changed" [R14].
- Ways out: (a) ViZDoom's synchronous mode, in which the game waits for the agent [R29]; (b) the LLM only decides every now and then, and the fast policy acts every frame.
- Monetary cost per episode with current APIs: **(not verified; needs to be measured in the project)**.

---

## 7. Tools and libraries

| Tool | Status | Windows | Source |
|---|---|---|---|
| Gymnasium | v1.3.0 | Yes | [R62] |
| Stable-Baselines3 | 2.9.0 (Jun/2026); requires Python 3.10+ and PyTorch >= 2.8 | Yes (docs recommend miniforge) | [R63][R64] |
| CleanRL | One file per algorithm; includes `ppo_rnd_envpool.py`; integrates W&B | Pure Python; **(Windows support not verified)** | [R65] |
| Sample Factory | v2, APPO, self-play, PBT, models on the HF Hub | **No**: "There is no Windows support at this time" | [R20][R38] |
| TorchRL | 0.14 (docs) | **(not verified)** | [R66] |
| RLlib (Ray) | New API stack enabled by default | **(not verified)** | [R67] |
| DreamerV3 | JAX, Python 3.11+, tested on Linux and Mac | **(not verified)** | [R52] |

**Experiment tracking**: CleanRL uses Weights & Biases and TensorBoard [R65]; Sample Factory integrates with the Hugging Face Hub [R38]. Comparison between W&B, TensorBoard and MLflow: **(not researched in a primary source)**.

**Hardware**:
- Hugging Face Deep RL Course: Health Gathering Supreme with Sample Factory, ~4M frames, free Colab T4 GPU, about 15 minutes **(seen in a search result, not on the course page)** [R68].
- Clearing E1M2: ~8h on an RTX 3080 12GB with 20 environments [R12].
- Sample Factory: 50K+ fps on a 10-core machine, 100K+ on 36 cores [R20].
- Conclusion: simple scenarios run even on CPU; original maps call for a consumer GPU and hours of training. The number of CPU cores limits experience collection as much as the GPU does.

---

## 8. What stands out in a portfolio in 2026

**Data**:
- Stanford AI Index 2026 (via Lightcast, the report's partner): Agentic AI skills went from 0.06% of job postings in 2024 to 0.23% in 2025, an increase of more than 280%, almost 90,000 postings in the US. Mentions of "ChatGPT" and "Chatbots" fell. Python is the most requested specialized skill [R26].
- LinkedIn Skills on the Rise 2026: AI engineering at the top; cites Prompt Engineering and Large Language Models [R27].
- Common AI engineer skills (LangChain, RAG, PyTorch): **(secondary source)** [R69].

**Researcher's opinion (marked as opinion)**:
- The project covers two worlds that rarely appear together: real RL (training a policy) and agentic engineering (harness, tools, memory, evals).
- What differentiates the most is **rigorous evaluation**: a table comparing approaches on the same map, with seeds, completion rate, time and cost. The cited papers do exactly that [R13][R14][R15].
- An honest negative result has value. "Pure LLM does not get past E1M1, hybrid does" is a strong story and aligned with the literature.
- A short video of the agent playing is the natural format for LinkedIn.
- Terms with apparent demand: agent harness, evals, tool use/MCP, world models, VLA, MLOps/experiment tracking. No primary data on demand per term: **(not verified)**.

---

## Feasibility of beating Doom

**Honest assessment: beating the whole game, from pixels, with a single agent, is an open problem.** No public record was found of anyone having done it.

Why it is hard:

1. **Sparse reward.** The default signal only comes at the map exit [R4].
2. **Long horizon.** Maps with colored keys, locked doors, switches and backtracking require remembering where you have been. GPT-4 failed precisely for lack of spatial awareness and object permanence [R13].
3. **Generalization.** The agent that clears E1M2 was trained on that map [R12]. The game has 36 maps (4 episodes of 9) [R4].
4. **Combat + navigation together.** Deathmatch bots were good at shooting and bad at navigation [R10].
5. **Difficulty (skill).** All the LLM results were at medium difficulty or lower [R13][R32].
6. **Bosses and final maps** (E2M8, E3M8): no results found.

Realistic goal ladder:

| Goal | Chance | Basis |
|---|---|---|
| Ready-made scenarios (Basic, DefendCenter, HealthGathering) | Very high | [R12][R68] |
| E1M1 on skill 1, agent trained on that map | High | [R12] cleared E1M2 |
| Episode 1 (E1M1–E1M8), one agent per map, low skill | Medium | Extrapolation from [R12]; E1 is the simplest episode **(opinion)** |
| Episode 1 with a single agent | Low to medium | Requires generalization |
| Whole game (4 episodes), skill 3+ | Low | No precedent found |

Approaches with the best chance, in order:

1. **Recurrent PPO + reward shaping + RND**, per map. This is what has direct evidence [R12].
2. **Human demos** to kick things off (BC, then fine-tuning with RL). Marvin won this way in 2017 [R10]; VPT uses the same logic [R46].
3. **Go-Explore with `save()`/`load()`** for maps with keys. No precedent in Doom, but the API allows it [R19][R29].
4. **Hybrid**: a planner (LLM or a classic algorithm over the automap) chooses subgoals; an RL policy executes [R59][R60].
5. **Pure LLM/VLM**: the evidence says it does not get past E1M1 [R13][R14]. It is worth having as a comparison baseline.

Scope decision to be made: what counts as "beating" the game? Suggestion: define levels (1 map, episode 1, whole game) and declare which information the agent uses (pixels only or pixels + game variables).

---

## Suggested learning path

| # | Milestone | Concept learned | Demonstrable deliverable |
|---|---|---|---|
| 0 | Setup | Gymnasium API, ViZDoom, tracking | Repo with the environment running on Windows, random agent, recorded video, first dashboard |
| 1 | Tabular RL | Q-learning, epsilon-greedy exploration, MDP | Q-table on a discretized scenario (e.g. Basic with the state reduced to game variables) + learning curve plot |
| 2 | DQN from pixels | Convolutional networks, replay buffer, target network, frame stacking/skip | Agent solves `VizdoomBasic-v1` and `DefendCenter`; reproduces the 2016 result [R7] |
| 3 | PPO + recurrence | Policy gradient, GAE, LSTM for partial observability, vectorized environments | Agent on `HealthGatheringSupreme` and `DeadlyCorridor`; DQN vs PPO comparison |
| 4 | First real map | Reward shaping, curriculum by skill, evaluation metrics | E1M1 completed on skill 1; eval suite with completion rate over N seeds |
| 5 | Exploration | ICM/RND, intrinsic reward | Map with a key (e.g. E1M2) completed; ablation with and without RND [R12][R17] |
| 6 | Imitation learning | BC, dataset of own demos, BC + RL fine-tuning | Dataset recorded in `SPECTATOR`; curve showing the sample-efficiency gain from demos |
| 7 | Memory and Go-Explore | State archive, `save()`/`load()`, topological map or automap | Agent that solves a map with backtracking; visualization of the explored map |
| 8 | Scale | APPO/Sample Factory on WSL2, throughput | Episode 1 (one agent per map or multi-map); table of fps and cost |
| 9 | LLM/VLM baseline | Agent harness, tool use, MCP, scratchpad memory, synchronous mode | MCP server for ViZDoom; eval reproducing the findings of [R13][R14] with current models; cost per episode |
| 10 | Hybrid | Hierarchy: planner + RL skills | LLM chooses subgoals, PPO executes; comparison with milestones 5 and 9 on the same suite |
| 11 | World model | Small DreamerV3 or a reduced DIAMOND-style diffusion model | Agent trained in imagination on a simple scenario, or a mini "neural Doom" playable at low resolution |
| 12 | Wrap-up | Evals, technical write-up | Final table of all approaches, summary video, post |

Notes:
- Milestones 9 to 11 do not depend on 8; they can be brought forward if the goal is to show the agentic side earlier.
- The evaluation suite from milestone 4 should be reused in all the following milestones. It is the common thread of the portfolio.

---

## Open questions / not verified

1. **User hardware (verified 2026-09-28)**: NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM, Intel i5-13420H (8 cores / 12 threads), 15.7 GB RAM, Windows 11. Sets the ceiling for scale and world-model milestones.
2. **Does the user have `doom.wad`?** Without it, only Freedoom (different maps) or shareware.
3. **Shareware `doom1.wad` in ViZDoom**: not verified whether it works with the `VizdoomDoomE1M*` environments. Resolved in `2026-09-28-fact-check-tooling.md`.
4. **Progression between maps and timeout** in the original-map environments: the documentation does not say [R4]. Needs a practical test. Resolved in `2026-09-28-fact-check-tooling.md`.
5. **Stability of ViZDoom on Windows** in long training runs: the maintainers warn that it is less tested [R1].
6. **Vanilla `.lmp` demos in ViZDoom**: compatibility not verified.
7. **Determinism of `save()`/`load()`** for Go-Explore: not verified; replays have known problems [R30].
8. **Details of the Gemini Plays Pokémon harness** in the technical report: not read.
9. **Game-TARS FPS benchmark**: not identified.
10. **Exact license of Freedoom and of the Doom source code (GPL, id Software)**: not checked in a primary source.
11. **"Gunner" paper (PeerJ CS 3410)** on Doom with scalable RL: access blocked (HTTP 403), content not read.
12. **"Laya vs Gemma" (Towards AI, Sep/2026)**: secondary source claiming 6 out of 6 on MAP01 with LoRA; not verified.
13. **Windows support** for CleanRL, TorchRL, RLlib, EnvPool and DreamerV3: not confirmed in official docs.
14. **Cost per episode of current LLMs** and performance of 2026 models in Doom: the papers read use 2024/2025 models; the picture may have changed.
15. **Absence of evidence is not evidence of absence**: the conclusion "nobody has beaten it" comes from searches with no results, not from a source that states it.

---

## References

- [R1] ViZDoom, official repository: https://github.com/Farama-Foundation/ViZDoom
- [R2] ViZDoom releases: https://github.com/Farama-Foundation/ViZDoom/releases (dates via https://api.github.com/repos/Farama-Foundation/ViZDoom/releases)
- [R3] ViZDoom on PyPI: https://pypi.org/project/vizdoom/ (metadata via https://pypi.org/pypi/vizdoom/json)
- [R4] ViZDoom docs, Playing original Doom levels: https://vizdoom.farama.org/environments/original_doom_levels/
- [R5] Freedoom, repository: https://github.com/freedoom/freedoom
- [R6] doomgeneric: https://github.com/ozkl/doomgeneric
- [R7] Kempka et al., ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning: https://arxiv.org/abs/1605.02097
- [R8] Lample & Chaplot, Playing FPS Games with Deep Reinforcement Learning: https://arxiv.org/abs/1609.05521
- [R9] Dosovitskiy & Koltun, Learning to Act by Predicting the Future: https://arxiv.org/abs/1611.01779
- [R10] Wydmuch, Kempka, Jaśkowski, ViZDoom Competitions: Playing Doom from Pixels: https://arxiv.org/abs/1809.03470
- [R11] Visual Doom AI Competition 2018, Singleplayer Track: https://www.aicrowd.com/challenges/visual-doom-ai-competition-2018-singleplayer-track-1
- [R12] callumhay/vizdoom_ppo_rnd: https://github.com/callumhay/vizdoom_ppo_rnd
- [R13] de Wynter, Will GPT-4 Run DOOM?: https://arxiv.org/abs/2403.05468
- [R14] Zhang et al., VideoGameBench: https://arxiv.org/abs/2505.18134
- [R15] Golchinfar, Vaziri, Marquardt, Playing DOOM with 1.3M Parameters: https://arxiv.org/abs/2604.07385
- [R16] Pathak et al., Curiosity-driven Exploration by Self-supervised Prediction: https://arxiv.org/abs/1705.05363
- [R17] Burda et al., Exploration by Random Network Distillation: https://arxiv.org/abs/1810.12894
- [R18] Savinov et al., Episodic Curiosity through Reachability: https://arxiv.org/abs/1810.02274
- [R19] Ecoffet et al., Go-Explore: https://arxiv.org/abs/1901.10995
- [R20] Sample Factory docs, VizDoom: https://www.samplefactory.dev/09-environment-integrations/vizdoom/
- [R21] Valevski et al., Diffusion Models Are Real-Time Game Engines (GameNGen): https://arxiv.org/abs/2408.14837 ; site: https://gamengen.github.io/
- [R22] Alonso et al., Diffusion for World Modeling (DIAMOND): https://arxiv.org/abs/2405.12399 ; code: https://github.com/eloialonso/diamond
- [R23] Anthropic, Claude's extended thinking: https://www.anthropic.com/news/visible-extended-thinking
- [R24] Gemini 2.5 technical report: https://arxiv.org/abs/2507.06261
- [R25] Karten et al., The PokeAgent Challenge: https://arxiv.org/abs/2603.15563
- [R26] Lightcast, Four Takeaways from the 2026 Stanford AI Index: https://lightcast.io/resources/blog/stanford-ai-2026 ; report: https://hai.stanford.edu/ai-index/2026-ai-index-report
- [R27] LinkedIn, Skills on the Rise 2026: https://news.linkedin.com/2026/Skills-on-the-rise-2026
- [R28] ViZDoom docs, Default scenarios: https://vizdoom.farama.org/environments/default/
- [R29] ViZDoom docs, DoomGame API: https://vizdoom.farama.org/api/python/doom_game/
- [R30] ViZDoom docs, FAQ: https://vizdoom.farama.org/faq/index.html
- [R31] ViZDoom issue #412: https://github.com/Farama-Foundation/ViZDoom/issues/412
- [R32] gunnargrosch/doom-mcp: https://github.com/gunnargrosch/doom-mcp
- [R33] Chocolate Doom: https://github.com/chocolate-doom/chocolate-doom
- [R34] VideoGameBench, repository: https://github.com/alexzhang13/videogamebench
- [R35] EnvPool docs, ViZDoom: https://envpool.readthedocs.io/en/latest/env/vizdoom.html
- [R36] Arnold, repository: https://github.com/glample/Arnold
- [R37] Petrenko et al., Sample Factory: https://arxiv.org/abs/2006.11751
- [R38] Sample Factory, repository: https://github.com/alex-petrenko/sample-factory
- [R39] Batth et al., Do We Need Transformers to Play FPS Video Games?: https://arxiv.org/abs/2504.17891
- [R40] Savinov et al., Semi-parametric Topological Memory for Navigation: https://arxiv.org/abs/1803.00653
- [R41] Parisotto & Salakhutdinov, Neural Map: https://arxiv.org/abs/1702.08360
- [R42] Bhatti et al., Playing Doom with SLAM-Augmented Deep Reinforcement Learning: https://arxiv.org/abs/1612.00380
- [R43] Mirowski et al., Learning to Navigate in Complex Environments: https://arxiv.org/abs/1611.03673
- [R44] ViZDoom docs, Third-party environments: https://vizdoom.farama.org/environments/third_party/
- [R45] Spick et al., Behavioural Cloning in VizDoom: https://arxiv.org/abs/2401.03993
- [R46] Baker et al., Video PreTraining (VPT): https://arxiv.org/abs/2206.11795
- [R47] Pearce & Zhu, Counter-Strike Deathmatch with Large-Scale Behavioural Cloning: https://arxiv.org/abs/2104.04258
- [R48] Chen et al., Decision Transformer: https://arxiv.org/abs/2106.01345
- [R49] Lee et al., Multi-Game Decision Transformers: https://arxiv.org/abs/2205.15241
- [R50] ViZDoom, record_episodes.py example: https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/record_episodes.py
- [R51] Hafner et al., Mastering Diverse Domains through World Models: https://arxiv.org/abs/2301.04104 ; Nature: https://www.nature.com/articles/s41586-025-08744-2
- [R52] DreamerV3, repository: https://github.com/danijar/dreamerv3
- [R53] Unofficial GameNGen: https://github.com/Masao-Taketani/GameNGen
- [R54] DeepMind, Genie 3: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- [R55] Hackaday, Claude Plays DOOM (secondary): https://hackaday.com/2026/08/26/claude-plays-doom/ ; project: https://hackaday.io/project/206493-claude-sonnet-plays-doom-on-scintix-p4
- [R56] Game-TARS: https://arxiv.org/abs/2510.23691
- [R57] SIMA 2: https://arxiv.org/abs/2512.04797 ; blog: https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/
- [R58] NitroGen: https://arxiv.org/abs/2601.02427 ; code: https://github.com/MineDojo/NitroGen
- [R59] Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution: https://arxiv.org/abs/2606.20014
- [R60] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical RL: https://arxiv.org/abs/2505.19761
- [R61] VLM as Action Advisor for Online Reinforcement Learning: https://arxiv.org/abs/2509.21126
- [R62] Gymnasium releases: https://github.com/Farama-Foundation/Gymnasium/releases
- [R63] Stable-Baselines3 changelog: https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html
- [R64] Stable-Baselines3 install: https://stable-baselines3.readthedocs.io/en/master/guide/install.html
- [R65] CleanRL: https://github.com/vwxyzjn/cleanrl ; paper: https://arxiv.org/abs/2111.08819
- [R66] TorchRL docs: https://docs.pytorch.org/rl/stable/
- [R67] RLlib, New API stack migration guide: https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html
- [R68] Hugging Face Deep RL Course, unit 8 hands-on Sample Factory: https://huggingface.co/learn/deep-rl-course/unit8/hands-on-sf
- [R69] Interview Query on LinkedIn data (secondary): https://www.interviewquery.com/p/linkedin-ai-engineering-fastest-growing-skills-2026
