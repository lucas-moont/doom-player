"""Figures for Post 2. Run with `uv run python docs/posts/make_figures_02.py`.

Plots read committed files only:
- results/curves/*.csv: each Training Run's curves, exported from its TensorBoard log
- results/behaviour/deadly-corridor-v1.json: what each DeadlyCorridor policy did on the eval seeds

`--from-checkpoints` first rebuilds those two from the local `checkpoints/`
(gitignored), so it runs only on the machine that trained them. Measuring
behaviour reads the button pressed, `KILLCOUNT` and `is_player_dead`: Privileged
Information, used for debugging and reporting only, never shown to a policy.

Outputs in docs/posts/media/:
- defend-center-seeds.png: three training seeds learning DefendCenter
- deadly-corridor.png: reward, shots and kills of the DeadlyCorridor policies
- defend-center.mp4, corridor-charge.mp4, corridor-shaped.mp4: short clips

Colours follow the dataviz skill's reference palette (as Post 1), slots 1-3 in
fixed order, validated for adjacent pairs in light mode. Slot 3 sits below 3:1
contrast on the surface, so every line is labelled directly and the post has tables.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt

from doom_player.paths import REPO_ROOT

OUT = REPO_ROOT / "docs" / "posts" / "media"
CURVES = REPO_ROOT / "results" / "curves"
BEHAVIOUR = REPO_ROOT / "results" / "behaviour" / "deadly-corridor-v1.json"
CHECKPOINTS = REPO_ROOT / "checkpoints"
VIDEOS = REPO_ROOT / "videos"
SURFACE, TEXT, TEXT_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
SLOT_1, SLOT_2, SLOT_3, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a7a69f"

# One entry per Training Run: (curve file name, checkpoint folder, W&B run)
RUNS = [
    ("basic-seed0-collapsed", "basic-seed0-collapsed", "6e2x96q3"),
    ("basic-seed0", "basic-seed0", "wdkr5yea"),
    ("defend-center-seed0", "defend-center-seed0", "5gqzcemn"),
    ("defend-center-seed1", "defend-center-seed1", "en65c3af"),
    ("defend-center-seed2", "defend-center-seed2", "9t6habd8"),
    ("deadly-corridor-seed0-unshaped", "deadly-corridor-seed0-unshaped", "c6ad79c0"),
    ("deadly-corridor-seed0-shaped-h1", "deadly-corridor-seed0", "qpjx6upd"),
    ("deadly-corridor-seed0-shaped-h5", "deadly-corridor-seed0-k100-h5", "xlch11xs"),
]
# The DeadlyCorridor policies, in a fixed order = fixed colour.
CORRIDOR = [
    ("random", "Random", None, NEUTRAL),
    ("unshaped", "PPO, Scenario reward", "deadly-corridor-seed0-unshaped", SLOT_2),
    ("shaped-h1", "PPO, + kills - health lost", "deadly-corridor-seed0", SLOT_1),
]
TAGS = ("rollout/ep_rew_mean", "train/approx_kl")


def export_curves() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    CURVES.mkdir(parents=True, exist_ok=True)
    for name, folder, wandb_run in RUNS:
        (events,) = (CHECKPOINTS / folder / "tensorboard").rglob("events.*")
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})  # 0 keeps every point
        acc.Reload()
        rows: dict[int, dict] = {}
        for tag in TAGS:
            for e in acc.Scalars(tag):
                rows.setdefault(e.step, {})[tag] = e.value
        with open(CURVES / f"{name}.csv", "w", newline="") as f:
            f.write(f"# W&B run {wandb_run}; ep_rew_mean is the Scenario's own reward, averaged over recent episodes\n")
            w = csv.writer(f)
            w.writerow(["step", "ep_rew_mean", "approx_kl"])
            for step in sorted(rows):
                w.writerow([step, *(_fmt(rows[step].get(t)) for t in TAGS)])
        print(f"{name}: {len(rows)} points")


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def measure_behaviour() -> None:
    """Replay each DeadlyCorridor policy on deadly-corridor-v1's seeds and count what it did."""
    from vizdoom import Button, GameVariable

    from doom_player.contenders.ppo import PPOContender
    from doom_player.contenders.random import RandomScenarioContender
    from doom_player.scenario_eval import DEADLY_CORRIDOR
    from doom_player.scenarios import make_scenario_env

    out = {"spec": DEADLY_CORRIDOR.name, "seeds": list(DEADLY_CORRIDOR.seeds), "policies": {}}
    for key, _, folder, _ in CORRIDOR + [("shaped-h5", "", "deadly-corridor-seed0-k100-h5", "")]:
        contender = RandomScenarioContender() if folder is None else PPOContender(CHECKPOINTS / folder / "model.zip")
        env = make_scenario_env(DEADLY_CORRIDOR.scenario)
        game = env.unwrapped.game
        attack = game.get_available_buttons().index(Button.ATTACK)
        totals = {"reward": 0.0, "decisions": 0, "attack": 0, "kills": 0, "deaths": 0}
        try:
            for seed in DEADLY_CORRIDOR.seeds:
                obs, _ = env.reset(seed=seed)
                contender.reset(seed, env.action_space)
                done = False
                while not done:
                    action = contender.act(obs)
                    totals["decisions"] += 1
                    totals["attack"] += int(env.unwrapped.button_map[action][attack])
                    obs, r, terminated, truncated, _ = env.step(action)
                    totals["reward"] += float(r)
                    done = terminated or truncated
                totals["kills"] += int(game.get_game_variable(GameVariable.KILLCOUNT))
                totals["deaths"] += int(game.is_player_dead())
        finally:
            env.close()
        totals["reward_mean"] = round(totals.pop("reward") / len(DEADLY_CORRIDOR.seeds), 2)
        out["policies"][key] = totals
        print(key, totals)
    BEHAVIOUR.parent.mkdir(parents=True, exist_ok=True)
    BEHAVIOUR.write_text(json.dumps(out, indent=2) + "\n")


def read_curve(name: str) -> tuple[list[int], list[float]]:
    with open(CURVES / f"{name}.csv") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r["ep_rew_mean"]]
    return [int(r["step"]) for r in rows], [float(r["ep_rew_mean"]) for r in rows]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def scoreboard_reward(contender: str, spec: str) -> float:
    rows = [json.loads(l) for l in (REPO_ROOT / "results" / "scoreboard.jsonl").read_text().splitlines() if l.strip()]
    return next(r["reward_mean"] for r in rows if r["contender"] == contender and r["spec"] == spec)


def seeds_chart() -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ends = []
    for seed, colour in zip(range(3), (SLOT_1, SLOT_2, SLOT_3)):
        steps, reward = read_curve(f"defend-center-seed{seed}")
        ax.plot([s / 1000 for s in steps], reward, color=colour, linewidth=2, label=f"Training seed {seed}")
        ends.append((reward[-1], seed, steps[-1] / 1000))
    # The three curves end almost together: spread their end labels so they never overlap.
    label_y = None
    for value, seed, x in sorted(ends, reverse=True):
        label_y = value if label_y is None else min(value, label_y - 0.55)
        ax.text(x + 12, label_y, f"seed {seed}", color=TEXT, fontsize=9, va="center")
    random = scoreboard_reward("random", "defend-center-v1")
    ax.axhline(random, color=NEUTRAL, linewidth=1.5)
    ax.text(1000, random - 0.45, f"random agent on the eval seeds: {random}", color=TEXT_2, fontsize=9,
            ha="right", va="top")
    ax.set_xlim(0, 1090)
    ax.set_ylim(-1.2, 11.5)
    ax.set_xlabel("Training steps (thousands)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Reward per episode while training\n(kills minus 1 for dying)", color=TEXT_2, fontsize=9)
    ax.set_title("DefendCenter: three training seeds, same settings", loc="left", color=TEXT, fontsize=12, pad=28)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=TEXT,
              handlelength=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "defend-center-seeds.png", facecolor=SURFACE)


def corridor_chart() -> None:
    data = json.loads(BEHAVIOUR.read_text())["policies"]
    panels = [("reward_mean", "Mean Scenario reward"), ("attack", "Shots fired"), ("kills", "Monsters killed")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    labels = [label for _, label, _, _ in CORRIDOR]
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        ax.grid(False)
        values = [data[k][key] for k, _, _, _ in CORRIDOR]
        ys = range(len(CORRIDOR))
        ax.barh(ys, values, height=0.56, color=[c for *_, c in CORRIDOR], edgecolor=SURFACE, linewidth=2)
        ax.axvline(0, color=RULE, linewidth=1)
        span = max(values) - min(0, min(values))
        ax.spines["left"].set_visible(False)  # the zero line below is the only vertical rule
        for y, v in zip(ys, values):
            text = f"{v:.1f}" if key == "reward_mean" else f"{v:d}"
            ax.text(v + span * 0.03 if v >= 0 else span * 0.03, y, text, va="center", fontsize=9, color=TEXT)
        ax.set_title(title, loc="left", color=TEXT, fontsize=10)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(min(0, min(values)) - span * 0.05, max(values) + span * 0.3)
    axes[0].set_yticks(range(len(CORRIDOR)), labels, color=TEXT, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("DeadlyCorridor, 10 eval seeds: every policy dies in all 10", x=0.01, ha="left", color=TEXT,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "deadly-corridor.png", facecolor=SURFACE)


def clip(source: Path, name: str) -> None:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
         "-vf", "scale=640:480:flags=neighbor", "-c:v", "libx264", "-crf", "30", "-preset", "slow",
         "-pix_fmt", "yuv420p", str(OUT / name)],
        check=True,
    )
    print(f"{name} <- {source.relative_to(REPO_ROOT)}")


def clips() -> None:
    best = max(_records("ppo-seed0", "defend-center-v1"), key=lambda r: (r["reward"], -r["seed"]))
    clip(VIDEOS / f"ppo-seed0-defend-center-seed{best['seed']}-episode-0.mp4", "defend-center.mp4")
    clip(VIDEOS / "deadly-corridor-unshaped" / "ppo-seed0-deadly-corridor-seed4-episode-0.mp4", "corridor-charge.mp4")
    clip(VIDEOS / "ppo-seed0-deadly-corridor-seed0-episode-0.mp4", "corridor-shaped.mp4")


def _records(contender: str, spec: str) -> list[dict]:
    path = REPO_ROOT / "results" / "attempts" / contender / f"{spec}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-checkpoints", action="store_true", help="rebuild curves and behaviour first")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.from_checkpoints:
        export_curves()
        measure_behaviour()
        clips()
    seeds_chart()
    corridor_chart()
