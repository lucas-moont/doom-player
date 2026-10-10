"""Figures for Post 4. Run with `uv run python docs/posts/make_figures_04.py`.

Plots read committed files only:
- results/scoreboard.jsonl: every Contender's row on e1m2-v1
- results/curves/e1m2-*.csv: the M5 Training Runs' curves (exported by docs/milestones/make_figures_m5.py)
- docs/milestones/img/m5-e1m2-route.png: E1M2's route through the red key, drawn from the WAD

`--from-checkpoints` first cuts the clips from the local `videos/` (gitignored),
so it runs only on the machine that filmed them. Progress, the key rate and
the route are Privileged Information, drawn for people to read; no policy
ever sees them.

Outputs in docs/posts/media/:
- e1m2-route.png: a copy of the route figure
- e1m2-scoreboard.png: keyed Progress and Clears on e1m2-v1 for every Contender
- e1m2-learning.png: the share of training Attempts that pick up the red key, and that Clear
- e1m2-clear.mp4: a learned Clear; e1m2-red-door.mp4: M4's recipe walking to the locked door, first 30 s;
  e1m2-circling.mp4: training seed 2 of keyed shaping circling a room, 30 s

Colours follow the dataviz skill's reference palette (as Posts 1 to 3): one
slot per training seed of keyed shaping, in fixed order, and the neutral grey
for every setting that Clears nothing. Every bar and line carries a direct label.
"""

import argparse
import csv
import json
import shutil
import subprocess

import imageio_ffmpeg
import matplotlib.pyplot as plt

from doom_player.paths import REPO_ROOT

OUT = REPO_ROOT / "docs" / "posts" / "media"
CURVES = REPO_ROOT / "results" / "curves"
SURFACE, TEXT, TEXT_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
SLOT_1, SLOT_2, SLOT_3, SLOT_4, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a7a69f"
SEED_COLOURS = (SLOT_1, SLOT_2, SLOT_3)  # keyed shaping, training seeds 0, 1, 2
GRID = "#ecebe8"
SMOOTH = 25  # each logged point averages only a few Attempts, so the learning chart shows a rolling mean

# Scoreboard rows in display order: (contender, label, colour).
ROWS = [
    ("random", "Random agent", NEUTRAL),
    ("ppo-e1m1-d3-seed0-shaped", "PPO trained on E1M1, seed 0", NEUTRAL),
    ("ppo-e1m1-d3-seed1-shaped", "PPO trained on E1M1, seed 1", NEUTRAL),
    ("ppo-e1m1-d3-seed2-shaped", "PPO trained on E1M1, seed 2", NEUTRAL),
    ("ppo-e1m2-d3-seed0-shaped", "Doors-open reward, seed 0", NEUTRAL),
    ("ppo-e1m2-d3-seed1-shaped", "Doors-open reward, seed 1", NEUTRAL),
    ("ppo-e1m2-d3-seed2-shaped", "Doors-open reward, seed 2", NEUTRAL),
    ("ppo-e1m2-d3-seed0-shaped-rnd", "Doors-open reward + RND, seed 0", SLOT_4),
    ("ppo-e1m2-d3-seed0-keyed-shaped-10m", "Key-route reward, seed 0", SLOT_1),
    ("ppo-e1m2-d3-seed1-keyed-shaped-10m", "Key-route reward, seed 1", SLOT_2),
    ("ppo-e1m2-d3-seed2-keyed-shaped-10m", "Key-route reward, seed 2", SLOT_3),
]
# (source in videos/, name in media/, ffmpeg cut options)
CLIPS = [
    ("ppo-e1m2-d3-seed0-keyed-shaped-10m-e1m2-v1-seed{fastest}.mp4", "e1m2-clear.mp4", []),
    ("ppo-e1m2-d3-seed1-shaped-e1m2-v1-seed0.mp4", "e1m2-red-door.mp4", ["-t", "30"]),
    ("ppo-e1m2-d3-seed2-keyed-shaped-10m-e1m2-v1-seed0.mp4", "e1m2-circling.mp4", ["-ss", "50", "-t", "30"]),
]


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clips() -> None:
    rows = _jsonl(REPO_ROOT / "results/attempts/ppo-e1m2-d3-seed0-keyed-shaped-10m/e1m2-v1.jsonl")
    fastest = min((r for r in rows if r["cleared"]), key=lambda r: r["tics"])
    for source, name, extra in CLIPS:
        source = source.format(fastest=fastest["seed"])
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(REPO_ROOT / "videos" / source), *extra,
             "-vf", "scale=640:480:flags=neighbor", "-c:v", "libx264", "-crf", "30", "-preset", "slow",
             "-pix_fmt", "yuv420p", str(OUT / name)],
            check=True,
        )
        print(f"{name} <- videos/{source}")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)


def scoreboard_rows() -> dict[str, dict]:
    return {r["contender"]: r for r in _jsonl(REPO_ROOT / "results" / "scoreboard.jsonl") if r["spec"] == "e1m2-v1"}


def scoreboard_chart() -> None:
    rows = scoreboard_rows()
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.spines["left"].set_visible(False)
    for y, (name, label, colour) in enumerate(ROWS):
        r = rows[name]
        ax.barh(y, r["progress_mean"], height=0.56, color=colour, edgecolor=SURFACE, linewidth=2)
        clears = round(r["clear_rate"] * len(r["seeds"]))
        ax.text(r["progress_mean"] + 0.02, y, f"{r['progress_mean']:.1%}   ·   {clears} of {len(r['seeds'])} finished",
                va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(range(len(ROWS)), [label for _, label, _ in ROWS], color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Progress along the route through the red key, E1M2, Difficulty 3, 5 fixed seeds", color=TEXT_2, fontsize=9)
    ax.set_title("How far each policy gets on E1M2, and how often it finishes", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e1m2-scoreboard.png", facecolor=SURFACE)


def read_curve(name: str, column: str) -> tuple[list[float], list[float]]:
    with open(CURVES / f"{name}.csv") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r[column]]
    return [int(r["step"]) / 1e6 for r in rows], [float(r[column]) for r in rows]


def keyed_curve(seed: int, column: str) -> tuple[list[float], list[float]]:
    """A keyed-shaping training seed's curve over 10M steps: the first 5M, then their continuation."""
    first, rest = read_curve(f"e1m2-d3-seed{seed}-keyed-shaped", column), read_curve(f"e1m2-d3-seed{seed}-keyed-shaped-10m", column)
    return first[0] + rest[0], first[1] + rest[1]


def rolling_mean(steps: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    """Each point becomes the mean of itself and the SMOOTH - 1 before it; the first SMOOTH - 1 have no mean."""
    means = [sum(values[i - SMOOTH + 1 : i + 1]) / SMOOTH for i in range(SMOOTH - 1, len(values))]
    return steps[SMOOTH - 1 :], means


def learning_chart() -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, column, ylabel in ((top, "key_rate", "Picked up the red key"), (bottom, "clear_rate", "Finished the level")):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_ylabel(f"{ylabel}\n(training Attempts,\nmean of {SMOOTH} points)", color=TEXT_2, fontsize=9)
        # Doors-open seed 0 trained before the key rate was logged; seeds 1 and 2 show the same.
        for seed in (1, 2):
            steps, values = rolling_mean(*read_curve(f"e1m2-d3-seed{seed}-shaped", column))
            ax.plot(steps, values, color=NEUTRAL, linewidth=1.5)
        ax.text(0.2 if column == "clear_rate" else 1.0, 0.2 if column == "clear_rate" else 0.05, "grey: doors-open reward, seeds 1 and 2 (5M steps)", color=TEXT_2, fontsize=8.5, va="bottom")
        for seed, colour in enumerate(SEED_COLOURS):
            steps, values = rolling_mean(*keyed_curve(seed, column))
            ax.plot(steps, values, color=colour, linewidth=1.5)
            # near 100% the three end together, so the top panel stacks its labels below them
            label_y = 0.84 - 0.09 * seed if column == "key_rate" else values[-1]
            ax.text(steps[-1] + 0.1, label_y, f"key-route reward, seed {seed}" if column == "key_rate" else f"seed {seed}",
                    color=colour, fontsize=8.5, va="center")
    bottom.set_xlim(0, 12.6)
    bottom.set_xticks([0, 2, 4, 6, 8, 10])
    bottom.set_xlabel("Training steps (millions)", color=TEXT_2, fontsize=9)
    top.set_title("Key-route reward: the key first, the exit much later, and not for every seed", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e1m2-learning.png", facecolor=SURFACE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-checkpoints", action="store_true", help="cut the clips first")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.from_checkpoints:
        clips()
    shutil.copyfile(REPO_ROOT / "docs" / "milestones" / "img" / "m5-e1m2-route.png", OUT / "e1m2-route.png")
    scoreboard_chart()
    learning_chart()
