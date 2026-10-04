"""Figures for Post 3. Run with `uv run python docs/posts/make_figures_03.py`.

Plots read committed files only:
- results/scoreboard.jsonl: every Contender's row on e1m1-v1
- results/curves/e1m1-*.csv: the Map Training Runs' curves, exported from TensorBoard

`--from-checkpoints` first re-exports the curves from the local `checkpoints/`
(gitignored) and cuts the clips from the local `videos/`, so it runs only on
the machine that trained them. `rollout/progress` is Privileged Information,
logged for people to read; no policy ever sees it.

Outputs in docs/posts/media/:
- e1m1-scoreboard.png: Clear Rate on e1m1-v1 for every Contender, with cost per Attempt
- e1m1-learning.png: Progress during training for training seeds 1 and 2
- e1m1-clear.mp4: a learned Clear; e1m1-sparse.mp4: the exit-reward-only policy, first 30 s

Colours follow the dataviz skill's reference palette (as Posts 1 and 2), slots
1-2 in fixed order plus the neutral grey for baselines, validated for adjacent
pairs in light mode. Every bar and line carries a direct label.
"""

import argparse
import csv
import json
import subprocess

import imageio_ffmpeg
import matplotlib.pyplot as plt

from doom_player.paths import REPO_ROOT

OUT = REPO_ROOT / "docs" / "posts" / "media"
CURVES = REPO_ROOT / "results" / "curves"
SURFACE, TEXT, TEXT_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
SLOT_1, SLOT_2, NEUTRAL = "#2a78d6", "#eb6834", "#a7a69f"

GRID = "#ecebe8"
SMOOTH = 25  # each logged point averages only a few Attempts, so the learning chart shows a rolling mean

# (checkpoint folder, which also names the curve file; W&B run)
RUNS = [
    ("e1m1-d3-seed0-sparse", "an8j66gw"),
    ("e1m1-d3-seed0-shaped", "oj834br1"),
    ("e1m1-d3-seed1-shaped", "7zv0jyx5"),
    ("e1m1-d3-seed2-shaped", "ffj5shht"),
]
TAGS = ("rollout/ep_rew_mean", "rollout/progress", "rollout/clear_rate")
# Scoreboard rows in display order: (contender, label, colour). Colour follows the family.
ROWS = [
    ("random", "Random agent", NEUTRAL),
    ("ppo-e1m1-d3-seed0-sparse", "PPO, exit reward only", NEUTRAL),
    ("opus-5.5-h0", "Opus 5.5, screen + actions", SLOT_1),
    ("opus-5.5-h1", "Opus 5.5, + automap", SLOT_1),
    ("opus-5.5-h2", "Opus 5.5, + automap + notes", SLOT_1),
    ("ppo-e1m1-d3-seed0-shaped", "PPO + Progress, training seed 0", SLOT_2),
    ("ppo-e1m1-d3-seed1-shaped", "PPO + Progress, training seed 1", SLOT_2),
    ("ppo-e1m1-d3-seed2-shaped", "PPO + Progress, training seed 2", SLOT_2),
]


def export_curves() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    CURVES.mkdir(parents=True, exist_ok=True)
    for name, wandb_run in RUNS:
        (events,) = (REPO_ROOT / "checkpoints" / name / "tensorboard").rglob("events.*")
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        rows: dict[int, dict] = {}
        for tag in TAGS:
            if tag in acc.Tags()["scalars"]:
                for e in acc.Scalars(tag):
                    rows.setdefault(e.step, {})[tag] = e.value
        with open(CURVES / f"{name}.csv", "w", newline="") as f:
            f.write(f"# W&B run {wandb_run}; progress and clear_rate are averages of training Attempts ended since the last point\n")
            w = csv.writer(f)
            w.writerow(["step", "ep_rew_mean", "progress", "clear_rate"])
            for step in sorted(rows):
                w.writerow([step, *(_fmt(rows[step].get(t)) for t in TAGS)])
        print(f"{name}: {len(rows)} points")


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clips() -> None:
    rows = _jsonl(REPO_ROOT / "results/attempts/ppo-e1m1-d3-seed0-shaped/e1m1-v1.jsonl")
    fastest = min((r for r in rows if r["cleared"]), key=lambda r: r["tics"])  # an early death is not a Clear
    cuts = [
        (f"ppo-e1m1-d3-seed0-shaped-e1m1-v1-seed{fastest['seed']}.mp4", "e1m1-clear.mp4", []),
        ("ppo-e1m1-d3-seed0-sparse-e1m1-v1-seed1.mp4", "e1m1-sparse.mp4", ["-t", "30"]),
    ]
    for source, name, extra in cuts:
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


def scoreboard_chart() -> None:
    rows = scoreboard_rows()
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.spines["left"].set_visible(False)
    for y, (name, label, colour) in enumerate(ROWS):
        r = rows[name]
        ax.barh(y, r["clear_rate"], height=0.56, color=colour, edgecolor=SURFACE, linewidth=2)
        cost = f"{r['wall_clock_s_per_attempt']:g} s"
        if r["tokens_per_attempt"]:
            cost = f"{r['tokens_per_attempt'] / 1e6:.1f}M tokens, {r['wall_clock_s_per_attempt']:.0f} s"
        ax.text(r["clear_rate"] + 0.02, y, f"{r['clear_rate']:.0%}   ·   {cost} per Attempt", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(range(len(ROWS)), [label for _, label, _ in ROWS], color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.75)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Clear Rate on E1M1, Difficulty 3, 5 fixed seeds", color=TEXT_2, fontsize=9)
    ax.set_title("Who finishes E1M1, and at what cost", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "e1m1-scoreboard.png", facecolor=SURFACE)


def scoreboard_rows() -> dict[str, dict]:
    return {r["contender"]: r for r in _jsonl(REPO_ROOT / "results" / "scoreboard.jsonl") if r["spec"] == "e1m1-v1"}


def read_curve(name: str, column: str) -> tuple[list[float], list[float]]:
    with open(CURVES / f"{name}.csv") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r[column]]
    return [int(r["step"]) / 1e6 for r in rows], [float(r[column]) for r in rows]


def rolling_mean(steps: list[float], values: list[float]) -> tuple[list[float], list[float]]:
    """Each point becomes the mean of itself and the SMOOTH - 1 before it; the first SMOOTH - 1 have no mean."""
    means = [sum(values[i - SMOOTH + 1 : i + 1]) / SMOOTH for i in range(SMOOTH - 1, len(values))]
    return steps[SMOOTH - 1 :], means


def learning_chart() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    last_step = 0.0
    for seed, colour in ((1, SLOT_1), (2, SLOT_2)):
        steps, progress = read_curve(f"e1m1-d3-seed{seed}-shaped", "progress")
        ax.plot(*rolling_mean(steps, progress), color=colour, linewidth=2, label=f"Training seed {seed}")
        last_step = max(last_step, steps[-1])
        first_clear = next(s for s, c in zip(*read_curve(f"e1m1-d3-seed{seed}-shaped", "clear_rate")) if c > 0)
        ax.axvline(first_clear, color=colour, linewidth=1, alpha=0.5)
        # below the curves, one line per seed so the labels never overlap
        ax.text(first_clear + 0.05, 0.62 - 0.08 * seed, f"seed {seed}: first training Clear at {first_clear:.2f}M steps",
                color=TEXT_2, fontsize=8.5)
    random = scoreboard_rows()["random"]["progress_mean"]
    ax.axhline(random, color=NEUTRAL, linewidth=1.5)
    ax.text(last_step, random - 0.07, f"random agent on the eval seeds: {random:.0%}", color=TEXT_2, fontsize=8.5, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, last_step * 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Training steps (millions)", color=TEXT_2, fontsize=9)
    ax.set_ylabel(f"Progress to the exit\n(training Attempts, mean of {SMOOTH} points)", color=TEXT_2, fontsize=9)
    ax.set_title("Learning E1M1's route: Progress first, Clears later", loc="left", color=TEXT, fontsize=12, pad=28)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "e1m1-learning.png", facecolor=SURFACE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-checkpoints", action="store_true", help="re-export curves and cut clips first")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.from_checkpoints:
        export_curves()
        clips()
    scoreboard_chart()
    learning_chart()
