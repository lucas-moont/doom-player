"""Figures for Post 1, from the committed results. Run with `uv run python docs/posts/make_figures.py`.

- outcomes.png: each Contender's five Attempts by outcome (stacked bar)
- paths.png: a looping H0 Attempt next to a Cleared H2 Attempt, over the distance field
- best-attempt.mp4: the fastest Clear, re-encoded small enough for the repository

Colours follow the dataviz skill's reference palette, slots 1-4 in fixed order,
validated for adjacent pairs (light mode). Two slots sit below 3:1 contrast on
the surface, so every segment carries a direct label and the post has a table.
"""

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
from PIL import Image

from doom_player.paths import REPO_ROOT
from doom_player.progress import render
from doom_player.replay import replay

OUT = REPO_ROOT / "docs" / "posts" / "media"
RESULTS = REPO_ROOT / "results" / "attempts"
SURFACE, TEXT, TEXT_2 = "#fcfcfb", "#0b0b0b", "#52514e"

# Failure categories validated by the owner on 2026-10-02 (M2 brief, Results).
FAILURES = {
    ("opus-5.5-h0", 0): "Lost in loops near the start",
    ("opus-5.5-h0", 1): "Lost in loops near the start",
    ("opus-5.5-h0", 2): "Lost in loops near the start",
    ("opus-5.5-h0", 4): "Out of time on the right route",
    ("opus-5.5-h1", 0): "Killed in combat",
}
OUTCOMES = {  # fixed order = fixed colour; never re-assigned by rank
    "Cleared": "#2a78d6",
    "Lost in loops near the start": "#eb6834",
    "Out of time on the right route": "#1baf7a",
    "Killed in combat": "#eda100",
    "Not cleared (random baseline)": "#a7a69f",  # neutral: the categories describe LLM failures
}
CONTENDERS = [("random", "Random agent"), ("opus-5.5-h0", "H0: screen + actions"),
              ("opus-5.5-h1", "H1: + automap"), ("opus-5.5-h2", "H2: + automap + notes")]


def records(contender: str) -> list[dict]:
    path = RESULTS / contender / "e1m1-v1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outcome(contender: str, record: dict) -> str:
    if record["cleared"]:
        return "Cleared"
    if contender == "random":
        return "Not cleared (random baseline)"
    return FAILURES[(contender, record["seed"])]


def outcomes_chart() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (contender, label) in enumerate(reversed(CONTENDERS)):
        counts = {name: 0 for name in OUTCOMES}
        for r in records(contender):
            counts[outcome(contender, r)] += 1
        left = 0
        for name, colour in OUTCOMES.items():
            n = counts[name]
            if n:
                # 2px surface gap between segments; label every segment (contrast relief)
                ax.barh(y, n, left=left, height=0.56, color=colour, edgecolor=SURFACE, linewidth=2)
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=10, color=TEXT, fontweight="bold")
                left += n
    ax.set_yticks(range(len(CONTENDERS)), [label for _, label in reversed(CONTENDERS)], color=TEXT, fontsize=10)
    ax.set_xticks(range(6), [str(i) for i in range(6)], color=TEXT_2, fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Attempts out of 5 (E1M1, Difficulty 3, 3 minutes each)", color=TEXT_2, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.tick_params(length=0)
    ax.set_title("How each Attempt ended: Claude Opus 5.5 by Harness rung", loc="left", color=TEXT, fontsize=12, pad=40)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in OUTCOMES.values()]
    ax.legend(handles, OUTCOMES.keys(), ncol=3, frameon=False, fontsize=8, loc="lower left",
              bbox_to_anchor=(0, 1.0), labelcolor=TEXT, handlelength=1, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "outcomes.png", facecolor=SURFACE)


def paths_figure() -> None:
    pictures = []
    for contender, seed in (("opus-5.5-h0", 0), ("opus-5.5-h2", 0)):
        record = next(r for r in records(contender) if r["seed"] == seed)
        session = replay(record, REPO_ROOT / record["transcript"])
        pictures.append(render(session.progress_meter.field, session.progress_meter.path))
    w, h = pictures[0].size
    sheet = Image.new("RGB", (w, h * 2 + 8), SURFACE)
    for i, picture in enumerate(pictures):
        sheet.paste(picture, (0, i * (h + 8)))
    sheet.save(OUT / "paths.png")


def best_clip() -> None:
    cleared = [r for c, _ in CONTENDERS[1:] for r in records(c) if r["cleared"]]
    best = min(cleared, key=lambda r: r["tics"])
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(REPO_ROOT / best["video"]),
         "-vf", "scale=640:480:flags=neighbor", "-c:v", "libx264", "-crf", "30", "-preset", "slow",
         "-pix_fmt", "yuv420p", str(OUT / "best-attempt.mp4")],
        check=True,
    )
    print(f"best Attempt: {best['contender']} seed {best['seed']}, {best['tics']} tics")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes_chart()
    paths_figure()
    best_clip()
