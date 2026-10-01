"""Write the Scoreboard table in README.md from results/scoreboard.jsonl.

Run with `uv run doom-scoreboard`. Only the lines between the two marker
comments are replaced, so the table is never edited by hand.
"""

import re

from doom_player.eval import RESULTS_DIR, read_records
from doom_player.paths import REPO_ROOT

README = REPO_ROOT / "README.md"
START, END = "<!-- scoreboard:start -->", "<!-- scoreboard:end -->"
HEADER = [
    "Contender",
    "Map",
    "Difficulty",
    "Clear Rate",
    "Progress",
    "Seeds",
    "Observation class",
    "Cost per Attempt",
    "Spec",
]


def cost(row: dict) -> str:
    return f"{row['tokens_per_attempt']:,} tokens, {row['wall_clock_s_per_attempt']} s"


def table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for r in sorted(rows, key=lambda r: (r["map"], -r["clear_rate"], -r["progress_mean"])):
        seeds = r["seeds"]
        cells = [
            r["contender"],
            f"`{r['map']}`",
            str(r["difficulty"]),
            f"{r['clear_rate']:.0%}",
            f"{r['progress_mean']:.0%}",
            f"{len(seeds)} ({seeds[0]}-{seeds[-1]})",
            r["observation_class"],
            cost(r),
            f"`{r['spec']}`",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = read_records(RESULTS_DIR / "scoreboard.jsonl")
    text = README.read_text()
    block = f"{START}\n{table(rows)}\n{END}"
    new, count = re.subn(re.escape(START) + ".*?" + re.escape(END), block, text, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"README.md needs exactly one {START} ... {END} block")
    README.write_text(new)
    print(table(rows))


if __name__ == "__main__":
    main()
