"""Write the Scoreboard tables in README.md from results/scoreboard.jsonl.

Run with `uv run doom-scoreboard`. Rows measured on a Map go to the Map table,
rows measured on a Scenario to the Scenario table. Only the lines between each
table's two marker comments are replaced, so the tables are never edited by hand.
"""

import re

from doom_player.eval import RESULTS_DIR, read_records
from doom_player.paths import REPO_ROOT

README = REPO_ROOT / "README.md"
START, END = "<!-- scoreboard:start -->", "<!-- scoreboard:end -->"
SCENARIO_START, SCENARIO_END = "<!-- scenarios:start -->", "<!-- scenarios:end -->"
HEADER = [
    "Contender",
    "Map",
    "Difficulty",
    "Clear Rate",
    "Progress",
    "Seeds",
    "Observation class",
    "Training cost",
    "Cost per Attempt",
    "Spec",
]
SCENARIO_HEADER = [
    "Contender",
    "Scenario",
    "Reward",
    "Seeds",
    "Observation class",
    "Training cost",
    "Cost per Attempt",
    "Spec",
]


def cost(row: dict) -> str:
    return f"{row['tokens_per_attempt']:,} tokens, {row['wall_clock_s_per_attempt']} s"


def training_cost(row: dict) -> str:
    if row["training_steps"] is None:
        return "none"
    return f"{row['training_steps']:,} steps, {round(row['training_wall_clock_s'] / 60)} min"


def seeds_cell(seeds: list[int]) -> str:
    return f"{len(seeds)} ({seeds[0]}-{seeds[-1]})"


def markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return "\n".join(lines)


def table(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: (r["map"], -r["clear_rate"], -r["progress_mean"]))
    return markdown(
        HEADER,
        [
            [
                r["contender"],
                f"`{r['map']}`",
                str(r["difficulty"]),
                f"{r['clear_rate']:.0%}",
                f"{r['progress_mean']:.0%}",
                seeds_cell(r["seeds"]),
                r["observation_class"],
                training_cost(r),
                cost(r),
                f"`{r['spec']}`",
            ]
            for r in rows
        ],
    )


def scenario_table(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: (r["scenario"], -r["reward_mean"]))
    return markdown(
        SCENARIO_HEADER,
        [
            [
                r["contender"],
                f"`{r['scenario']}`",
                f"{r['reward_mean']:.1f}",
                seeds_cell(r["seeds"]),
                r["observation_class"],
                training_cost(r),
                f"{r['wall_clock_s_per_attempt']} s",
                f"`{r['spec']}`",
            ]
            for r in rows
        ],
    )


def render_readme(text: str, rows: list[dict]) -> str:
    """Return `text` with both tables rebuilt from Scoreboard `rows`."""
    text = _replace_block(text, START, END, table([r for r in rows if "map" in r]))
    scenario_rows = [r for r in rows if "scenario" in r]
    return _replace_block(text, SCENARIO_START, SCENARIO_END, scenario_table(scenario_rows))


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    block = f"{start}\n{body}\n{end}"
    pattern = re.escape(start) + ".*?" + re.escape(end)
    new, count = re.subn(pattern, lambda _: block, text, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"README.md needs exactly one {start} ... {end} block")
    return new


def main() -> None:
    rows = read_records(RESULTS_DIR / "scoreboard.jsonl")
    README.write_text(render_readme(README.read_text(), rows))
    print(table([r for r in rows if "map" in r]))
    print(scenario_table([r for r in rows if "scenario" in r]))


if __name__ == "__main__":
    main()
