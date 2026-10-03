"""The README shows Map rows and Scenario rows in two tables, rebuilt from results."""

from doom_player.scoreboard import render_readme

README = """# doom-player

<!-- scoreboard:start -->
old map table
<!-- scoreboard:end -->

## Scenarios

<!-- scenarios:start -->
old scenario table
<!-- scenarios:end -->
"""

MAP_ROW = {
    "contender": "random", "spec": "e1m1-v1", "map": "E1M1", "difficulty": 3,
    "seeds": [0, 1, 2, 3, 4], "clear_rate": 0.0, "progress_mean": 0.05,
    "observation_class": "human-equivalent", "tokens_per_attempt": 0,
    "wall_clock_s_per_attempt": 12.3,
}
PPO_ROW = {
    "contender": "ppo", "spec": "basic-v1", "scenario": "basic",
    "seeds": [0, 1, 2, 3, 4], "reward_mean": 71.4,
    "observation_class": "human-equivalent", "wall_clock_s_per_attempt": 0.42,
    "training_steps": 200000, "training_wall_clock_s": 2460.0,
}
RANDOM_ROW = {
    "contender": "random", "spec": "basic-v1", "scenario": "basic",
    "seeds": [0, 1, 2, 3, 4], "reward_mean": -112.0,
    "observation_class": "human-equivalent", "wall_clock_s_per_attempt": 0.61,
    "training_steps": None, "training_wall_clock_s": None,
}


def test_scoreboard_renders_a_scenario_table_next_to_the_map_table():
    text = render_readme(README, [MAP_ROW, RANDOM_ROW, PPO_ROW])

    map_block = text.split("<!-- scoreboard:start -->")[1].split("<!-- scoreboard:end -->")[0]
    scenario_block = text.split("<!-- scenarios:start -->")[1].split("<!-- scenarios:end -->")[0]

    assert "| random | `E1M1` |" in map_block
    assert "basic" not in map_block
    assert scenario_block.strip().splitlines() == [
        "| Contender | Scenario | Reward | Seeds | Observation class | Training cost | Cost per Attempt | Spec |",
        "|---|---|---|---|---|---|---|---|",
        "| ppo | `basic` | 71.4 | 5 (0-4) | human-equivalent | 200,000 steps, 41 min | 0.42 s | `basic-v1` |",
        "| random | `basic` | -112.0 | 5 (0-4) | human-equivalent | none | 0.61 s | `basic-v1` |",
    ]
