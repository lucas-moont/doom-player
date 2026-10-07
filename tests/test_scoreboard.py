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
    "wall_clock_s_per_attempt": 12.3, "training_steps": None, "training_wall_clock_s": None,
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


PPO_MAP_ROW = {
    "contender": "ppo-e1m1-d3-seed0", "spec": "e1m1-v1", "map": "E1M1", "difficulty": 3,
    "seeds": [0, 1, 2, 3, 4], "clear_rate": 0.2, "progress_mean": 0.61,
    "observation_class": "human-equivalent", "tokens_per_attempt": 0,
    "wall_clock_s_per_attempt": 3.1, "training_steps": 5000000, "training_wall_clock_s": 10800.0,
}


def test_map_scoreboard_shows_training_cost():
    text = render_readme(README, [MAP_ROW, PPO_MAP_ROW])
    map_block = text.split("<!-- scoreboard:start -->")[1].split("<!-- scoreboard:end -->")[0]

    assert map_block.strip().splitlines() == [
        "| Contender | Map | Difficulty | Clear Rate | Progress | Seeds | Observation class | Training cost | Cost per Attempt | Spec |",
        "|---|---|---|---|---|---|---|---|---|---|",
        "| ppo-e1m1-d3-seed0 | `E1M1` | 3 | 20% | 61% | 5 (0-4) | human-equivalent | 5,000,000 steps, 180 min | 0 tokens, 3.1 s | `e1m1-v1` |",
        "| random | `E1M1` | 3 | 0% | 5% | 5 (0-4) | human-equivalent | none | 0 tokens, 12.3 s | `e1m1-v1` |",
    ]


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


def test_a_row_measured_on_another_map_says_where_its_contender_trained():
    transfer = {**PPO_MAP_ROW, "spec": "e1m2-v1", "map": "E1M2", "trained_on": "E1M1"}
    home = {**PPO_MAP_ROW, "trained_on": "E1M1"}
    text = render_readme(README, [home, transfer])
    assert "| ppo-e1m1-d3-seed0 (trained on `E1M1`) | `E1M2` |" in text
    assert "| ppo-e1m1-d3-seed0 | `E1M1` |" in text
