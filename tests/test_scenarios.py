"""The Scenario environment: what a learned Driver sees and does.

Scenarios ship inside the vizdoom package, so these tests need no WAD.
"""

import numpy as np
import pytest

from vizdoom import GameVariable

from doom_player.scenarios import RewardShaping, make_scenario_env


@pytest.mark.parametrize("scenario", ["basic", "defend-center", "deadly-corridor"])
def test_scenario_observation_is_a_stacked_grayscale_screen(scenario):
    env = make_scenario_env(scenario)
    try:
        obs, _ = env.reset(seed=0)
        # 4 recent frames, each an 84x84 grayscale image, one byte per pixel
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def play_fixed_actions(seed: int) -> np.ndarray:
    """Reset with `seed`, press the same buttons every time, return the last view."""
    env = make_scenario_env("basic")
    try:
        obs, _ = env.reset(seed=seed)
        for action in [0, 1, 2, 3, 2, 1]:
            obs, *_ = env.step(action)
        return obs
    finally:
        env.close()


def test_same_seed_gives_the_same_game():
    assert np.array_equal(play_fixed_actions(seed=7), play_fixed_actions(seed=7))
    # Basic places its monster at random, so another seed shows another view
    assert not np.array_equal(play_fixed_actions(seed=7), play_fixed_actions(seed=8))


def test_shaped_reward_adds_kills_and_charges_health_lost():
    env = RewardShaping(make_scenario_env("defend-center"), kill_reward=10.0, health_penalty=0.5)
    game = env.unwrapped.game
    try:
        env.reset(seed=0)
        shaped = raw = 0.0
        done, step = False, 0
        while not done:
            # DefendCenter's actions: 0 nothing, 1 attack, 2 turn right, 3 turn left
            _, reward, terminated, truncated, info = env.step(1 if step % 2 else 2)
            shaped += reward
            raw += info["raw_reward"]
            done, step = terminated or truncated, step + 1
        kills = game.get_game_variable(GameVariable.KILLCOUNT)
        health_lost = 100 - max(0.0, game.get_game_variable(GameVariable.HEALTH))
        assert kills > 0 and health_lost > 0  # the Attempt exercised both terms
        assert shaped - raw == pytest.approx(10.0 * kills - 0.5 * health_lost)
    finally:
        env.close()


def test_unknown_scenario_is_refused_with_the_valid_names():
    with pytest.raises(ValueError, match="basic.*defend-center"):
        make_scenario_env("E1M1")
