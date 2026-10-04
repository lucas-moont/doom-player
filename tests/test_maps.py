"""The Map environment: an original Map seen and played the way a learned Driver trains on it.

It is a Gymnasium face on the AttemptSession, so training and the Eval Suite
play the very same game (ADR 0011). Needs the purchased WAD.
"""

from itertools import cycle

import gymnasium as gym
import numpy as np
import pytest

from doom_player.eval import SPECS
from doom_player.maps import ACTIONS, ProgressShaping, make_map_env
from doom_player.paths import WAD_PATH
from doom_player.scenarios import FRAME_SKIP
from doom_player.session import AttemptSession

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")

ACTION = {name: i for i, name in enumerate(ACTIONS)}  # action index by name


def test_map_observation_is_a_stacked_grayscale_screen():
    env = make_map_env("E1M1")
    try:
        obs, _ = env.reset(seed=0)
        # 4 recent frames, each an 84x84 grayscale image of the screen, HUD included
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        # The screen only: no position, depth or labels (ADR 0002).
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.unwrapped.session.record.observation_class == "human-equivalent"
    finally:
        env.close()


def test_each_action_presses_a_fixed_set_of_the_sessions_buttons():
    env = make_map_env("E1M1")
    try:
        env.reset(seed=0)
        buttons = env.unwrapped.session.buttons
        assert env.action_space.n == len(ACTIONS)
        for names in ACTIONS.values():
            assert set(names) <= set(buttons)  # every name is a real button
        assert env.unwrapped.pressed == [[b in names for b in buttons] for names in ACTIONS.values()]
        # Doors and the exit switch need USE; monsters need ATTACK.
        assert {"USE", "ATTACK", "MOVE_FORWARD"} <= {b for names in ACTIONS.values() for b in names}
    finally:
        env.close()


ACTION_SEQUENCE = [ACTION["forward"], ACTION["forward"], ACTION["forward + turn left"], ACTION["forward"], ACTION["forward + turn right"],
                   ACTION["forward"], ACTION["attack"], ACTION["forward"], ACTION["use"], ACTION["turn left"], ACTION["forward"], ACTION["forward"]] * 6


def test_map_env_plays_the_same_game_as_the_attempt_session():
    # 72 actions of 4 tics overrun the 200-tic limit, so both Attempts end by timeout.
    env = make_map_env("E1M1", tic_limit=200)
    try:
        env.reset(options={"game_seed": 3})
        pressed = env.unwrapped.pressed
        for action in ACTION_SEQUENCE:
            *_, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        through_env = env.unwrapped.session.record.to_dict()
        env_path = list(env.unwrapped.session.progress_meter.path)
    finally:
        env.close()

    session = AttemptSession("training", "E1M1", 3, seed=3, tic_limit=200)
    try:
        for action in ACTION_SEQUENCE:
            if session.finished:
                break
            session.act(pressed[action], FRAME_SKIP)
        direct = session.record.to_dict()
        direct_path = list(session.progress_meter.path)
    finally:
        session.close()
    del through_env["wall_clock_s"], direct["wall_clock_s"]
    assert through_env["seed"] == 3 and through_env["truncated"]
    assert through_env == direct
    assert env_path == direct_path  # same position after every action


def test_map_attempt_ends_at_the_tic_limit():
    env = make_map_env("E1M1", tic_limit=40)
    try:
        env.reset(options={"game_seed": 0})
        steps = 0
        while True:
            _, _, terminated, truncated, info = env.step(ACTION["turn left"])  # safe, goes nowhere
            steps += 1
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert info["record"]["truncated"] and not info["record"]["cleared"]
        assert steps == 40 // FRAME_SKIP
    finally:
        env.close()


def test_training_games_never_use_an_eval_seed():
    eval_seeds = {s for spec in SPECS.values() for s in spec.seeds}
    env = make_map_env("E1M1")

    def game_for(seed):
        env.reset(seed=seed)
        return env.unwrapped.session.seed

    try:
        games = [game_for(seed) for seed in (0, 1, None, None)]
        again = game_for(0)
    finally:
        env.close()
    assert not eval_seeds & set(games)
    assert games[0] != games[1]  # each training seed gets its own games...
    assert again == games[0]  # ...reproducibly


def play_shaped(actions: list[int], difficulty: int, tic_limit: int, seed: int = 0) -> tuple[float, float, dict]:
    """Play `actions` in a loop until the Attempt ends; return (shaped total, raw total, record)."""
    env = ProgressShaping(make_map_env("E1M1", difficulty, tic_limit), progress_reward=100.0, death_penalty=50.0)
    try:
        env.reset(options={"game_seed": seed})
        shaped = raw = 0.0
        for action in cycle(actions):
            _, reward, terminated, truncated, info = env.step(action)
            shaped += reward
            raw += info["raw_reward"]
            if terminated or truncated:
                return shaped, raw, info["record"]
    finally:
        env.close()


def test_progress_shaping_pays_the_best_progress_once():
    # Walk toward the exit, back, and toward it again: only new ground pays.
    shaped, raw, record = play_shaped([ACTION["forward"]] * 12 + [ACTION["back"]] * 12, difficulty=3, tic_limit=480)
    assert not record["died"] and record["progress"] > 0.05
    assert shaped - raw == pytest.approx(100.0 * record["progress"], abs=0.01)  # the record rounds Progress to 4 places


def test_progress_shaping_charges_death():
    zigzag = [ACTION["forward + turn left"]] * 6 + [ACTION["forward"]] * 10 + [ACTION["forward + turn right"]] * 6
    shaped, raw, record = play_shaped(zigzag, difficulty=5, tic_limit=6300)
    assert record["died"]
    assert shaped - raw == pytest.approx(100.0 * record["progress"] - 50.0, abs=0.01)
