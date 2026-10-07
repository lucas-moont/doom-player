"""RND pays more for a screen it has rarely seen than for one it has seen many times.

Fast and without the WAD: frames are made-up 84x84 gray pictures, the size
of one frame of the policy view.
"""

import io

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from doom_player.rnd import RND, RNDBonus, RNDUpdate


def frames(seed: int, n: int = 64) -> np.ndarray:
    """`n` gray frames of one made-up place: noise around a pattern of its own."""
    rng = np.random.default_rng(seed)
    place = rng.integers(0, 256, (1, 84, 84))
    noise = rng.integers(-8, 9, (n, 84, 84))
    return np.clip(place + noise, 0, 255).astype(np.uint8)


def test_a_familiar_screen_pays_less_than_a_new_one():
    rnd = RND(seed=0, device="cpu")
    familiar, new = frames(1), frames(2)
    rnd.update_obs_stats(np.concatenate([familiar, new]))
    for _ in range(30):
        rnd.fit(familiar)
    assert rnd.bonus(familiar).mean() < 0.5 * rnd.bonus(new).mean()


def test_fitting_lowers_the_prediction_error():
    rnd = RND(seed=0, device="cpu")
    seen = frames(1)
    rnd.update_obs_stats(seen)
    first = rnd.fit(seen)
    for _ in range(10):
        last = rnd.fit(seen)
    assert last < first


def test_the_same_seed_builds_the_same_networks():
    a, b, c = RND(seed=3, device="cpu"), RND(seed=3, device="cpu"), RND(seed=4, device="cpu")
    shot = frames(5, n=4)
    assert np.array_equal(a.bonus(shot), b.bonus(shot))
    assert not np.array_equal(a.bonus(shot), c.bonus(shot))


def test_state_survives_a_round_trip():
    rnd = RND(seed=0, device="cpu")
    seen = frames(1)
    rnd.update_obs_stats(seen)
    rnd.fit(seen)
    saved = io.BytesIO()  # as a checkpoint keeps it: through a file
    torch.save(rnd.state_dict(), saved)
    saved.seek(0)
    copy = RND(seed=9, device="cpu")
    copy.load_state_dict(torch.load(saved))
    assert np.allclose(copy.bonus(seen), rnd.bonus(seen))
    assert copy.fit(seen) == pytest.approx(rnd.fit(seen), rel=1e-4)  # the optimiser's state came along too


class Rooms(gym.Env):
    """Three steps through three made-up rooms, one frame stack each, paying 1 per step."""

    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.room = 0
        return self._view(), {}

    def step(self, action):
        self.room += 1
        return self._view(), 1.0, self.room == 3, False, {}

    def _view(self):
        return np.repeat(frames(10 + self.room, n=1), 4, axis=0)


def rooms(n_envs: int = 2) -> DummyVecEnv:
    return DummyVecEnv([Rooms] * n_envs)


def play(env: RNDBonus, steps: int) -> list[np.ndarray]:
    env.reset()
    rewards = []
    for _ in range(steps):
        _, reward, _, _ = env.step(np.zeros(env.num_envs, dtype=int))
        rewards.append(reward)
    return rewards


def test_no_coefficient_leaves_the_reward_unchanged():
    env = RNDBonus(rooms(), coef=0.0, rnd=RND(device="cpu"))
    assert all(np.array_equal(r, [1.0, 1.0]) for r in play(env, 6))


def test_the_bonus_is_added_to_the_reward_once_warmed_up():
    # Same seed, same rooms: the two wrappers compute the same bonus and pay it at different rates.
    half, full = (RNDBonus(rooms(), coef=c, rnd=RND(device="cpu"), warmup_frames=4) for c in (0.5, 1.0))
    paid_half, paid_full = (np.array(play(env, 6)) - 1.0 for env in (half, full))
    assert not paid_half[:2].any() and not paid_full[:2].any()  # 2 steps x 2 copies: still warming up
    assert np.all(paid_half[2:] > 0)
    assert np.allclose(paid_full[2:], 2 * paid_half[2:])


def test_the_last_screen_of_an_attempt_is_the_one_paid_for():
    # When an Attempt ends, the vec env has already reset that copy: its observation
    # is the next Attempt's spawn, and the screen the step reached is in the info.
    rnd = RND(device="cpu")
    paid_for = []
    bonus = rnd.bonus
    rnd.bonus = lambda f: paid_for.append(f.copy()) or bonus(f)
    play(RNDBonus(rooms(), coef=0.5, rnd=rnd), 3)
    assert np.array_equal(paid_for[-1], np.repeat(frames(13, n=1), 2, axis=0))


def test_each_finished_attempt_reports_its_bonus_and_reward():
    env = RNDBonus(rooms(), coef=0.5, rnd=RND(device="cpu"))
    env.reset()
    for _ in range(3):
        _, _, dones, infos = env.step(np.zeros(2, dtype=int))
    assert dones.all()
    for info in infos:
        assert info["rnd"]["scaled_reward"] == pytest.approx(3.0)
        assert info["rnd"]["bonus"] > 0


def test_the_predictor_trains_on_each_rollout_and_learns_the_rooms():
    env = RNDBonus(rooms(), coef=0.5, rnd=RND(device="cpu"))
    play(env, 6)
    first = env.train_on_rollout()
    for _ in range(10):
        play(env, 6)
        last = env.train_on_rollout()
    assert last["rnd/predictor_loss"] < first["rnd/predictor_loss"]


def test_ppo_trains_the_predictor_after_every_rollout():
    env = RNDBonus(rooms(), coef=0.5, rnd=RND(device="cpu"))
    trained = []
    train_on_rollout = env.train_on_rollout
    env.train_on_rollout = lambda: trained.append(1) or train_on_rollout()
    model = PPO("CnnPolicy", env, n_steps=6, batch_size=12, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=24, callback=RNDUpdate())  # 2 copies x 6 steps per rollout: 2 rollouts
    assert len(trained) == 2


def test_the_wrapper_state_survives_a_round_trip(tmp_path):
    env = RNDBonus(rooms(), coef=0.5, rnd=RND(device="cpu"))
    play(env, 6)
    env.train_on_rollout()
    env.save(tmp_path / "rnd.pt")
    copy = RNDBonus(rooms(), coef=0.5, rnd=RND(seed=9, device="cpu"))
    copy.load(tmp_path / "rnd.pt")
    assert np.allclose(play(copy, 2), play(env, 2))  # the same bonus for the same screens


def test_the_warm_up_leaves_the_bonus_scale_untouched():
    # Before the pixel statistics settle the errors are noise; a running spread never forgets them.
    env = RNDBonus(rooms(), coef=0.5, rnd=RND(device="cpu"), warmup_frames=4)
    play(env, 2)
    assert env.returns.count < 1  # only its starting epsilon
    play(env, 1)
    assert env.returns.count >= 1
