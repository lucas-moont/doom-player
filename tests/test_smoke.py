"""Checks that need no purchased WAD, so they pass on a fresh clone."""

from doom_player.random_agent import DEFAULT_ENV, make_env


def test_freedoom_env_resets_and_steps():
    env = make_env(DEFAULT_ENV, frame_skip=4)
    try:
        obs, _ = env.reset(seed=0)
        assert obs["screen"].shape == (240, 320, 3)
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(float(reward), float)
        assert not truncated
    finally:
        env.close()
