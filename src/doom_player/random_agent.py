"""Play one Attempt with an agent that presses random buttons.

Run with `uv run doom-random`. The random agent is the floor every later
Contender must beat, and a check that the whole pipeline works.
"""

import argparse

import gymnasium as gym
from vizdoom import gymnasium_wrapper  # noqa: F401  (registers the Vizdoom* environments)

DEFAULT_ENV = "VizdoomFreedoom1E1M1-S1-v0"


def make_env(env_id: str, frame_skip: int) -> gym.Env:
    return gym.make(env_id, frame_skip=frame_skip)


def play_attempt(env: gym.Env, seed: int) -> dict:
    """Play until the Attempt ends; return its summary."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    steps, total_reward = 0, 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += float(reward)
    return {
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium environment id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=4, help="tics per action")
    args = parser.parse_args()

    env = make_env(args.env, args.frame_skip)
    try:
        summary = play_attempt(env, args.seed)
    finally:
        env.close()
    print(summary)


if __name__ == "__main__":
    main()
