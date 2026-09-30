"""Play one Attempt with an agent that presses random buttons.

Run with `uv run doom-random`. The random agent is the floor every later
Contender must beat, and a check that the whole pipeline works.

The Attempt is recorded as video in `videos/` and logged to Weights & Biases.
Set `WANDB_MODE=offline` to keep the log on disk, or `WANDB_MODE=disabled`
to skip W&B.
"""

import argparse

import gymnasium as gym
import vizdoom
import wandb
from gymnasium.wrappers import RecordVideo
from vizdoom import gymnasium_wrapper  # noqa: F401  (registers the Vizdoom* environments)

from doom_player.paths import REPO_ROOT, WAD_PATH

DEFAULT_ENV = "VizdoomFreedoom1E1M1-S1-v0"
VIDEO_DIR = REPO_ROOT / "videos"
WANDB_PROJECT = "doom-player"


def make_env(
    env_id: str,
    frame_skip: int,
    max_steps: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    options = {}
    if env_id.startswith("VizdoomDoomE"):
        # Original Maps need the purchased WAD. Without a path, ViZDoom looks
        # for doom.wad inside its own package; point it at wads/ instead.
        if not WAD_PATH.exists():
            raise SystemExit(f"{env_id} needs the purchased WAD at {WAD_PATH}")
        options["doom_game_path"] = str(WAD_PATH)
    return gym.make(
        env_id,
        frame_skip=frame_skip,
        max_episode_steps=max_steps,
        render_mode=render_mode,
        **options,
    )


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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2100,
        help="end the Attempt as truncated after this many steps; "
        "the Map's own limit is 126000 tics, an hour of video",
    )
    args = parser.parse_args()

    run = wandb.init(
        project=WANDB_PROJECT,
        job_type="random-agent",
        config={**vars(args), "contender": "random", "vizdoom": vizdoom.__version__},
    )
    name_prefix = f"{args.env}-seed{args.seed}-{run.id}"

    env = make_env(args.env, args.frame_skip, args.max_steps, render_mode="rgb_array")
    # One frame per step, played back at the game's real speed (35 tics per second).
    env = RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda _: True,
        name_prefix=name_prefix,
        fps=round(vizdoom.DEFAULT_TICRATE / args.frame_skip),
    )
    try:
        summary = play_attempt(env, args.seed)
    finally:
        env.close()  # writes the video file

    video_path = VIDEO_DIR / f"{name_prefix}-episode-0.mp4"
    run.summary.update(summary)
    run.log({"attempt_video": wandb.Video(str(video_path), format="mp4")})
    run.finish()

    print(summary)
    print(f"video: {video_path}")


if __name__ == "__main__":
    main()
