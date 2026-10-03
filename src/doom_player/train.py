"""Train a PPO policy on one Scenario: one Training Run, one W&B run, one checkpoint.

Run with `uv run doom-train --scenario basic --seed 0`. Several copies of the
Scenario play in parallel processes and feed one CNN policy on the GPU.
Stable-Baselines3 writes its training curves (episode reward, episode length,
losses) in TensorBoard format; W&B copies them into the run.
Set `WANDB_MODE=offline` to keep the log on disk.
"""

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from doom_player.paths import REPO_ROOT
from doom_player.scenarios import SCENARIOS, RewardShaping, make_scenario_env

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
WANDB_PROJECT = "doom-player"


@dataclass(frozen=True)
class TrainConfig:
    scenario: str
    seed: int  # the training seed: network weights and the games played while learning
    total_steps: int = 200_000
    n_envs: int = 8
    n_steps: int = 256  # steps each copy plays before one PPO update
    batch_size: int = 256
    # The next four follow RL Zoo's PPO settings for Atari, the closest pixel-input
    # benchmark. SB3's defaults (10 epochs, clip 0.2, no entropy bonus) made the
    # first Basic run learn, then collapse at 100k steps (M3 brief, Results).
    learning_rate: float = 2.5e-4
    n_epochs: int = 4  # passes over each batch of experience per update
    clip_range: float = 0.1  # how far one update may move the policy
    ent_coef: float = 0.01  # bonus for keeping some randomness, so exploration never stops
    # Extra training reward on top of the Scenario's own (`scenarios.RewardShaping`).
    # Zero keeps the Scenario's reward unchanged. DeadlyCorridor needs them: on its
    # own reward the policy learned to run forward and die (M3 brief, Results).
    kill_reward: float = 0.0
    health_penalty: float = 0.0
    out_dir: Path = CHECKPOINT_DIR

    @property
    def reward_shaping(self) -> dict | None:
        if not (self.kill_reward or self.health_penalty):
            return None
        return {"kill_reward": self.kill_reward, "health_penalty": self.health_penalty}


def train(config: TrainConfig) -> Path:
    """Run one Training Run; return the path of the saved checkpoint."""
    run_dir = config.out_dir / f"{config.scenario}-seed{config.seed}"
    vec_env_cls = SubprocVecEnv if config.n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        make_scenario_env,
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=vec_env_cls,
        env_kwargs={"scenario": config.scenario},
        # Applied outside SB3's Monitor, so the logged curve stays the Scenario's own reward.
        wrapper_class=RewardShaping if config.reward_shaping else None,
        wrapper_kwargs=config.reward_shaping,
    )
    # Scenario rewards run from about -500 to +100 in Basic but -1 to +30 in
    # DefendCenter. Rescaling them to a running unit size keeps the value
    # network's errors small on every Scenario. Training only: the Monitor
    # wrapper inside make_vec_env still logs the raw reward, and the Eval
    # Suite scores the raw reward.
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    with wandb.init(
        project=WANDB_PROJECT,
        job_type="train",
        name=f"train-ppo-{config.scenario}-seed{config.seed}",
        config={"algorithm": "PPO", **asdict(config), "out_dir": str(config.out_dir)},
        sync_tensorboard=True,
    ):
        try:
            model = PPO(
                "CnnPolicy",
                env,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                n_epochs=config.n_epochs,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                seed=config.seed,
                tensorboard_log=str(run_dir / "tensorboard"),
                verbose=0,
            )
            start = time.perf_counter()
            model.learn(total_timesteps=config.total_steps)
            wall_clock_s = round(time.perf_counter() - start, 1)
        finally:
            env.close()
        checkpoint = run_dir / "model.zip"
        model.save(checkpoint)
        record = {
            "run_id": uuid.uuid4().hex[:12],  # tells this checkpoint apart from a retrain with the same seed
            "scenario": config.scenario,
            "seed": config.seed,
            "steps": model.num_timesteps,
            "wall_clock_s": wall_clock_s,
            "reward_shaping": config.reward_shaping,
            "wandb_run": wandb.run.url if wandb.run else None,
        }
        _record_path(checkpoint).write_text(json.dumps(record, indent=2) + "\n")
    return checkpoint


def training_record(checkpoint: Path) -> dict:
    """What the Training Run behind a checkpoint cost: steps and wall-clock seconds."""
    return json.loads(_record_path(checkpoint).read_text())


def _record_path(checkpoint: Path) -> Path:
    return checkpoint.with_name("training.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    parser.add_argument("--kill-reward", type=float, default=0.0, help="extra training reward per kill")
    parser.add_argument("--health-penalty", type=float, default=0.0, help="training cost per health point lost")
    args = parser.parse_args()
    checkpoint = train(
        TrainConfig(
            args.scenario,
            args.seed,
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            kill_reward=args.kill_reward,
            health_penalty=args.health_penalty,
        )
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
