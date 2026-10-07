"""Train a PPO policy on one Scenario or one original Map: one Training Run, one W&B run, one checkpoint.

Run with `uv run doom-train --scenario basic --seed 0`, or on a Map with
`uv run doom-train --map E1M1 --difficulty 3 --seed 0`. Several copies of the
game play in parallel processes and feed one CNN policy on the GPU.
Stable-Baselines3 writes its training curves (episode reward, episode length,
losses) in TensorBoard format; W&B copies them into the run.
Set `WANDB_MODE=offline` to keep the log on disk.
"""

import argparse
import re
import json
import time
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import wandb
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, unwrap_vec_wrapper

from doom_player.eval import MAP_SPECS
from doom_player.maps import ACTIONS, DEFAULT_PROGRESS_RULE, DEFAULT_TIC_LIMIT, ProgressShaping, make_map_env
from doom_player.paths import REPO_ROOT
from doom_player.rnd import RND, RNDBonus, RNDUpdate
from doom_player.scenarios import SCENARIOS, RewardShaping, make_scenario_env

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
WANDB_PROJECT = "doom-player"
SEED_SPACING = 1000  # at most this many game copies per Training Run; see train() for why
REWARD_SCALE_FILE = "vecnormalize.pkl"  # VecNormalize's running reward statistics, saved beside model.zip
RND_FILE = "rnd.pt"  # the RND bonus's networks and statistics, saved beside model.zip when it was paid
SCENARIO_SHAPING = ("kill_reward", "health_penalty")
MAP_SHAPING = ("progress_reward", "death_penalty")
SHAPING_PROGRESS_RULE = "doors-open"  # M4's recipe pays every-door-open Progress, on every Map (ADR 0013)


class MapRules(NamedTuple):
    """How a Map Training Run's Attempts are played and measured."""

    tic_limit: int
    progress_rule: str  # the rule the record and the Progress curve keep; shaping pays its own


@dataclass(frozen=True)
class TrainConfig:
    scenario: str | None = None  # a Scenario, or...
    seed: int = 0  # the training seed: network weights and the games played while learning
    map: str | None = None  # ...an original Map, at `difficulty`
    difficulty: int = 3
    tic_limit: int | None = None  # Maps: each Attempt's length; None takes the Map's Eval Spec's
    total_steps: int = 200_000
    n_envs: int = 8
    n_steps: int = 256  # steps each copy plays before one PPO update
    batch_size: int = 256
    # The next four follow RL Zoo's PPO settings for Atari, the closest pixel-input
    # benchmark. SB3's defaults (10 epochs, clip 0.2, no entropy bonus) made the
    # first Basic Training Run learn, then collapse at 100k steps (M3 brief, Results).
    learning_rate: float = 2.5e-4
    n_epochs: int = 4  # passes over each batch of experience per update
    clip_range: float = 0.1  # how far one update may move the policy
    ent_coef: float = 0.01  # bonus for keeping some randomness, so exploration never stops
    # Extra training reward on top of the game's own. Zero keeps it unchanged.
    # Scenarios (`scenarios.RewardShaping`): DeadlyCorridor needed them, since on its
    # own reward the policy learned to run forward and die (M3 brief, Results).
    kill_reward: float = 0.0
    health_penalty: float = 0.0
    # Maps (`maps.ProgressShaping`): the exit alone pays too rarely to learn from.
    progress_reward: float = 0.0
    death_penalty: float = 0.0
    # Exploration bonus (`rnd.RNDBonus`): how much a new screen pays next to the
    # rescaled game reward. Zero leaves training as in M4.
    rnd_coef: float = 0.0
    recurrent: bool = False  # an LSTM policy (RecurrentPPO): a memory of the last seconds
    init_from: Path | None = None  # continue from this checkpoint, e.g. a lower Difficulty
    checkpoint_every: int = 0  # steps between intermediate checkpoints; 0 for none
    label: str = ""  # tells settings apart in folder and Contender names
    out_dir: Path = CHECKPOINT_DIR

    def __post_init__(self):
        if (self.scenario is None) == (self.map is None):
            raise ValueError("train on a scenario or on a map, not both")
        other_game = MAP_SHAPING if self.scenario else SCENARIO_SHAPING
        if any(getattr(self, name) for name in other_game):
            raise ValueError(f"{', '.join(other_game)} apply to {'Maps' if self.scenario else 'Scenarios'} only")
        if self.map and not re.fullmatch(r"E\dM\d|MAP\d\d", self.map):
            raise ValueError(f"{self.map!r} is not a Map name such as E1M1 or MAP01")
        if self.recurrent and self.scenario:
            raise ValueError("--recurrent applies to Maps only: the Scenario Contender keeps no memory between steps")
        if self.checkpoint_every < 0:
            raise ValueError("--checkpoint-every must be 0 (none) or a positive number of steps")
        if self.map and not 1 <= self.difficulty <= 5:
            raise ValueError("--difficulty is Doom's skill level, 1 to 5")
        if not re.fullmatch(r"[a-z0-9-]*", self.label):
            raise ValueError("--label may use lowercase letters, digits and dashes only: it names folders")
        if self.tic_limit is not None and (self.scenario or self.tic_limit < 1):
            raise ValueError("--tic-limit applies to Maps only, and must be positive")
        if self.rnd_coef < 0:
            raise ValueError("--rnd-coef must be 0 (no bonus) or positive")
        if bool(self.rnd_coef) != ("rnd" in self.label.split("-")):
            raise ValueError("--rnd-coef and a label with an 'rnd' part, such as shaped-rnd, go together")
        if self.tic_limit is not None and not self.label:
            raise ValueError("a Training Run with --tic-limit needs a --label naming it: its Attempts differ")
        if self.n_envs > SEED_SPACING:
            raise ValueError(f"at most {SEED_SPACING} copies, or neighbouring training seeds would share games")

    @property
    def checkpoint_folder(self) -> str:
        """Game, training seed and label, such as `e1m1-d3-seed0-shaped`; also names the W&B run."""
        game = self.scenario if self.scenario else f"{self.map.lower()}-d{self.difficulty}"
        return f"{game}-seed{self.seed}{self._label_suffix}"

    @property
    def contender(self) -> str:
        """The Scoreboard name; M3's Scenario policies kept the shorter `ppo-seed<N>`."""
        if self.scenario:
            return f"ppo-seed{self.seed}{self._label_suffix}"
        return f"ppo-{self.checkpoint_folder}"

    @property
    def _label_suffix(self) -> str:
        return f"-{self.label}" if self.label else ""

    @property
    def reward_shaping(self) -> dict | None:
        terms = {n: getattr(self, n) for n in (SCENARIO_SHAPING if self.scenario else MAP_SHAPING)}
        if not any(terms.values()):
            return None
        return terms if self.scenario else {**terms, "progress_rule": SHAPING_PROGRESS_RULE}

    @property
    def map_rules(self) -> MapRules:
        """Maps: the Map's Eval Spec's rules, so training is read on the Scoreboard's ruler; `tic_limit` overrides."""
        spec = MAP_SPECS.get(self.map)
        rules = MapRules(spec.tic_limit, spec.progress_rule) if spec else MapRules(DEFAULT_TIC_LIMIT, DEFAULT_PROGRESS_RULE)
        return rules._replace(tic_limit=self.tic_limit or rules.tic_limit)

    def env_factory(self):
        """How each training copy is built: the env, and the shaping wrapper (None without shaping)."""
        if self.scenario:
            return partial(make_scenario_env, self.scenario), RewardShaping if self.reward_shaping else None
        rules = self.map_rules
        env = partial(make_map_env, self.map, self.difficulty, rules.tic_limit, progress_rule=rules.progress_rule)
        return env, ProgressShaping if self.reward_shaping else None


def train(config: TrainConfig) -> Path:
    """Run one Training Run; return the path of the saved checkpoint."""
    run_dir = config.out_dir / config.checkpoint_folder
    # Checked before any game process starts, so a refusal leaves nothing running.
    # Any leftover counts: a Training Run stopped halfway leaves steps-<N>/ checkpoints behind.
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; give this Training Run a --label, or move it aside")
    parent = training_record(config.init_from) if config.init_from else None
    if parent and parent.get("recurrent", False) != config.recurrent:
        raise ValueError("--recurrent must match the checkpoint training continues from")
    if parent and config.map and parent.get("actions") not in (None, list(ACTIONS)):
        raise ValueError("the checkpoint training continues from learned another Action set than maps.ACTIONS")
    factory, shaping = config.env_factory()
    env = make_vec_env(
        factory,
        n_envs=config.n_envs,
        # SB3 seeds copy k with `seed + k`, so training seeds 0 and 1 would share
        # 7 of 8 copies' streams of games. Spacing them keeps each seed's games its own.
        seed=config.seed * SEED_SPACING,
        vec_env_cls=SubprocVecEnv if config.n_envs > 1 else DummyVecEnv,
        # Applied outside SB3's Monitor, so the logged curve stays the game's own reward.
        wrapper_class=shaping,
        wrapper_kwargs=config.reward_shaping,
    )
    # Scenario rewards run from about -500 to +100 in Basic but -1 to +30 in
    # DefendCenter. Rescaling them to a running unit size keeps the value
    # network's errors small on every Scenario. Training only: the Monitor
    # wrapper inside make_vec_env still logs the raw reward, and the Eval
    # Suite scores the raw reward. A continued Training Run keeps its parent's
    # scale, which its value network learned on.
    if parent_scale := _beside(config.init_from, REWARD_SCALE_FILE):
        env = VecNormalize.load(str(parent_scale), env)
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
    if config.rnd_coef:
        # Outside VecNormalize, so the game's reward is rescaled the same with or
        # without the bonus, and the bonus is scaled on its own (M5 brief). No
        # bonus during the first rollout, while the pixel statistics settle.
        env = RNDBonus(env, config.rnd_coef, RND(seed=config.seed), warmup_frames=config.n_envs * config.n_steps)
        if parent_rnd := _beside(config.init_from, RND_FILE):
            env.load(parent_rnd)
    algorithm = RecurrentPPO if config.recurrent else PPO
    hyperparameters = dict(
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        n_epochs=config.n_epochs,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
    )
    with wandb.init(
        project=WANDB_PROJECT,
        job_type="train",
        name=f"train-ppo-{config.checkpoint_folder}",
        config={"algorithm": algorithm.__name__, **asdict(config), "out_dir": str(config.out_dir)},
        sync_tensorboard=True,
    ):
        try:
            if config.init_from:
                # The weights come from the checkpoint; the settings, as for a new Training Run, from this config.
                model = algorithm.load(
                    config.init_from, env=env, tensorboard_log=str(run_dir / "tensorboard"), **hyperparameters
                )
                model.set_random_seed(config.seed)
            else:
                model = algorithm(
                    "CnnLstmPolicy" if config.recurrent else "CnnPolicy",
                    env,
                    **hyperparameters,
                    seed=config.seed,
                    tensorboard_log=str(run_dir / "tensorboard"),
                    verbose=0,
                )
            callbacks = [IntermediateCheckpoints(run_dir, config, parent)] if config.checkpoint_every else []
            if config.map:
                callbacks.append(ProgressCurve())
            if config.rnd_coef:
                callbacks.append(RNDUpdate())
            start = time.perf_counter()
            # A continued Training Run keeps counting steps from where its checkpoint stopped.
            model.learn(
                total_timesteps=config.total_steps,
                callback=callbacks,  # SB3 wraps a list in a CallbackList
                reset_num_timesteps=not config.init_from,
            )
            wall_clock_s = time.perf_counter() - start
        finally:
            env.close()
        return save_checkpoint(model, run_dir, config, wall_clock_s, parent)


def save_checkpoint(
    model, folder: Path, config: TrainConfig, wall_clock_s: float, parent: dict | None, halfway: bool = False
) -> Path:
    """Save the policy as `folder/model.zip`, with what its Training Run cost so far beside it.

    A checkpoint saved `halfway` through a Training Run gets a Contender name of its
    own (`...-at-<steps>`), so the Eval Suite can measure it next to the final one.
    """
    checkpoint = folder / "model.zip"
    folder.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    model.get_vec_normalize_env().save(str(folder / REWARD_SCALE_FILE))  # for a Training Run continued from here
    if bonus := unwrap_vec_wrapper(model.get_env(), RNDBonus):
        bonus.save(folder / RND_FILE)
    rules = config.map_rules if config.map else MapRules(None, None)
    record = {
        "run_id": uuid.uuid4().hex[:12],  # tells this checkpoint apart from a retrain with the same seed
        "contender": config.contender + (f"-at-{model.num_timesteps}" if halfway else ""),
        "scenario": config.scenario,
        "map": config.map,
        "difficulty": config.difficulty if config.map else None,
        "tic_limit": rules.tic_limit,
        "progress_rule": rules.progress_rule,  # the rule the record and curve keep
        "seed": config.seed,
        "steps": model.num_timesteps,
        # A continued Training Run's cost includes the Training Run it started from.
        "wall_clock_s": round(wall_clock_s + (parent["wall_clock_s"] if parent else 0), 1),
        "reward_shaping": config.reward_shaping,
        "rnd": bonus.settings if bonus else None,
        "recurrent": config.recurrent,
        "actions": list(ACTIONS) if config.map else None,  # the Action set, in the order the policy learned it
        "init_from": parent["run_id"] if parent else None,
        "wandb_run": wandb.run.url if wandb.run else None,
    }
    _record_path(checkpoint).write_text(json.dumps(record, indent=2) + "\n")
    return checkpoint


class IntermediateCheckpoints(BaseCallback):
    """Every `config.checkpoint_every` steps, save a checkpoint into `steps-<N>/`, to look at a Training Run halfway."""

    def __init__(self, run_dir: Path, config: TrainConfig, parent: dict | None):
        super().__init__()
        self.run_dir, self.config, self.parent = run_dir, config, parent

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        self._schedule_next()  # a continued Training Run counts on from its checkpoint's steps, not from 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            folder = self.run_dir / f"steps-{self.num_timesteps}"
            elapsed = time.perf_counter() - self.started
            save_checkpoint(self.model, folder, self.config, elapsed, self.parent, halfway=True)
            self._schedule_next()
        return True

    def _schedule_next(self) -> None:
        # The next multiple of checkpoint_every: one step of all game copies may jump past several.
        every = self.config.checkpoint_every
        self._next = (self.num_timesteps // every + 1) * every


class ProgressCurve(BaseCallback):
    """Maps only: log each finished Attempt's Progress and Clear next to the reward curve.

    A Map's own reward is 0 until the exit, so its curve stays flat while the
    policy learns the route; Progress shows the learning before the first Clear.
    Each point averages only the Attempts that ended since the last one (a few,
    since a full Attempt is thousands of steps long), so it is noisier than the
    reward curve, which averages the last 100 Attempts. On a Map with keys it
    also logs the share of Attempts that ended holding one. Progress and keys
    held are Privileged Information: the policy never sees these curves; they
    are declared in Results.
    """

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if record := info.get("record"):  # the Attempt that copy was playing just ended
                self.logger.record_mean("rollout/progress", record["progress"])
                self.logger.record_mean("rollout/clear_rate", float(record["cleared"]))
                if record["keys_held"] is not None:  # keys were read: a keyed Map with keys on it
                    self.logger.record_mean("rollout/key_rate", float(bool(record["keys_held"])))
        return True


def _beside(checkpoint: Path | None, name: str) -> Path | None:
    """A file saved next to `checkpoint`, if there is one."""
    path = checkpoint.with_name(name) if checkpoint else None
    return path if path and path.exists() else None


def training_record(checkpoint: Path) -> dict:
    """What the Training Run behind a checkpoint cost: steps and wall-clock seconds."""
    return json.loads(_record_path(checkpoint).read_text())


def _record_path(checkpoint: Path) -> Path:
    return checkpoint.with_name("training.json")


def main() -> None:
    # SUPPRESS leaves unset flags out, so TrainConfig's defaults are the only defaults.
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    game = parser.add_mutually_exclusive_group(required=True)
    game.add_argument("--scenario", choices=sorted(SCENARIOS))
    game.add_argument("--map", type=str.upper, help="an original Map, such as E1M1")
    parser.add_argument("--difficulty", type=int)
    parser.add_argument("--tic-limit", type=int, help="Maps: each Attempt's length; default, the Map's Eval Spec's")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--kill-reward", type=float, help="Scenarios: extra training reward per kill")
    parser.add_argument("--health-penalty", type=float, help="Scenarios: training cost per health point lost")
    parser.add_argument("--progress-reward", type=float, help="Maps: training reward for reaching Progress 1")
    parser.add_argument("--death-penalty", type=float, help="Maps: training cost of dying")
    parser.add_argument("--rnd-coef", type=float, help="exploration bonus (RND) for new screens; 0 for none")
    parser.add_argument("--recurrent", action="store_true", help="an LSTM policy (RecurrentPPO)")
    parser.add_argument("--init-from", type=Path, help="continue training from this checkpoint")
    parser.add_argument("--checkpoint-every", type=int, help="steps between intermediate checkpoints")
    parser.add_argument("--label", help="tells settings apart in folder and Contender names")
    print(train(TrainConfig(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
