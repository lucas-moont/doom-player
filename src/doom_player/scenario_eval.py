"""The Eval Suite for Scenarios: the same fixed procedure, on a Scenario instead of a Map.

A Scenario's referee is its own Gymnasium environment, the very one a learned
Driver trains on (ADR 0010). The score is the Scenario's built-in reward,
summed over one Attempt.
"""

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np
import vizdoom
from gymnasium.wrappers import RecordVideo

from doom_player.eval import (
    RESULTS_DIR,
    _git_commit,
    append_record,
    attempts_path,
    check_checkpoint,
    check_trained_on,
    checkpoint_id,
    read_records,
    training_columns,
)
from doom_player.scenarios import FRAME_SKIP, make_scenario_env
from doom_player.session import OBSERVATION_CLASS


class ScenarioContender(Protocol):
    name: str  # the fixed name it has on the Scoreboard
    training: dict | None  # what its Training Run cost (`train.training_record`); None if untrained

    def reset(self, seed: int, action_space: gym.spaces.Discrete) -> None:
        """Prepare for a new Attempt."""

    def act(self, observation: np.ndarray) -> int:
        """Choose the next action, an index into the Scenario's action space."""


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything that must be equal for two Scenario rows to be comparable."""

    name: str  # change the name whenever any other field changes
    scenario: str
    seeds: tuple[int, ...]


# Ten seeds: a Scenario Attempt lasts seconds, so more seeds cost little and cut noise.
BASIC = ScenarioSpec(name="basic-v1", scenario="basic", seeds=tuple(range(10)))
DEFEND_CENTER = ScenarioSpec(name="defend-center-v1", scenario="defend-center", seeds=tuple(range(10)))
DEADLY_CORRIDOR = ScenarioSpec(name="deadly-corridor-v1", scenario="deadly-corridor", seeds=tuple(range(10)))
SCENARIO_SPECS = {s.name: s for s in (BASIC, DEFEND_CENTER, DEADLY_CORRIDOR)}


def play_scenario(
    contender: ScenarioContender, scenario: str, seed: int, video_dir: Path | None = None
) -> dict:
    """Play one Attempt. With `video_dir`, also save it as an MP4 of the full-color screen."""
    if video_dir is None:
        env = make_scenario_env(scenario)
    else:
        env = RecordVideo(
            make_scenario_env(scenario, render_mode="rgb_array"),
            video_folder=str(video_dir),
            episode_trigger=lambda _: True,
            name_prefix=f"{contender.name}-{scenario}-seed{seed}",
            # One frame per step, played back at the game's real speed.
            fps=round(vizdoom.DEFAULT_TICRATE / FRAME_SKIP),
        )
    start = time.perf_counter()
    try:
        obs, _ = env.reset(seed=seed)
        contender.reset(seed, env.action_space)
        reward, steps, done = 0.0, 0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(contender.act(obs))
            reward += float(r)
            steps += 1
            done = terminated or truncated
    finally:
        env.close()
    return {
        "contender": contender.name,
        "scenario": scenario,
        "seed": seed,
        "reward": reward,
        "steps": steps,
        "observation_class": OBSERVATION_CLASS,
        "wall_clock_s": round(time.perf_counter() - start, 2),
        "checkpoint": checkpoint_id(contender.training),
    }


def run_scenario_eval(
    contender: ScenarioContender,
    spec: ScenarioSpec,
    results_dir: Path = RESULTS_DIR,
    stop_after: int | None = None,
    video_dir: Path | None = None,
) -> dict | None:
    """Play the spec's missing Attempts; return the Scoreboard row once all are done.

    Records go to the same `results/attempts/` files as Map Attempts, so a
    stopped evaluation resumes where it left off, as `run_eval` does.
    With `video_dir`, every Attempt played is also saved as video.
    """
    path = attempts_path(results_dir, contender.name, spec)
    records = read_records(path)
    check_scenario_rules(records, spec, contender.training)
    check_trained_on(contender, spec.scenario)
    done = {r["seed"] for r in records}
    played = 0
    for seed in spec.seeds:
        if seed in done:
            continue
        if stop_after is not None and played == stop_after:
            return None
        record = play_scenario(contender, spec.scenario, seed, video_dir)
        append_record(path, {"spec": spec.name, **record})
        played += 1
        print(f"seed {seed}: reward {record['reward']}")
    return scenario_row(contender.name, spec, read_records(path), contender.training)


def check_scenario_rules(records: list[dict], spec: ScenarioSpec, training: dict | None = None) -> None:
    """Refuse to mix Attempts played under different rules, or by different policies, in one file."""
    for r in records:
        if r["scenario"] != spec.scenario:
            raise SystemExit(
                f"seed {r['seed']} in {spec.name} was played on {r['scenario']!r}, but the spec "
                f"now says {spec.scenario!r}. Give the changed spec a new name."
            )
        check_checkpoint(r, spec.name, training)


def scenario_row(
    contender: str, spec: ScenarioSpec, records: list[dict], training: dict | None = None
) -> dict:
    by_seed = {r["seed"]: r for r in records}
    records = [by_seed[s] for s in spec.seeds]
    n = len(records)
    return {
        "contender": contender,
        "spec": spec.name,
        "scenario": spec.scenario,
        "seeds": list(spec.seeds),
        "reward_mean": round(sum(r["reward"] for r in records) / n, 2),
        "reward_by_seed": [r["reward"] for r in records],
        "observation_class": records[0]["observation_class"],
        "wall_clock_s_per_attempt": round(sum(r["wall_clock_s"] for r in records) / n, 2),
        **training_columns(training),
        "measured_on": date.today().isoformat(),
        "commit": _git_commit(),
    }


def main_scenario(contender_name: str, spec: ScenarioSpec, checkpoint: Path | None, video: bool) -> None:
    """`doom-eval` on a Scenario spec: play, write the Scoreboard row, log to W&B."""
    from doom_player.eval import log_to_wandb, write_row
    from doom_player.paths import REPO_ROOT

    if contender_name == "ppo":
        if checkpoint is None:
            raise SystemExit("--contender ppo needs --checkpoint checkpoints/<scenario>-seed<N>/model.zip")
        from doom_player.contenders.ppo import PPOContender

        contender = PPOContender(checkpoint)
    elif contender_name == "random":
        from doom_player.contenders.random import RandomScenarioContender

        contender = RandomScenarioContender()
    else:
        raise SystemExit(f"{contender_name!r} plays Maps only")
    video_dir = REPO_ROOT / "videos" if video else None
    row = run_scenario_eval(contender, spec, video_dir=video_dir)
    write_row(row)
    log_to_wandb(row, read_records(attempts_path(RESULTS_DIR, contender.name, spec)), spec)
    print(json.dumps(row, indent=2))
