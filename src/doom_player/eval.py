"""The Eval Suite: one fixed procedure that measures any Contender.

Run with `uv run doom-eval --contender random`. For each seed of the spec it
plays one Attempt and appends its record to a JSON Lines file (one JSON object
per line) under `results/attempts/`. A record is written the moment its
Attempt ends, so an evaluation stopped halfway resumes where it left off. When
every seed is done, the Contender's Scoreboard row is written to
`results/scoreboard.jsonl` and logged to Weights & Biases.
"""

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from doom_player.contenders import CONTENDERS
from doom_player.contenders.base import Contender, WholeAttemptContender
from doom_player.paths import REPO_ROOT
from doom_player.scenarios import FRAME_SKIP
from doom_player.session import AttemptSession
from doom_player.video import VIDEO_FPS, VideoWriter

RESULTS_DIR = REPO_ROOT / "results"
WANDB_PROJECT = "doom-player"


@dataclass(frozen=True)
class EvalSpec:
    """Everything that must be equal for two Scoreboard rows to be comparable."""

    name: str  # change the name whenever any other field changes
    map: str
    difficulty: int
    seeds: tuple[int, ...]
    tic_limit: int


# Settled with the owner on 2026-10-01 (M1 brief, design question 2).
STANDARD_E1M1 = EvalSpec(name="e1m1-v1", map="E1M1", difficulty=3, seeds=(0, 1, 2, 3, 4), tic_limit=6300)
SPECS = {STANDARD_E1M1.name: STANDARD_E1M1}


def attempts_path(results_dir: Path, contender: str, spec: EvalSpec) -> Path:
    return results_dir / "attempts" / contender / f"{spec.name}.jsonl"


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_record(path: Path, record: dict) -> None:
    """Door A and Door B both store an Attempt through here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def play(contender: Contender, session: AttemptSession, on_frame=None) -> dict:
    """Door A: the suite's loop asks the Contender for every decision."""
    contender.reset(session.seed, session.buttons)
    try:
        while not session.finished:
            session.act(contender.act(session.observe()), contender.tics_per_action, on_frame)
    finally:
        session.close()
    return session.record.to_dict()


def run_eval(
    contender: Contender | WholeAttemptContender,
    spec: EvalSpec = STANDARD_E1M1,
    results_dir: Path = RESULTS_DIR,
    stop_after: int | None = None,
    video_dir: Path | None = None,
) -> dict | None:
    """Play the spec's missing Attempts; return the Scoreboard row once all are done.

    `stop_after` ends the run early after that many new Attempts, the way a
    subscription limit would; calling again resumes. With `video_dir`, every
    Attempt played is also filmed, one frame per tic.
    """
    path = attempts_path(results_dir, contender.name, spec)
    records = read_records(path)
    check_rules(records, spec, contender.training)
    check_trained_on(contender, spec.map)  # before any video file is opened
    done = {r["seed"] for r in records}
    played = 0
    for seed in spec.seeds:
        if seed in done:
            continue
        if stop_after is not None and played == stop_after:
            return None
        whole = isinstance(contender, WholeAttemptContender)
        video = None
        if video_dir:
            # A learned Map Contender is filmed once per decision, which leaves its game unchanged (ADR 0011).
            fps = VIDEO_FPS / FRAME_SKIP if whole else VIDEO_FPS
            video = VideoWriter(video_dir / f"{contender.name}-{spec.name}-seed{seed}.mp4", fps)
        on_frame = video.add if video else None
        try:
            if whole:
                record = contender.play_attempt(spec, seed, on_frame)
            else:
                session = AttemptSession(contender.name, spec.map, spec.difficulty, seed, spec.tic_limit)
                record = play(contender, session, on_frame)
        finally:
            if video:
                video.close()
        append_record(path, {"spec": spec.name, **record, "checkpoint": checkpoint_id(contender.training)})
        played += 1
        print(f"seed {seed}: progress {record['progress']}, cleared {record['cleared']}")
    return scoreboard_row(contender.name, spec, read_records(path), contender.training)


def check_rules(records: list[dict], spec: EvalSpec, training: dict | None = None) -> None:
    """Refuse to mix Attempts played under different rules, or by different policies, in one file."""
    for r in records:
        played = (r["map"], r["difficulty"], r["tic_limit"])
        if played != (spec.map, spec.difficulty, spec.tic_limit):
            raise SystemExit(
                f"seed {r['seed']} in {spec.name} was played as {played}, but the spec now says "
                f"{(spec.map, spec.difficulty, spec.tic_limit)}. Give the changed spec a new name."
            )
        check_checkpoint(r, spec.name, training)


def checkpoint_id(training: dict | None) -> str | None:
    """Which Training Run a learned Contender comes from; None for an untrained one."""
    return training["run_id"] if training else None


def check_checkpoint(record: dict, spec_name: str, training: dict | None) -> None:
    """Refuse an Attempt played by another checkpoint under the same Contender name (Maps and Scenarios)."""
    # Records from before M4 (Maps) have no checkpoint field: nothing was trained.
    if record.get("checkpoint") != checkpoint_id(training):
        raise SystemExit(
            f"seed {record['seed']} in {spec_name} was played by another checkpoint "
            f"({record.get('checkpoint')}) under the name {record['contender']!r}. Move its Attempts file "
            "aside before measuring the new checkpoint."
        )


def check_trained_on(contender, game: str) -> None:
    """Refuse to measure a learned Contender on a Map or Scenario other than the one it trained on."""
    if contender.training:
        trained_on = contender.training.get("map") or contender.training.get("scenario")
        if trained_on != game:
            raise SystemExit(f"{contender.name} was trained on {trained_on}, not {game}")


def training_columns(training: dict | None) -> dict:
    """The Scoreboard's training-cost fields, for Map and Scenario rows alike."""
    return {
        "training_steps": training["steps"] if training else None,
        "training_wall_clock_s": training["wall_clock_s"] if training else None,
    }


def scoreboard_row(contender: str, spec: EvalSpec, records: list[dict], training: dict | None = None) -> dict:
    by_seed = {r["seed"]: r for r in records}
    records = [by_seed[s] for s in spec.seeds]
    n = len(records)
    tokens = [r["tokens"] for r in records if r["tokens"] is not None]
    return {
        "contender": contender,
        "spec": spec.name,
        "map": spec.map,
        "difficulty": spec.difficulty,
        "seeds": list(spec.seeds),
        "tic_limit": spec.tic_limit,
        "clear_rate": sum(r["cleared"] for r in records) / n,
        "progress_mean": round(sum(r["progress"] for r in records) / n, 4),
        "progress_by_seed": [r["progress"] for r in records],
        "observation_class": records[0]["observation_class"],
        "tokens_per_attempt": round(sum(tokens) / n) if tokens else 0,
        "wall_clock_s_per_attempt": round(sum(r["wall_clock_s"] for r in records) / n, 1),
        **training_columns(training),
        "measured_on": date.today().isoformat(),
        "commit": _git_commit(),
    }


def write_row(row: dict, results_dir: Path = RESULTS_DIR) -> None:
    """Replace this Contender's row for this spec, keep everyone else's."""
    path = results_dir / "scoreboard.jsonl"
    rows = [r for r in read_records(path) if (r["contender"], r["spec"]) != (row["contender"], row["spec"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, row]))


def log_to_wandb(row: dict, records: list[dict], spec: EvalSpec) -> None:
    import wandb

    with wandb.init(
        project=WANDB_PROJECT,
        job_type="eval",
        name=f"eval-{row['contender']}-{spec.name}",
        config={"contender": row["contender"], **asdict(spec)},
    ) as run:
        columns = list(records[0])
        # Tables hold scalars; nested values (an LLM's token breakdown) go in as JSON text.
        cell = lambda v: json.dumps(v) if isinstance(v, dict | list) else v  # noqa: E731
        run.log({"attempts": wandb.Table(columns=columns, data=[[cell(r.get(c)) for c in columns] for r in records])})
        run.summary.update({k: v for k, v in row.items() if not isinstance(v, list)})


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        return out.stdout.strip()
    except OSError:
        return ""


def main() -> None:
    # Imported here: Scenario support pulls in Stable-Baselines3 and torch,
    # which a Map evaluation does not need.
    from doom_player.scenario_eval import SCENARIO_SPECS, main_scenario

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contender", required=True, choices=sorted({*CONTENDERS, "ppo"}))
    parser.add_argument("--spec", default=STANDARD_E1M1.name, choices=sorted({*SPECS, *SCENARIO_SPECS}))
    parser.add_argument("--checkpoint", type=Path, help="the model.zip a `ppo` Contender plays from")
    parser.add_argument("--video", action="store_true", help="save each Attempt to videos/")
    args = parser.parse_args()
    if args.checkpoint and args.contender != "ppo":
        parser.error(f"{args.contender!r} plays without a checkpoint")

    if args.spec in SCENARIO_SPECS:
        main_scenario(args.contender, SCENARIO_SPECS[args.spec], args.checkpoint, args.video)
        return
    spec = SPECS[args.spec]
    if args.contender == "ppo":
        if args.checkpoint is None:
            parser.error("--contender ppo needs --checkpoint checkpoints/<run>/model.zip")
        from doom_player.contenders.ppo import PPOMapContender

        contender = PPOMapContender(args.checkpoint)
    else:
        contender = CONTENDERS[args.contender]()
    row = run_eval(contender, spec, video_dir=REPO_ROOT / "videos" if args.video else None)
    write_row(row)
    log_to_wandb(row, read_records(attempts_path(RESULTS_DIR, contender.name, spec)), spec)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
