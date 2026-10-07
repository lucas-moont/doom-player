"""Replay an LLM's Attempt from its transcript, to see where it went.

The game is deterministic for a given seed, so the `act` calls recorded in the
CLI's stream-json transcript are enough to play the Attempt again, with no
LLM and no tokens. The replay gives the player's path, for a debugging
picture over the distance field (ADR 0002 allows Privileged Information in
debugging visualisations), and it checks the transcript: the replayed record
must match the stored one.

Run with `uv run doom-replay results/attempts/opus-5.5-h1/e1m1-v1.jsonl` (every seed) or add `--seed N`.
"""

import argparse
import json
from pathlib import Path

from doom_player.paths import REPO_ROOT
from doom_player.progress import render
from doom_player.session import AttemptSession

ACT = "mcp__doom__act"


def acts_from_transcript(path: Path) -> list[tuple[list[str], int]]:
    """The `act` calls that the server accepted, in the order they ran."""
    requested, accepted = {}, []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = event.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use" and block.get("name") == ACT:
                requested[block["id"]] = block.get("input", {})
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in requested:
                if not block.get("is_error"):
                    args = requested[block["tool_use_id"]]
                    accepted.append((list(args.get("buttons", [])), int(args.get("tics", 8))))
    return accepted


def replay(record: dict, transcript: Path) -> AttemptSession:
    session = AttemptSession(
        record["contender"],
        record["map"],
        record["difficulty"],
        record["seed"],
        record["tic_limit"],
        record.get("progress_rule", "doors-open"),  # records from before M5 have none
    )
    reward = 0.0
    for buttons, tics in acts_from_transcript(transcript):
        if session.finished:
            break
        # Tic by tic, so the clock is exact even when the exit is reached mid-action.
        reward += session.press(buttons, tics, on_frame=lambda frame: None)
    session.close()
    session.exit_reward = reward  # ViZDoom pays +1 only for reaching the exit alive
    return session


def check(record: dict, out_dir: Path) -> bool:
    session = replay(record, REPO_ROOT / record["transcript"])
    r = session.record
    fields = ("actions", "tics", "progress", "died", "cleared")
    same = all(getattr(r, f) == record[f] for f in fields)
    exit_ok = (session.exit_reward >= 1.0) == record["cleared"]
    print(
        f"seed {record['seed']}: {'MATCH' if same and exit_ok else 'MISMATCH'} | "
        + ", ".join(f"{f} {record[f]}->{getattr(r, f)}" for f in fields)
        + f", exit reward {session.exit_reward}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    render(session.progress_meter.field, session.progress_meter.path, out_dir / f"{record['contender']}-seed{record['seed']}.png")
    return same and exit_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="a results/attempts/... JSON Lines file")
    parser.add_argument("--seed", type=int, help="only this seed (default: every record)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "paths", help="where path pictures go")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if args.seed is not None:
        records = [r for r in records if r["seed"] == args.seed]
    results = [check(r, args.out_dir) for r in records]
    print(f"{sum(results)} of {len(results)} records replay exactly")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
