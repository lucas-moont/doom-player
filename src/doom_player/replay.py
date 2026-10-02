"""Replay an LLM's Attempt from its transcript, to see where it went.

The game is deterministic for a given seed, so the `act` calls recorded in the
CLI's stream-json transcript are enough to play the Attempt again, with no
LLM and no tokens. The replay gives the player's path, for a debugging
picture over the distance field (ADR 0002 allows Privileged Information in
debugging visualisations), and it checks the transcript: the replayed record
must match the stored one.

Run with `uv run doom-replay results/attempts/opus-5.5-h0/e1m1-v1.jsonl --seed 1`.
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
    session = AttemptSession(record["contender"], record["map"], record["difficulty"], record["seed"], record["tic_limit"])
    for buttons, tics in acts_from_transcript(transcript):
        if session.finished:
            break
        session.press(buttons, tics)
    session.close()
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="a results/attempts/... JSON Lines file")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, help="where to save the path picture")
    args = parser.parse_args()

    record = next(r for r in map(json.loads, args.records.read_text().splitlines()) if r["seed"] == args.seed)
    session = replay(record, REPO_ROOT / record["transcript"])
    replayed = session.record
    same = (replayed.actions, replayed.tics, replayed.progress, replayed.died, replayed.cleared) == (
        record["actions"], record["tics"], record["progress"], record["died"], record["cleared"]
    )
    print(f"replayed: actions {replayed.actions}, progress {replayed.progress}, died {replayed.died}")
    print(f"stored:   actions {record['actions']}, progress {record['progress']}, died {record['died']}")
    print("MATCH" if same else "MISMATCH")
    out = args.out or REPO_ROOT / "runs" / "paths" / f"{record['contender']}-seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(session.progress_meter.field, session.progress_meter.path, out)
    print(f"path picture: {out}")


if __name__ == "__main__":
    main()
