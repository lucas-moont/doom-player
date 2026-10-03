"""The LLM launcher: the Eval Suite's loop for Door B.

Run with `uv run doom-llm-eval --harness h0`. For each seed of the spec not yet
in `results/`, it starts the Doom MCP server for that seed as a local HTTP
service, then lets the Claude Code CLI play it headless (`claude -p`). The
CLI is isolated: no built-in tools (no files, no shell, no web), only the
Doom server's tools, its own config directory, an empty working directory.

If the CLI stops before the Attempt is over, the same CLI session is resumed;
the server keeps the game, so play continues where it stopped. When the
Attempt ends, the server writes its record; the launcher adds the token
counts and the model id from the CLI's output and stores it, exactly like a
Door A record. A subscription limit stops the run; the next run starts the
interrupted seed again from spawn and skips every finished one.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from doom_player.eval import (
    RESULTS_DIR,
    SPECS,
    STANDARD_E1M1,
    EvalSpec,
    append_record,
    attempts_path,
    check_rules,
    log_to_wandb,
    read_records,
    scoreboard_row,
    write_row,
)
from doom_player.harness import HARNESSES, RESUME, TASK, Harness
from doom_player.paths import REPO_ROOT

MODEL = "claude-opus-5-5"
MODEL_LABEL = "opus-5.5"
CLAUDE = Path.home() / ".local" / "bin" / "claude"
CLAUDE_CONFIG_DIR = Path.home() / ".config" / "doom-player-claude"
RUNS_DIR = REPO_ROOT / "runs" / "llm"
VIDEO_DIR = REPO_ROOT / "videos"
MAX_RESUMES = 20
SERVER_START_TIMEOUT_S = 120


class LimitReached(Exception):
    """The subscription's usage limit stopped the CLI."""


@dataclass
class CliUsage:
    """What the CLI reports about one Attempt, summed over resumes."""

    model: str | None = None
    session_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0
    tools_offered: list[str] = field(default_factory=list)
    runs: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def read_stream(self, lines: list[str]) -> dict | None:
        """Fold one CLI run's stream-json output in; return its final result event."""
        result = None
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init":
                self.session_id = event.get("session_id", self.session_id)
                self.model = event.get("model", self.model)
                self.tools_offered = event.get("tools", self.tools_offered)
            elif kind == "assistant":
                content = event.get("message", {}).get("content", [])
                self.tool_calls += sum(1 for block in content if block.get("type") == "tool_use")
            elif kind == "result":
                result = event
                usage = event.get("usage", {})
                for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
                    setattr(self, key, getattr(self, key) + int(usage.get(key, 0) or 0))
                self.turns += int(event.get("num_turns", 0) or 0)
                self.session_id = event.get("session_id", self.session_id)
        self.runs += 1
        return result


def contender_name(harness: Harness) -> str:
    return f"{MODEL_LABEL}-{harness.name}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, server: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the Doom MCP server exited with code {server.returncode}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"the Doom MCP server did not open port {port}")


def claude_command(claude: Path, harness: Harness, mcp_config: Path, tic_limit: int, resume: str | None) -> list[str]:
    command = [
        str(claude),
        "-p",
        RESUME if resume else TASK,
        "--model",
        MODEL,
        "--tools",
        "",  # no built-in tools: no files, no shell, no web
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
        "--allowedTools",
        *harness.allowed_tools,
        "--system-prompt",
        harness.system_prompt(tic_limit),
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if resume:
        command += ["--resume", resume]
    return command


def play_llm_attempt(
    harness: Harness,
    spec: EvalSpec,
    seed: int,
    claude: Path = CLAUDE,
    runs_dir: Path = RUNS_DIR,
    video_dir: Path = VIDEO_DIR,
) -> dict:
    """Play one Attempt through the CLI; return its complete record."""
    contender = contender_name(harness)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{contender}-{spec.name}-seed{seed}-{stamp}"
    transcript = runs_dir / contender / f"{name}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    video = video_dir / f"{name}.mp4"

    with tempfile.TemporaryDirectory(prefix="doom-llm-") as workdir:
        workdir = Path(workdir)
        record_out = workdir / "record.json"
        port = free_port()
        server = subprocess.Popen(
            [
                sys.executable, "-m", "doom_player.mcp_server",
                "--contender", contender,
                "--seed", str(seed),
                "--difficulty", str(spec.difficulty),
                "--tic-limit", str(spec.tic_limit),
                "--record-out", str(record_out),
                "--video", str(video),
                "--http", str(port),
                *harness.server_flags,
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=(runs_dir / contender / f"{name}.server.log").open("w"),
        )
        try:
            wait_for_port(port, server, SERVER_START_TIMEOUT_S)
            mcp_config = workdir / "mcp.json"
            mcp_config.write_text(json.dumps({"mcpServers": {"doom": {"type": "http", "url": f"http://127.0.0.1:{port}/mcp"}}}))
            # The CLI plays from an empty folder of its own, never from the repository.
            play_dir = workdir / "play"
            play_dir.mkdir()
            env = {**os.environ, "CLAUDE_CONFIG_DIR": str(CLAUDE_CONFIG_DIR)}

            usage = CliUsage()
            while not record_out.exists():
                if usage.runs > MAX_RESUMES:
                    raise RuntimeError(f"seed {seed}: still not over after {MAX_RESUMES} resumes")
                resume = usage.session_id if usage.runs else None
                run = subprocess.run(
                    claude_command(claude, harness, mcp_config, spec.tic_limit, resume),
                    cwd=play_dir, env=env, capture_output=True, text=True,
                    stdin=subprocess.DEVNULL,  # otherwise the CLI waits 3 s for piped input
                )
                with transcript.open("a") as f:
                    f.write(run.stdout)
                result = usage.read_stream(run.stdout.splitlines())
                if result is None or result.get("is_error"):
                    message = (result or {}).get("result") or run.stderr[-500:]
                    if "limit" in str(message).lower():
                        raise LimitReached(str(message))
                    if result is None:
                        raise RuntimeError(f"seed {seed}: the CLI gave no result: {message}")
        finally:
            server.terminate()
            server.wait(timeout=30)

        record = json.loads(record_out.read_text())

    return {
        **record,
        "spec": spec.name,
        "tokens": usage.tokens,
        "model": usage.model,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
        },
        "llm_turns": usage.turns,
        "tool_calls": usage.tool_calls,
        "cli_runs": usage.runs,
        "tools_offered": usage.tools_offered,
        "transcript": str(transcript.relative_to(REPO_ROOT)) if transcript.is_relative_to(REPO_ROOT) else str(transcript),
        "video": str(video.relative_to(REPO_ROOT)) if video.is_relative_to(REPO_ROOT) else str(video),
    }


def run_llm_eval(
    harness: Harness,
    spec: EvalSpec = STANDARD_E1M1,
    results_dir: Path = RESULTS_DIR,
    claude: Path = CLAUDE,
    runs_dir: Path = RUNS_DIR,
    video_dir: Path = VIDEO_DIR,
    seeds: list[int] | None = None,
) -> dict | None:
    """Door B's version of eval.run_eval: same files, same rules, same row."""
    contender = contender_name(harness)
    path = attempts_path(results_dir, contender, spec)
    records = read_records(path)
    check_rules(records, spec)
    done = {r["seed"] for r in records}
    for seed in seeds if seeds is not None else spec.seeds:
        if seed in done:
            continue
        record = play_llm_attempt(harness, spec, seed, claude, runs_dir, video_dir)
        append_record(path, record)
        print(f"seed {seed}: progress {record['progress']}, cleared {record['cleared']}, tokens {record['tokens']:,}", flush=True)
    records = read_records(path)
    if {r["seed"] for r in records} >= set(spec.seeds):
        return scoreboard_row(contender, spec, records)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    parser.add_argument("--spec", default=STANDARD_E1M1.name, choices=sorted(SPECS))
    parser.add_argument("--seeds", type=int, nargs="+", help="only these seeds of the spec (default: all)")
    parser.add_argument("--practice", action="store_true", help="store records under runs/, not results/")
    args = parser.parse_args()

    harness, spec = HARNESSES[args.harness], SPECS[args.spec]
    results_dir = RUNS_DIR / "practice" if args.practice else RESULTS_DIR
    try:
        row = run_llm_eval(harness, spec, results_dir, seeds=args.seeds)
    except LimitReached as limit:
        raise SystemExit(f"Stopped by the subscription limit: {limit}\nRun the same command after the reset to continue.")
    if row is None:
        print("Not every seed is done yet; no Scoreboard row written.")
        return
    if not args.practice:
        write_row(row)
        records = read_records(attempts_path(RESULTS_DIR, row["contender"], spec))
        log_to_wandb(row, records, spec)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
