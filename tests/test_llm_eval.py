"""The LLM launcher, driven by a fake CLI: resume, skip, tokens, isolation."""

import stat
import sys
from pathlib import Path

import pytest

from doom_player.eval import EvalSpec, attempts_path, read_records
from doom_player.harness import HARNESSES
from doom_player.llm_eval import CliUsage, run_llm_eval
from doom_player.paths import WAD_PATH

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")

SHORT = EvalSpec(name="test-llm", map="E1M1", difficulty=3, seeds=(0, 1), tic_limit=210)


@pytest.fixture
def fake_claude(tmp_path):
    # A shell wrapper: the venv's path has spaces, which a #! line cannot hold.
    script = tmp_path / "claude"
    fake = Path(__file__).parent / "fake_claude.py"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_launcher_resumes_records_tokens_and_skips_done_seeds(tmp_path, fake_claude):
    dirs = {"results_dir": tmp_path / "results", "runs_dir": tmp_path / "runs", "video_dir": tmp_path / "videos"}
    row = run_llm_eval(HARNESSES["h0"], SHORT, claude=fake_claude, **dirs)

    records = read_records(attempts_path(dirs["results_dir"], "opus-5.5-h0", SHORT))
    assert [r["seed"] for r in records] == [0, 1]
    for r in records:
        assert r["spec"] == "test-llm" and r["truncated"]
        assert r["cli_runs"] == 2  # stopped early once, then resumed into the same game
        assert r["tokens"] == 2 * (100 + 10 + 1000)
        assert r["model"] == "claude-opus-5-5"
        assert r["tools_offered"] == ["mcp__doom__look", "mcp__doom__act"]  # H0: no automap
        assert (tmp_path / r["video"]).exists() or Path(r["video"]).exists()
    assert row["contender"] == "opus-5.5-h0" and row["tokens_per_attempt"] == 2220

    fake_claude.write_text("#!/bin/sh\nexit 3\n")  # any call now would fail
    assert run_llm_eval(HARNESSES["h0"], SHORT, claude=fake_claude, **dirs)["seeds"] == [0, 1]


def test_usage_sums_over_resumed_runs():
    usage = CliUsage()
    result = '{"type": "result", "is_error": false, "session_id": "s", "num_turns": 2, "usage": {"input_tokens": 5, "output_tokens": 1}}'
    usage.read_stream(['{"type": "system", "subtype": "init", "session_id": "s", "model": "m", "tools": []}', result])
    usage.read_stream([result, "not json"])
    assert (usage.tokens, usage.turns, usage.runs, usage.model) == (12, 4, 2, "m")
