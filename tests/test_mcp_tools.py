"""Every MCP tool returns Human-equivalent Observations only (ADR 0002), tool by tool."""

import asyncio
import base64
import io
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image

from doom_player.mcp_server import Game, build_server
from doom_player.paths import WAD_PATH
from doom_player.session import HUD_VARIABLES, AttemptSession

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")

ALLOWED_STATUS_KEYS = {"hud", "events", "tics_left", "attempt_over"}
# Words that would mean Privileged Information slipped out.
FORBIDDEN = ("position", "depth", "label", "sector", "progress", "distance", "coordinate", "seed")


@pytest.fixture
def server(tmp_path):
    session = AttemptSession("mcp-test", tic_limit=70)
    game = Game(session, tmp_path / "records.jsonl", spec=None)
    yield build_server(game), game, tmp_path
    if not game.saved:
        session.close()


def call(server, name, **arguments):
    return asyncio.run(server.call_tool(name, arguments)).content


def check_only_human_equivalent(content):
    images, texts = [], []
    for block in content:
        if block.type == "image":
            assert block.mime_type == "image/png"
            image = Image.open(io.BytesIO(base64.b64decode(block.data)))
            assert image.size == (320, 240)
            images.append(image)
        else:
            assert block.type == "text"
            texts.append(block.text)
    for text in texts:
        assert not any(word in text.lower() for word in FORBIDDEN), text
        if text.startswith("{"):
            status = json.loads(text)
            assert set(status) <= ALLOWED_STATUS_KEYS
            assert set(status["hud"]) == set(HUD_VARIABLES)
            assert all(isinstance(v, int) for v in status["hud"].values())
            assert all(isinstance(e, str) for e in status["events"])
    return images, texts


def test_tool_list_is_look_act_automap(server):
    mcp, _, _ = server
    tools = asyncio.run(mcp.list_tools())
    assert sorted(t.name for t in tools) == ["act", "automap", "look"]
    for tool in tools:
        assert not any(word in (tool.description or "").lower() for word in FORBIDDEN)


def test_look_returns_screen_and_hud(server):
    mcp, _, _ = server
    images, texts = check_only_human_equivalent(call(mcp, "look"))
    assert len(images) == 1 and len(texts) == 1
    assert json.loads(texts[0])["tics_left"] == 70


def test_automap_returns_an_image_only(server):
    mcp, _, _ = server
    images, texts = check_only_human_equivalent(call(mcp, "automap"))
    assert len(images) == 1 and texts == []


def test_act_moves_time_forward_and_returns_screen_and_hud(server):
    mcp, _, _ = server
    images, texts = check_only_human_equivalent(call(mcp, "act", buttons=["MOVE_FORWARD"], tics=20))
    assert len(images) == 1
    assert json.loads(texts[0])["tics_left"] == 50


def test_act_rejects_unknown_buttons_and_long_actions(server):
    mcp, _, _ = server
    # The reason must reach the model, not a generic "Error executing tool".
    for args, reason in (({"buttons": ["NOCLIP"]}, "unknown buttons"), ({"buttons": ["ATTACK"], "tics": 36}, "between 1 and 35")):
        with pytest.raises(ToolError, match=reason):
            asyncio.run(mcp.call_tool("act", args))


def test_end_of_attempt_is_reported_and_recorded(server):
    mcp, game, tmp_path = server
    call(mcp, "act", buttons=[], tics=35)
    _, texts = check_only_human_equivalent(call(mcp, "act", buttons=[], tics=35))
    assert json.loads(texts[0])["attempt_over"] == "time up"
    record = json.loads((tmp_path / "records.jsonl").read_text())
    assert record["truncated"] and record["progress"] is not None  # measured, never returned
    _, texts = check_only_human_equivalent(call(mcp, "act", buttons=["ATTACK"], tics=1))
    assert "already over" in texts[0]


@pytest.mark.parametrize(
    ("automap_tool", "notes_tools", "expected"),
    [
        (False, False, ["act", "look"]),  # H0
        (True, False, ["act", "automap", "look"]),  # H1
        (True, True, ["act", "automap", "look", "read_notes", "write_note"]),  # H2
    ],
)
def test_harness_rungs_expose_their_tools(tmp_path, automap_tool, notes_tools, expected):
    session = AttemptSession("mcp-test", tic_limit=35)
    try:
        mcp = build_server(Game(session, tmp_path / "r.jsonl", None), automap_tool, notes_tools)
        assert sorted(t.name for t in asyncio.run(mcp.list_tools())) == expected
    finally:
        session.close()


def test_notes_return_exactly_what_was_written(tmp_path):
    session = AttemptSession("mcp-test", tic_limit=35)
    try:
        mcp = build_server(Game(session, tmp_path / "r.jsonl", None), notes_tools=True)
        assert call(mcp, "read_notes")[0].text == "No notes yet."
        call(mcp, "write_note", text="Door east of the start room opens with USE.")
        call(mcp, "write_note", text="Zombie behind the pillar.")
        assert call(mcp, "read_notes")[0].text == (
            "1. Door east of the start room opens with USE.\n2. Zombie behind the pillar."
        )
    finally:
        session.close()


def test_record_out_and_video(tmp_path):
    session = AttemptSession("mcp-test", tic_limit=70)
    video, record_out = tmp_path / "attempt.mp4", tmp_path / "record.json"
    game = Game(session, tmp_path / "unused.jsonl", "e1m1-v1", record_out, video)
    mcp = build_server(game)
    call(mcp, "act", buttons=["MOVE_FORWARD"], tics=35)
    call(mcp, "act", buttons=["TURN_LEFT"], tics=35)
    record = json.loads(record_out.read_text())
    assert record["spec"] == "e1m1-v1" and record["truncated"]
    assert not (tmp_path / "unused.jsonl").exists()

    import imageio_ffmpeg

    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert frames >= 68  # one per tic, the final tic ends the Attempt
