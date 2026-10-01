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
