"""A stand-in for `claude -p`, so the launcher is tested without a subscription.

It refuses to run unless the isolation flags are present, plays through the
Doom MCP server over HTTP like the real CLI, stops early on its first run (to
force a resume), and prints stream-json events shaped like the real ones.
"""

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def emit(event):
    print(json.dumps(event), flush=True)


async def play(url, max_calls):
    async with streamable_http_client(url) as (read, write, *_):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [f"mcp__doom__{t.name}" for t in (await session.list_tools()).tools]
            emit({"type": "system", "subtype": "init", "session_id": "fake-session", "model": "claude-opus-5-5", "tools": tools})
            for _ in range(max_calls):
                emit({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__doom__act"}]}})
                result = await session.call_tool("act", {"buttons": ["MOVE_FORWARD"], "tics": 35})
                text = next((b.text for b in result.content if b.type == "text"), "")
                if "attempt_over" in text or "already over" in text:
                    break


def main():
    assert arg("--tools") == "", "built-in tools must be off"
    assert "--strict-mcp-config" in sys.argv and arg("--model") == "claude-opus-5-5"
    url = json.load(open(arg("--mcp-config")))["mcpServers"]["doom"]["url"]
    resumed = arg("--resume") is not None
    asyncio.run(play(url, max_calls=1000 if resumed else 2))
    emit({
        "type": "result", "subtype": "success", "is_error": False, "session_id": "fake-session", "num_turns": 3,
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000},
    })


if __name__ == "__main__":
    main()
