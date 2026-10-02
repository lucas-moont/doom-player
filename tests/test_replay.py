"""Reading `act` calls back from a CLI transcript."""

import json

from doom_player.replay import acts_from_transcript


def test_only_accepted_act_calls_are_replayed_in_order(tmp_path):
    def use(id_, name, args):
        return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": id_, "name": name, "input": args}]}}

    def result(id_, error=False):
        return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id_, "is_error": error}]}}

    events = [
        use("a", "mcp__doom__look", {}), result("a"),
        use("b", "mcp__doom__act", {"buttons": ["MOVE_FORWARD"], "tics": 20}), result("b"),
        use("c", "mcp__doom__act", {"buttons": ["JUMP"]}), result("c", error=True),  # rejected: game did not move
        use("d", "mcp__doom__act", {"buttons": []}), result("d"),  # tics left to the default
    ]
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(map(json.dumps, events)) + "\nnot json\n")
    assert acts_from_transcript(transcript) == [(["MOVE_FORWARD"], 20), ([], 8)]
