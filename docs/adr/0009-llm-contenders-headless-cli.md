# LLM Contenders play through an isolated headless CLI, and the game outlives it

An LLM Contender needs a decision loop around the model. We could write our own loop against a model API, or use the agent loop the owner already has through the Claude Code CLI subscription (ADR 0003). We use the CLI's loop, run headless (`claude -p`). That makes the Harness three switches: the Doom MCP server's tool flags, the allowed tools, and a manual-level system prompt (`src/doom_player/harness.py`). The CLI is isolated so it can see only Human-equivalent Observations (ADR 0002): no built-in tools (`--tools ""`), only our server (`--strict-mcp-config`), a config directory of its own, and an empty working directory. The launcher (`src/doom_player/llm_eval.py`) starts the game server as a local HTTP service, rather than letting the CLI start it over stdio, so a CLI session that stops early can be resumed into the same game. Settled with the owner on 2026-10-02.

## Considered Options

- **Our own loop against the API**: full control of context and memory, but it bills per token instead of using the subscription, and every Harness feature would be code we maintain.
- **The CLI starting the server over stdio** (as in M1's `.mcp.json`): simpler, but a resume would start a fresh game from spawn.

## Consequences

- The CLI's own behaviour is part of the Harness: its context handling, its retries (19 in the pilot), and a context note the account attaches (the owner's e-mail address). Results name the CLI version along with the model.
- Token counts come from the CLI's stream-json `result` events; with the subscription there is no invoice, so cost is tokens and wall-clock time (ADR 0003).
- Every LLM Attempt is replayable from its transcript (`uv run doom-replay`), which is how M2's results were verified.
