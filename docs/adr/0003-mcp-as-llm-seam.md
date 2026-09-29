# The game is exposed to LLMs through an MCP server, driven by subscription CLIs

The owner runs LLMs through the Claude Code CLI subscription, may add Codex, and wants to switch providers without rewriting the project. Instead of writing one adapter per provider, the game is exposed once as an MCP server and any MCP-capable client connects to it. Provider choice becomes configuration.

## Consequences

- LLM usage is bounded by subscription limits, not by a money budget. When a limit is hit, evaluation pauses until it resets; the Eval Suite must be resumable.
- Cost on the Scoreboard is reported as tokens and wall-clock time, since there is no per-call invoice.
- `gunnargrosch/doom-mcp` is prior art for tool design (structured state plus a thumbnail). It is a reference, not a dependency: it wraps doomgeneric, and this project needs ViZDoom.
