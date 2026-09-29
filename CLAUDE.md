# doom-player

An agent that beats the original Doom (1993), built milestone by milestone, in public. The repository doubles as a portfolio, so every milestone ends in something measurable and showable.

## Before you start any task

1. Read `docs/ROADMAP.md` and find the current milestone (the first one not marked `done`).
2. Read that milestone's brief in `docs/milestones/`. Its completion criteria define done.
3. Use the vocabulary in `CONTEXT.md`. Several everyday words are ambiguous here ("episode", "skill", "run"); the glossary fixes one meaning for each.

## Hard rules

- **WAD files stay out of git.** `doom.wad` is copyrighted. It lives in `wads/`, which is gitignored. Freedoom is the only WAD that may be referenced as downloadable.
- **Contenders see human-equivalent observations only**: screen pixels, HUD values, automap. Privileged information (positions, depth, labels, sectors) is allowed for computing training reward and for debugging visualisations, and each use is declared in the milestone's results. See `docs/adr/0002-human-equivalent-observations.md`.
- **The LLM stays out of the per-tic loop.** It acts as Navigator, through the MCP server, with the game in synchronous mode. See `docs/adr/0001-driver-navigator-hybrid.md`.
- **Every result goes through the eval suite** and lands on the Scoreboard with seeds, observation class, and cost. A number measured any other way is a note, not a result.
- **Inside WSL2, install only user-space packages.** The NVIDIA driver lives on the Windows side; the `cuda`, `cuda-12-x` and `cuda-drivers` apt packages break GPU passthrough.

## Working with the owner

The owner is learning ML through this project and reads every change.

- Finish each milestone with a study guide in `docs/learn/`, following `docs/learn/README.md`.
- Prefer the library a working engineer would use (Stable-Baselines3, Gymnasium, W&B) over a hand-rolled version. Hand-rolled versions belong to the study repo.
- Explain each new term on first use in docs and PR descriptions, analogy first.
- Write everything in English.
- Commit and push only when the owner asks. Publishing to GitHub, W&B, or LinkedIn is the owner's action.

## When a fact is uncertain

Research notes in `docs/research/` mark claims as `(not verified)` or `(inference, untested)`. Test such a claim before building on it, then record the outcome in the milestone brief.

## Recording decisions

- A new or sharpened term: update `CONTEXT.md` in the same change.
- A decision that is hard to reverse, surprising, and the result of a real trade-off: add an ADR in `docs/adr/`, numbered sequentially. One paragraph is enough.
