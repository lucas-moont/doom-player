# Development happens in Ubuntu on WSL2, not native Windows

The owner's machine runs Windows 11. ViZDoom's maintainers call the Windows build less tested and recommend WSL or Docker for long experiments, and Sample Factory (the fastest ViZDoom trainer, wanted at M10) has no Windows support. Switching operating systems mid-project would cost more than starting on Linux, so all development happens in an Ubuntu distro on WSL2, with `uv` managing Python. Docker is deferred: a Dockerfile is added later for reproducibility, because Docker plus GPU passthrough plus game rendering is too many failure points for a first setup.

## Consequences

- ViZDoom documents WSL as "seems to be working fine, it is not officially supported". M0 verifies it on this machine.
- The repository lives on the Windows filesystem today. If file I/O under `/mnt/c` proves slow during training, move the working copy into the Linux filesystem.

## Update 2026-10-01: measured in M0, move before training

M0 measured the cost of `/mnt/c`. The same smoke test ran in 45.8 s from `/mnt/c` and in 1.3 s from a clone in the Linux filesystem, about 35 times slower. The first `uv sync` (torch and its NVIDIA libraries, written into `.venv/` on `/mnt/c`) took 7 min 52 s. For M0 and M1 this is tolerable. A Training Run reads and writes checkpoints, logs and videos for hours, so the working copy moves into the Linux filesystem before the first Training Run, in M3 at the latest.

The move, when it happens:

1. Inside Ubuntu: `git clone https://github.com/lucas-moont/doom-player.git ~/doom-player`
2. Copy the WAD: `mkdir -p ~/doom-player/wads && cp "/mnt/c/Users/thewo/Documents/Doom Player/doom/DOOM.WAD" ~/doom-player/wads/doom.wad`
3. `cd ~/doom-player && uv sync && uv run doom-check`. `UV_LINK_MODE=copy` is no longer needed.
4. Edit from Windows through `\\wsl$\Ubuntu-24.04\home\lucas\doom-player`, or with VS Code's WSL extension.
5. Push any unpushed work from the old copy first, then delete it so there is one working copy.

The workspace `CLAUDE.md` assumes both repositories sit side by side in `Documents\Doom Player`. Update it in the same change.

## Update 2026-10-03: moved

Steps 1 to 4 were done at the start of M3. In `~/doom-player` the full test suite (30 tests) passes in 25 s, and the MCP server answers `initialize` 1.5 s after launch, so `.mcp.json` no longer needs the separate server venv from M1; it now runs `cd ~/doom-player && uv run doom-mcp`. W&B credentials live in `~/.netrc` and carried over unchanged. Step 5, deleting the old copy, is the owner's.

The same day the whole workspace followed: `~/doom-workspace/` now holds the workspace `CLAUDE.md`, `doom/` (the owner's game files), this repository as `doom-player/` and the study repository as `doom-player-study/`, mirroring the old Windows folder. The owner keeps running Claude Code on Windows and opens the folders through `\\wsl$`. After moving a working copy, delete `.venv/` and run `uv sync`: the environment stores absolute paths.
