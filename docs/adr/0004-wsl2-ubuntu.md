# Development happens in Ubuntu on WSL2, not native Windows

The owner's machine runs Windows 11. ViZDoom's maintainers call the Windows build less tested and recommend WSL or Docker for long experiments, and Sample Factory (the fastest ViZDoom trainer, wanted at M9) has no Windows support. Switching operating systems mid-project would cost more than starting on Linux, so all development happens in an Ubuntu distro on WSL2, with `uv` managing Python. Docker is deferred: a Dockerfile is added later for reproducibility, because Docker plus GPU passthrough plus game rendering is too many failure points for a first setup.

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
