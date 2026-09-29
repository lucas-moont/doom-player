# Development happens in Ubuntu on WSL2, not native Windows

The owner's machine runs Windows 11. ViZDoom's maintainers call the Windows build less tested and recommend WSL or Docker for long experiments, and Sample Factory (the fastest ViZDoom trainer, wanted at M9) has no Windows support. Switching operating systems mid-project would cost more than starting on Linux, so all development happens in an Ubuntu distro on WSL2, with `uv` managing Python. Docker is deferred: a Dockerfile is added later for reproducibility, because Docker plus GPU passthrough plus game rendering is too many failure points for a first setup.

## Consequences

- ViZDoom documents WSL as "seems to be working fine, it is not officially supported". M0 verifies it on this machine.
- The repository lives on the Windows filesystem today. If file I/O under `/mnt/c` proves slow during training, move the working copy into the Linux filesystem.
