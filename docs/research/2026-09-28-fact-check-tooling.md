# Fact check: tooling and environment (2026-09-28)

> Verified against primary sources. ViZDoom claims were checked in the source code at tag 1.3.1 (commit f771231).
> Items marked **(not verified)** or **(inference, untested)** must be tested before being relied on.

## 1. Weights & Biases free plan

- A "Free" plan exists at $0/mo, "Designed for personal development of AI applications and models". Includes experiment tracking, registry, Reports and Tables. Community support only. Source: https://wandb.ai/site/pricing/
- Storage: 5 GB/mo (artifacts + run data, averaged over the last 30 days). Weave ingestion: 1 GB/mo. Same source.
- Free plan has both public and private projects. No team-based access controls. Same source.
- Reports have a Share button with "Copy report link". Source: https://docs.wandb.ai/models/reports/collaborate-on-reports
- **(not verified)** Limits on tracked hours and number of projects on the Free plan. Do not claim "unlimited".
- **(not verified)** That a public share link works on the Free plan without login. Safe path: make the project public, which the Free plan supports.

## 2. ViZDoom and the shareware `doom1.wad`

- The docs require a purchased original WAD (Steam/GOG), named `doom.wad` / `doom2.wad` in lowercase, placed in the package directory (`vizdoom.install_path`). Shareware is not mentioned. Source: https://vizdoom.farama.org/environments/original_doom_levels/
- `VizdoomDoom{MAP}-S{1..5}-v0` environments use `doom.cfg`, whose first line is `doom_game_path = doom.wad`. Sources: https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/doom.cfg and `gymnasium_wrapper/__init__.py` (lines 262-278)
- IWAD lookup is by exact base name, in `./` and then the package directory. A file named `doom1.wad` will not match. Source: `src/lib/ViZDoomController.cpp` (lines 1267-1298)
- The engine identifies the shareware by content, not name: entry "DOOM Shareware", `MustContain = "E1M1"`. Source: `src/vizdoom/wadsrc/static/iwadinfo.txt` (lines 296-305)
- **(inference, untested)** Renaming `doom1.wad` to `doom.wad` should make E1M1 to E1M9 load; E2/E3/E4 do not exist in that file and would fail.
- Guaranteed fallback: `VizdoomFreedoom1{MAP}-S{X}-v0` with `freedoom1.wad`, which ships inside the package. Same map names, but Freedoom maps, not id Software's.

## 3. Episode timeout and termination on original maps

- `episode_timeout = 126000` tics (60 minutes, per the file's own comment). `episode_start_time = 1`. `map_exit_reward = 1`. Resolution 320x240, 19 buttons. Source: `scenarios/doom.cfg`
- The episode ends at the map exit. There is **no progression to the next map**. Sources: `src/vizdoom/src/viz_game.cpp:388`, `src/lib/ViZDoomController.cpp:603`, `src/lib/ViZDoomGame.cpp:215-216`
- Player death also ends the episode; the exit reward is only added if the player did not die.
- In the Gymnasium wrapper: `terminated = game.is_episode_finished()`, `truncated = game.is_episode_timeout_reached()` (`treat_episode_timeout_as_truncation=True` by default). No `max_episode_steps` is registered. Source: `gymnasium_wrapper/base_gymnasium_env.py` (lines 70, 236-237)
- Consequence: a custom "campaign" wrapper is needed to chain maps and carry health, ammo and weapons across them.

## 4. NVIDIA GPU inside WSL2

- Install only the Windows NVIDIA driver (R495 or newer). "Do not install any Linux display driver in WSL." Source: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- Do not install the `cuda`, `cuda-12-x` or `cuda-drivers` meta-packages inside WSL. If the toolkit is needed, use the "WSL-Ubuntu" installer. Same source.
- Limitations: no Unified Memory, limited pinned memory.
- Requirements: Windows 11 or Windows 10 21H2+, WSL kernel 5.10.43.3 or newer (`wsl cat /proc/version`), glibc distro such as Ubuntu. Source: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl
- **(not verified)** That the PyTorch pip wheel removes the need for a separate CUDA toolkit install.

## 5. ViZDoom without a display

- "ViZDoom with window disabled can be used on Linux systems without X Server." Source: https://vizdoom.farama.org/api/python/doom_game/
- The Gymnasium wrapper already calls `set_window_visible(False)` by default.
- On Linux, `pip install vizdoom` uses a prebuilt wheel. Only system dependency mentioned: `apt install libopenal1`, for the audio buffer. `doom.cfg` sets `audio_buffer_enabled = true`, so `libopenal1` is probably required on original maps **(inference)**. Source: https://vizdoom.farama.org/introduction/python_quickstart/
- The FAQ says WSL "seems to be working fine, it is not officially supported". Source: https://vizdoom.farama.org/faq/

## Open items to test in milestone M0

1. Does the RTX 4050 show up in `nvidia-smi` inside Ubuntu on WSL2?
2. Does a renamed shareware WAD load E1M1 in ViZDoom?
3. Is `libopenal1` required for the original-map environments?
4. Does a public W&B project render for a logged-out visitor?
