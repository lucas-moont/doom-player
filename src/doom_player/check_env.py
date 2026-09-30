"""Report whether this machine is ready to run Doom Attempts.

Run with `uv run doom-check`. Prints the GPU state, the ViZDoom version,
and whether the purchased WAD is in place.
"""

import hashlib
from pathlib import Path

import torch
import vizdoom

from doom_player.paths import WAD_PATH

# The Ultimate Doom, verified 2026-09-28 (see docs/milestones/M0-workshop.md).
EXPECTED_WAD_MD5 = "c4fe9fd920207691a9f493668e0a2083"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    print(f"torch          {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available {cuda}")
    if cuda:
        print(f"GPU            {torch.cuda.get_device_name(0)}")

    print(f"vizdoom        {vizdoom.__version__}")

    if not WAD_PATH.exists():
        print(f"WAD            missing: copy your doom.wad to {WAD_PATH}")
        return
    md5 = md5_of(WAD_PATH)
    verdict = "matches The Ultimate Doom" if md5 == EXPECTED_WAD_MD5 else "UNKNOWN version"
    print(f"WAD            {WAD_PATH} (MD5 {md5}, {verdict})")


if __name__ == "__main__":
    main()
