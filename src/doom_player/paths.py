"""Where the project keeps files that live outside git."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The purchased WAD. Copyrighted, so `wads/` is gitignored.
WAD_PATH = REPO_ROOT / "wads" / "doom.wad"
