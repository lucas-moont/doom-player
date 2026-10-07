"""Pictures of E1M2's route for the M5 brief, from the WAD alone.

Run with `uv run python docs/milestones/make_figures_m5.py`. Writes to
`docs/milestones/img/`:

- `m5-e1m2-route.png`: walking distance to the exit with every door open
  (bright = near the exit), and the keyed route traced over it: spawn to the
  red key with the red door shut, then on to the exit.
Rings: spawn green, red key red, red door yellow.

- `m5-e1m2-misleads.png`: in magenta, where the doors-open rule pays Progress
  to a player who cannot have the red key yet: everything it pays for before
  the detour is ground toward a door that will not open.
"""

from pathlib import Path

import numpy as np
from PIL import ImageDraw

import omg
import omg.mapedit

from doom_player.paths import WAD_PATH
from doom_player.progress import LOCKS, KeyedRoute, distance_field, render

OUT = Path(__file__).parent / "img"
RED = frozenset({"red"})
STEPS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def descend(field, x: float, y: float) -> list[tuple[float, float]]:
    """The walk from (x, y) down the distance field to its goal, one cell at a time."""
    row, col = field.cell_of(x, y)
    path = []
    while field.distance[row, col] > 0:
        path.append((field.origin[0] + (col + 0.5) * 8, field.origin[1] + (row + 0.5) * 8))
        row, col = min(((row + dr, col + dc) for dr, dc in STEPS), key=lambda rc: field.distance[rc])
    return path


def mark(image, field, spots: dict[tuple[float, float], tuple[int, int, int]]) -> None:
    """Draw a ring around each spot, in the image's flipped, doubled coordinates."""
    draw = ImageDraw.Draw(image)
    height = field.distance.shape[0]
    for (x, y), colour in spots.items():
        row, col = field.cell_of(x, y)
        cx, cy = 2 * col, 2 * (height - 1 - row)
        draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), outline=colour, width=3)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    route = KeyedRoute("E1M2")
    field = route.field
    key = route.keys["red"][0]
    to_key = distance_field("E1M2", locked=RED, goal=key)
    path = descend(to_key, *field.start) + descend(field, *key)
    editor = omg.mapedit.MapEditor(omg.WAD(str(WAD_PATH)).maps["E1M2"])
    door = [line for line in editor.linedefs if line.action in LOCKS["red"]]
    ends = [editor.vertexes[v] for line in door for v in (line.vx_a, line.vx_b)]
    door_middle = (sum(v.x for v in ends) / len(ends), sum(v.y for v in ends) / len(ends))
    spots = {field.start: (0, 255, 0), key: (255, 0, 0), door_middle: (255, 255, 0)}

    image = render(field, path)
    mark(image, field, spots)
    image.save(OUT / "m5-e1m2-route.png")

    # Cells a player reaches from the spawn without the red key, where the
    # doors-open rule already pays: closer to the exit than the spawn is.
    reach = np.isfinite(distance_field("E1M2", locked=RED, goal=field.start).distance)
    misleads = reach & (field.distance < field.start_distance)
    image = render(field)
    pixels = image.load()
    height = field.distance.shape[0]
    for row, col in zip(*np.nonzero(misleads)):
        for dy in (0, 1):
            for dx in (0, 1):
                pixels[2 * col + dx, 2 * (height - 1 - row) + dy] = (255, 0, 255)
    mark(image, field, spots)
    image.save(OUT / "m5-e1m2-misleads.png")
    best = field.distance[reach].min()
    print(f"doors-open pays up to {1 - best / field.start_distance:.1%} without the red key")


if __name__ == "__main__":
    main()
