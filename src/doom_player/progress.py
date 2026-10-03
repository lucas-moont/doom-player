"""Progress: how much of the walk to the exit an Attempt covered.

Clear Rate is 0 for every weak Contender, which hides the differences between
them. Progress tells them apart. The Map's walls are read from the WAD and laid
on a grid of small cells. A Dijkstra pass from the exit gives every free cell
its walking distance to the exit, going around walls (a distance field). For
one Attempt:

    Progress = 1 - (closest distance reached) / (distance at spawn)

clipped to 0..1, and 1 for a Clear. It reads the player's position, which is
Privileged Information, so it measures and never feeds a Contender (ADR 0002).
Simplifications are listed in ADR 0008.
"""

import heapq
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import omg
import omg.mapedit

from doom_player.paths import WAD_PATH

CELL = 8  # map units per grid cell; a player is 32 units wide
MAX_STEP = 24  # the highest ledge a player can walk up
EXIT_ACTIONS = {11, 52}  # exit switch, walk-over exit (51 and 124 are secret exits)
EXIT_REACH = 32  # cells this close to the exit line count as "at the exit"
NO_SIDE = 0xFFFF
PLAYER_HEIGHT = 56  # an opening lower than this is closed
# Doors a player opens by pressing USE on them (DR/D1, normal, locked and fast).
# A closed sector without one of these opens only from elsewhere, if at all.
MANUAL_DOOR_ACTIONS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}


@dataclass
class DistanceField:
    map: str
    origin: tuple[int, int]  # map coordinates of cell (0, 0)
    blocked: np.ndarray  # bool, [row, col], row grows with y
    distance: np.ndarray  # float, map units to the exit; inf where unreachable
    start: tuple[float, float]  # player 1 start

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.origin[1]) // CELL), int((x - self.origin[0]) // CELL)

    def distance_at(self, x: float, y: float) -> float:
        """Walking distance to the exit; the nearest reachable cell if needed."""
        row, col = self.cell_of(x, y)
        for radius in range(4):
            window = self.distance[
                max(0, row - radius) : row + radius + 1,
                max(0, col - radius) : col + radius + 1,
            ]
            if window.size and np.isfinite(window).any():
                return float(window[np.isfinite(window)].min())
        return math.inf

    @property
    def start_distance(self) -> float:
        return self.distance_at(*self.start)


@dataclass
class ProgressMeter:
    """Follows one Attempt and keeps its closest approach to the exit."""

    field: DistanceField
    closest: float = math.inf
    path: list[tuple[float, float]] = field(default_factory=list)

    def visit(self, x: float, y: float) -> None:
        self.path.append((x, y))
        self.closest = min(self.closest, self.field.distance_at(x, y))

    @property
    def progress(self) -> float:
        start = self.field.start_distance
        return min(1.0, max(0.0, 1.0 - self.closest / start))


def _walls(editor: omg.mapedit.MapEditor) -> list[tuple[float, float, float, float]]:
    """Lines a player cannot cross.

    One-sided and impassable lines; ledges higher than a step; and closed
    openings, unless the closed sector is a door the player can open with USE.
    """
    sector_of = lambda side: editor.sidedefs[side].sector  # noqa: E731
    openable = {sector_of(line.back) for line in editor.linedefs if line.action in MANUAL_DOOR_ACTIONS and line.back != NO_SIDE}
    walls = []
    for line in editor.linedefs:
        a, b = editor.vertexes[line.vx_a], editor.vertexes[line.vx_b]
        blocking = line.back == NO_SIDE or line.impassable
        if not blocking:
            sides = (sector_of(line.front), sector_of(line.back))
            front, back = (editor.sectors[i] for i in sides)
            closed = [i for i in sides if editor.sectors[i].z_ceil - editor.sectors[i].z_floor < PLAYER_HEIGHT]
            blocking = abs(front.z_floor - back.z_floor) > MAX_STEP or any(i not in openable for i in closed)
        if blocking:
            walls.append((a.x, a.y, b.x, b.y))
    return walls


def _rasterize(blocked: np.ndarray, origin: tuple[int, int], segment) -> None:
    x0, y0, x1, y1 = segment
    samples = max(1, int(math.hypot(x1 - x0, y1 - y0) / 2))
    for t in np.linspace(0.0, 1.0, samples + 1):
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        blocked[int((y - origin[1]) // CELL), int((x - origin[0]) // CELL)] = True


@cache
def distance_field(map_name: str = "E1M1", wad_path: Path = WAD_PATH) -> DistanceField:
    editor = omg.mapedit.MapEditor(omg.WAD(str(wad_path)).maps[map_name])
    xs = [v.x for v in editor.vertexes]
    ys = [v.y for v in editor.vertexes]
    origin = (min(xs) - CELL, min(ys) - CELL)
    shape = ((max(ys) - origin[1]) // CELL + 2, (max(xs) - origin[0]) // CELL + 2)

    blocked = np.zeros(shape, dtype=bool)
    for wall in _walls(editor):
        _rasterize(blocked, origin, wall)

    # Sources: free cells within reach of an exit line's midpoint, on its front
    # side (the right of a walk from vertex A to B), where the player stands.
    # The back of a one-sided exit switch is outside the Map.
    distance = np.full(shape, math.inf)
    heap = []
    for line in editor.linedefs:
        if line.action not in EXIT_ACTIONS:
            continue
        a, b = editor.vertexes[line.vx_a], editor.vertexes[line.vx_b]
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        r = EXIT_REACH // CELL + 1
        row0, col0 = int((my - origin[1]) // CELL), int((mx - origin[0]) // CELL)
        for row in range(row0 - r, row0 + r + 1):
            for col in range(col0 - r, col0 + r + 1):
                cx = origin[0] + (col + 0.5) * CELL
                cy = origin[1] + (row + 0.5) * CELL
                in_front = (b.x - a.x) * (cy - a.y) - (b.y - a.y) * (cx - a.x) < 0
                if in_front and not blocked[row, col] and math.hypot(cx - mx, cy - my) <= EXIT_REACH:
                    distance[row, col] = 0.0
                    heap.append((0.0, row, col))
    if not heap:
        raise ValueError(f"{map_name} has no exit line")

    # Dijkstra over 8 neighbours; a diagonal step may not cut a wall's corner.
    heapq.heapify(heap)
    steps = [(dr, dc, CELL * math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    rows, cols = shape
    while heap:
        d, row, col = heapq.heappop(heap)
        if d > distance[row, col]:
            continue
        for dr, dc, cost in steps:
            r2, c2 = row + dr, col + dc
            if not (0 <= r2 < rows and 0 <= c2 < cols) or blocked[r2, c2]:
                continue
            if dr and dc and (blocked[row + dr, col] or blocked[row, col + dc]):
                continue
            if d + cost < distance[r2, c2]:
                distance[r2, c2] = d + cost
                heapq.heappush(heap, (d + cost, r2, c2))

    start = next((t.x, t.y) for t in editor.things if t.type == 1)
    return DistanceField(map_name, origin, blocked, distance, start)


def render(field: DistanceField, path: list[tuple[float, float]] = (), out: Path | None = None):
    """Debug picture: walls black, distance as colour (near exit = bright), path white."""
    from PIL import Image

    finite = np.isfinite(field.distance)
    shade = np.zeros(field.distance.shape)
    shade[finite] = 1.0 - field.distance[finite] / field.distance[finite].max()
    rgb = np.zeros((*shade.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * shade).astype(np.uint8)
    rgb[..., 1] = (180 * shade**2).astype(np.uint8)
    rgb[..., 2] = (90 * (1 - shade) * finite).astype(np.uint8)
    rgb[field.blocked] = (230, 230, 230)
    # Join consecutive positions, so a path sampled once per action reads as a line.
    for (x0, y0), (x1, y1) in zip(path, path[1:] or path):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (CELL / 2)))
        for t in np.linspace(0.0, 1.0, steps + 1):
            row, col = field.cell_of(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            rgb[row, col] = (0, 255, 255)
    image = Image.fromarray(rgb[::-1]).resize((shade.shape[1] * 2, shade.shape[0] * 2), Image.NEAREST)
    if out is not None:
        image.save(out)
    return image
