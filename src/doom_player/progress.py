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
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import omg
import omg.mapedit

from doom_player.paths import WAD_PATH

CELL = 8  # map units per grid cell; a player is 32 units wide
MAX_STEP = 24  # the highest ledge a player can walk up
EXIT_ACTIONS = {11, 52}  # exit switch, walk-over exit (51 and 124 are secret exits)
EXIT_REACH = 32  # cells this close to the exit line count as "at the exit"
PICKUP_REACH = 36  # a player (radius 16) touches an item (radius 20) this close, per axis
NO_SIDE = 0xFFFF
PLAYER_HEIGHT = 56  # an opening lower than this is closed
# Doors a player opens by pressing USE on them (DR/D1, normal, locked and fast).
# A closed sector without one of these opens only from elsewhere, if at all.
MANUAL_DOOR_ACTIONS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
# Doors that open only for a player holding the key of their colour (DR and D1).
LOCKS = {"blue": {26, 32}, "yellow": {27, 34}, "red": {28, 33}}
# Thing types of the keys of each colour: keycard and skull key.
KEY_THINGS = {"blue": {5, 40}, "yellow": {6, 39}, "red": {13, 38}}
# Thing flags: which Difficulties a thing appears at, and multiplayer only.
SKILL_FLAGS = {1: 0x1, 2: 0x1, 3: 0x2, 4: 0x4, 5: 0x4}
MULTIPLAYER_ONLY = 0x10
# Doors opened from elsewhere, by a switch (103) or a shot (46), found by sector tag.
# A whitelist: other remote openers (63 on E1M1, for one) would change E1M1's field.
TAGGED_DOOR_ACTIONS = {46, 103}
# Lifts a player rides by walking on (88) or pressing a switch (62). A lift's
# ledge is not a wall: the floor comes down to meet the player.
LIFT_ACTIONS = {62, 88}


@dataclass
class DistanceField:
    map: str
    origin: tuple[int, int]  # map coordinates of cell (0, 0)
    blocked: np.ndarray  # bool, [row, col], row grows with y
    distance: np.ndarray  # float, map units to the exit; inf where unreachable
    start: tuple[float, float]  # player 1 start

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.origin[1]) // CELL), int((x - self.origin[0]) // CELL)

    def distance_at(self, x: float, y: float, strict: bool = False) -> float:
        """Walking distance to the exit; the nearest reachable cell if needed.

        A position read off the game can land in a cell a wall was drawn
        through, so nearby cells stand in for it. Without `strict` they also
        stand in for a free cell the exit cannot reach, which on a field with
        a locked door would borrow the distance from the door's other side.
        """
        row, col = self.cell_of(x, y)
        if strict and not self.blocked[row, col]:
            return float(self.distance[row, col])
        for radius in range(4):
            window = self.distance[
                max(0, row - radius) : row + radius + 1,
                max(0, col - radius) : col + radius + 1,
            ]
            if window.size and np.isfinite(window).any():
                return float(window[np.isfinite(window)].min())
        return math.inf

    @cached_property  # read on every step while training; it never changes
    def start_distance(self) -> float:
        return self.distance_at(*self.start)

    # The ruler interface ProgressMeter reads; this rule ignores keys.
    def remaining(self, x: float, y: float, keys: frozenset[str] = frozenset()) -> float:
        return self.distance_at(x, y)

    @property
    def start_remaining(self) -> float:
        return self.start_distance


@dataclass(frozen=True)
class KeyedRoute:
    """Walking distance still to cover, through the keys the Map needs.

    With keys K held, the rest of the route is the shorter of: straight to the
    exit through the doors K opens; or to a key k not yet held, then the rest
    of the route from there with K + k. A key that does not shorten the route
    never wins, so no list of required keys is written by hand, and on a Map
    without keys this is the distance field itself. Each leg weighs its own
    walking length, so Progress stays "the share of the route covered".
    Positions are looked up strictly, so a cell behind a locked door does not
    borrow the distance from the door's open side.
    """

    map: str
    difficulty: int = 3
    wad_path: Path = WAD_PATH

    @cached_property
    def field(self) -> DistanceField:
        """Every door open: the M4 rule, and the picture's background."""
        return distance_field(self.map, self.wad_path)

    @property
    def keys(self) -> dict[str, list[tuple[float, float]]]:
        """Where each colour's keys lie at this Difficulty, from the WAD."""
        return _key_spots(self.map, self.difficulty, self.wad_path)

    def _field(self, held: frozenset[str], goal=None) -> DistanceField:
        return distance_field(self.map, self.wad_path, locked=frozenset(LOCKS) - held, goal=goal)

    def remaining(self, x: float, y: float, keys: frozenset[str] = frozenset()) -> float:
        if not self.keys:
            return self.field.distance_at(x, y)
        best = self._field(keys).distance_at(x, y, strict=True)
        for colour, spots in self.keys.items():
            if colour in keys:
                continue
            for spot in spots:
                leg = self._field(keys, goal=spot).distance_at(x, y, strict=True)
                if leg < best:
                    best = min(best, leg + self._from_key(spot, keys | {colour}))
        return best

    def _from_key(self, spot: tuple[float, float], keys: frozenset[str]) -> float:
        # The rest of the route once a key is picked up; at most 2^3 key sets.
        return _route_from(self, spot, keys)

    @cached_property  # read on every step while training; it never changes
    def start_remaining(self) -> float:
        return self.remaining(*self.field.start)


@cache  # every Attempt builds a new KeyedRoute; the WAD is read once per Map and Difficulty
def _key_spots(map_name: str, difficulty: int, wad_path: Path) -> dict[str, list[tuple[float, float]]]:
    editor = omg.mapedit.MapEditor(omg.WAD(str(wad_path)).maps[map_name])
    present = lambda t: t.flags & SKILL_FLAGS[difficulty] and not t.flags & MULTIPLAYER_ONLY  # noqa: E731
    return {
        colour: spots
        for colour, types in KEY_THINGS.items()
        if (spots := [(t.x, t.y) for t in editor.things if t.type in types and present(t)])
    }


@cache
def _route_from(route: KeyedRoute, spot: tuple[float, float], keys: frozenset[str]) -> float:
    return route.remaining(*spot, keys)


@dataclass
class ProgressMeter:
    """Follows one Attempt and keeps its closest approach to the exit.

    The ruler is a DistanceField (every door open, the M4 rule) or a
    KeyedRoute; either answers `remaining(x, y, keys)`.
    """

    ruler: DistanceField | KeyedRoute
    closest: float = math.inf
    path: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # An unreachable spawn would make Progress NaN, or a false 1 the first
        # time the player stands on a cell the exit reaches.
        if not math.isfinite(self.ruler.start_remaining):
            raise ValueError(f"{self.ruler.map}: the spawn cannot reach the exit under {type(self.ruler).__name__}")

    @property
    def field(self) -> DistanceField:
        """The distance field a picture of the Attempt draws."""
        return self.ruler if isinstance(self.ruler, DistanceField) else self.ruler.field

    def visit(self, x: float, y: float, keys: frozenset[str] = frozenset()) -> None:
        self.path.append((x, y))
        self.closest = min(self.closest, self.ruler.remaining(x, y, keys))

    @property
    def progress(self) -> float:
        start = self.ruler.start_remaining
        return min(1.0, max(0.0, 1.0 - self.closest / start))


def _walls(editor: omg.mapedit.MapEditor, locked: frozenset[str] = frozenset()) -> list[tuple[float, float, float, float]]:
    """Lines a player cannot cross.

    One-sided and impassable lines; ledges higher than a step, unless one side
    is a lift; and closed openings, unless the closed sector is a door the
    player can open, with USE or from elsewhere. Doors of a `locked` colour
    stay shut, unless another, unlocked line opens them too.
    """
    shut = set().union(*(LOCKS[colour] for colour in locked))
    sector_of = lambda side: editor.sidedefs[side].sector  # noqa: E731
    tagged = lambda actions: {line.tag for line in editor.linedefs if line.action in actions and line.tag}  # noqa: E731
    door_tags, lift_tags = tagged(TAGGED_DOOR_ACTIONS), tagged(LIFT_ACTIONS)
    openable = {sector_of(line.back) for line in editor.linedefs if line.action in MANUAL_DOOR_ACTIONS - shut and line.back != NO_SIDE}
    openable |= {i for i, sector in enumerate(editor.sectors) if sector.tag in door_tags}
    lifts = {i for i, sector in enumerate(editor.sectors) if sector.tag in lift_tags}
    walls = []
    for line in editor.linedefs:
        a, b = editor.vertexes[line.vx_a], editor.vertexes[line.vx_b]
        blocking = line.back == NO_SIDE or line.impassable
        if not blocking:
            sides = (sector_of(line.front), sector_of(line.back))
            front, back = (editor.sectors[i] for i in sides)
            closed = [i for i in sides if editor.sectors[i].z_ceil - editor.sectors[i].z_floor < PLAYER_HEIGHT]
            ledge = abs(front.z_floor - back.z_floor) > MAX_STEP and not lifts.intersection(sides)
            blocking = ledge or any(i not in openable for i in closed)
        if blocking:
            walls.append((a.x, a.y, b.x, b.y))
    return walls


def _rasterize(blocked: np.ndarray, origin: tuple[int, int], segment) -> None:
    x0, y0, x1, y1 = segment
    samples = max(1, int(math.hypot(x1 - x0, y1 - y0) / 2))
    for t in np.linspace(0.0, 1.0, samples + 1):
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        blocked[int((y - origin[1]) // CELL), int((x - origin[0]) // CELL)] = True


def _exit_sources(editor, origin, blocked) -> list[tuple[float, int, int]]:
    # Free cells within reach of an exit line's midpoint, on its front side
    # (the right of a walk from vertex A to B), where the player stands.
    # The back of a one-sided exit switch is outside the Map.
    sources = []
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
                    sources.append((0.0, row, col))
    return sources


def _pickup_sources(goal, origin, blocked) -> list[tuple[float, int, int]]:
    # Free cells whose centre is close enough to touch an item at `goal`.
    gx, gy = goal
    r = PICKUP_REACH // CELL + 1
    row0, col0 = int((gy - origin[1]) // CELL), int((gx - origin[0]) // CELL)
    sources = []
    for row in range(row0 - r, row0 + r + 1):
        for col in range(col0 - r, col0 + r + 1):
            cx = origin[0] + (col + 0.5) * CELL
            cy = origin[1] + (row + 0.5) * CELL
            if not blocked[row, col] and max(abs(cx - gx), abs(cy - gy)) < PICKUP_REACH:
                sources.append((0.0, row, col))
    return sources


@cache
def distance_field(
    map_name: str = "E1M1",
    wad_path: Path = WAD_PATH,
    *,
    locked: frozenset[str] = frozenset(),
    goal: tuple[float, float] | None = None,
) -> DistanceField:
    """Walking distance to the exit, or to an item at `goal`; doors of the `locked` key colours stay shut."""
    editor = omg.mapedit.MapEditor(omg.WAD(str(wad_path)).maps[map_name])
    xs = [v.x for v in editor.vertexes]
    ys = [v.y for v in editor.vertexes]
    origin = (min(xs) - CELL, min(ys) - CELL)
    shape = ((max(ys) - origin[1]) // CELL + 2, (max(xs) - origin[0]) // CELL + 2)

    blocked = np.zeros(shape, dtype=bool)
    for wall in _walls(editor, locked):
        _rasterize(blocked, origin, wall)

    distance = np.full(shape, math.inf)
    heap = _exit_sources(editor, origin, blocked) if goal is None else _pickup_sources(goal, origin, blocked)
    if not heap:
        raise ValueError(f"{map_name} has no reachable {'exit line' if goal is None else f'cell at {goal}'}")
    for _, row, col in heap:
        distance[row, col] = 0.0

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
