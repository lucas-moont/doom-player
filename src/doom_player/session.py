"""One Attempt on one Map, under fixed rules, ending in a record.

The AttemptSession is the ruler's core. Every Contender plays through one:
an in-process Contender calls `act` in a loop (Door A, see `eval.py`), and an
LLM calls it through the MCP server's tools (Door B). Either way the Attempt
ends in the same `AttemptRecord`.

What a Contender may see is built in `observe` and limited to Human-equivalent
Observations (ADR 0002): the screen, the automap, and numbers the HUD shows.
The player's position is Privileged Information; it is read only here, to
measure Progress, and never leaves the session.
"""

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import vizdoom as vzd

from doom_player.paths import WAD_PATH

# Numbers a human reads off the status bar. Key cards have no game variable;
# they are visible in the screen's HUD like any other part of the status bar.
HUD_VARIABLES = {
    "HEALTH": vzd.GameVariable.HEALTH,
    "ARMOR": vzd.GameVariable.ARMOR,
    "SELECTED_WEAPON": vzd.GameVariable.SELECTED_WEAPON,
    "SELECTED_WEAPON_AMMO": vzd.GameVariable.SELECTED_WEAPON_AMMO,
    "BULLETS": vzd.GameVariable.AMMO2,
    "SHELLS": vzd.GameVariable.AMMO3,
    "ROCKETS": vzd.GameVariable.AMMO5,
    "CELLS": vzd.GameVariable.AMMO6,
}

OBSERVATION_CLASS = "human-equivalent"
MAX_TICS_PER_ACTION = 35


@dataclass
class AttemptRecord:
    """What the Eval Suite keeps from one Attempt."""

    contender: str
    map: str
    difficulty: int
    seed: int
    tic_limit: int
    observation_class: str
    actions: int = 0
    tics: int = 0
    terminated: bool = False
    truncated: bool = False
    died: bool = False
    cleared: bool = False
    progress: float | None = None
    wall_clock_s: float = 0.0
    tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    """Everything a Contender is allowed to see after one action."""

    screen: np.ndarray  # (240, 320, 3) RGB, HUD included
    automap: np.ndarray  # (240, 320, 3) RGB, what Tab shows
    hud: dict[str, int]
    tics_left: int


@dataclass
class AttemptSession:
    contender: str
    map: str = "E1M1"
    difficulty: int = 3
    seed: int = 0
    tic_limit: int = 6300
    progress_meter: object | None = None  # doom_player.progress.ProgressMeter
    record: AttemptRecord = field(init=False)

    def __post_init__(self) -> None:
        if not WAD_PATH.exists():
            raise SystemExit(f"Original Maps need the purchased WAD at {WAD_PATH}")
        self.game = self._build_game()
        self.buttons = [b.name for b in self.game.get_available_buttons()]
        self.record = AttemptRecord(
            contender=self.contender,
            map=self.map,
            difficulty=self.difficulty,
            seed=self.seed,
            tic_limit=self.tic_limit,
            observation_class=OBSERVATION_CLASS,
        )
        self._started = time.perf_counter()
        self._measure()

    def _build_game(self) -> vzd.DoomGame:
        game = vzd.DoomGame()
        game.load_config(os.path.join(vzd.scenarios_path, "doom.cfg"))
        game.set_doom_game_path(str(WAD_PATH))
        game.set_doom_map(self.map)
        game.set_doom_skill(self.difficulty)
        game.set_window_visible(False)
        game.set_audio_buffer_enabled(False)  # no Contender listens yet
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_available_game_variables(list(HUD_VARIABLES.values()))
        # PLAYER mode (doom.cfg's default) is synchronous: the game waits for each action.
        game.set_mode(vzd.Mode.PLAYER)
        game.set_episode_timeout(self.tic_limit)
        game.set_seed(self.seed)
        game.init()
        return game

    @property
    def finished(self) -> bool:
        return self.game.is_episode_finished()

    def observe(self) -> Observation:
        state = self.game.get_state()
        values = state.game_variables
        return Observation(
            screen=state.screen_buffer,
            automap=state.automap_buffer,
            hud={name: int(v) for name, v in zip(HUD_VARIABLES, values)},
            tics_left=max(0, self.tic_limit - self._tics_played()),
        )

    def act(self, pressed: list[bool], tics: int) -> float:
        """Hold the given buttons for `tics` tics; return the reward."""
        if self.finished:
            raise RuntimeError("The Attempt is over")
        if len(pressed) != len(self.buttons):
            raise ValueError(f"expected {len(self.buttons)} button states")
        if not 1 <= tics <= MAX_TICS_PER_ACTION:
            raise ValueError(f"tics must be between 1 and {MAX_TICS_PER_ACTION}")
        reward = self.game.make_action([float(p) for p in pressed], tics)
        self.record.actions += 1
        self._measure()
        if self.finished:
            self._finish()
        return reward

    def press(self, names: list[str], tics: int) -> float:
        """Like `act`, with buttons given by name (as the MCP tools do)."""
        unknown = set(names) - set(self.buttons)
        if unknown:
            raise ValueError(f"unknown buttons: {sorted(unknown)}")
        return self.act([b in names for b in self.buttons], tics)

    def _tics_played(self) -> int:
        # doom.cfg starts the episode clock at tic 1.
        return self.game.get_episode_time() - 1

    def _measure(self) -> None:
        # Privileged Information, used for measurement only.
        if self.progress_meter is not None:
            x = self.game.get_game_variable(vzd.GameVariable.POSITION_X)
            y = self.game.get_game_variable(vzd.GameVariable.POSITION_Y)
            self.progress_meter.visit(x, y)

    def _finish(self) -> None:
        r = self.record
        r.tics = self._tics_played()
        r.truncated = self.game.is_episode_timeout_reached()
        r.died = self.game.is_player_dead()
        r.terminated = not r.truncated
        r.cleared = r.terminated and not r.died
        if self.progress_meter is not None:
            r.progress = 1.0 if r.cleared else round(self.progress_meter.progress, 4)
        r.wall_clock_s = round(time.perf_counter() - self._started, 2)

    def close(self) -> None:
        self.game.close()
