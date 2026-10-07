"""Original Maps as a Gymnasium environment, for training a learned Driver.

`MapEnv` is a Gymnasium face on the AttemptSession: every episode is one
Attempt, refereed by the same code the Eval Suite uses, so a policy trains on
exactly the screen, seeds, time limit and Progress it is measured on (ADR 0011).
`make_map_env` adds the same policy view as Scenarios: gray, 84x84, 4 frames.
"""

import gymnasium as gym
import numpy as np

from doom_player.scenarios import FRAME_SKIP, policy_view
from doom_player.session import AttemptSession

# The Action set: the button combinations a learned Driver may press, one per
# decision. A short list makes exploring easier than 19 free buttons; it limits
# what the policy does, not what it sees. No weapon switching.
ACTIONS = {
    "forward": ("MOVE_FORWARD",),
    "forward + turn left": ("MOVE_FORWARD", "TURN_LEFT"),
    "forward + turn right": ("MOVE_FORWARD", "TURN_RIGHT"),
    "turn left": ("TURN_LEFT",),
    "turn right": ("TURN_RIGHT",),
    "back": ("MOVE_BACKWARD",),
    "strafe left": ("MOVE_LEFT",),
    "strafe right": ("MOVE_RIGHT",),
    "attack": ("ATTACK",),
    "forward + attack": ("MOVE_FORWARD", "ATTACK"),
    "use": ("USE",),
}

TRAINING_SEEDS_FROM = 1000  # Eval Specs use small game seeds (e1m1-v1: 0 to 4)


class MapEnv(gym.Env):
    """One Attempt per episode on an original Map, through a fresh AttemptSession."""

    def __init__(
        self,
        map: str = "E1M1",
        difficulty: int = 3,
        tic_limit: int = 6300,
        contender: str = "training",
        progress_rule: str = "doors-open",
    ):
        self.map, self.difficulty, self.tic_limit = map, difficulty, tic_limit
        self.progress_rule = progress_rule  # the rule the record keeps; shaping may read another
        self.contender = contender
        self.observation_space = gym.spaces.Box(0, 255, (240, 320, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.session: AttemptSession | None = None
        self.pressed: list[list[bool]] = []  # each action's button states, in the session's button order
        self.on_frame = None  # set to a video writer's `add` to film one frame per decision

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new Attempt.

        The Eval Suite names the game with `options={"game_seed": s}`. Training
        leaves it out, and the game seed is drawn from this env's random
        generator (seeded by `seed`), always at or above TRAINING_SEEDS_FROM, so
        a policy never practises on the games it is measured on.
        """
        super().reset(seed=seed)
        self._end_attempt()
        if options and "game_seed" in options:
            game_seed = int(options["game_seed"])
        else:
            game_seed = int(self.np_random.integers(TRAINING_SEEDS_FROM, 2**31 - 1))
        self.session = AttemptSession(self.contender, self.map, self.difficulty, game_seed, self.tic_limit, self.progress_rule)
        if not self.pressed:  # the buttons are fixed by doom.cfg: build the table once
            unknown = {b for names in ACTIONS.values() for b in names} - set(self.session.buttons)
            if unknown:
                raise ValueError(f"ACTIONS names unknown buttons: {sorted(unknown)}")
            self.pressed = [[b in names for b in self.session.buttons] for names in ACTIONS.values()]
        self._screen = self.session.screen()
        if self.on_frame:  # the spawn, first frame of a video
            self.on_frame(self._screen)
        return self._screen, {}

    def step(self, action: int):
        # Never tic by tic for video: that redraws the status bar face differently,
        # and the policy, which reads the screen, would play another game (ADR 0011).
        reward = self.session.act(self.pressed[int(action)], FRAME_SKIP)
        if not self.session.finished:
            self._screen = self.session.screen()
            if self.on_frame:
                self.on_frame(self._screen)
            return self._screen, reward, False, False, {}
        # Over: the game has no screen left, so the last one stands in for it.
        record = self.session.record
        return self._screen, reward, record.terminated, record.truncated, {"record": record.to_dict()}

    def _end_attempt(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def close(self):
        self._end_attempt()


class ProgressShaping(gym.Wrapper):
    """Training only: pay `progress_reward` for new ground toward the exit, charge `death_penalty` on death.

    Progress counts only the best distance reached so far, so walking back and
    forth earns nothing: an Attempt's shaping adds up to `progress_reward` times
    its recorded Progress, minus the penalty if it died. Progress is measured
    from the player's position, Privileged Information the policy never sees;
    it is declared in Results (ADR 0002). The Map's own reward stays in
    `info["raw_reward"]`.

    `progress_rule` picks the Progress paid for, which may differ from the one
    the record keeps: M4's recipe pays doors-open Progress, where a locked door
    counts as open, even on a Map scored by the keyed rule (ADR 0013).
    """

    def __init__(self, env: gym.Env, progress_reward: float, death_penalty: float, progress_rule: str = "doors-open"):
        super().__init__(env)
        self.progress_reward = progress_reward
        self.death_penalty = death_penalty
        self.progress_rule = progress_rule

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._paid = self.unwrapped.session.progress_under(self.progress_rule)  # Progress already paid for
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        progress = self.unwrapped.session.progress_under(self.progress_rule)
        bonus = self.progress_reward * (progress - self._paid)
        self._paid = progress
        if "record" in info and info["record"]["died"]:
            bonus -= self.death_penalty
        return obs, reward + bonus, terminated, truncated, {**info, "raw_reward": reward}


def make_map_env(*args, **kwargs) -> gym.Env:
    """A MapEnv (same arguments) seen through the policy view."""
    return policy_view(MapEnv(*args, **kwargs))
