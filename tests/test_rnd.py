"""RND pays more for a screen it has rarely seen than for one it has seen many times.

Fast and without the WAD: frames are made-up 84x84 gray pictures, the size
of one frame of the policy view.
"""

import io

import numpy as np
import pytest
import torch

from doom_player.rnd import RND


def frames(seed: int, n: int = 64) -> np.ndarray:
    """`n` gray frames of one made-up place: noise around a pattern of its own."""
    rng = np.random.default_rng(seed)
    place = rng.integers(0, 256, (1, 84, 84))
    noise = rng.integers(-8, 9, (n, 84, 84))
    return np.clip(place + noise, 0, 255).astype(np.uint8)


def test_a_familiar_screen_pays_less_than_a_new_one():
    rnd = RND(seed=0, device="cpu")
    familiar, new = frames(1), frames(2)
    rnd.update_obs_stats(np.concatenate([familiar, new]))
    for _ in range(30):
        rnd.fit(familiar)
    assert rnd.bonus(familiar).mean() < 0.5 * rnd.bonus(new).mean()


def test_fitting_lowers_the_prediction_error():
    rnd = RND(seed=0, device="cpu")
    seen = frames(1)
    rnd.update_obs_stats(seen)
    first = rnd.fit(seen)
    for _ in range(10):
        last = rnd.fit(seen)
    assert last < first


def test_the_same_seed_builds_the_same_networks():
    a, b, c = RND(seed=3, device="cpu"), RND(seed=3, device="cpu"), RND(seed=4, device="cpu")
    shot = frames(5, n=4)
    assert np.array_equal(a.bonus(shot), b.bonus(shot))
    assert not np.array_equal(a.bonus(shot), c.bonus(shot))


def test_state_survives_a_round_trip():
    rnd = RND(seed=0, device="cpu")
    seen = frames(1)
    rnd.update_obs_stats(seen)
    rnd.fit(seen)
    saved = io.BytesIO()  # as a checkpoint keeps it: through a file
    torch.save(rnd.state_dict(), saved)
    saved.seek(0)
    copy = RND(seed=9, device="cpu")
    copy.load_state_dict(torch.load(saved))
    assert np.allclose(copy.bonus(seen), rnd.bonus(seen))
    assert copy.fit(seen) == pytest.approx(rnd.fit(seen), rel=1e-4)  # the optimiser's state came along too
