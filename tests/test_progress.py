"""Progress on E1M1: 0 at spawn, 1 at the exit, walls respected."""

import math

import numpy as np
import pytest

from doom_player.paths import WAD_PATH
from doom_player.progress import ProgressMeter, distance_field

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")

E1M1_START = (1056, -3616)
E1M1_EXIT_SWITCH = (2912, -4768)  # midpoint of linedef 326, action 11


@pytest.fixture(scope="module")
def field():
    return distance_field("E1M1")


def test_start_is_reachable_and_farther_than_a_straight_line(field):
    straight = math.dist(E1M1_START, E1M1_EXIT_SWITCH)
    assert field.start == E1M1_START
    assert straight < field.start_distance < math.inf


def test_exit_is_at_distance_zero(field):
    assert field.distance_at(E1M1_EXIT_SWITCH[0] - 16, E1M1_EXIT_SWITCH[1]) == 0.0


def test_outside_the_map_is_unreachable(field):
    # Seeds sit only on the exit switch's front side, so nothing leaks outside.
    assert field.distance_at(-700, -2100) == math.inf
    assert np.isfinite(field.distance).sum() < field.distance.size / 2


def test_meter_starts_at_zero_and_reaches_one_at_the_exit(field):
    meter = ProgressMeter(field)
    meter.visit(*E1M1_START)
    assert meter.progress == pytest.approx(0.0, abs=0.01)
    meter.visit(E1M1_EXIT_SWITCH[0] - 16, E1M1_EXIT_SWITCH[1])
    assert meter.progress == 1.0


def test_meter_keeps_the_best_moment(field):
    meter = ProgressMeter(field)
    meter.visit(*E1M1_START)
    meter.visit(E1M1_EXIT_SWITCH[0] - 16, E1M1_EXIT_SWITCH[1])
    meter.visit(*E1M1_START)  # walking back does not undo progress
    assert meter.progress == 1.0
