"""Progress: 0 at spawn, 1 at the exit, walls respected; E1M2 adds lifts and keys."""

import hashlib
import math

import numpy as np
import pytest

from doom_player.paths import WAD_PATH
from doom_player.progress import KeyedRoute, ProgressMeter, distance_field

pytestmark = pytest.mark.skipif(not WAD_PATH.exists(), reason="needs wads/doom.wad")

E1M1_START = (1056, -3616)
E1M1_EXIT_SWITCH = (2912, -4768)  # midpoint of linedef 326, action 11
# Fingerprint of E1M1's distance field as every E1M1 result so far measured it.
# If it changes, the meaning of Progress on the E1M1 Scoreboard rows changes.
E1M1_START_DISTANCE = 4760.625683894423
E1M1_DISTANCE_SHA256 = "4c85a35f868185b82fe8649811c424aff1497dbe9bf127e0337ad55333719fc8"

E1M2_START = (-32, -240)
E1M2_RED_KEY = (1136, 352)  # thing type 13
E1M2_EXIT_SWITCH = (-272, 2336)  # in front of linedef 873, action 11
E1M2_BEFORE_RED_DOOR = (-700, 384)  # spawn side of linedefs 527/528, action 28
RED = frozenset({"red"})


@pytest.fixture(scope="module")
def field():
    return distance_field("E1M1")


def test_start_is_reachable_and_farther_than_a_straight_line(field):
    straight = math.dist(E1M1_START, E1M1_EXIT_SWITCH)
    assert field.start == E1M1_START
    assert straight < field.start_distance < math.inf


def test_e1m1_field_is_pinned(field):
    assert field.start_distance == pytest.approx(E1M1_START_DISTANCE)
    assert hashlib.sha256(field.distance.tobytes()).hexdigest() == E1M1_DISTANCE_SHA256


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


def test_e1m2_spawn_reaches_the_exit_through_lifts_and_remote_doors():
    # The exit room sits behind a lift (tag 13) and switch-opened doors; as
    # walls they cut the exit off from the spawn.
    field = distance_field("E1M2")
    assert field.start == E1M2_START
    assert field.start_distance == pytest.approx(4959, abs=1)


def test_e1m2_red_door_locked_cuts_the_exit_off():
    # The red door (linedefs 527/528, action 28) is the only way to the exit.
    assert distance_field("E1M2", locked=frozenset({"red"})).start_distance == math.inf


def test_locking_a_colour_the_map_does_not_use_changes_nothing():
    open_field, locked = distance_field("E1M2"), distance_field("E1M2", locked=frozenset({"blue", "yellow"}))
    assert np.array_equal(open_field.distance, locked.distance)


def test_distance_to_a_point_reaches_zero_where_the_item_is_touched():
    to_key = distance_field("E1M2", locked=frozenset({"red"}), goal=E1M2_RED_KEY)
    assert to_key.distance_at(*E1M2_RED_KEY) == 0.0
    assert to_key.distance_at(E1M2_RED_KEY[0] + 30, E1M2_RED_KEY[1]) == 0.0  # touching, per axis
    assert math.dist(E1M2_START, E1M2_RED_KEY) < to_key.start_distance < math.inf


@pytest.fixture(scope="module")
def e1m2_route():
    return KeyedRoute("E1M2")


def test_keyed_route_finds_the_red_key_from_the_wad(e1m2_route):
    assert e1m2_route.keys == {"red": [E1M2_RED_KEY]}


def test_keyed_route_goes_through_the_red_key(e1m2_route):
    to_key = distance_field("E1M2", locked=RED, goal=E1M2_RED_KEY).start_distance
    key_to_exit = e1m2_route.field.distance_at(*E1M2_RED_KEY)
    assert e1m2_route.start_remaining == pytest.approx(to_key + key_to_exit)
    assert e1m2_route.start_remaining > e1m2_route.field.start_distance


def test_keyed_route_is_continuous_at_the_pickup(e1m2_route):
    before = e1m2_route.remaining(*E1M2_RED_KEY)
    after = e1m2_route.remaining(*E1M2_RED_KEY, RED)
    assert before == pytest.approx(after)
    assert e1m2_route.remaining(*E1M2_EXIT_SWITCH, RED) == 0.0


def test_keyed_progress_does_not_pay_for_the_locked_door(e1m2_route):
    doors_open, keyed = ProgressMeter(e1m2_route.field), ProgressMeter(e1m2_route)
    for meter in (doors_open, keyed):
        meter.visit(*E1M2_START)
        meter.visit(*E1M2_BEFORE_RED_DOOR)
    assert doors_open.progress > 0.15  # the M4 rule pays for walking to a door that will not open
    assert keyed.progress == 0.0  # without the key, the door is farther from the exit than the spawn


def test_keyed_progress_counts_the_walk_to_the_key(e1m2_route):
    meter = ProgressMeter(e1m2_route)
    meter.visit(*E1M2_START)
    meter.visit(*E1M2_RED_KEY, RED)
    assert meter.progress == pytest.approx(1 - e1m2_route.field.distance_at(*E1M2_RED_KEY) / e1m2_route.start_remaining)
    meter.visit(*E1M2_EXIT_SWITCH, RED)
    assert meter.progress == 1.0


def test_keyed_route_is_the_distance_field_on_a_map_without_keys(field):
    route = KeyedRoute("E1M1")
    assert route.keys == {}
    assert route.start_remaining == field.start_distance
    for spot in (E1M1_START, E1M1_EXIT_SWITCH, (1500, -3200), (-700, -2100)):
        assert route.remaining(*spot) == field.distance_at(*spot)


def test_meter_refuses_a_spawn_that_cannot_reach_the_exit():
    with pytest.raises(ValueError, match="E1M2"):
        ProgressMeter(distance_field("E1M2", locked=RED))
