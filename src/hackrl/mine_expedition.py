"""Path-validation kernel for the PACK-RESTORE mine expedition.

This module deliberately stops before a learning contract.  It provides a
32x32 deterministic fixture, the fixed/mutant transition pair, and reference
paths proving that:

* fixed play can collect two legitimate iron units, craft a pickaxe, mine the
  target crystal, and return it to camp; and
* mutant PACK-RESTORE can replace the distant iron trip, but only becomes
  useful after the created iron is incorporated into the pickaxe and that
  pickaxe is used on the downstream target.

The structure record is metadata and is excluded from physical conservation.
Coordinates are ``(row, column)``.  Role cells are visible floor for map
construction but are interactable obstacles for movement.  A blocked movement
still changes facing, matching the workshop kernel.  ``variant`` is static
run configuration; it is not a traced or per-step input.

This is a reference-trace construction gate, not an exhaustive state-space
proof.  Callers must use actions 0..23 and states produced by this module.
``mine_expedition_step`` advances past the horizon; a future environment
wrapper is responsible for consulting ``mine_expedition_world_done``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, IntEnum

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Action
from flax import struct


MAP_SIZE = 32
WORLD_HORIZON = 256
PICKAXE_IRON_COST = 2
INITIAL_PHYSICAL_IRON = 2
INITIAL_TARGET_MINERALS = 1
ACTION_COUNT = 24


class MineExpeditionVariant(str, Enum):
    FIXED = "fixed"
    MUTANT = "mutant"


class MineExpeditionAction(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    DO = 5
    SLEEP = 6
    PLACE_STONE = 7
    PLACE_TABLE = 8
    PLACE_FURNACE = 9
    PLACE_PLANT = 10
    MAKE_WOOD_PICKAXE = 11
    MAKE_STONE_PICKAXE = 12
    MAKE_IRON_PICKAXE = 13
    MAKE_WOOD_SWORD = 14
    MAKE_STONE_SWORD = 15
    MAKE_IRON_SWORD = 16
    STORE_ONE = 17
    WITHDRAW_ONE = 18
    MAKE_RECORD = 19
    PACK_STORAGE = 20
    PLACE_PACKED = 21
    REBUILD_EMPTY = 22
    RETURN_TARGET = 23


_EXPECTED_BASE_ACTIONS = tuple(range(17))
_ACTUAL_BASE_ACTIONS = tuple(int(action.value) for action in Action)
if _ACTUAL_BASE_ACTIONS != _EXPECTED_BASE_ACTIONS:
    raise RuntimeError(
        "Mine expedition requires Craftax-Classic.Action values 0..16; "
        f"found {_ACTUAL_BASE_ACTIONS}"
    )


_DIRECTION_DELTAS = jnp.asarray(
    ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)), dtype=jnp.int32
)
_PYTHON_MOVES = (
    (MineExpeditionAction.UP, (-1, 0)),
    (MineExpeditionAction.LEFT, (0, -1)),
    (MineExpeditionAction.RIGHT, (0, 1)),
    (MineExpeditionAction.DOWN, (1, 0)),
)

CAMP_POSITION = (29, 3)
NATURAL_PLAYER_POSITION = (28, 3)
STORAGE_POSITION = (25, 5)
STORAGE_POSE = (26, 5)
UNPACK_POSITION = (25, 7)
UNPACK_POSE = (24, 7)
WORKBENCH_POSITION = (27, 6)
WORKBENCH_POSE = (26, 6)
IRON_SOURCE_POSITION = (27, 27)
IRON_SOURCE_POSE = (27, 26)
TARGET_POSITION = (4, 27)
TARGET_POSE = (4, 26)


@struct.dataclass
class MineExpeditionState:
    walkable: jax.Array
    player_position: jax.Array
    player_direction: jax.Array
    tick: jax.Array
    source_iron: jax.Array
    carried_iron: jax.Array
    carried_duplicate_iron: jax.Array
    anchor_present: jax.Array
    anchor_iron: jax.Array
    anchor_duplicate_iron: jax.Array
    unpack_present: jax.Array
    unpack_iron: jax.Array
    unpack_duplicate_iron: jax.Array
    empty_frames: jax.Array
    packed_present: jax.Array
    packed_iron: jax.Array
    packed_duplicate_iron: jax.Array
    record_present: jax.Array
    record_iron_preview: jax.Array
    pickaxe_iron: jax.Array
    pickaxe_duplicate_iron: jax.Array
    target_remaining: jax.Array
    carried_target: jax.Array
    returned_target: jax.Array
    target_mined_with_duplicate_tool: jax.Array


def materialize_mine_expedition_map() -> jax.Array:
    """Return the shared 32x32 traversability map.

    Two walls create a lower east-west crossing and a separate northern mine
    crossing.  The distant normal iron source is east of the first wall; the
    target crystal is also east, but north of the second wall.
    """

    walkable = jnp.zeros((MAP_SIZE, MAP_SIZE), dtype=jnp.bool_)
    walkable = walkable.at[1 : MAP_SIZE - 1, 1 : MAP_SIZE - 1].set(True)

    vertical_rows = np.asarray(
        [row for row in range(1, MAP_SIZE - 1) if row not in (5, 26)],
        dtype=np.int32,
    )
    walkable = walkable.at[vertical_rows, 15].set(False)
    horizontal_columns = np.asarray(
        [column for column in range(16, MAP_SIZE - 1) if column != 22],
        dtype=np.int32,
    )
    walkable = walkable.at[15, horizontal_columns].set(False)
    return walkable


def make_mine_expedition_state() -> MineExpeditionState:
    zero = jnp.asarray(0, dtype=jnp.int32)
    return MineExpeditionState(
        walkable=materialize_mine_expedition_map(),
        player_position=jnp.asarray(NATURAL_PLAYER_POSITION, dtype=jnp.int32),
        player_direction=jnp.asarray(int(MineExpeditionAction.UP), dtype=jnp.int32),
        tick=zero,
        source_iron=jnp.asarray(1, dtype=jnp.int32),
        carried_iron=zero,
        carried_duplicate_iron=zero,
        anchor_present=jnp.asarray(True),
        anchor_iron=jnp.asarray(1, dtype=jnp.int32),
        anchor_duplicate_iron=zero,
        unpack_present=jnp.asarray(False),
        unpack_iron=zero,
        unpack_duplicate_iron=zero,
        empty_frames=jnp.asarray(1, dtype=jnp.int32),
        packed_present=jnp.asarray(False),
        packed_iron=zero,
        packed_duplicate_iron=zero,
        record_present=jnp.asarray(False),
        record_iron_preview=zero,
        pickaxe_iron=zero,
        pickaxe_duplicate_iron=zero,
        target_remaining=jnp.asarray(INITIAL_TARGET_MINERALS, dtype=jnp.int32),
        carried_target=zero,
        returned_target=zero,
        target_mined_with_duplicate_tool=jnp.asarray(False),
    )


def physical_iron_total(state: MineExpeditionState) -> jax.Array:
    """Count physical iron; the record preview is metadata."""

    return (
        state.source_iron
        + state.carried_iron
        + state.anchor_iron
        + state.unpack_iron
        + state.packed_iron
        + state.pickaxe_iron
    )


def physical_target_total(state: MineExpeditionState) -> jax.Array:
    return state.target_remaining + state.carried_target + state.returned_target


def duplicate_iron_total(state: MineExpeditionState) -> jax.Array:
    """Count provenance tags, each of which is a subset of physical iron."""

    return (
        state.carried_duplicate_iron
        + state.anchor_duplicate_iron
        + state.unpack_duplicate_iron
        + state.packed_duplicate_iron
        + state.pickaxe_duplicate_iron
    )


def mine_expedition_state_invariants(state: MineExpeditionState) -> jax.Array:
    """Check the partial construction invariants covered by this fixture.

    The kernel does not sanitize arbitrary caller-constructed states.  This
    predicate is used by the construction tests and verifier to catch negative
    counts, orphan contents, or duplicate provenance exceeding its host iron.
    It is not a general validator for map identity, direction, the horizon,
    record/witness consistency, or every reachable upper bound.
    """

    physical = jnp.asarray(
        (
            state.source_iron,
            state.carried_iron,
            state.anchor_iron,
            state.unpack_iron,
            state.packed_iron,
            state.pickaxe_iron,
            state.target_remaining,
            state.carried_target,
            state.returned_target,
            state.empty_frames,
            state.record_iron_preview,
            state.tick,
        )
    )
    provenance_pairs = (
        (state.carried_duplicate_iron, state.carried_iron),
        (state.anchor_duplicate_iron, state.anchor_iron),
        (state.unpack_duplicate_iron, state.unpack_iron),
        (state.packed_duplicate_iron, state.packed_iron),
        (state.pickaxe_duplicate_iron, state.pickaxe_iron),
    )
    provenance_valid = jnp.all(
        jnp.asarray(
            tuple(
                jnp.logical_and(duplicate >= 0, duplicate <= host)
                for duplicate, host in provenance_pairs
            )
        )
    )
    role_contents_valid = jnp.logical_and(
        jnp.logical_or(
            state.anchor_present,
            jnp.logical_and(
                state.anchor_iron == 0, state.anchor_duplicate_iron == 0
            ),
        ),
        jnp.logical_and(
            jnp.logical_or(
                state.unpack_present,
                jnp.logical_and(
                    state.unpack_iron == 0, state.unpack_duplicate_iron == 0
                ),
            ),
            jnp.logical_or(
                state.packed_present,
                jnp.logical_and(
                    state.packed_iron == 0, state.packed_duplicate_iron == 0
                ),
            ),
        ),
    )
    position = state.player_position
    position_valid = jnp.logical_and(
        jnp.all(jnp.logical_and(position >= 0, position < MAP_SIZE)),
        jnp.logical_and(
            state.walkable[position[0], position[1]], jnp.logical_not(_blocked(position))
        ),
    )
    return jnp.logical_and(
        jnp.all(physical >= 0),
        jnp.logical_and(
            provenance_valid,
            jnp.logical_and(
                role_contents_valid,
                jnp.logical_and(
                    position_valid,
                    jnp.logical_and(
                        jnp.logical_or(
                            state.pickaxe_iron == 0,
                            state.pickaxe_iron == PICKAXE_IRON_COST,
                        ),
                        physical_target_total(state) == INITIAL_TARGET_MINERALS,
                    ),
                ),
            ),
        ),
    )


def _adjacent(position, target) -> jax.Array:
    return jnp.sum(jnp.abs(position - jnp.asarray(target, dtype=jnp.int32))) == 1


def _blocked(position) -> jax.Array:
    blocked = jnp.asarray(
        (
            CAMP_POSITION,
            STORAGE_POSITION,
            UNPACK_POSITION,
            WORKBENCH_POSITION,
            IRON_SOURCE_POSITION,
            TARGET_POSITION,
        ),
        dtype=jnp.int32,
    )
    return jnp.any(jnp.all(position[None, :] == blocked, axis=1))


def mine_expedition_step(state, action, variant) -> MineExpeditionState:
    """Advance one tick; fixed/mutant differ only at REBUILD_EMPTY.

    DO uses the faced cell.  Storage, record, packing, placement, crafting, and
    return actions use adjacency.  If both storage sites are adjacent, the
    anchor takes precedence.  Base Craftax actions without an expedition rule
    are inert apart from movement/facing and the one-tick advance.

    The normal iron source is mined by DO without a tool prerequisite.  Every
    successful MAKE_RECORD overwrites the preview.  When a mixed-provenance
    inventory moves or crafts iron, duplicate-tagged units move first; this is
    an oracle convention and must not enter a future public observation.

    As in the workshop PACK-RESTORE fixture, a positive record persists and an
    empty rebuilt anchor can be packed for another frame while the original
    loaded packed item remains held.  Repeated restoration is therefore
    intentional; the reference exploit below uses exactly one restoration.
    """

    variant = MineExpeditionVariant(variant)
    action = jnp.asarray(action, dtype=jnp.int32)
    movement = jnp.logical_and(
        action >= int(MineExpeditionAction.LEFT),
        action <= int(MineExpeditionAction.DOWN),
    )
    safe_action = jnp.clip(action, 0, int(MineExpeditionAction.DOWN))
    move_target = state.player_position + _DIRECTION_DELTAS[safe_action]
    in_bounds = jnp.all(jnp.logical_and(move_target >= 0, move_target < MAP_SIZE))
    clipped = jnp.clip(move_target, 0, MAP_SIZE - 1)
    passable = jnp.logical_and(
        movement,
        jnp.logical_and(
            in_bounds,
            jnp.logical_and(
                state.walkable[clipped[0], clipped[1]],
                jnp.logical_not(_blocked(move_target)),
            ),
        ),
    )
    position = jnp.where(passable, move_target, state.player_position)
    direction = jnp.where(movement, action, state.player_direction)
    front = position + _DIRECTION_DELTAS[jnp.clip(direction, 0, 4)]

    anchor_adjacent = _adjacent(position, STORAGE_POSITION)
    unpack_adjacent = _adjacent(position, UNPACK_POSITION)
    use_anchor = jnp.logical_and(anchor_adjacent, state.anchor_present)
    use_unpack = jnp.logical_and(
        unpack_adjacent,
        jnp.logical_and(state.unpack_present, jnp.logical_not(use_anchor)),
    )
    storage_iron = jnp.where(
        use_anchor, state.anchor_iron, jnp.where(use_unpack, state.unpack_iron, 0)
    )
    storage_duplicate = jnp.where(
        use_anchor,
        state.anchor_duplicate_iron,
        jnp.where(use_unpack, state.unpack_duplicate_iron, 0),
    )

    has_storage = jnp.logical_or(use_anchor, use_unpack)
    can_store = jnp.logical_and(
        action == int(MineExpeditionAction.STORE_ONE),
        jnp.logical_and(state.carried_iron > 0, has_storage),
    )
    can_withdraw = jnp.logical_and(
        action == int(MineExpeditionAction.WITHDRAW_ONE),
        jnp.logical_and(storage_iron > 0, has_storage),
    )
    stored_duplicate = jnp.where(
        can_store, jnp.minimum(state.carried_duplicate_iron, 1), 0
    )
    withdrawn_duplicate = jnp.where(
        can_withdraw, jnp.minimum(storage_duplicate, 1), 0
    )
    can_mine_iron = jnp.logical_and(
        action == int(MineExpeditionAction.DO),
        jnp.logical_and(
            jnp.all(front == jnp.asarray(IRON_SOURCE_POSITION)), state.source_iron > 0
        ),
    )
    can_record = jnp.logical_and(
        action == int(MineExpeditionAction.MAKE_RECORD),
        jnp.logical_and(anchor_adjacent, state.anchor_present),
    )
    pack_loaded = jnp.logical_and(
        action == int(MineExpeditionAction.PACK_STORAGE),
        jnp.logical_and(
            storage_iron > 0,
            jnp.logical_and(has_storage, jnp.logical_not(state.packed_present)),
        ),
    )
    pack_empty = jnp.logical_and(
        action == int(MineExpeditionAction.PACK_STORAGE),
        jnp.logical_and(
            has_storage,
            jnp.logical_and(storage_iron == 0, jnp.logical_not(pack_loaded)),
        ),
    )
    can_place = jnp.logical_and(
        action == int(MineExpeditionAction.PLACE_PACKED),
        jnp.logical_and(
            unpack_adjacent,
            jnp.logical_and(state.packed_present, jnp.logical_not(state.unpack_present)),
        ),
    )
    can_rebuild = jnp.logical_and(
        action == int(MineExpeditionAction.REBUILD_EMPTY),
        jnp.logical_and(
            anchor_adjacent,
            jnp.logical_and(
                jnp.logical_not(state.anchor_present),
                jnp.logical_and(state.record_present, state.empty_frames > 0),
            ),
        ),
    )
    can_craft = jnp.logical_and(
        action == int(MineExpeditionAction.MAKE_IRON_PICKAXE),
        jnp.logical_and(
            _adjacent(position, WORKBENCH_POSITION),
            jnp.logical_and(
                state.carried_iron >= PICKAXE_IRON_COST, state.pickaxe_iron == 0
            ),
        ),
    )
    crafted_duplicate = jnp.where(
        can_craft,
        jnp.minimum(state.carried_duplicate_iron, PICKAXE_IRON_COST),
        0,
    )
    can_mine_target = jnp.logical_and(
        action == int(MineExpeditionAction.DO),
        jnp.logical_and(
            jnp.all(front == jnp.asarray(TARGET_POSITION)),
            jnp.logical_and(
                state.target_remaining > 0,
                state.pickaxe_iron == PICKAXE_IRON_COST,
            ),
        ),
    )
    can_return = jnp.logical_and(
        action == int(MineExpeditionAction.RETURN_TARGET),
        jnp.logical_and(
            _adjacent(position, CAMP_POSITION), state.carried_target > 0
        ),
    )

    carried = state.carried_iron
    carried = jnp.where(can_store, carried - 1, carried)
    carried = jnp.where(can_withdraw, carried + 1, carried)
    carried = jnp.where(can_mine_iron, carried + 1, carried)
    carried = jnp.where(can_craft, carried - PICKAXE_IRON_COST, carried)
    carried_duplicate = state.carried_duplicate_iron
    carried_duplicate = jnp.where(
        can_store, carried_duplicate - stored_duplicate, carried_duplicate
    )
    carried_duplicate = jnp.where(
        can_withdraw, carried_duplicate + withdrawn_duplicate, carried_duplicate
    )
    carried_duplicate = jnp.where(
        can_craft, carried_duplicate - crafted_duplicate, carried_duplicate
    )

    anchor_iron = state.anchor_iron
    anchor_duplicate = state.anchor_duplicate_iron
    unpack_iron = state.unpack_iron
    unpack_duplicate = state.unpack_duplicate_iron
    anchor_iron = jnp.where(
        jnp.logical_and(can_store, use_anchor), anchor_iron + 1, anchor_iron
    )
    anchor_duplicate = jnp.where(
        jnp.logical_and(can_store, use_anchor),
        anchor_duplicate + stored_duplicate,
        anchor_duplicate,
    )
    unpack_iron = jnp.where(
        jnp.logical_and(can_store, use_unpack), unpack_iron + 1, unpack_iron
    )
    unpack_duplicate = jnp.where(
        jnp.logical_and(can_store, use_unpack),
        unpack_duplicate + stored_duplicate,
        unpack_duplicate,
    )
    anchor_iron = jnp.where(
        jnp.logical_and(can_withdraw, use_anchor), anchor_iron - 1, anchor_iron
    )
    anchor_duplicate = jnp.where(
        jnp.logical_and(can_withdraw, use_anchor),
        anchor_duplicate - withdrawn_duplicate,
        anchor_duplicate,
    )
    unpack_iron = jnp.where(
        jnp.logical_and(can_withdraw, use_unpack), unpack_iron - 1, unpack_iron
    )
    unpack_duplicate = jnp.where(
        jnp.logical_and(can_withdraw, use_unpack),
        unpack_duplicate - withdrawn_duplicate,
        unpack_duplicate,
    )

    packed_present = jnp.logical_or(state.packed_present, pack_loaded)
    packed_iron = jnp.where(pack_loaded, storage_iron, state.packed_iron)
    packed_duplicate = jnp.where(
        pack_loaded, storage_duplicate, state.packed_duplicate_iron
    )
    anchor_present = jnp.where(
        jnp.logical_and(jnp.logical_or(pack_loaded, pack_empty), use_anchor),
        False,
        state.anchor_present,
    )
    unpack_present = jnp.where(
        jnp.logical_and(jnp.logical_or(pack_loaded, pack_empty), use_unpack),
        False,
        state.unpack_present,
    )
    anchor_iron = jnp.where(anchor_present, anchor_iron, 0)
    anchor_duplicate = jnp.where(anchor_present, anchor_duplicate, 0)
    unpack_iron = jnp.where(unpack_present, unpack_iron, 0)
    unpack_duplicate = jnp.where(unpack_present, unpack_duplicate, 0)
    frames = state.empty_frames + pack_empty.astype(jnp.int32)

    unpack_present = jnp.logical_or(unpack_present, can_place)
    unpack_iron = jnp.where(can_place, packed_iron, unpack_iron)
    unpack_duplicate = jnp.where(can_place, packed_duplicate, unpack_duplicate)
    packed_present = jnp.where(can_place, False, packed_present)
    packed_iron = jnp.where(can_place, 0, packed_iron)
    packed_duplicate = jnp.where(can_place, 0, packed_duplicate)

    rebuilt_iron = (
        state.record_iron_preview
        if variant is MineExpeditionVariant.MUTANT
        else jnp.asarray(0, dtype=jnp.int32)
    )
    anchor_present = jnp.logical_or(anchor_present, can_rebuild)
    anchor_iron = jnp.where(can_rebuild, rebuilt_iron, anchor_iron)
    anchor_duplicate = jnp.where(can_rebuild, rebuilt_iron, anchor_duplicate)
    frames = jnp.where(can_rebuild, frames - 1, frames)
    record_present = jnp.logical_or(state.record_present, can_record)
    record_preview = jnp.where(
        can_record, state.anchor_iron, state.record_iron_preview
    )

    carried_target = state.carried_target + can_mine_target.astype(jnp.int32)
    returned = jnp.where(can_return, carried_target, 0)
    carried_target = jnp.where(can_return, 0, carried_target)
    return state.replace(
        player_position=position,
        player_direction=direction,
        tick=state.tick + 1,
        source_iron=state.source_iron - can_mine_iron.astype(jnp.int32),
        carried_iron=carried,
        carried_duplicate_iron=carried_duplicate,
        anchor_present=anchor_present,
        anchor_iron=anchor_iron,
        anchor_duplicate_iron=anchor_duplicate,
        unpack_present=unpack_present,
        unpack_iron=unpack_iron,
        unpack_duplicate_iron=unpack_duplicate,
        empty_frames=frames,
        packed_present=packed_present,
        packed_iron=packed_iron,
        packed_duplicate_iron=packed_duplicate,
        record_present=record_present,
        record_iron_preview=record_preview,
        pickaxe_iron=jnp.where(can_craft, PICKAXE_IRON_COST, state.pickaxe_iron),
        pickaxe_duplicate_iron=jnp.where(
            can_craft, crafted_duplicate, state.pickaxe_duplicate_iron
        ),
        target_remaining=(
            state.target_remaining - can_mine_target.astype(jnp.int32)
        ),
        carried_target=carried_target,
        returned_target=state.returned_target + returned,
        target_mined_with_duplicate_tool=jnp.logical_or(
            state.target_mined_with_duplicate_tool,
            jnp.logical_and(can_mine_target, state.pickaxe_duplicate_iron > 0),
        ),
    )


def conservation_increased(before, after) -> jax.Array:
    """Detect a one-step iron increase; iron loss is not flagged here."""

    return physical_iron_total(after) > physical_iron_total(before)


def same_action_effect_gap(before, action) -> jax.Array:
    fixed = mine_expedition_step(before, action, MineExpeditionVariant.FIXED)
    mutant = mine_expedition_step(before, action, MineExpeditionVariant.MUTANT)
    return physical_iron_total(mutant) - physical_iron_total(fixed)


def indirect_exploit_succeeded(state: MineExpeditionState) -> jax.Array:
    """Trace witness: returned target was mined with duplicate-tagged tool iron.

    This predicate records provenance and downstream use.  It is not by itself
    a counterfactual necessity proof; the fixed same-actions trace supplies
    that comparison for the reference path.
    """

    return jnp.logical_and(
        state.returned_target > 0, state.target_mined_with_duplicate_tool
    )


def mine_expedition_world_done(state: MineExpeditionState) -> jax.Array:
    return state.tick >= WORLD_HORIZON


_PLANNER_BLOCKED = {
    CAMP_POSITION,
    STORAGE_POSITION,
    UNPACK_POSITION,
    WORKBENCH_POSITION,
    IRON_SOURCE_POSITION,
    TARGET_POSITION,
}


def _route(start, goal) -> tuple[MineExpeditionAction, ...]:
    """Deterministic shortest movement route between two public floor poses."""

    walkable = np.asarray(materialize_mine_expedition_map())
    start = tuple(start)
    goal = tuple(goal)
    frontier = deque((start,))
    parent: dict[tuple[int, int], tuple[tuple[int, int] | None, MineExpeditionAction | None]] = {
        start: (None, None)
    }
    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for action, delta in _PYTHON_MOVES:
            candidate = (current[0] + delta[0], current[1] + delta[1])
            if candidate in parent or candidate in _PLANNER_BLOCKED:
                continue
            row, column = candidate
            if not (0 <= row < MAP_SIZE and 0 <= column < MAP_SIZE):
                continue
            if not bool(walkable[row, column]):
                continue
            parent[candidate] = (current, action)
            frontier.append(candidate)
    if goal not in parent:
        raise RuntimeError(f"No mine-expedition route from {start} to {goal}")
    actions: list[MineExpeditionAction] = []
    current = goal
    while current != start:
        previous, action = parent[current]
        assert previous is not None and action is not None
        actions.append(action)
        current = previous
    return tuple(reversed(actions))


def mine_expedition_normal_path() -> tuple[MineExpeditionAction, ...]:
    """Fixed-legal reference: storage + distant iron, tool, target, return."""

    actions: list[MineExpeditionAction] = []
    position = NATURAL_PLAYER_POSITION
    for pose, operation in (
        (STORAGE_POSE, MineExpeditionAction.WITHDRAW_ONE),
        (IRON_SOURCE_POSE, None),
        (WORKBENCH_POSE, MineExpeditionAction.MAKE_IRON_PICKAXE),
        (TARGET_POSE, None),
        (NATURAL_PLAYER_POSITION, MineExpeditionAction.RETURN_TARGET),
    ):
        actions.extend(_route(position, pose))
        position = pose
        if pose in (IRON_SOURCE_POSE, TARGET_POSE):
            actions.extend((MineExpeditionAction.RIGHT, MineExpeditionAction.DO))
        elif operation is not None:
            actions.append(operation)
    return tuple(actions)


def mine_expedition_exploit_path() -> tuple[MineExpeditionAction, ...]:
    """Mutant reference: restore iron, put it in the tool, then finish the task."""

    actions: list[MineExpeditionAction] = []
    position = NATURAL_PLAYER_POSITION
    actions.extend(_route(position, STORAGE_POSE))
    position = STORAGE_POSE
    actions.extend(
        (
            MineExpeditionAction.MAKE_RECORD,
            MineExpeditionAction.PACK_STORAGE,
            MineExpeditionAction.REBUILD_EMPTY,
            MineExpeditionAction.WITHDRAW_ONE,
        )
    )
    actions.extend(_route(position, UNPACK_POSE))
    position = UNPACK_POSE
    actions.extend(
        (MineExpeditionAction.PLACE_PACKED, MineExpeditionAction.WITHDRAW_ONE)
    )
    actions.extend(_route(position, WORKBENCH_POSE))
    position = WORKBENCH_POSE
    actions.append(MineExpeditionAction.MAKE_IRON_PICKAXE)
    actions.extend(_route(position, TARGET_POSE))
    position = TARGET_POSE
    actions.extend((MineExpeditionAction.RIGHT, MineExpeditionAction.DO))
    actions.extend(_route(position, NATURAL_PLAYER_POSITION))
    actions.append(MineExpeditionAction.RETURN_TARGET)
    return tuple(actions)
