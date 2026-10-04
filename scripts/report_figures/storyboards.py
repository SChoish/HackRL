"""Verified action sequences for the three defects. These are not learned policies."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hackrl.craft_remain import replay as craft_replay
from hackrl.craft_remain_env import (
    DELIVERY as CRAFT_DELIVERY,
    GRID as CRAFT_GRID,
    PATH_PLAYER,
    SOURCE as CRAFT_SOURCE,
    CraftRemainAction,
    CraftRemainStart,
    CraftRemainVariant,
    REFERENCE_ACTION_INDEX,
    craft_remain_step,
    make_craft_remain_state,
    observe_craft_remain,
    physical_total as craft_physical,
)
from hackrl.pack_restore import (
    INITIAL_GRAIN_TOTAL,
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreStart,
    PackRestoreVariant,
    conservation_increased,
    make_pack_restore_state,
    observe_pack_restore,
    pack_restore_step,
    physical_grain_total,
)
from hackrl.tick_claim import (
    GROWTH_WAIT_TICKS,
    TickClaimAction,
    TickClaimPhase,
    TickClaimStart,
    TickClaimVariant,
    make_tick_claim_state,
    observe_tick_claim,
    tick_claim_setup_prefix,
    tick_claim_step,
)

from report_figures.render import draw_frame, draw_pair, draw_strip

ROOT = Path("/home/ext_csv/HackRL")
OUT = ROOT / "runs" / "figures_report_v1" / "storyboards"
CAPTION = "Verified action sequence, not a learned policy. Lengths are for this start, not a policy score."
DIRS = {0: "stay", 1: "W", 2: "E", 3: "N", 4: "S"}


def _xy(value):
    array = np.asarray(value).reshape(-1)
    return int(array[0]), int(array[1])


def _obs_cells(observation, roles):
    tiles = np.asarray(observation.map_tiles)
    channels = np.asarray(observation.role_channels)
    rows, cols = tiles.shape
    cells = []
    for row in range(rows):
        for col in range(cols):
            kind = "floor"
            text = ""
            for index, name in roles:
                if channels[row, col, index]:
                    kind = name
                    break
            else:
                if int(tiles[row, col]) != int(tiles[rows // 2, cols // 2]):
                    kind = "wall" if kind == "floor" else kind
            cells.append({"r": row, "c": col, "kind": kind, "text": text})
    return cells, (rows // 2, cols // 2)


def _mark(cells, position, kind, text):
    row, col = _xy(position)
    cells.append({"r": row, "c": col, "kind": kind, "text": text})


def _floor_box(cells, walkable):
    mask = np.asarray(walkable)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return
    for row, col in zip(rows.tolist(), cols.tolist()):
        cells.append({"r": int(row), "c": int(col), "kind": "floor", "text": ""})


def tick_frames():
    layout = 0
    start = make_tick_claim_state(layout, int(TickClaimPhase.RIPE), start=TickClaimStart.NATURAL)
    prefix = tuple(int(action) for action in np.asarray(tick_claim_setup_prefix(layout)))
    prepared = start
    for action in prefix:
        prepared = tick_claim_step(prepared, action, TickClaimVariant.FIXED)
    delay = int(prepared.reservation_delay)
    exploit = (
        (int(TickClaimAction.ARM_HARVEST),)
        + (int(TickClaimAction.NOOP),) * (delay - 1)
        + (int(TickClaimAction.DO),)
        + (int(TickClaimAction.NOOP),) * GROWTH_WAIT_TICKS
        + (int(TickClaimAction.DO),)
        + (int(TickClaimAction.DOWN),)
        + (int(TickClaimAction.DELIVER),)
    )
    names = {int(action): action.name for action in TickClaimAction}
    fixed = prepared
    mutant = prepared
    frames = []
    for action in exploit:
        before_fixed, before_mutant = fixed, mutant
        fixed = tick_claim_step(fixed, action, TickClaimVariant.FIXED)
        mutant = tick_claim_step(mutant, action, TickClaimVariant.MUTANT)
        frames.append(
            {
                "action_id": action,
                "action": names[action],
                "fixed": _tick_view(fixed, names[action], before_fixed, before_mutant, mutant),
                "mutant": _tick_view(mutant, names[action], before_mutant, before_fixed, fixed),
            }
        )
    arm = next(index for index, item in enumerate(frames) if item["action"] == "ARM_HARVEST")
    expiry = next(
        index
        for index, item in enumerate(frames)
        if item["action"] == "NOOP" and "reservation remaining 0" in " ".join(item["mutant"]["oracle"])
    )
    settle = next(index for index, item in enumerate(frames) if item["mutant"]["diverges"])
    deliver = next(index for index, item in enumerate(frames) if item["action"] == "DELIVER")
    beats = [
        (arm, "1  reservation created"),
        (expiry, "2  expiry in front of the ripe crop"),
        (settle, "3  DO settles the reservation"),
        (deliver, "4  duplicate grain delivered"),
    ]
    note = (
        f"{CAPTION} Layout 0, ripe natural start, setup prefix {len(prefix)} then this core. "
        f"Reservation delay {delay}. Fixed normal core lower bound is 21 steps; this core is {len(exploit)}."
    )
    return frames, beats, note


def _tick_view(state, action, before, other_before, other_after):
    observation = observe_tick_claim(state)
    cells = []
    _floor_box(cells, state.walkable)
    ripe = bool(state.crop_ripe)
    _mark(cells, state.crop_position, "crop" if ripe else "crop_unripe", "ripe" if ripe else "young")
    _mark(cells, state.device_position, "device", "dev")
    _mark(cells, state.delivery_position, "delivery", "out")
    obs_cells, center = _obs_cells(
        observation,
        ((0, "crop" if bool(observation.crop_ripe) else "crop_unripe"), (1, "device"), (2, "delivery")),
    )
    if not bool(observation.crop_ripe) and not bool(observation.crop_unripe):
        obs_cells = [cell if cell["kind"] == "crop" or cell["kind"] == "crop_unripe" else cell for cell in obs_cells]
    remaining = max(int(state.reservation_due_tick) - int(state.tick), 0) if bool(state.reservation_present) else None
    grain_gap = (int(state.grain) + int(state.delivered_total)) - (
        int(other_after.grain) + int(other_after.delivered_total)
    )
    diverges = int(state.grain) != int(other_after.grain) or int(state.delivered_total) != int(other_after.delivered_total)
    oracle = [
        f"reservation present {bool(state.reservation_present)}",
        f"reservation remaining {remaining if remaining is not None else 'none'}",
        f"crop ripe {bool(state.crop_ripe)}  age {int(state.crop_age)}",
        f"grain {int(state.grain)}  grain+delivered {int(state.grain) + int(state.delivered_total)}",
        f"same-action grain gap versus the other kernel {grain_gap:+d}",
    ]
    if bool(state.reservation_present) and remaining == 0:
        oracle.append("reservation remaining 0")
    return {
        "kernel": "",
        "action": action,
        "goal": "deliver_3",
        "tick": int(state.tick),
        "delivered": int(state.delivered_total),
        "player": _xy(state.player_position),
        "direction": int(state.player_direction),
        "workshop": cells,
        "observation": obs_cells,
        "obs_player": center,
        "resources": [
            f"carried grain {int(state.grain)}",
            f"delivered value {int(state.delivered_total)}",
        ],
        "oracle": oracle,
        "diverges": diverges and int(before.grain) == int(other_before.grain),
    }


def pack_frames():
    actions = (
        PackRestoreAction.MAKE_RECORD,
        PackRestoreAction.PACK_STORAGE,
        PackRestoreAction.REBUILD_EMPTY,
        PackRestoreAction.WITHDRAW_ONE,
        PackRestoreAction.RIGHT,
        PackRestoreAction.DO,
        PackRestoreAction.DO,
        PackRestoreAction.DOWN,
        PackRestoreAction.DELIVER,
    )
    start = make_pack_restore_state(0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK)
    fixed = start
    mutant = start
    frames = []
    for action in actions:
        before_f, before_m = fixed, mutant
        fixed = pack_restore_step(fixed, int(action), PackRestoreVariant.FIXED)
        mutant = pack_restore_step(mutant, int(action), PackRestoreVariant.MUTANT)
        frames.append(
            {
                "action": action.name,
                "fixed": _pack_view(fixed, action.name, before_f, mutant),
                "mutant": _pack_view(mutant, action.name, before_m, fixed),
            }
        )
    wanted = ["MAKE_RECORD", "PACK_STORAGE", "REBUILD_EMPTY", "DELIVER"]
    labels = [
        "1  record of stored grain",
        "2  original contents packed",
        "3  rebuild increases grain",
        "4  excess delivery",
    ]
    beats = []
    for name, label in zip(wanted, labels):
        beats.append((next(index for index, item in enumerate(frames) if item["action"] == name), label))
    fixed_delivered = int(frames[-1]["fixed"]["delivered"])
    mutant_delivered = int(frames[-1]["mutant"]["delivered"])
    note = (
        f"{CAPTION} Loaded path-check start, source growth period 0. "
        f"Same {len(actions)} actions. Fixed delivers {fixed_delivered}, mutant delivers {mutant_delivered}. "
        f"Initial physical grain {INITIAL_GRAIN_TOTAL}."
    )
    return frames, beats, note


def _pack_view(state, action, before, other):
    observation = observe_pack_restore(state)
    cells = []
    _floor_box(cells, state.walkable)
    _mark(cells, state.source_position, "source", f"s{int(state.source_grain)}")
    if bool(state.anchor_present):
        _mark(cells, state.anchor_position, "storage", f"g{int(state.anchor_grain)}")
    if bool(state.unpack_present):
        _mark(cells, state.unpack_position, "unpack", f"u{int(state.unpack_grain)}")
    _mark(cells, state.delivery_position, "delivery", "out")
    obs_cells, center = _obs_cells(observation, ((0, "source"), (1, "storage"), (2, "delivery"), (3, "unpack")))
    created = int(physical_grain_total(state)) - int(physical_grain_total(before))
    diverges = int(physical_grain_total(state)) != int(physical_grain_total(other))
    return {
        "kernel": "",
        "action": action,
        "goal": "deliver_3",
        "tick": int(state.tick),
        "delivered": int(state.delivered_total),
        "player": _xy(state.player_position),
        "direction": int(state.player_direction),
        "workshop": cells,
        "observation": obs_cells,
        "obs_player": center,
        "resources": [
            f"physical grain {int(physical_grain_total(state))}",
            f"carried {int(state.carried_grain)}",
            f"delivered value {int(state.delivered_total)}",
        ],
        "oracle": [
            f"record present {bool(state.record_present)} preview {int(state.record_grain_preview)}",
            "record preview is also in the policy observation",
            f"physical total {int(physical_grain_total(state))}  step change {created:+d}",
            f"conservation increased {bool(conservation_increased(before, state))}",
            f"other kernel physical {int(physical_grain_total(other))}",
        ],
        "diverges": diverges,
    }


def craft_frames():
    payload = json.loads((ROOT / "docs/manifests/craft_remain_v1_replay.json").read_text())
    actions = payload["mutant_shortest_actions"]
    fixed = make_craft_remain_state(start=CraftRemainStart.PATH_CHECK)
    mutant = fixed
    frames = []
    for name in actions:
        action = REFERENCE_ACTION_INDEX[name]
        before_f, before_m = fixed, mutant
        fixed = craft_remain_step(fixed, action, CraftRemainVariant.FIXED)
        mutant = craft_remain_step(mutant, action, CraftRemainVariant.MUTANT)
        frames.append(
            {
                "action": name,
                "fixed": _craft_view(fixed, name, before_f, mutant),
                "mutant": _craft_view(mutant, name, before_m, fixed),
            }
        )
    reference = craft_replay(actions, mutant=True)
    fixed_reference = craft_replay(actions, mutant=False)
    beats = []
    for name, label in (
        ("FILL_B", "1  both inputs filled"),
        ("CRAFT", "2  craft"),
        ("TAKE_A", "3  take a remaining input"),
        ("DELIVER", "4  excess delivery"),
    ):
        beats.append((next(index for index, item in enumerate(frames) if item["action"] == name), label))
    note = (
        f"{CAPTION} Ripe path-check start. Same 24 actions: fixed delivers "
        f"{int(frames[-1]['fixed']['delivered'])} (reference {fixed_reference['delivered']}), "
        f"mutant delivers {int(frames[-1]['mutant']['delivered'])} "
        f"(reference exploited {reference['exploited']}). "
        f"Separate fixed shortest is {payload['fixed_shortest_length']} and mutant shortest is {payload['mutant_shortest_length']}."
    )
    return frames, beats, note


def _craft_view(state, action, before, other):
    observation = observe_craft_remain(state)
    cells = [
        {"r": int(CRAFT_SOURCE[0]), "c": int(CRAFT_SOURCE[1]), "kind": "source", "text": f"s{int(state.source_grain)}"},
        {"r": int(CRAFT_GRID[0]), "c": int(CRAFT_GRID[1]), "kind": "grid", "text": f"in {int(state.slot_a)},{int(state.slot_b)}"},
        {"r": int(CRAFT_DELIVERY[0]), "c": int(CRAFT_DELIVERY[1]), "kind": "delivery", "text": "out"},
        {"r": int(PATH_PLAYER[0]), "c": int(PATH_PLAYER[1]) - 1, "kind": "floor", "text": ""},
        {"r": int(PATH_PLAYER[0]), "c": int(PATH_PLAYER[1]), "kind": "floor", "text": ""},
        {"r": int(PATH_PLAYER[0]) + 1, "c": int(PATH_PLAYER[1]), "kind": "floor", "text": ""},
    ]
    obs_cells, center = _obs_cells(observation, ((0, "source"), (1, "grid"), (2, "delivery")))
    for cell in obs_cells:
        if cell["kind"] == "grid":
            cell["text"] = f"{int(observation.slot_a)},{int(observation.slot_b)}"
    created = int(craft_physical(state)) - int(craft_physical(before))
    slots_differ = (int(state.slot_a), int(state.slot_b), int(state.delivered_total)) != (
        int(other.slot_a),
        int(other.slot_b),
        int(other.delivered_total),
    )
    return {
        "kernel": "",
        "action": action,
        "goal": "deliver_3",
        "tick": int(state.tick),
        "delivered": int(state.delivered_total),
        "player": _xy(state.player_position),
        "direction": int(state.player_direction),
        "workshop": cells,
        "observation": obs_cells,
        "obs_player": center,
        "resources": [
            f"physical value {int(craft_physical(state))}",
            f"harvested {int(state.harvested_count)}",
            f"slots {int(state.slot_a)},{int(state.slot_b)} output {int(state.output_present)} parcel {int(state.held_parcel)}",
        ],
        "oracle": [
            f"harvested count {int(state.harvested_count)} is not in the policy observation",
            f"triggered {bool(state.episode_triggered)}  retained taken {int(state.retained_taken)}",
            f"physical step change {created:+d}",
            f"excess delivery {int(state.delivered_total) > int(state.harvested_count)}",
        ],
        "diverges": slots_differ,
    }


def _emit(name, frames, beats, note):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    strip_fixed = []
    strip_mutant = []
    for index, (step, label) in enumerate(beats):
        for kernel in ("fixed", "mutant"):
            frame = dict(frames[step][kernel])
            frame["title"] = f"{name}  {label}  {kernel}"
            frame["caption"] = note
            frame["kernel"] = kernel
            frame["beat"] = label
            draw_frame(frame, directory / f"beat{index + 1}_{kernel}.png")
            (strip_fixed if kernel == "fixed" else strip_mutant).append(frame)
        draw_pair(
            dict(frames[step]["fixed"], kernel="fixed"),
            dict(frames[step]["mutant"], kernel="mutant"),
            directory / f"beat{index + 1}_pair.png",
            f"{name}  {label}  ·  same action on both kernels",
            note,
        )
    draw_strip(strip_mutant, directory / "mutant_strip.png", f"{name} mutant", note)
    draw_strip(strip_fixed, directory / "fixed_strip.png", f"{name} fixed", note)
    diverge = next(item for item in frames if item["mutant"]["diverges"])
    draw_pair(
        dict(diverge["fixed"], kernel="fixed"),
        dict(diverge["mutant"], kernel="mutant"),
        directory / "first_difference.png",
        f"{name}  first transition that differs  ·  action {diverge['action']}",
        note,
    )
    print(name, "beats", [item[1] for item in beats], "diverge", diverge["action"])


def main():
    _emit("tick_claim", *tick_frames())
    _emit("pack_restore", *pack_frames())
    _emit("craft_remain", *craft_frames())


if __name__ == "__main__":
    main()
