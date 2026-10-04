"""Render a recorded PACK-RESTORE trace as pixel-art PNG/GIF/MP4.

This module intentionally does not import the environment. It only reads the
recorded JSON and holds exact states between transitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/ext_csv/HackRL")
TRACE = ROOT / "runs/figures_report_v2/pack_dual_seed20/trace.json"
OUT = ROOT / "runs/figures_report_v2/pack_dual_seed20"

SCALE = 2
BASE_W = 960
PRESENTATION_H = 360
ANALYSIS_H = 500
TILE = 42
MAP_MIN = 7
MAP_MAX = 11

PALETTE = {
    "ink": "#2b2118",
    "paper": "#f5ead4",
    "panel": "#ead8b8",
    "panel_dark": "#c6a978",
    "stone_a": "#d8c8a7",
    "stone_b": "#cbb894",
    "wall": "#55483b",
    "wall_light": "#746252",
    "shadow": "#806a52",
    "fixed": "#3f6d8a",
    "mutant": "#44745d",
    "violation": "#ad3f35",
    "wheat": "#d89b2b",
    "wheat_light": "#f0c65d",
    "green": "#6c8c45",
    "wood": "#8b5a32",
    "wood_light": "#b87942",
    "wood_dark": "#543621",
    "metal": "#8b9693",
    "blueprint": "#427f8e",
    "shirt": "#2e6f79",
    "skin": "#d9a36f",
    "hat": "#c78a35",
    "white": "#fff8e8",
    "black": "#191512",
}


def _font(size, bold=False):
    # Pillow's bundled bitmap font keeps the output portable and pixel-like.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text(draw, xy, text, size=12, fill=None, anchor=None):
    draw.text(
        xy,
        str(text),
        font=_font(size),
        fill=fill or PALETTE["ink"],
        anchor=anchor,
    )


def _box(draw, box, fill, outline=None, width=1):
    draw.rectangle(box, fill=fill, outline=outline, width=width)


def _line(draw, points, fill, width=1):
    draw.line(points, fill=fill, width=width)


def _wheat(draw, cx, cy, scale=1):
    gold = PALETTE["wheat"]
    light = PALETTE["wheat_light"]
    green = PALETTE["green"]
    _line(draw, [(cx, cy + 8 * scale), (cx, cy - 7 * scale)], green, 2 * scale)
    for offset in (-5, -1, 3):
        _box(
            draw,
            (
                cx - 4 * scale,
                cy + offset * scale,
                cx - 1 * scale,
                cy + (offset + 3) * scale,
            ),
            gold,
        )
        _box(
            draw,
            (
                cx + 1 * scale,
                cy + (offset - 1) * scale,
                cx + 4 * scale,
                cy + (offset + 2) * scale,
            ),
            light,
        )


def _stone_tile(draw, x, y, row, col):
    shade = PALETTE["stone_a"] if (row + col) % 2 == 0 else PALETTE["stone_b"]
    _box(draw, (x, y, x + TILE - 1, y + TILE - 1), shade, PALETTE["shadow"])
    crack = (row * 13 + col * 7) % 3
    if crack == 0:
        _line(
            draw,
            [(x + 6, y + 12), (x + 11, y + 15), (x + 9, y + 20)],
            "#ad9875",
        )
    elif crack == 1:
        _line(
            draw,
            [(x + 29, y + 28), (x + 34, y + 25), (x + 37, y + 29)],
            "#ad9875",
        )


def _wall_tile(draw, x, y, row, col):
    _box(
        draw,
        (x, y, x + TILE - 1, y + TILE - 1),
        PALETTE["wall"],
        PALETTE["black"],
    )
    offset = 0 if row % 2 == 0 else TILE // 2
    _line(draw, [(x, y + TILE // 2), (x + TILE, y + TILE // 2)], PALETTE["wall_light"])
    _line(
        draw,
        [(x + (offset % TILE), y), (x + (offset % TILE), y + TILE // 2)],
        PALETTE["wall_light"],
    )


def _chest(draw, x, y, grain):
    _box(draw, (x + 7, y + 15, x + 35, y + 34), PALETTE["shadow"])
    _box(
        draw,
        (x + 5, y + 10, x + 35, y + 30),
        PALETTE["wood"],
        PALETTE["wood_dark"],
        2,
    )
    _box(
        draw,
        (x + 4, y + 7, x + 36, y + 15),
        PALETTE["wood_light"],
        PALETTE["wood_dark"],
        2,
    )
    _box(draw, (x + 17, y + 14, x + 22, y + 22), PALETTE["metal"])
    for index in range(min(int(grain), 3)):
        _wheat(draw, x + 11 + index * 9, y + 21, 1)


def _empty_storage_pad(draw, x, y):
    _box(draw, (x + 7, y + 24, x + 35, y + 31), PALETTE["metal"], PALETTE["ink"])
    for px in (10, 31):
        _box(draw, (x + px, y + 21, x + px + 3, y + 25), PALETTE["black"])


def _source(draw, x, y, grain, ripe):
    _box(draw, (x + 5, y + 27, x + 37, y + 35), "#85663e")
    if grain > 0:
        color = PALETTE["wheat"] if ripe else PALETTE["green"]
        for dx in (11, 21, 31):
            _line(draw, [(x + dx, y + 29), (x + dx, y + 12)], color, 2)
            if ripe:
                _box(draw, (x + dx - 3, y + 9, x + dx + 3, y + 16), PALETTE["wheat_light"])


def _delivery(draw, x, y):
    _box(draw, (x + 5, y + 11, x + 37, y + 35), PALETTE["wood"], PALETTE["wood_dark"], 2)
    _box(draw, (x + 8, y + 7, x + 34, y + 14), PALETTE["wood_light"], PALETTE["wood_dark"])
    _wheat(draw, x + 21, y + 23, 1)


def _unpack(draw, x, y, present, grain):
    _box(draw, (x + 5, y + 27, x + 37, y + 34), PALETTE["metal"], PALETTE["ink"])
    if present:
        _box(draw, (x + 8, y + 12, x + 34, y + 29), PALETTE["wood_light"], PALETTE["wood_dark"], 2)
        for index in range(min(int(grain), 3)):
            _wheat(draw, x + 13 + index * 8, y + 20, 1)


def _worker(draw, x, y, direction):
    # Shadow and body.
    _box(draw, (x + 12, y + 31, x + 31, y + 35), PALETTE["shadow"])
    _box(draw, (x + 14, y + 19, x + 29, y + 31), PALETTE["shirt"], PALETTE["black"])
    _box(draw, (x + 16, y + 10, x + 27, y + 21), PALETTE["skin"], PALETTE["black"])
    _box(draw, (x + 12, y + 7, x + 31, y + 12), PALETTE["hat"], PALETTE["wood_dark"])
    _box(draw, (x + 16, y + 4, x + 27, y + 9), PALETTE["hat"], PALETTE["wood_dark"])
    offsets = {1: (-11, 0), 2: (11, 0), 3: (0, -11), 4: (0, 11)}
    dx, dy = offsets.get(int(direction), (0, 0))
    _line(
        draw,
        [(x + 21, y + 20), (x + 21 + dx, y + 20 + dy)],
        PALETTE["white"],
        2,
    )
    _box(
        draw,
        (
            x + 19 + dx,
            y + 18 + dy,
            x + 23 + dx,
            y + 22 + dy,
        ),
        PALETTE["white"],
    )


def _record_icon(draw, x, y, preview, active):
    color = PALETTE["blueprint"] if active else "#9b917f"
    _box(draw, (x, y, x + 34, y + 30), PALETTE["paper"], color, 2)
    _line(draw, [(x + 7, y + 9), (x + 27, y + 9)], color, 2)
    _line(draw, [(x + 7, y + 15), (x + 23, y + 15)], color, 2)
    if active:
        _wheat(draw, x + 17, y + 22, 1)
    _text(draw, (x + 38, y + 15), f"record {preview if active else '-'}", 10, anchor="lm")


def _packed_icon(draw, x, y, grain, active):
    if not active:
        _box(draw, (x, y, x + 31, y + 27), "#b7aa92", "#8f826d")
        _text(draw, (x + 38, y + 14), "packed none", 10, anchor="lm")
        return
    _box(draw, (x, y, x + 31, y + 27), PALETTE["wood_light"], PALETTE["wood_dark"], 2)
    _line(draw, [(x + 15, y), (x + 15, y + 27)], PALETTE["metal"], 3)
    _wheat(draw, x + 8, y + 14, 1)
    _text(draw, (x + 38, y + 14), f"packed {grain}", 10, anchor="lm")


def _state_at(kernel, timeline_index):
    steps = kernel["steps"]
    if timeline_index == 0:
        return kernel["initial"], None, False
    if timeline_index <= len(steps):
        step = steps[timeline_index - 1]
        return step["after"], step, False
    return steps[-1]["after"], steps[-1], True


def _map_origin(panel_x):
    return panel_x + 18, 58


def _draw_map(draw, panel_x, state, observation, analysis):
    origin_x, origin_y = _map_origin(panel_x)
    for row in range(MAP_MIN - 1, MAP_MAX + 2):
        for col in range(MAP_MIN - 1, MAP_MAX + 2):
            x = origin_x + (col - (MAP_MIN - 1)) * TILE
            y = origin_y + (row - (MAP_MIN - 1)) * TILE
            if MAP_MIN <= row <= MAP_MAX and MAP_MIN <= col <= MAP_MAX:
                _stone_tile(draw, x, y, row, col)
            else:
                _wall_tile(draw, x, y, row, col)

    def at(position):
        row, col = position
        return (
            origin_x + (int(col) - (MAP_MIN - 1)) * TILE,
            origin_y + (int(row) - (MAP_MIN - 1)) * TILE,
        )

    x, y = at(state["anchor_position"])
    if state["anchor_present"]:
        _chest(draw, x, y, state["anchor_grain"])
    else:
        _empty_storage_pad(draw, x, y)

    x, y = at(state["source_position"])
    ripe = (
        state["source_growth_period"] == 0
        or state["source_age"] >= state["source_growth_period"]
    )
    _source(draw, x, y, state["source_grain"], ripe)

    x, y = at(state["delivery_position"])
    _delivery(draw, x, y)

    x, y = at(state["unpack_position"])
    _unpack(draw, x, y, state["unpack_present"], state["unpack_grain"])

    x, y = at(state["player_position"])
    _worker(draw, x, y, state["player_direction"])

    if analysis:
        # Exact 7x9 view centered on the player. This is a view boundary only;
        # the detailed visible values come from the stored observation.
        player_row, player_col = state["player_position"]
        top = max(int(player_row) - 3, MAP_MIN - 1)
        bottom = min(int(player_row) + 3, MAP_MAX + 1)
        left = max(int(player_col) - 4, MAP_MIN - 1)
        right = min(int(player_col) + 4, MAP_MAX + 1)
        x0 = origin_x + (left - (MAP_MIN - 1)) * TILE
        y0 = origin_y + (top - (MAP_MIN - 1)) * TILE
        x1 = origin_x + (right - (MAP_MIN - 1) + 1) * TILE - 1
        y1 = origin_y + (bottom - (MAP_MIN - 1) + 1) * TILE - 1
        _box(draw, (x0, y0, x1, y1), None, PALETTE["wheat_light"], 2)
        _text(draw, (x0 + 4, y0 + 3), "policy view", 8, PALETTE["ink"])


def _event_label(step):
    if step is None:
        return None
    events = step["events"]
    if events["conservation_violation"]:
        return "CONSERVATION +1"
    if events["violation_delivery"]:
        return "DUPLICATED GRAIN DELIVERED"
    if events["record_created"]:
        return "RECORD CAPTURED"
    if events["storage_packed"]:
        return (
            "EMPTY STORAGE SHELL PACKED"
            if events.get("empty_frame_created")
            else "STORAGE PACKED"
        )
    if events["storage_rebuilt"]:
        return "STORAGE REBUILT"
    if events["goal_success"]:
        return "GOAL COMPLETE"
    return None


def _draw_panel(
    draw,
    panel_x,
    panel_w,
    state,
    step,
    held,
    kernel_name,
    analysis,
    height,
):
    accent = PALETTE["fixed"] if kernel_name == "fixed" else PALETTE["mutant"]
    violation = bool(step and step["events"]["conservation_violation"] and not held)
    border = PALETTE["violation"] if violation else accent
    _box(
        draw,
        (panel_x + 4, 4, panel_x + panel_w - 5, height - 5),
        PALETTE["paper"],
        border,
        3,
    )
    _box(
        draw,
        (panel_x + 7, 7, panel_x + panel_w - 8, 46),
        accent,
    )
    _text(draw, (panel_x + 18, 25), kernel_name.upper(), 18, PALETTE["white"], "lm")
    _text(
        draw,
        (panel_x + 92, 25),
        "PACK · Dual s20 · u4096 · positive 4/5",
        8,
        PALETTE["white"],
        "lm",
    )
    status = (
        f"step {state['tick']:02d}"
        if not held
        else f"held after success at {state['tick']:02d}"
    )
    _text(draw, (panel_x + panel_w - 18, 25), status, 11, PALETTE["white"], "rm")

    observation = step["observation_after"] if step else None
    _draw_map(draw, panel_x, state, observation, analysis)

    hud_x = panel_x + 329
    _text(draw, (hud_x, 66), "DELIVER", 10)
    delivered = int(state["delivered_total"])
    for index in range(3):
        color = PALETTE["wheat"] if index < delivered else "#d1c5ae"
        _box(draw, (hud_x + index * 30, 83, hud_x + 22 + index * 30, 105), color, PALETTE["ink"])
        if index < delivered:
            _wheat(draw, hud_x + 11 + index * 30, 94, 1)
    _text(draw, (hud_x + 98, 94), f"{delivered}/3", 14, anchor="lm")

    _text(draw, (hud_x, 124), "INVENTORY", 10)
    _wheat(draw, hud_x + 13, 150, 1)
    _text(draw, (hud_x + 31, 150), f"x {state['carried_grain']}", 13, anchor="lm")

    _record_icon(
        draw,
        hud_x,
        176,
        state["record_grain_preview"],
        state["record_present"],
    )
    _packed_icon(
        draw,
        hud_x,
        218,
        state["packed_grain"],
        state["packed_present"],
    )
    _text(
        draw,
        (hud_x, 263),
        f"empty frames  {state['empty_frames']}",
        10,
    )

    action = "START" if step is None else step["action"]
    if held:
        action = "HELD  ·  episode already complete"
    _box(
        draw,
        (panel_x + 18, 309, panel_x + panel_w - 18, 339),
        PALETTE["panel"],
        PALETTE["panel_dark"],
    )
    _text(draw, (panel_x + 30, 324), action, 12, anchor="lm")
    label = _event_label(step)
    if label and not held:
        badge_fill = PALETTE["violation"] if violation else accent
        _box(
            draw,
            (panel_x + 185, 278, panel_x + panel_w - 18, 303),
            badge_fill,
            PALETTE["black"],
        )
        _text(
            draw,
            (panel_x + panel_w - 26, 290),
            label,
            9,
            PALETTE["white"],
            "rm",
        )

    if not analysis:
        return

    section_y = 355
    _box(
        draw,
        (panel_x + 18, section_y, panel_x + panel_w - 18, height - 18),
        "#e5d7bd",
        PALETTE["panel_dark"],
    )
    _text(draw, (panel_x + 28, section_y + 16), "ANALYSIS  ·  NOT POLICY INPUT", 10, PALETTE["violation"] if violation else PALETTE["ink"])
    physical = (
        int(state["source_grain"])
        + int(state["carried_grain"])
        + int(state["anchor_grain"])
        + int(state["unpack_grain"])
        + int(state["packed_grain"])
        + int(state["delivered_total"])
    )
    formula = (
        f"physical = source {state['source_grain']} + carried {state['carried_grain']} "
        f"+ stores {int(state['anchor_grain']) + int(state['unpack_grain'])} "
        f"+ packed {state['packed_grain']} + delivered {state['delivered_total']} = {physical}"
    )
    _text(draw, (panel_x + 28, section_y + 38), formula, 9)
    if step is not None:
        events = step["events"]
        _text(
            draw,
            (panel_x + 28, section_y + 59),
            (
                f"action p={step['policy']['argmax_probability']:.3f}   "
                f"value={step['policy']['value']:.3f}   "
                f"physical created={events['physical_created']:+d}"
            ),
            9,
        )
        _text(
            draw,
            (panel_x + 28, section_y + 80),
            (
                f"record preview={state['record_grain_preview']}   "
                f"source age={state['source_age']}/{state['source_growth_period']}   "
                f"violation delivery={events['violation_delivery']}"
            ),
            9,
        )
        if violation:
            _text(
                draw,
                (panel_x + 28, section_y + 103),
                "The recorded grain was withdrawn; reconstruction adds another grain.",
                10,
                PALETTE["violation"],
            )


def _render(trace, index, analysis):
    height = ANALYSIS_H if analysis else PRESENTATION_H
    canvas = Image.new("RGB", (BASE_W, height), PALETTE["paper"])
    draw = ImageDraw.Draw(canvas)
    panel_w = BASE_W // 2
    fixed_state, fixed_step, fixed_held = _state_at(
        trace["kernels"]["fixed"], index
    )
    mutant_state, mutant_step, mutant_held = _state_at(
        trace["kernels"]["mutant"], index
    )
    _draw_panel(
        draw,
        0,
        panel_w,
        fixed_state,
        fixed_step,
        fixed_held,
        "fixed",
        analysis,
        height,
    )
    _draw_panel(
        draw,
        panel_w,
        panel_w,
        mutant_state,
        mutant_step,
        mutant_held,
        "mutant",
        analysis,
        height,
    )
    return canvas.resize(
        (BASE_W * SCALE, height * SCALE),
        Image.Resampling.NEAREST,
    )


def _write_video(frames, directory, stem, fps=2):
    arrays = [np.asarray(frame) for frame in frames]
    imageio.mimsave(directory / f"{stem}.gif", arrays, duration=500, loop=0)
    imageio.mimsave(
        directory / f"{stem}.mp4",
        arrays,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )


def main():
    trace = json.loads(TRACE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    frame_count = max(
        len(trace["kernels"]["fixed"]["steps"]),
        len(trace["kernels"]["mutant"]["steps"]),
    )
    presentation = []
    analysis = []
    for index in range(frame_count + 1):
        p = _render(trace, index, analysis=False)
        a = _render(trace, index, analysis=True)
        p.save(OUT / f"presentation_{index:03d}.png")
        a.save(OUT / f"analysis_{index:03d}.png")
        presentation.append(p)
        analysis.append(a)

    _write_video(presentation, OUT, "presentation")
    _write_video(analysis, OUT, "analysis")

    mutant_steps = trace["kernels"]["mutant"]["steps"]
    violation = next(
        item["step"]
        for item in mutant_steps
        if item["events"]["conservation_violation"]
    )
    record = next(
        item["step"] for item in mutant_steps if item["events"]["record_created"]
    )
    packed = next(
        item["step"] for item in mutant_steps if item["events"]["storage_packed"]
    )
    final = frame_count
    for label, index in (
        ("initial", 0),
        ("record", record),
        ("packed", packed),
        ("violation", violation),
        ("final", final),
    ):
        presentation[index].save(OUT / f"key_{label}.png")
        analysis[index].save(OUT / f"key_{label}_analysis.png")
    print(
        OUT,
        "frames",
        frame_count + 1,
        "record",
        record,
        "packed",
        packed,
        "violation",
        violation,
        flush=True,
    )


if __name__ == "__main__":
    main()
