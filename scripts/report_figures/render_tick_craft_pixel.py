"""Render recorded TICK-CLAIM and CRAFT-REMAIN traces in one pixel-art style."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from report_figures.render_pack_pixel import (
    ANALYSIS_H,
    FOOTER_H,
    _draw_footer,
    _draw_view_bounds,
    _map_origin,
    BASE_W,
    PALETTE,
    PRESENTATION_H,
    SCALE,
    TILE,
    _box,
    _font,
    _fit_text,
    _line,
    _stone_tile,
    _text,
    _wall_tile,
    _wheat,
    _worker,
)

from report_figures.common import ROOT, state_at as _state_at, frame_status, world_geometry
SPECS = {
    "tick": {
        "trace": ROOT / "runs/figures_report_v2/tick_dual_seed20/trace.json",
        "output": ROOT / "runs/figures_report_v2/tick_dual_seed20",
    },
    "craft": {
        "trace": ROOT / "runs/figures_report_v2/craft_dual_seed21/trace.json",
        "output": ROOT / "runs/figures_report_v2/craft_dual_seed21",
    },
}


def _delivery_hud(draw, x, delivered):
    _text(draw, (x, 66), "DELIVER", 10)
    for index in range(3):
        color = PALETTE["wheat"] if index < delivered else "#d1c5ae"
        _box(draw, (x + index * 30, 83, x + 22 + index * 30, 105), color, PALETTE["ink"])
        if index < delivered:
            _wheat(draw, x + 11 + index * 30, 94, 1)
    _text(draw, (x + 98, 94), f"{delivered}/3", 14, anchor="lm")


def _crop(draw, x, y, ripe, available=True):
    _box(draw, (x + 5, y + 28, x + 37, y + 35), "#85663e")
    if not available:
        for dx in (10, 22, 34):
            _line(draw, [(x + dx, y + 31), (x + dx, y + 25)], "#827257", 2)
        return
    stem = PALETTE["wheat"] if ripe else PALETTE["green"]
    for dx in (10, 20, 30):
        _line(draw, [(x + dx, y + 29), (x + dx, y + 12)], stem, 2)
        if ripe:
            _box(draw, (x + dx - 3, y + 8, x + dx + 3, y + 16), PALETTE["wheat_light"])
        else:
            _line(draw, [(x + dx, y + 18), (x + dx - 5, y + 15)], "#7ea655", 2)


def _delivery_crate(draw, x, y):
    _box(draw, (x + 5, y + 11, x + 37, y + 35), PALETTE["wood"], PALETTE["wood_dark"], 2)
    _box(draw, (x + 8, y + 7, x + 34, y + 14), PALETTE["wood_light"], PALETTE["wood_dark"])
    _wheat(draw, x + 21, y + 23, 1)


def _clock_machine(draw, x, y, state):
    _box(draw, (x + 6, y + 10, x + 36, y + 35), "#6e567d", PALETTE["black"], 2)
    _box(draw, (x + 10, y + 5, x + 32, y + 12), PALETTE["metal"], PALETTE["black"])
    _box(draw, (x + 12, y + 14, x + 30, y + 31), PALETTE["paper"], PALETTE["black"], 2)
    cx, cy = x + 21, y + 22
    _line(draw, [(cx, cy), (cx, cy - 6)], PALETTE["violation"] if state["reservation_present"] else PALETTE["ink"], 2)
    if state["reservation_present"]:
        remaining = max(int(state["reservation_due_tick"]) - int(state["tick"]), 0)
        _box(draw, (x + 31, y + 7, x + 39, y + 15), PALETTE["wheat"], PALETTE["black"])
        _text(draw, (x + 35, y + 11), remaining, 7, anchor="mm")


def _workbench(draw, x, y, state):
    _box(draw, (x + 3, y + 8, x + 39, y + 34), PALETTE["wood"], PALETTE["wood_dark"], 2)
    for index, present in enumerate((state["slot_a"], state["slot_b"])):
        sx = x + 7 + index * 13
        _box(draw, (sx, y + 11, sx + 10, y + 22), "#ead8b8", PALETTE["wood_dark"])
        if present:
            _wheat(draw, sx + 5, y + 17, 1)
    _box(draw, (x + 30, y + 10, x + 38, y + 23), "#d8c8a7", PALETTE["wood_dark"])
    if state["output_present"]:
        _box(draw, (x + 31, y + 13, x + 37, y + 20), "#b54c38", PALETTE["black"])
    _line(draw, [(x + 8, y + 34), (x + 8, y + 39)], PALETTE["wood_dark"], 3)
    _line(draw, [(x + 34, y + 34), (x + 34, y + 39)], PALETTE["wood_dark"], 3)


def _parcel(draw, x, y, active):
    _text(draw, (x, y), "PARCEL", 10)
    if active:
        _box(draw, (x, y + 17, x + 31, y + 42), "#b54c38", PALETTE["black"], 2)
        _line(draw, [(x + 15, y + 17), (x + 15, y + 42)], PALETTE["wheat_light"], 3)
        _text(draw, (x + 38, y + 29), "value 2", 10, anchor="lm")
    else:
        _box(draw, (x, y + 17, x + 31, y + 42), "#b7aa92", "#8f826d")
        _text(draw, (x + 38, y + 29), "none", 10, anchor="lm")


def _bbox(trace, environment):
    return world_geometry(trace, environment)[1]


def _map_position(panel_x, bbox, position):
    top, _bottom, left, _right = bbox
    row, col = position
    origin_x, origin_y = _map_origin(panel_x, bbox)
    return origin_x + (int(col) - left) * TILE, origin_y + (int(row) - top) * TILE


def _draw_base_map(draw, panel_x, trace, environment):
    bbox = _bbox(trace, environment)
    top, bottom, left, right = bbox
    walkable, _ = world_geometry(trace, environment)
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            x, y = _map_position(panel_x, bbox, (row, col))
            floor = (row, col) in walkable
            if floor:
                _stone_tile(draw, x, y, row, col)
            else:
                _wall_tile(draw, x, y, row, col)
    return bbox


def _draw_tick_map(draw, panel_x, trace, state, observation, analysis):
    bbox = _draw_base_map(draw, panel_x, trace, "tick")
    x, y = _map_position(panel_x, bbox, state["crop_position"])
    _crop(draw, x, y, bool(state["crop_ripe"]), True)
    x, y = _map_position(panel_x, bbox, state["device_position"])
    _clock_machine(draw, x, y, state)
    x, y = _map_position(panel_x, bbox, state["delivery_position"])
    _delivery_crate(draw, x, y)
    x, y = _map_position(panel_x, bbox, state["player_position"])
    _worker(draw, x, y, state["player_direction"])
    if analysis:
        _draw_view_bounds(draw, panel_x, state, observation, bbox)


def _draw_craft_map(draw, panel_x, trace, state, observation, analysis):
    bbox = _draw_base_map(draw, panel_x, trace, "craft")
    x, y = _map_position(panel_x, bbox, trace["geometry"]["source"])
    _crop(draw, x, y, bool(state["source_grain"]), bool(state["source_grain"]))
    x, y = _map_position(panel_x, bbox, trace["geometry"]["grid"])
    _workbench(draw, x, y, state)
    x, y = _map_position(panel_x, bbox, trace["geometry"]["delivery"])
    _delivery_crate(draw, x, y)
    x, y = _map_position(panel_x, bbox, state["player_position"])
    _worker(draw, x, y, state["player_direction"])
    if analysis:
        _draw_view_bounds(draw, panel_x, state, observation, bbox)


def _tick_event(step):
    if step is None:
        return None
    events = step["events"]
    if events["conservation_violation"]:
        return "MANUAL + RESERVED PAYOUT"
    if events["violation_delivery"]:
        return "EXCESS DELIVERY (ORACLE)"
    if events["goal_success"]:
        return "GOAL COMPLETE"
    if events["reservation_created"]:
        return "RESERVATION ARMED"
    if events["reservation_due_before_action"]:
        return "RESERVATION EXPIRES NOW"
    return None


def _craft_event(step):
    if step is None:
        return None
    events = step["events"]
    if events["conservation_violation_amount"] > 0:
        return "OUTPUT CREATED · INPUTS REMAIN"
    if events["excess_delivery"]:
        return "EXCESS DELIVERY"
    if events["retained_recovered"]:
        return "REMAINING INPUT RECOVERED"
    if events["output_taken"]:
        return "OUTPUT PARCEL TAKEN"
    if events["filled_b"]:
        return "TWO INPUT SLOTS FILLED"
    if events["filled_a"]:
        return "INPUT A FILLED"
    if events["goal_success"]:
        return "GOAL COMPLETE"
    return None


def _draw_header(draw, panel_x, panel_w, height, kernel_name, state, violation):
    accent = PALETTE["fixed"] if kernel_name == "fixed" else PALETTE["mutant"]
    border = PALETTE["violation"] if violation else accent
    _box(draw, (panel_x + 4, 4, panel_x + panel_w - 5, height - 5), PALETTE["paper"], border, 3)
    _box(draw, (panel_x + 7, 7, panel_x + panel_w - 8, 46), accent)
    _text(draw, (panel_x + 18, 25), kernel_name.upper(), 18, PALETTE["white"], "lm")
    return accent


def _draw_tick_panel(draw, panel_x, panel_w, trace, state, step, held, kernel_name, analysis, height):
    violation = bool(step and step["events"]["conservation_violation"] and not held)
    accent = _draw_header(draw, panel_x, panel_w, height, kernel_name, state, violation)
    policy = trace["policy"]
    _text(draw, (panel_x + 120, 25), f"TICK | {policy['method']} s{policy['seed']} | {trace.get('provenance', {}).get('action_selection', 'mode')}", 8, PALETTE["white"], "lm")
    status = frame_status(trace["kernels"][kernel_name], state, held)
    _text(draw, (panel_x + panel_w - 18, 25), status, 11, PALETTE["white"], "rm")
    observation = step["observation_after"] if step else trace["kernels"][kernel_name]["initial_observation"]
    _draw_tick_map(draw, panel_x, trace, state, observation, analysis)
    hud_x = panel_x + 329
    _delivery_hud(draw, hud_x, int(state["delivered_total"]))
    _text(draw, (hud_x, 124), "GRAIN", 10)
    _wheat(draw, hud_x + 13, 150, 1)
    _text(draw, (hud_x + 31, 150), f"x {state['grain']}", 13, anchor="lm")
    _text(draw, (hud_x, 180), "RESERVATION", 10)
    if state["reservation_present"]:
        remaining = max(int(state["reservation_due_tick"]) - int(state["tick"]), 0)
        _box(draw, (hud_x, 199, hud_x + 31, 229), "#6e567d", PALETTE["black"], 2)
        _text(draw, (hud_x + 15, 214), remaining, 13, PALETTE["white"], "mm")
        _text(draw, (hud_x + 39, 214), f"due in {remaining}", 10, anchor="lm")
    else:
        _box(draw, (hud_x, 199, hud_x + 31, 229), "#b7aa92", "#8f826d")
        _text(draw, (hud_x + 39, 214), "none", 10, anchor="lm")
    _text(draw, (hud_x, 244), "observer crop state", 10)
    _text(draw, (hud_x, 263), "ripe" if state["crop_ripe"] else "growing", 10)
    action = "START" if step is None else step["action"]
    if held:
        action = "HELD  |  no further transitions"
    _box(draw, (panel_x + 18, 389, panel_x + panel_w - 18, 419), PALETTE["panel"], PALETTE["panel_dark"])
    _text(draw, (panel_x + 30, 404), action, 12, anchor="lm")
    label = _tick_event(step)
    if label and not held:
        fill = PALETTE["violation"] if violation else accent
        _box(draw, (panel_x + 185, 357, panel_x + panel_w - 18, 382), fill, PALETTE["black"])
        _text(draw, (panel_x + panel_w - 26, 370), label, 9, PALETTE["white"], "rm")
    if not analysis:
        return
    y = 435
    _box(draw, (panel_x + 18, y, panel_x + panel_w - 18, height - 18), "#e5d7bd", PALETTE["panel_dark"])
    _text(draw, (panel_x + 28, y + 16), "ANALYSIS  ·  NOT POLICY INPUT", 10, PALETTE["violation"] if violation else PALETTE["ink"])
    _text(draw, (panel_x + 28, y + 39), f"physical = carried {state['grain']} + delivered {state['delivered_total']} = {state['physical_grain_total']}", 9)
    if step:
        event = step["events"]
        _text(draw, (panel_x + 28, y + 60), f"action p={step['policy']['argmax_probability']:.3f}   value={step['policy']['value']:.3f}   cycle payout={event['cycle_payout_total']}", 9)
        before = step["before"]
        _text(draw, (panel_x + 28, y + 81), f"pre-step reservation object/cycle={before['reservation_object_id']}/{before['reservation_cycle_id']}   violation created={event['violation_grain_created']}", 9)
        if violation:
            _text(draw, (panel_x + 28, y + 104), "One DO action receives both manual and scheduled payout.", 10, PALETTE["violation"])


def _draw_craft_panel(draw, panel_x, panel_w, trace, state, step, held, kernel_name, analysis, height):
    violation = bool(step and step["events"]["conservation_violation_amount"] > 0 and not held)
    accent = _draw_header(draw, panel_x, panel_w, height, kernel_name, state, violation)
    policy = trace["policy"]
    _text(draw, (panel_x + 120, 25), f"CRAFT | {policy['method']} s{policy['seed']} | {trace.get('provenance', {}).get('action_selection', 'mode')}", 8, PALETTE["white"], "lm")
    status = frame_status(trace["kernels"][kernel_name], state, held)
    _text(draw, (panel_x + panel_w - 18, 25), status, 11, PALETTE["white"], "rm")
    observation = step["observation_after"] if step else trace["kernels"][kernel_name]["initial_observation"]
    _draw_craft_map(draw, panel_x, trace, state, observation, analysis)
    hud_x = panel_x + 329
    _delivery_hud(draw, hud_x, int(state["delivered_total"]))
    _text(draw, (hud_x, 124), "RAW GRAIN", 10)
    _wheat(draw, hud_x + 13, 150, 1)
    _text(draw, (hud_x + 31, 150), f"x {state['carried_grain']}", 13, anchor="lm")
    _parcel(draw, hud_x, 170, bool(state["held_parcel"]))
    _text(draw, (hud_x, 226), f"input A  {state['slot_a']}", 10)
    _text(draw, (hud_x, 245), f"input B  {state['slot_b']}", 10)
    _text(draw, (hud_x, 264), f"output   {int(state['output_present'])}", 10)
    action = "START" if step is None else step["action"]
    if held:
        action = "HELD  |  no further transitions"
    _box(draw, (panel_x + 18, 389, panel_x + panel_w - 18, 419), PALETTE["panel"], PALETTE["panel_dark"])
    _text(draw, (panel_x + 30, 404), action, 12, anchor="lm")
    label = _craft_event(step)
    if label and not held:
        fill = PALETTE["violation"] if violation else accent
        _box(draw, (panel_x + 185, 357, panel_x + panel_w - 18, 382), fill, PALETTE["black"])
        _text(draw, (panel_x + panel_w - 26, 370), label, 9, PALETTE["white"], "rm")
    if not analysis:
        return
    y = 435
    _box(draw, (panel_x + 18, y, panel_x + panel_w - 18, height - 18), "#e5d7bd", PALETTE["panel_dark"])
    _text(draw, (panel_x + 28, y + 16), "ANALYSIS  ·  NOT POLICY INPUT", 10, PALETTE["violation"] if violation else PALETTE["ink"])
    formula = (
        f"physical = source {state['source_grain']} + carried {state['carried_grain']} "
        f"+ inputs {int(state['slot_a']) + int(state['slot_b'])} + output {2 * int(state['output_present'])} "
        f"+ held {2 * int(state['held_parcel'])} + delivered {state['delivered_total']} = {state['physical_total']}"
    )
    _fit_text(draw, (panel_x + 28, y + 39), formula, panel_w - 56)
    if step:
        event = step["events"]
        _text(draw, (panel_x + 28, y + 60), f"action p={step['policy']['argmax_probability']:.3f}   value={step['policy']['value']:.3f}   violation amount={event['conservation_violation_amount']:+d}", 9)
        _text(draw, (panel_x + 28, y + 81), f"harvested={state['harvested_count']}   trigger={state['episode_triggered']}   retained recovered={state['retained_taken']}", 9)
        if violation:
            _text(draw, (panel_x + 28, y + 104), "The parcel is created while both input slots remain filled.", 10, PALETTE["violation"])


def _render(trace, environment, index, analysis):
    height = ANALYSIS_H if analysis else PRESENTATION_H
    canvas = Image.new("RGB", (BASE_W, height + FOOTER_H), PALETTE["paper"])
    draw = ImageDraw.Draw(canvas)
    panel_w = BASE_W // 2
    for panel_index, kernel_name in enumerate(("fixed", "mutant")):
        state, step, held = _state_at(trace["kernels"][kernel_name], index)
        args = (
            draw,
            panel_index * panel_w,
            panel_w,
            trace,
            state,
            step,
            held,
            kernel_name,
            analysis,
            height,
        )
        if environment == "tick":
            _draw_tick_panel(*args)
        else:
            _draw_craft_panel(*args)
    _draw_footer(draw, trace, height)
    return canvas.resize((BASE_W * SCALE, (height + FOOTER_H) * SCALE), Image.Resampling.NEAREST)


def _write_video(frames, directory, stem, fps=1):
    arrays = [np.asarray(frame) for frame in frames]
    imageio.mimsave(directory / f"{stem}.gif", arrays, duration=1000 / fps, loop=0)
    imageio.mimsave(
        directory / f"{stem}.mp4",
        arrays,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )


def _key_steps(trace, environment, final):
    mutant = trace["kernels"]["mutant"]["steps"]
    wanted = (
        (("armed", "reservation_created"), ("violation", "conservation_violation"), ("delivery", "violation_delivery"))
        if environment == "tick" else
        (("inputs", "filled_b"), ("violation", "conservation_violation_amount"), ("recovered", "retained_recovered"))
    )
    keys = [("initial", 0)]
    for label, event in wanted:
        found = next((item["step"] for item in mutant if item["events"][event]), None)
        if found is not None:
            keys.append((label, found))
    return (*keys, ("final", final))


def render_environment(environment, trace_path=None, output=None, *, step_seconds=1., event_seconds=4.):
    spec = SPECS[environment]
    trace_path = Path(trace_path or spec["trace"])
    trace = json.loads(trace_path.read_text())
    output = Path(output) if output else trace_path.parent
    output.mkdir(parents=True, exist_ok=True)
    final = max(len(trace["kernels"]["fixed"]["steps"]), len(trace["kernels"]["mutant"]["steps"]))
    presentation = []
    analysis = []
    for index in range(final + 1):
        p = _render(trace, environment, index, False)
        a = _render(trace, environment, index, True)
        p.save(output / f"presentation_{index:03d}.png")
        a.save(output / f"analysis_{index:03d}.png")
        presentation.append(p)
        analysis.append(a)
    _write_video(presentation, output, "presentation")
    _write_video(analysis, output, "analysis")
    from report_figures.explainer import write_explainer
    timeline = write_explainer(trace, environment, presentation, output, step_seconds=step_seconds, event_seconds=event_seconds)
    print("explainer", timeline["duration_seconds"], "seconds", flush=True)
    for label, index in _key_steps(trace, environment, final):
        presentation[index].save(output / f"key_{label}.png")
        analysis[index].save(output / f"key_{label}_analysis.png")
    print(environment, output, "frames", final + 1, flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("tick", "craft"))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--event-seconds", type=float, default=4.0)
    args = parser.parse_args()
    if (args.trace or args.output) and not args.environment:
        parser.error("--trace/--output requires --environment")
    for environment in (args.environment,) if args.environment else ("tick", "craft"):
        render_environment(environment, args.trace, args.output, step_seconds=args.step_seconds, event_seconds=args.event_seconds)


if __name__ == "__main__":
    main()
