"""Event-paced explanation of recorded transitions, without simulating a world."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from report_figures.common import state_at, world_geometry

NAMES = {"pack": "PACK-RESTORE", "tick": "TICK-CLAIM", "craft": "CRAFT-REMAIN"}
LEGENDS = {
    "pack": (("A", "source_position", "grain source"), ("B", "anchor_position", "storage"), ("C", "unpack_position", "unpack site"), ("D", "delivery_position", "delivery")),
    "tick": (("A", "crop_position", "crop"), ("B", "device_position", "reservation timer"), ("C", "delivery_position", "delivery")),
    "craft": (("A", "source", "grain source"), ("B", "grid", "crafting table"), ("C", "delivery", "delivery")),
}
MOVE = {"LEFT": "Move left", "RIGHT": "Move right", "UP": "Move up", "DOWN": "Move down"}


def describe_step(environment, step, held=False):
    """Explain only observed events. A failed action is never a successful event."""
    if step is None:
        return {"title": "Starting state", "detail": "No action has been taken yet.", "focus": [], "important": False}
    if held:
        return {"title": "Episode ended", "detail": "This side is held still. No more environment transitions occur.", "focus": [], "important": False}
    events, before, after = step["events"], step["before"], step["after"]
    action = step["action"]
    total = "physical_total" if environment == "craft" else "physical_grain_total"
    change = f"Physical total: {before[total]} -> {after[total]}."
    def note(title, detail, *focus):
        return {"title": title, "detail": detail, "focus": list(focus), "important": True}
    if environment == "pack":
        if events.get("conservation_violation"):
            return note("Rebuilding creates extra grain", f"{change} +{events['physical_created']} grain without collecting it from the source.", "anchor_position")
        if events.get("record_created"):
            return note("Save the storage record", f"The record remembers {after['record_grain_preview']} grain. A record is metadata, not physical grain.", "anchor_position")
        if events.get("storage_packed"):
            storage = "anchor_position" if before["anchor_present"] and not after["anchor_present"] else "unpack_position"
            if events.get("empty_frame_created"):
                return note("Pack the empty storage shell", f"No grain is inside this shell. {change}", storage)
            return note("Pack the storage", f"Packed grain: {before['packed_grain']} -> {after['packed_grain']}. {change}", storage)
        if events.get("storage_rebuilt"):
            return note("Rebuild the storage", f"Storage is present again. {change}", "anchor_position")
        if action == "WITHDRAW_ONE" and after["carried_grain"] > before["carried_grain"]:
            storage = "anchor_position" if after["anchor_grain"] < before["anchor_grain"] else "unpack_position"
            return note("Withdraw one grain", f"Carried grain: {before['carried_grain']} -> {after['carried_grain']}. {change}", storage)
        if action == "STORE_ONE" and after["carried_grain"] < before["carried_grain"]:
            storage = "anchor_position" if after["anchor_grain"] > before["anchor_grain"] else "unpack_position"
            return note("Put one grain into storage", f"Carried grain: {before['carried_grain']} -> {after['carried_grain']}. {change}", storage)
        if action == "PLACE_PACKED" and not before["unpack_present"] and after["unpack_present"]:
            return note("Unpack the original grain", f"Grain at the unpack site: {after['unpack_grain']}. {change}", "unpack_position")
    elif environment == "tick":
        if events.get("conservation_violation"):
            return note("One crop cycle pays twice", f"Cycle payout: {events['cycle_payout_total']} grain; allowed: 1. Manual harvest and reservation payout overlap.", "crop_position", "device_position")
        if events.get("reservation_created"):
            return note("Arm a harvest reservation", f"The reservation is due at tick {after['reservation_due_tick']}. Watch the crop when the timer expires.", "device_position")
        if events.get("reservation_due_before_action"):
            return note("The reservation expires now", f"Payout on this transition: {events['created_amount']}. This cycle has paid {events['cycle_payout_total']} grain.", "crop_position", "device_position")
        if events.get("created_amount", 0) > 0:
            return note("Receive grain from the crop", f"Received {events['created_amount']} grain on this transition. Cycle payout: {events['cycle_payout_total']}.", "crop_position")
    elif environment == "craft":
        if events.get("conservation_violation_amount", 0) > 0:
            return note("A parcel appears; inputs remain", f"{change} Input A={after['slot_a']}, B={after['slot_b']}; an output parcel is also present.", "grid")
        if events.get("crafted"):
            return note("Craft a parcel worth two grain", f"Inputs after crafting: A={after['slot_a']}, B={after['slot_b']}. {change}", "grid")
        if events.get("retained_recovered"):
            return note("Recover a retained ingredient", f"Take an input left behind after crafting. Carried grain: {before['carried_grain']} -> {after['carried_grain']}.", "grid")
        if events.get("filled_a") or events.get("filled_b"):
            return note("Place an ingredient in the table", f"Input A={after['slot_a']}, input B={after['slot_b']}. Crafting needs both slots filled.", "grid")
        if events.get("output_taken"):
            return note("Take the finished parcel", "The carried parcel has delivery value 2.", "grid")
        if events.get("harvested"):
            return note("Harvest one grain", f"Carried grain: {before['carried_grain']} -> {after['carried_grain']}.", "source")
    if events.get("delivered", 0) > 0:
        extra = events.get("violation_grain_delivered", 0)
        detail = f"Delivered value: {before['delivered_total']} -> {after['delivered_total']} (goal: 3)."
        if extra:
            detail += f" Oracle attributes {extra} to the excess-grain balance."
        elif events.get("excess_delivery"):
            detail += " Delivered value exceeds harvested grain."
        return note("Deliver at the destination", detail, "delivery" if environment == "craft" else "delivery_position")
    if environment == "pack" and action == "DO" and after["carried_grain"] > before["carried_grain"]:
        return note("Collect grain from the source", f"Carried grain: {before['carried_grain']} -> {after['carried_grain']}. {change}", "source_position")
    if action in MOVE:
        moved = before["player_position"] != after["player_position"]
        return {"title": MOVE[action] if moved else "Try moving; path is blocked", "detail": "White marker shows the facing direction. No harvest or delivery event is recorded on this step.", "focus": [], "important": False}
    if action == "NOOP":
        return {"title": "Wait one environment tick", "detail": "The world clock advances. Timers or crops may progress.", "focus": [], "important": False}
    return {"title": "Action: " + action.replace("_", " ").lower(), "detail": "Read the recorded state changes above; this action alone does not imply success.", "focus": [], "important": False}


def build_timeline(trace, environment, step_seconds=1., event_seconds=4., intro_seconds=7., outro_seconds=7., fps=10):
    durations = (step_seconds, event_seconds, intro_seconds, outro_seconds)
    if not np.isfinite(fps) or fps <= 0 or any(not np.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("durations and fps must be finite and positive")
    final = max(len(k["steps"]) for k in trace["kernels"].values())
    segments = []
    def add(kind, index, seconds):
        count = max(1, round(seconds * fps))
        start = sum(s["video_frames"] for s in segments)
        segments.append({"kind": kind, "trace_index": index, "video_frames": count,
                         "start_frame": start, "end_frame_exclusive": start + count,
                         "duration_seconds": count / fps})
    add("intro", 0, intro_seconds)
    for index in range(1, final + 1):
        important = any(describe_step(environment, *state_at(k, index)[1:])["important"] for k in trace["kernels"].values())
        add("transition", index, event_seconds if important else step_seconds)
    add("outro", final, outro_seconds)
    return {"fps": fps, "segments": segments, "total_frames": sum(s["video_frames"] for s in segments),
            "duration_seconds": sum(s["duration_seconds"] for s in segments),
            "semantics": "Pauses repeat recorded states; playback seconds are not environment ticks."}


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _paragraph(draw, xy, text, width, size=12, fill="#30291f"):
    x, y = xy
    # Wrap by measured pixels, including long captions, not by character count.
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=_font(size)) > width:
            draw.text((x, y), line, font=_font(size), fill=fill)
            y += size + 5
            line = word
        else:
            line = candidate
    draw.text((x, y), line, font=_font(size), fill=fill)
    return y + size + 5


def _annotate_objects(image, trace, environment, index, descriptions):
    from report_figures.render_pack_pixel import _map_origin, TILE, PALETTE
    draw = ImageDraw.Draw(image)
    bbox = world_geometry(trace, environment)[1]
    for side, name in enumerate(("fixed", "mutant")):
        state, _, _ = state_at(trace["kernels"][name], index)
        ox, oy = _map_origin(side * 480, bbox)
        for letter, key, _label in LEGENDS[environment]:
            row, col = trace["geometry"][key] if environment == "craft" else state[key]
            x, y = ox + (col - bbox[2]) * TILE, oy + (row - bbox[0]) * TILE
            if key in descriptions[side]["focus"]:
                draw.rectangle((x + 1, y + 1, x + TILE - 2, y + TILE - 2), outline="#ca4d24", width=3)
            draw.rectangle((x + 1, y + 1, x + 13, y + 15), fill="#fff5db", outline=PALETTE["ink"])
            draw.text((x + 3, y), letter, font=_font(11), fill=PALETTE["ink"])


def explain_frame(base, trace, environment, segment):
    """Add labels and commentary outside the exact recorded world image."""
    small = base.resize((base.width // 2, base.height // 2), Image.Resampling.NEAREST)
    index, kind = segment["trace_index"], segment["kind"]
    descriptions = [describe_step(environment, *state_at(trace["kernels"][name], index)[1:]) for name in ("fixed", "mutant")]
    _annotate_objects(small, trace, environment, index, descriptions)
    header, explanation = 80, 144
    out = Image.new("RGB", (960, header + small.height + explanation), "#f5ead4")
    out.paste(small, (0, header))
    draw = ImageDraw.Draw(out)
    draw.text((18, 8), f"{NAMES[environment]}  |  Goal: deliver grain value 3", font=_font(22), fill="#30291f")
    legend = "     ".join(f"{letter}: {label}" for letter, _, label in LEGENDS[environment])
    draw.text((18, 39), legend, font=_font(12), fill="#30291f")
    draw.text((18, 59), "LEFT: fixed rules     RIGHT: defect enabled     Highlight = object involved in the recorded event", font=_font(11), fill="#5b4d3d")
    y = header + small.height
    if kind == "intro":
        scripted = trace.get("provenance", {}).get("comparison") == "same_action_sequence"
        intro = {
            "pack": "A storage record remembers contents. Watch whether rebuilding the storage restores grain while the original grain still exists elsewhere.",
            "tick": "A crop should pay once per growth cycle. Watch what happens when manual harvest coincides with a reservation payout.",
            "craft": "Two ingredients normally become one parcel worth two. Watch whether the input ingredients disappear when crafting succeeds.",
        }[environment]
        draw.text((20, y + 8), "WHAT TO WATCH", font=_font(17), fill="#30291f")
        _paragraph(draw, (20, y + 35), intro, 900, 14)
        comparison = "Scripted verification: identical actions on both kernels. Not a learned-policy result." if scripted else "One saved policy, two separate kernel rollouts. Actions may differ after the states diverge."
        _paragraph(draw, (20, y + 92), comparison, 900, 12)
    elif kind == "outro":
        draw.text((20, y + 8), "WHAT THIS EPISODE SHOWS", font=_font(17), fill="#30291f")
        for side, name in enumerate(("fixed", "mutant")):
            kernel = trace["kernels"][name]
            summary = kernel["summary"]
            state, _, _ = state_at(kernel, index)
            text = f"{name.upper()}: delivered {state['delivered_total']}/3; {summary['length']} steps; {'goal reached' if summary['success'] else 'goal not reached'}."
            _paragraph(draw, (20 + 480 * side, y + 36), text, 435, 14)
        success = all(k["summary"]["success"] for k in trace["kernels"].values())
        gain = trace["kernels"]["fixed"]["summary"]["length"] - trace["kernels"]["mutant"]["summary"]["length"]
        conclusion = f"Both succeeded. This example uses {abs(gain)} {'fewer' if gain >= 0 else 'more'} steps with the defect." if success else "At least one side missed the goal: do not interpret the length difference as a speed gain."
        _paragraph(draw, (20, y + 92), conclusion + " One selected episode, not a population estimate.", 915, 12)
    else:
        for side, info in enumerate(descriptions):
            x = side * 480 + 12
            draw.rectangle((x, y + 6, x + 455, y + 133), fill="#ead8b8", outline="#b99c71")
            end = _paragraph(draw, (x + 12, y + 15), info["title"], 430, 16)
            _paragraph(draw, (x + 12, end + 6), info["detail"], 430, 12)
    return out.resize((out.width * 2, out.height * 2), Image.Resampling.NEAREST)


def write_explainer(trace, environment, base_frames, output, *, step_seconds=1., event_seconds=4.):
    output = Path(output)
    timeline = build_timeline(trace, environment, step_seconds, event_seconds)
    timeline["canonical_trace_sha256"] = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    frames = [explain_frame(base_frames[entry["trace_index"]], trace, environment, entry) for entry in timeline["segments"]]
    durations = [round(entry["duration_seconds"] * 1000) for entry in timeline["segments"]]
    imageio.mimsave(output / "explainer.gif", [np.asarray(frame) for frame in frames], duration=durations, loop=0)
    with imageio.get_writer(output / "explainer.mp4", fps=timeline["fps"], codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame, entry in zip(frames, timeline["segments"]):
            array = np.asarray(frame)
            for _ in range(entry["video_frames"]):
                writer.append_data(array)
    frames[0].save(output / "explainer_intro.png")
    frames[-1].save(output / "explainer_summary.png")
    for frame, entry in zip(frames, timeline["segments"]):
        if entry["kind"] == "transition":
            frame.save(output / f"explainer_step_{entry['trace_index']:03d}.png")
    (output / "explainer_timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    return timeline
