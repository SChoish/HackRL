"""Replay recorded spatial paths, or explicit reference demonstrations, for rendering.

Never loads a checkpoint or trains a policy. Recorded evaluation positions and
outcomes must match the actual kernel before a trace can be exported.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np

from hackrl import spatial_wall_pass as env


def state_dict(state):
    return {field.name: np.asarray(getattr(state, field.name)).tolist()
            for field in fields(state) if field.name not in ("walkable", "wall_mask")}


def replay(actions, *, layout=0, split="validation", kernel="fixed", recorded=None):
    state = env.make_spatial_wall_pass_state(layout, split=split)
    initial = state_dict(state)
    step_fn = jax.jit(lambda s, a: env.spatial_wall_pass_step_with_transition(s, a, kernel))
    mutant_fn = jax.jit(lambda s, a: env.spatial_wall_pass_step_with_transition(s, a, "mutant"))
    steps = []
    for index, action in enumerate(actions):
        if int(state.tick) >= env.WORLD_HORIZON or int(state.delivered_total) > 0:
            raise ValueError("actions continue beyond the episode end")
        before = state_dict(state)
        next_state, transition = step_fn(state, int(action))
        mutant_state, mutant_transition = mutant_fn(state, int(action))
        blocked = bool(mutant_transition.crossed_wall) and np.array_equal(
            np.asarray(state.player_position), np.asarray(next_state.player_position))
        after = state_dict(next_state)
        if recorded is not None:
            row = recorded[index]
            origin = row.get("from", row.get("from_position"))
            destination = row.get("to", row.get("to_position"))
            if origin != before["player_position"] or destination != after["player_position"]:
                raise ValueError(f"recorded path disagrees with kernel at step {index + 1}")
            for key in ("blocked_dash", "blocked"):
                if key in row and bool(row[key]) != blocked:
                    raise ValueError(f"recorded blocked event disagrees at step {index + 1}")
            if "wall_pass" in row and bool(row["wall_pass"]) != bool(transition.crossed_wall):
                raise ValueError(f"recorded crossing disagrees at step {index + 1}")
        events = {f.name: np.asarray(getattr(transition, f.name)).tolist()
                  for f in fields(transition)}
        events["blocked_dash"] = blocked
        steps.append({"before": before, "after": after, "action": int(action),
                      "action_name": env.SpatialWallPassAction(action).name, "events": events})
        state = next_state
    success = bool(state.delivered_total > 0)
    return {"kernel": kernel, "initial": initial, "steps": steps,
            "summary": {"success": success, "length": len(steps),
                        "wall_passes": sum(s["events"]["crossed_wall"] for s in steps),
                        "blocked_dashes": sum(s["events"]["blocked_dash"] for s in steps),
                        "end_reason": "success" if success else (
                            "horizon" if int(state.tick) == env.WORLD_HORIZON else "trace end")}}


def geometry(layout=0, split="validation"):
    state = env.make_spatial_wall_pass_state(layout, split=split)
    return {"map_size": env.MAP_SIZE, "walkable": np.argwhere(state.walkable).tolist(),
            "walls": np.argwhere(state.wall_mask).tolist(),
            "camp": np.asarray(state.camp_position).tolist(),
            "item": np.asarray(state.item_position).tolist()}


def demo_trace(layout=0):
    normal = list(map(int, env.normal_reference_actions(layout)))
    shortcut = list(map(int, env.exploit_reference_actions(layout)))
    prefix = [int(env.transform_action(a, layout)) for a in
              (env.SpatialWallPassAction.RIGHT, env.SpatialWallPassAction.DASH_RIGHT,
               env.SpatialWallPassAction.LEFT)]
    scenes = []
    for title, subtitle, kernel, actions in (
        ("NORMAL ROUTE", "Go around the wall", "fixed", normal),
        ("WALL SHORTCUT", "Dash through the wall", "mutant", shortcut),
        ("BLOCKED, THEN DETOUR", "Try the shortcut, then go around", "fixed", prefix + normal),
    ):
        scenes.append({"title": title, "subtitle": subtitle,
                       **replay(actions, layout=layout, kernel=kernel)})
    return {"schema_version": "hackrl_spatial_render_v1", "kind": "reference_demo",
            "split": "validation", "layout": layout, "geometry": geometry(layout),
            "scenes": scenes,
            "provenance": {"selection": "Constructive reference actions, not learned policies",
                           "environment_source_sha256": hashlib.sha256(Path(env.__file__).read_bytes()).hexdigest()}}


def json_pointer(document, pointer):
    current = document
    if not pointer:
        return current
    if not pointer.startswith("/"):
        raise ValueError("episode pointer must be a JSON pointer starting with /")
    for key in pointer[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


def from_spec(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    split, layout = spec["split"], int(spec["layout"])
    if split not in ("train", "validation", "test") or not 0 <= layout < 16:
        raise ValueError("only the original spatial fixture splits and layouts 0..15 are supported")
    scenes = []
    sources = []
    for entry in spec["scenes"]:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = spec_path.parent / path
        raw = path.read_bytes()
        record = json_pointer(json.loads(raw), entry["pointer"])
        if int(record["layout_index"]) != layout:
            raise ValueError("selected episode has a different layout")
        rows = record["path"]
        result = replay([r["action"] for r in rows], layout=layout, split=split,
                        kernel=entry["kernel"], recorded=rows)
        if bool(record["success"]) != result["summary"]["success"]:
            raise ValueError("replayed success disagrees with evaluation")
        if int(record["length"]) != len(rows):
            raise ValueError("episode path is incomplete")
        scenes.append({"title": entry["title"], "subtitle": entry["subtitle"], **result})
        sources.append({"file": str(path.resolve()), "pointer": entry["pointer"],
                        "sha256": hashlib.sha256(raw).hexdigest()})
    if not 1 <= len(scenes) <= 3:
        raise ValueError("choose one to three scenes")
    return {"schema_version": "hackrl_spatial_render_v1", "kind": "recorded_evaluation",
            "split": split, "layout": layout, "geometry": geometry(layout, split),
            "scenes": scenes, "provenance": {"sources": sources,
                "selection": spec["selection"],
                "environment_source_sha256": hashlib.sha256(Path(env.__file__).read_bytes()).hexdigest()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--demo", action="store_true")
    choice.add_argument("--spec", type=Path)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = demo_trace(args.layout) if args.demo else from_spec(args.spec)
    root = Path(__file__).resolve().parents[2]
    trace["provenance"]["recording_code_sha"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    trace["provenance"]["recording_tree_dirty"] = bool(subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "kind": trace["kind"],
                      "scenes": [s["summary"] for s in trace["scenes"]]}))


if __name__ == "__main__":
    main()
