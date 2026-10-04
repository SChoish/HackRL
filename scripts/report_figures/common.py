"""Trace provenance and display semantics; no environment imports."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HACKRL_ROOT", CODE_ROOT)).resolve()


def provenance(checkpoint):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(CODE_ROOT), *args], text=True
        ).strip()

    hashes = {}
    for name in ("config.json", "state.msgpack"):
        digest = hashlib.sha256()
        with (Path(checkpoint) / name).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return {
        "recording_code_sha": git("rev-parse", "HEAD"),
        "recording_tree_dirty": bool(git("status", "--porcelain")),
        "checkpoint_sha256": hashes,
        "action_selection": "mode",
        "comparison": "same_policy_separate_kernel_rollouts",
        "environment_origin": "synthetic; not an original-game code reproduction",
    }


def state_at(kernel, index):
    if index < 0:
        raise ValueError("timeline index must be nonnegative")
    steps = kernel["steps"]
    if index == 0 or not steps:
        return kernel["initial"], None, index > 0
    step = steps[min(index, len(steps)) - 1]
    return step["after"], step, index > len(steps)


def frame_status(kernel, state, held):
    if not held:
        return f"step {state['tick']:02d}"
    reason = "success" if kernel["summary"]["success"] else "end"
    return f"held: {reason} {state['tick']:02d}"


def metadata_lines(trace):
    policy, start = trace["policy"], trace["start"]
    recorded = trace.get("provenance", {})
    selection = trace.get("selection", {})
    comparison = (
        "same scripted actions on both kernels"
        if recorded.get("comparison") == "same_action_sequence" else
        "same policy, separate kernel rollouts; actions may diverge"
    )
    first = (
        "Synthetic workshop | observer view, not policy input | "
        f"{recorded.get('action_selection', 'mode')} | "
        f"{comparison}"
    )
    second = (
        f"{start.get('split', 'n/a')} / {start['family']} / "
        f"layout {start.get('layout', 'n/a')} / {start['phase']} | "
        f"selection: {selection.get('rule', 'not recorded')} | one episode, not a mean"
    )
    checkpoint_hash = recorded.get("checkpoint_sha256", {}).get("state.msgpack", "unrecorded")
    third = (
        f"{policy['method']} seed {policy['seed']} update {policy['adaptation_updates']} | "
        f"checkpoint SHA256 {checkpoint_hash[:12]} | "
        f"recording code {recorded.get('recording_code_sha', 'unrecorded')[:12]}"
        + (" (dirty)" if recorded.get("recording_tree_dirty") else "")
    )
    return first, second, third


def world_geometry(trace, environment):
    """Return actual floor cells and a crop containing the whole recorded path."""
    geometry = trace["geometry"]
    size = int(geometry["map_size"])
    if "walkable" in geometry:
        walkable = {tuple(cell) for cell in geometry["walkable"]}
    elif environment == "craft":
        # Compatibility with v1 traces: this engine has open floor, not a room.
        blocked = {tuple(geometry[name]) for name in ("source", "grid", "delivery")}
        walkable = {(r, c) for r in range(size) for c in range(size)} - blocked
    else:
        raise ValueError("trace must record walkable geometry")
    if environment == "craft":
        points = [geometry[name] for name in ("source", "grid", "delivery")]
    else:
        points = list(walkable)
    for kernel in trace["kernels"].values():
        states = [kernel["initial"], *(step["after"] for step in kernel["steps"])]
        points.extend(state["player_position"] for state in states)
        points.extend(value for key, value in kernel["initial"].items() if key.endswith("_position"))
    rows, cols = zip(*points)
    bbox = (min(rows) - 1, max(rows) + 1, min(cols) - 1, max(cols) + 1)
    if bbox[1] - bbox[0] + 1 > 7 or bbox[3] - bbox[2] + 1 > 7:
        raise ValueError("recorded path exceeds the 7x7 movie viewport; enlarge the renderer instead of clipping the path")
    return walkable, bbox


def visible_crop(state, observation, bbox):
    """Intersect the actual stored observation window with the displayed crop."""
    tiles = observation["map_tiles"]
    rows, cols = len(tiles), len(tiles[0])
    row, col = state["player_position"]
    top = max(bbox[0], row - rows // 2)
    bottom = min(bbox[1], row + rows // 2)
    left = max(bbox[2], col - cols // 2)
    right = min(bbox[3], col + cols // 2)
    return None if top > bottom or left > right else (top, bottom, left, right)
