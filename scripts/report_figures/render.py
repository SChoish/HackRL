"""Five-panel stills for workshop, observation, and oracle traces.

The observation panel is only what observe_* returned. Conservation, trigger,
and excess delivery stay in the oracle strip.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

KINDS = {
    "floor": ("#f3efe6", ""),
    "wall": ("#cfcfcf", ""),
    "hidden": ("#ececec", ""),
    "crop": ("#2e7d32", "crop"),
    "crop_unripe": ("#a5d6a7", "young"),
    "device": ("#6a1b9a", "device"),
    "storage": ("#1565c0", "store"),
    "source": ("#2e7d32", "source"),
    "delivery": ("#ef6c00", "deliver"),
    "unpack": ("#00838f", "unpack"),
    "grid": ("#6d4c41", "grid"),
    "parcel": ("#c62828", "parcel"),
}
PLAYER = "#1f4e79"
DIVERGE = "#b71c1c"
ARROW = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _cell(ax, x, y, kind, text, size=1.0):
    color, fallback = KINDS.get(kind, ("#ffffff", ""))
    ax.add_patch(Rectangle((x, y), size, size, facecolor=color, edgecolor="#666666", linewidth=0.6))
    label = text or fallback
    if label:
        ax.text(x + size / 2, y + size / 2, label, ha="center", va="center", fontsize=7, color="#111111")


def _grid(ax, cells, player, direction, title):
    xs = [cell["c"] for cell in cells] or [0]
    ys = [cell["r"] for cell in cells] or [0]
    for cell in cells:
        _cell(ax, cell["c"], -cell["r"], cell["kind"], cell.get("text", ""))
    if player is not None:
        row, col = player
        ax.scatter([col + 0.5], [-row + 0.5], s=80, c=PLAYER, zorder=3)
        drow, dcol = ARROW.get(int(direction or 0), (0, 0))
        if drow or dcol:
            ax.annotate(
                "",
                xy=(col + 0.5 + 0.28 * dcol, -row + 0.5 - 0.28 * drow),
                xytext=(col + 0.5, -row + 0.5),
                arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.4},
            )
    pad = 0.4
    ax.set_xlim(min(xs) - pad, max(xs) + 1 + pad)
    ax.set_ylim(-max(ys) - pad, -min(ys) + 1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _lines(ax, title, rows, face="#ffffff"):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="square,pad=0", facecolor=face, edgecolor="#bbbbbb"))
    ax.set_title(title, fontsize=11, loc="left")
    text = "\n".join(rows) if rows else ""
    ax.text(0.06, 0.90, text, ha="left", va="top", fontsize=9, family="DejaVu Sans", linespacing=1.45)


def draw_frame(frame, path):
    fig = plt.figure(figsize=(14.5, 7.2), dpi=140, facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[4.2, 1.5], width_ratios=[1.3, 1.15, 1.05], hspace=0.28, wspace=0.18)
    workshop = fig.add_subplot(grid[0, 0])
    observation = fig.add_subplot(grid[0, 1])
    side = fig.add_subplot(grid[0, 2])
    oracle = fig.add_subplot(grid[1, :])
    _grid(workshop, frame["workshop"], frame.get("player"), frame.get("direction"), "Workshop state")
    _grid(observation, frame["observation"], frame.get("obs_player"), frame.get("direction"), "Observed map 7x9 (numeric inputs not shown)")
    _lines(
        side,
        "Action and resources",
        [
            f"action  {frame.get('action') or '—'}",
            f"goal    {frame.get('goal')}",
            f"tick    {frame.get('tick')}",
            f"delivered {frame.get('delivered')}",
            "",
            *frame.get("resources", []),
        ],
    )
    face = "#fdecea" if frame.get("diverges") else "#f7f7f7"
    _lines(oracle, "Diagnostics (may include public fields)", frame.get("oracle", []), face=face)
    banner = frame["title"]
    if frame.get("diverges"):
        banner += "   ·   kernels differ in displayed quantities"
    fig.suptitle(banner, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.01, frame.get("caption", ""), fontsize=8, color="#333333")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_pair(left, right, path, title, caption):
    fig = plt.figure(figsize=(16.5, 8.4), dpi=130, facecolor="white")
    outer = fig.add_gridspec(1, 2, wspace=0.08)
    for index, frame in enumerate((left, right)):
        inner = outer[index].subgridspec(2, 2, height_ratios=[4.2, 1.6], width_ratios=[1.2, 1.0], hspace=0.25, wspace=0.12)
        workshop = fig.add_subplot(inner[0, 0])
        observation = fig.add_subplot(inner[0, 1])
        oracle = fig.add_subplot(inner[1, :])
        _grid(workshop, frame["workshop"], frame.get("player"), frame.get("direction"), frame["kernel"])
        _grid(observation, frame["observation"], frame.get("obs_player"), frame.get("direction"), "Observed map (not full input)")
        face = "#fdecea" if frame.get("diverges") else "#f7f7f7"
        _lines(
            oracle,
            "Oracle",
            [f"action {frame.get('action') or '—'}    tick {frame.get('tick')}    delivered {frame.get('delivered')}", *frame.get("oracle", [])],
            face=face,
        )
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.01, caption, fontsize=8, color="#333333")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_strip(frames, path, title, caption):
    fig = plt.figure(figsize=(3.4 * len(frames), 4.6), dpi=140, facecolor="white")
    grid = fig.add_gridspec(1, len(frames), wspace=0.18)
    for index, frame in enumerate(frames):
        ax = fig.add_subplot(grid[index])
        edge = DIVERGE if frame.get("diverges") else "#444444"
        for spine in ax.spines.values():
            spine.set_color(edge)
            spine.set_linewidth(2.0 if frame.get("diverges") else 0.8)
        _grid(ax, frame["workshop"], frame.get("player"), frame.get("direction"), frame.get("beat", frame["kernel"]))
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.01, caption, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
