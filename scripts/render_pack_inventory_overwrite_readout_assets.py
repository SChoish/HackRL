#!/usr/bin/env python3
"""Render slide-insert assets for the harmful-bug adaptation readout.

The script performs no rollout or training. It reads pinned result/evaluation JSON,
checks their SHA-256 digests, and writes matching PNG and SVG assets plus a contact
sheet and machine-readable source receipt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs/figures/pack_inventory_overwrite_readout_v1"
SOURCES = {
    "mixed_path": Path(
        "/raid/ext_csv/HackRL/runs/spatial_wall_pass_reliability_mix_v1/"
        "fixed_01pct_seed_132/checkpoints/adapt_4096/evaluation.json"
    ),
    "ppo": REPOSITORY / "docs/manifests/pack_inventory_overwrite_ppo_v1_results.json",
    "lr": REPOSITORY / "docs/manifests/pack_inventory_overwrite_lr_ppo_v1_results.json",
    "transfer_compact": REPOSITORY
    / "docs/manifests/pack_inventory_overwrite_train_probability_transfer_v1_results.json",
    "transfer_raw": Path(
        "/raid/ext_csv/HackRL/runs/"
        "pack_inventory_overwrite_train_probability_transfer_v1/result.json"
    ),
}
EXPECTED_SHA256 = {
    "mixed_path": "092d487154b6652e5298c4dba1048af5849fc6969be04d9d77fdb2d3c72c7029",
    "ppo": "3cc5791c40540692c2a282a8f71d3fc81d109f1f8ea5e2cd29550bbe6302cac0",
    "lr": "6fb4a73d5920e1daac33b3d5ea39389507618c4dd59127a39824758e892d3344",
    "transfer_compact": "62365377984a24a9748f840b5eee61169713a557f4842acf047620f6e49a894e",
    "transfer_raw": "5e7444899a162d27e8425b42522fd13086d8c7d17e33f2e820096ba1de47f1c7",
}

INK = "#17232d"
MUTED = "#66737f"
GRID = "#dfe5ea"
GROUND = "#f6f8fa"
BLUE = "#3274a1"
LIGHT_BLUE = "#55a6d9"
RED = "#d95f59"
ORANGE = "#e69542"
PURPLE = "#6f4c9b"
GREEN = "#3b8f6b"
GRAY = "#8a949e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources() -> tuple[dict[str, dict], dict[str, str]]:
    documents = {}
    receipts = {}
    for key, path in SOURCES.items():
        actual = sha256(path)
        expected = EXPECTED_SHA256[key]
        if actual != expected:
            raise RuntimeError(f"{key} hash changed: expected {expected}, got {actual}")
        try:
            receipt_path = str(path.relative_to(REPOSITORY))
        except ValueError:
            receipt_path = str(path)
        receipts[receipt_path] = actual
        documents[key] = json.loads(path.read_text(encoding="utf-8"))
    return documents, receipts


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    )


def clean(ax, *, grid: str | None = "y") -> None:
    ax.set_facecolor(GROUND)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)


def save(fig, stem: str) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    svg = OUTPUT / f"{stem}.svg"
    fig.savefig(png, dpi=160, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png, svg


def label_panel(ax, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        zorder=10,
    )


WALKABLE = {
    (row, col)
    for row in range(5, 12)
    for col in range(6, 13)
    if not (col == 9 and row in range(6, 11))
    and not (row == 11 and col == 12)
}
WALL = {(row, 9) for row in range(6, 11)}
CAMP = (8, 6)
ITEM = (8, 11)


def route_points(record: dict) -> list[tuple[int, int]]:
    steps = record["path"]
    return [tuple(steps[0]["from"]), *(tuple(step["to"]) for step in steps)]


def draw_world(ax) -> None:
    for row, col in WALKABLE:
        ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, facecolor="#eef2f5", edgecolor="white", lw=1.1))
    for row, col in WALL:
        ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, facecolor="#59636e", edgecolor="white", lw=1.1))
    ax.scatter([CAMP[1]], [CAMP[0]], marker="s", s=180, c=GREEN, edgecolors="white", linewidths=1.5, zorder=5)
    ax.scatter([ITEM[1]], [ITEM[0]], marker="D", s=180, c="#f2c14e", edgecolors="white", linewidths=1.5, zorder=5)
    ax.text(CAMP[1], CAMP[0] + 0.08, "C", color="white", ha="center", va="center", fontweight="bold", zorder=6)
    ax.text(ITEM[1], ITEM[0] + 0.08, "I", color=INK, ha="center", va="center", fontweight="bold", zorder=6)
    ax.set_xlim(5.3, 12.7)
    ax.set_ylim(11.7, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_path(ax, points, color, *, start_index=0, linewidth=3.2, alpha=1.0) -> None:
    xs = [point[1] for point in points[start_index:]]
    ys = [point[0] for point in points[start_index:]]
    ax.plot(xs, ys, color=color, lw=linewidth, alpha=alpha, zorder=3)
    ax.scatter(xs, ys, s=20, color=color, alpha=alpha, zorder=4)
    ax.scatter([xs[0]], [ys[0]], s=95, facecolor="white", edgecolor=color, linewidth=2.0, zorder=6)


def render_routes(doc: dict) -> tuple[Path, Path]:
    mutant = doc["mutant"]["mode"]["episode_records"][0]
    fixed = doc["fixed"]["mode"]["episode_records"][0]
    if not (mutant["wall_pass"] and mutant["success"] and fixed["success_after_first_block"]):
        raise AssertionError("pinned path no longer demonstrates shortcut and recovery")
    mutant_points = route_points(mutant)
    fixed_points = route_points(fixed)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), dpi=120)
    for ax in axes:
        draw_world(ax)

    draw_path(axes[0], mutant_points, PURPLE)
    axes[0].annotate("wall pass", xy=(9, 8), xytext=(9, 6.4), ha="center", color=PURPLE, fontweight="bold", arrowprops={"arrowstyle": "->", "color": PURPLE})
    label_panel(axes[0], "Shortcut works\n8 steps · success")

    blocked_from = tuple(fixed["path"][1]["from"])
    axes[1].plot([blocked_from[1], 10], [blocked_from[0], blocked_from[0]], color=RED, lw=4, zorder=3)
    axes[1].scatter([9], [8], marker="X", s=220, color=RED, edgecolor="white", linewidth=1.2, zorder=6)
    axes[1].text(9, 7.2, "4 blocked dashes", color=RED, ha="center", fontweight="bold")
    axes[1].scatter([blocked_from[1]], [blocked_from[0]], s=95, facecolor="white", edgecolor=RED, linewidth=2.0, zorder=6)
    label_panel(axes[1], "Patch blocks the shortcut\nrepeated same-state attempts")

    draw_path(axes[2], fixed_points, BLUE)
    axes[2].scatter([9], [8], marker="X", s=180, color=RED, edgecolor="white", linewidth=1.2, zorder=7)
    axes[2].annotate("detour", xy=(9.5, 11), xytext=(7.0, 10.7), color=BLUE, fontweight="bold", arrowprops={"arrowstyle": "->", "color": BLUE})
    label_panel(axes[2], "Mixed training recovers\n22 steps · success after block")

    fig.text(0.5, 0.025, "Actual deterministic paths · q=1% fixed-kernel mixture · seed 132 · validation layout 0", ha="center", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.0)
    return save(fig, "01_exploitation_and_recovery_paths")


def rounded(ax, xy, width, height, text, *, facecolor, edgecolor="none", fontsize=13, color=INK) -> None:
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold")


def arrow(ax, start, end, *, color=INK, style="-|>", lw=2.2, connectionstyle="arc3") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15, linewidth=lw, color=color, connectionstyle=connectionstyle))


def render_harmful_ppo(doc: dict) -> tuple[Path, Path]:
    block = doc["primary_final_mode_harmful_kernel"]
    labels = ["Frozen source", "Normal continue", "Defect PPO"]
    keys = ["frozen_harmful", "normal_continue", "harmful_adapt"]
    success = [100 * block[key]["success_rate"] for key in keys]
    loss = [100 * block[key]["loss_episode_rate"] for key in keys]

    fig = plt.figure(figsize=(12.5, 7.0), dpi=130)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.75, 1.35), hspace=0.28, wspace=0.22)
    normal = fig.add_subplot(grid[0, 0]); normal.axis("off"); normal.set_xlim(0, 1); normal.set_ylim(0, 1)
    harmful = fig.add_subplot(grid[0, 1]); harmful.axis("off"); harmful.set_xlim(0, 1); harmful.set_ylim(0, 1)
    rounded(normal, (0.02, 0.28), 0.22, 0.42, "carry c", facecolor="#e9eef3")
    rounded(normal, (0.38, 0.28), 0.22, 0.42, "harvest", facecolor="#dcecf6")
    rounded(normal, (0.74, 0.28), 0.24, 0.42, "c + 1", facecolor="#dcefe7", color=GREEN)
    arrow(normal, (0.24, 0.49), (0.38, 0.49)); arrow(normal, (0.60, 0.49), (0.74, 0.49))
    normal.text(0.02, 0.87, "Normal transition", fontsize=15, fontweight="bold")
    normal.text(0.50, 0.12, "source stock: r -> r - 1", ha="center", color=MUTED, fontsize=11)
    rounded(harmful, (0.02, 0.28), 0.22, 0.42, "carry c", facecolor="#e9eef3")
    rounded(harmful, (0.38, 0.28), 0.22, 0.42, "harvest", facecolor="#f7e3e1")
    rounded(harmful, (0.74, 0.28), 0.24, 0.42, "1", facecolor="#f7d8d5", color=RED)
    arrow(harmful, (0.24, 0.49), (0.38, 0.49)); arrow(harmful, (0.60, 0.49), (0.74, 0.49), color=RED)
    harmful.text(0.02, 0.87, "Overwrite defect", fontsize=15, fontweight="bold", color=RED)
    harmful.text(0.50, 0.12, "c grain disappears when c > 0", ha="center", color=RED, fontsize=11, fontweight="bold")

    ax = fig.add_subplot(grid[1, :]); clean(ax, grid="y")
    x = np.arange(len(labels)); width = 0.33
    ax.bar(x - width / 2, success, width, color=BLUE, label="Task success", zorder=3)
    ax.bar(x + width / 2, loss, width, color=RED, label="Loss episode", zorder=3)
    for index, values in enumerate((success, loss)):
        offsets = x - width / 2 if index == 0 else x + width / 2
        for xpos, value in zip(offsets, values, strict=True):
            ax.text(xpos, value + 2.2, f"{value:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 108); ax.set_ylabel("Episode rate (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.text(2, 48, "80% success\n20% loss", ha="center", va="center", fontsize=17, color=GREEN, fontweight="bold")
    fig.text(0.5, 0.038, "Protect before first harvest: 22.5% → 60.0%", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.015, "Seeds 30–34 · Dual-history source policies · no teacher updates or BC · final normal-kernel success: 100%", ha="center", color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.13)
    return save(fig, "02_ppo_avoids_harmful_bug")


def render_lr_tradeoff(doc: dict) -> tuple[Path, Path]:
    final = doc["final_aggregates"]
    contrasts = doc["paired_learning_rate_contrasts"]
    probabilities = ["5%", "30%"]
    standard = [final["p05_lr1"], final["p30_lr1"]]
    lower = [final["p05_lr025"], final["p30_lr025"]]
    delta = [contrasts["p05_lower_minus_standard"]["mean_exact_return"], contrasts["p30_lower_minus_standard"]["mean_exact_return"]]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), dpi=130, gridspec_kw={"width_ratios": (0.8, 1.05, 1.05)})
    ax = axes[0]; clean(ax, grid="x")
    colors = [GREEN if value > 0 else RED for value in delta]
    ax.barh(probabilities[::-1], delta[::-1], color=colors[::-1], height=0.46, zorder=3)
    ax.axvline(0, color=INK, linewidth=1)
    for y, value in enumerate(delta[::-1]):
        ax.text(value / 2, y, f"{value:+.5f}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.set_xlim(-0.045, 0.042); ax.set_xlabel("1/4 LR minus standard LR return")
    ax.set_title("Return within the same budget", loc="left", fontsize=14)

    ax = axes[1]; clean(ax, grid="x")
    for y, (probability, left, right) in enumerate(zip(probabilities, standard, lower, strict=True)):
        a = 100 * left["normal_kernel_success_rate"]; b = 100 * right["normal_kernel_success_rate"]
        ax.plot([a, b], [y, y], color=GRAY, lw=3)
        ax.scatter([a], [y], s=90, color=GRAY, label="Standard LR" if y == 0 else None, zorder=3)
        ax.scatter([b], [y], s=100, color=LIGHT_BLUE, marker="D", label="1/4 LR" if y == 0 else None, zorder=4)
        ax.text(a - 0.25, y + 0.16, f"{a:.2f}%", ha="right", fontsize=10); ax.text(b + 0.12, y + 0.16, f"{b:.0f}%", fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(probabilities); ax.set_xlim(95.5, 100.6)
    ax.set_xlabel("Normal-kernel success")
    ax.set_title("Performance preservation", loc="left", fontsize=14)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.03, 0.5))

    ax = axes[2]; clean(ax, grid="x")
    for y, (probability, left, right) in enumerate(zip(probabilities, standard, lower, strict=True)):
        a = left["risky_starts"]; b = right["risky_starts"]
        ax.plot([a, b], [y, y], color=GRAY, lw=3)
        ax.scatter([a], [y], s=90, color=GRAY, zorder=3); ax.scatter([b], [y], s=100, color=ORANGE, marker="D", zorder=4)
        ax.text(a, y + 0.16, str(a), ha="center", fontsize=10); ax.text(b, y + 0.16, str(b), ha="center", fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(probabilities); ax.set_xlim(36, 92)
    ax.set_xlabel("Risky starts / 160  (lower is safer)")
    ax.set_title("Risk reduction", loc="left", fontsize=14)
    fig.text(0.5, 0.042, "Smaller updates preserved normal execution, but also reduced useful adaptation.", ha="center", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.015, "At 5%, the mean advantage is driven by 2 of 5 seeds.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.98), w_pad=2.4)
    return save(fig, "03_smaller_updates_tradeoff")


CONDITION_STYLE = (
    ("source", "Source", GRAY, "o"),
    ("p00_lr1", "Train 0%", "#9aa3ab", "o"),
    ("p05_lr1", "Train 5%", RED, "o"),
    ("p30_lr1", "Train 30%", BLUE, "o"),
    ("p100_lr1", "Train 100%", PURPLE, "o"),
    ("p05_lr025", "5% · 1/4 LR", "#e2948f", "D"),
    ("p30_lr025", "30% · 1/4 LR", LIGHT_BLUE, "D"),
)


def transfer_metrics(compact: dict, raw: dict, condition: str) -> tuple[float, float, float, list[float]]:
    if condition == "source":
        mean = compact["source_baseline"]
        values = [float(row["cross_probability_exact"]["p05"]["existing_expected_discounted_return"]) for row in raw["source_baseline"]["per_seed"]]
        return mean["p05_exact_expected_discounted_return"], 100 * mean["normal_kernel_success_rate"], float(mean["risky_starts"]), values
    mean = compact["p05_evaluation"][condition]
    rows = sorted(raw["policy_rows"][condition], key=lambda row: int(row["seed"]))
    values = [float(row["cross_probability_exact"]["p05"]["existing_expected_discounted_return"]) for row in rows]
    return mean["exact_expected_discounted_return"], 100 * mean["normal_success_rate"], float(mean["risky_starts"]), values


def render_probability_transfer(compact: dict, raw: dict) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), dpi=130, gridspec_kw={"width_ratios": (1.2, 1.0)})
    ax = axes[0]; clean(ax, grid="x")
    y = np.arange(len(CONDITION_STYLE))[::-1]
    source = transfer_metrics(compact, raw, "source")[0]
    ax.axvline(source, color=GRAY, linestyle=(0, (4, 4)), lw=1.4)
    rng = np.random.default_rng(17)
    for ypos, (condition, label, color, marker) in zip(y, CONDITION_STYLE, strict=True):
        mean, _, _, values = transfer_metrics(compact, raw, condition)
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(values, ypos + jitter, s=28, color=color, alpha=0.55, zorder=3)
        ax.scatter([mean], [ypos], s=110, color=color, marker=marker, edgecolor="white", linewidth=1.3, zorder=4)
        ax.text(mean + 0.0022, ypos, f"{mean:.5f}", va="center", fontsize=10, fontweight="bold" if condition == "p30_lr025" else "normal")
    ax.set_yticks(y); ax.set_yticklabels([item[1] for item in CONDITION_STYLE])
    ax.set_xlim(0.755, 0.917); ax.set_xlabel("Expected discounted return at 5% deployment")
    ax.set_title("All policies evaluated at 5%", loc="left", fontsize=15)
    ax.text(0.03, 0.04, "dots = seeds   large marker = mean", transform=ax.transAxes, color=MUTED, fontsize=9)

    ax = axes[1]; clean(ax, grid="both")
    offsets = {"source": (2, 0.10), "p00_lr1": (2, -0.33), "p05_lr1": (2, 0.08), "p30_lr1": (-3, -0.23), "p100_lr1": (2, -0.15), "p05_lr025": (-3, 0.28), "p30_lr025": (-3, -0.22)}
    for condition, label, color, marker in CONDITION_STYLE:
        value, normal, risky, _ = transfer_metrics(compact, raw, condition)
        ax.scatter(risky, normal, s=125, color=color, marker=marker, edgecolor="white", linewidth=1.4, zorder=4)
        dx, dy = offsets[condition]
        ax.text(risky + dx, normal + dy, label, color=color, fontsize=9, ha="right" if dx < 0 else "left")
        if condition == "p100_lr1":
            ax.annotate("6 risky starts, but 97.5% normal success", xy=(risky, normal), xytext=(22, 95.7), color=PURPLE, fontsize=10, fontweight="bold", arrowprops={"arrowstyle": "->", "color": PURPLE})
    ax.set_xlim(-4, 106); ax.set_ylim(95.2, 100.55)
    ax.set_xlabel("Risky starts / 160  (lower is safer)"); ax.set_ylabel("Normal-kernel success (%)")
    ax.set_title("Less risk does not imply better execution", loc="left", fontsize=15)
    fig.text(0.5, 0.040, "30% · 1/4 LR has the highest observed mean; it is not an identified optimum.", ha="center", fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.014, "Seeds 35–39 · final checkpoint after 32 updates · exact finite-panel expectation · risky starts measured on fixed paths", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.98), w_pad=2.8)
    return save(fig, "04_training_frequency_transfer")


def render_seed_failure(compact: dict, raw: dict) -> tuple[Path, Path]:
    rows = compact["p100_per_seed"]
    seeds = [int(row["seed"]) for row in rows]
    normal = [100 * float(row["normal_success"]) for row in rows]
    p100_by_seed = {int(row["seed"]): row for row in raw["policy_rows"]["p100_lr1"]}
    p30_quarter_by_seed = {int(row["seed"]): row for row in raw["policy_rows"]["p30_lr025"]}
    differences = np.asarray(
        [
            p100_by_seed[seed]["cross_probability_exact"]["p05"]["existing_expected_discounted_return"]
            - p30_quarter_by_seed[seed]["cross_probability_exact"]["p05"]["existing_expected_discounted_return"]
            for seed in seeds
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(12.4, 6.0), dpi=130)
    grid = fig.add_gridspec(2, 1, height_ratios=(0.52, 1.48), hspace=0.07)
    top = fig.add_subplot(grid[0, 0]); clean(top, grid="y")
    top.bar(np.arange(5), normal, color=[RED if seed == 35 else BLUE for seed in seeds], width=0.58, zorder=3)
    for index, value in enumerate(normal):
        top.text(index, value + 0.8, f"{value:g}%", ha="center", fontsize=10, fontweight="bold")
    top.set_ylim(84, 104); top.set_yticks([87.5, 100]); top.set_ylabel("Normal success")
    top.set_xticks([]); top.set_title("The p=1 policy removes risk but seed 35 loses the base task", loc="left", fontsize=15)

    ax = fig.add_subplot(grid[1, 0]); clean(ax, grid="y")
    colors = [RED if value < 0 else LIGHT_BLUE for value in differences]
    ax.axhline(0, color=INK, lw=1.0)
    ax.vlines(np.arange(5), 0, differences, colors=colors, linewidth=4, zorder=2)
    ax.scatter(np.arange(5), differences, s=100, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for index, value in enumerate(differences):
        if value < 0:
            ax.text(index + 0.06, value - 0.002, f"{value:+.5f}", ha="left", va="top", fontsize=10, fontweight="bold")
        else:
            ax.text(index, value + 0.005, f"{value:+.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels([f"Seed {seed}" for seed in seeds])
    ax.set_ylim(-0.125, 0.047)
    ax.set_ylabel("5% return: p=1 minus 30% · 1/4 LR")
    ax.text(0.17, 0.07, "Seed 35\n0 risky starts\n4/32 normal starts fail", transform=ax.transAxes, color=RED, fontsize=12, fontweight="bold", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fae9e7", "edgecolor": "none"})
    ax.text(0.66, 0.10, "4/5 improve\none failure reverses the mean", transform=ax.transAxes, color=PURPLE, fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.12)
    return save(fig, "05_seed_specific_failure")


def render_return_decomposition(compact: dict, lr: dict) -> tuple[Path, Path]:
    source = compact["source_baseline"]
    p100 = compact["p05_evaluation"]["p100_lr1"]
    normal_change = p100["normal_return"] - source["normal_kernel_return"]
    source_cost = source["normal_kernel_return"] - source["p05_exact_expected_discounted_return"]
    p100_cost = p100["normal_return"] - p100["exact_expected_discounted_return"]
    defect_loss_reduction = source_cost - p100_cost
    net = p100["exact_expected_discounted_return"] - source["p05_exact_expected_discounted_return"]
    if abs((normal_change + defect_loss_reduction) - net) > 1e-12:
        raise AssertionError("return decomposition no longer closes")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=130, gridspec_kw={"width_ratios": (1.15, 0.85)})
    ax = axes[0]; clean(ax, grid="y")
    x = np.arange(3)
    ax.bar(0, normal_change, bottom=0, color=RED, width=0.6, zorder=3)
    ax.bar(1, defect_loss_reduction, bottom=normal_change, color=GREEN, width=0.6, zorder=3)
    ax.bar(2, net, bottom=0, color=PURPLE, width=0.6, zorder=3)
    ax.plot([0.3, 0.7], [normal_change, normal_change], color=GRAY, lw=1.5)
    ax.plot([1.3, 1.7], [net, net], color=GRAY, lw=1.5)
    for xpos, value, bottom in [(0, normal_change, 0), (1, defect_loss_reduction, normal_change), (2, net, 0)]:
        ax.text(xpos, bottom + value / 2, f"{value:+.5f}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(["Normal task\nchange", "Defect-loss\nreduction", "Final change"])
    ax.set_ylabel("Return change: p=1 policy minus source")
    ax.set_title("Risk reduction is offset by task degradation", loc="left", fontsize=15)
    ax.set_ylim(-0.033, 0.031)

    ax = axes[1]; clean(ax, grid="x")
    harms = [
        lr["final_aggregates"]["p05_lr1"]["collection"]["cumulative_lost_grain"],
        lr["final_aggregates"]["p30_lr1"]["collection"]["cumulative_lost_grain"],
        compact["new_training_collection"]["p100_lr1"]["cumulative_lost_grain"],
    ]
    labels = ["Train 5%", "Train 30%", "Train 100%"]
    ypos = np.arange(3)[::-1]
    ax.barh(ypos, harms, color=[ORANGE, BLUE, PURPLE], height=0.55, zorder=3)
    for y, value in zip(ypos, harms, strict=True):
        ax.text(value + 350, y, f"{value:,}", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(ypos); ax.set_yticklabels(labels); ax.set_xlim(0, 17000)
    ax.set_xlabel("Grain lost during five training runs")
    ax.set_title("Adaptation itself incurs harm", loc="left", fontsize=15)
    fig.text(0.5, 0.02, "Measured-return decomposition; it does not identify whether degradation is forgetting, over-avoidance, or state-distribution shift.", ha="center", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98), w_pad=3.0)
    return save(fig, "06_risk_and_task_decomposition")


def render_switch_diagnostic() -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(15.0, 5.8), dpi=130)
    ax.set_xlim(0, 14.8); ax.set_ylim(-0.2, 4.6); ax.axis("off")
    ax.text(0.2, 4.33, "Frozen-policy execution diagnostic", fontsize=18, fontweight="bold")
    ax.text(14.6, 4.33, "0 updates", ha="right", fontsize=13, color=PURPLE, fontweight="bold")
    lanes = [
        (3.55, "Source", [(1.9, 2.4, "pi_S", "#e9eef3")], "Original execution"),
        (2.65, "Adapted", [(1.9, 2.4, "pi_H", "#ede6f5")], "Avoidance + observed task damage"),
        (1.75, "One-step", [(1.9, 2.1, "pi_S", "#e9eef3"), (6.2, 1.45, "pi_H", "#ede6f5"), (9.8, 2.1, "pi_S", "#e9eef3")], "Adapted for one action"),
        (0.85, "Persistent", [(1.9, 2.1, "pi_S", "#e9eef3"), (6.2, 3.0, "pi_H", "#ede6f5"), (11.2, 1.7, "pi_S", "#e9eef3")], "Adapted until empty"),
    ]
    for y, label, boxes, note in lanes:
        ax.text(0.2, y, label, va="center", fontsize=14, fontweight="bold")
        ax.text(14.6, y, note, va="center", ha="right", fontsize=11, color=MUTED)
        for x, width, text, color in boxes:
            rounded(ax, (x, y - 0.28), width, 0.56, text, facecolor=color, edgecolor="#cad2d9", fontsize=15, color=PURPLE if text == "pi_H" else INK)
        for first, second in zip(boxes, boxes[1:]):
            arrow(ax, (first[0] + first[1], y), (second[0], y), color=GRAY)
    # Trigger diamonds and return gates for switching lanes.
    for y in (1.75, 0.85):
        ax.scatter([5.35], [y], marker="D", s=270, color=RED, edgecolor="white", linewidth=1.4, zorder=5)
        ax.text(5.35, y, "B", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)
        arrow(ax, (4.0, y), (5.05, y), color=GRAY)
        arrow(ax, (5.65, y), (6.2, y), color=RED)
    ax.text(5.35, 2.14, "risky harvest B", color=RED, ha="center", fontsize=9, fontweight="bold")
    ax.scatter([10.55], [0.85], marker="D", s=270, color=GREEN, edgecolor="white", linewidth=1.4, zorder=5)
    ax.text(10.55, 0.85, "0", color="white", ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)
    ax.text(10.55, 0.46, "inventory empty", color=GREEN, ha="center", fontsize=8.5, fontweight="bold")
    arrow(ax, (9.2, 0.85), (10.25, 0.85), color=GRAY)
    arrow(ax, (10.85, 0.85), (11.2, 0.85), color=GREEN)
    ax.text(5.35, 0.23, "Simulator oracle only detects the risky-harvest condition; it does not force a safe action.", ha="center", fontsize=10, color=RED)
    ax.text(14.6, 0.23, "5 seeds x 32 starts x 2 kernels x 4 modes = 1,280 episodes", ha="right", fontsize=10, color=MUTED)
    fig.tight_layout()
    return save(fig, "07_frozen_policy_switch_diagnostic")


def contact_sheet(png_paths: list[Path]) -> Path:
    thumb_size = (720, 405)
    margin = 28
    label_height = 54
    canvas = Image.new("RGB", (thumb_size[0] * 2 + margin * 3, (thumb_size[1] + label_height) * 4 + margin * 5), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(png_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size)
        column = index % 2; row = index // 2
        x = margin + column * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + label_height + margin)
        paste_x = x + (thumb_size[0] - image.width) // 2
        paste_y = y + (thumb_size[1] - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
        draw.text((x, y + thumb_size[1] + 10), path.stem, fill=INK)
    path = OUTPUT / "contact_sheet.png"
    canvas.save(path)
    return path


def main() -> None:
    setup()
    documents, receipts = load_sources()
    rendered = [
        render_routes(documents["mixed_path"]),
        render_harmful_ppo(documents["ppo"]),
        render_lr_tradeoff(documents["lr"]),
        render_probability_transfer(documents["transfer_compact"], documents["transfer_raw"]),
        render_seed_failure(documents["transfer_compact"], documents["transfer_raw"]),
        render_return_decomposition(documents["transfer_compact"], documents["lr"]),
        render_switch_diagnostic(),
    ]
    pngs = [pair[0] for pair in rendered]
    sheet = contact_sheet(pngs)
    receipt = {
        "schema_version": "hackrl_pack_inventory_overwrite_readout_assets_v1",
        "render_script": str(Path(__file__).relative_to(REPOSITORY)),
        "source_sha256": receipts,
        "assets": [
            {"png": str(png.relative_to(REPOSITORY)), "svg": str(svg.relative_to(REPOSITORY))}
            for png, svg in rendered
        ],
        "contact_sheet": str(sheet.relative_to(REPOSITORY)),
        "notes": [
            "All quantitative panels use recorded values; no illustrative numeric data were introduced.",
            "Asset 01 is one selected deterministic path example, not a population average.",
            "Asset 07 is a proposed oracle-assisted execution diagnostic, not a completed result.",
        ],
    }
    (OUTPUT / "sources.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
