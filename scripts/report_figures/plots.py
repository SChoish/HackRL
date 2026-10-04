"""Result figures from saved curves. No new rollouts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from report_figures.common import ROOT as REPO_ROOT

ROOT = REPO_ROOT / "runs"
OUT = ROOT / "figures_report_v1" / "results"
SEEDS = (20, 21, 22, 23, 24)
LOADED_SOURCES = {}
UPDATES = (0, 32, 128, 256, 512, 1024, 2048, 4096)
TRANSITIONS_PER_UPDATE = 512 * 64
COLORS = {
    "GC-PPO": "#4c78a8",
    "Dual": "#f58518",
    "BC-off": "#54a24b",
    "teacher only": "#e45756",
    "frozen imitation": "#b279a2",
}


def _load(path):
    path = Path(path)
    raw = path.read_bytes()
    LOADED_SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def _family(document, kernel, slice_name):
    return document[kernel][slice_name]["natural_reset"]


def _exploit(block):
    if "violation_delivery_rate" in block:
        return float(block["violation_delivery_rate"])
    return float(block["exploit_rate"])


def _success(block):
    return float(block["success_rate"])


def _return(block):
    return float(block["mean_discounted_return"])


def _curve(root, env, method, variant, seed, update):
    base = root if env is None else root / env
    return _load(base / method / variant / f"seed{seed}" / "curve" / f"adapt_{update}.json")


def _bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(4000, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _panel(ax, groups, ylabel, title):
    rng = np.random.default_rng(0)
    for index, (label, values) in enumerate(groups):
        mean, low, high = _bootstrap_mean(values, rng)
        ax.plot([index, index], [low, high], color="#222222", lw=1.2, zorder=2)
        ax.scatter([index], [mean], s=36, c="#222222", zorder=3)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=28, c="#4c78a8", zorder=4)
        buckets = {}
        for seed, value in zip(SEEDS, values):
            buckets.setdefault(round(float(value), 4), []).append(str(seed))
        for value, names in buckets.items():
            ax.text(index + 0.18, value, ",".join(names), fontsize=6, va="center", color="#333333")
    ax.set_xlim(-0.45, len(groups) - 0.25)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for label, _ in groups], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    values = [v for _, group in groups for v in group]
    ax.set_ylim(min(-0.05, min(values) - 0.05), max(1.18, max(values) + 0.05))
    ax.axhline(0, color="#aaaaaa", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _edv_groups(env, methods, slice_name, root_for):
    groups = {"E": [], "D": [], "V": []}
    for label, method, root in methods:
        success, excess, gain = [], [], []
        for seed in SEEDS:
            mutant = _curve(root, env, method, "mutant", seed, 4096)
            fixed_branch = _curve(root, env, method, "fixed", seed, 4096)
            on_fixed = _family(mutant, "fixed", slice_name)
            on_mutant = _family(mutant, "mutant", slice_name)
            continued = _family(fixed_branch, "mutant", slice_name)
            success.append(_success(on_fixed))
            excess.append(_exploit(on_mutant) - _exploit(continued))
            gain.append(_return(on_mutant) - _return(on_fixed))
        groups["E"].append((label, success))
        groups["D"].append((label, excess))
        groups["V"].append((label, gain))
    return groups


def draw_edv():
    compare = ROOT / "dual_leo_compare_v1"
    methods = (("Tick GC-PPO", "gc", compare), ("Tick Dual", "dual", compare), ("Pack GC-PPO", "gc", compare), ("Pack Dual", "dual", compare))
    # Build per-env then interleave by taking tick from env tick and pack from env pack.
    tick = _edv_groups("tick", (("GC-PPO", "gc", compare), ("Dual", "dual", compare)), "mode", None)
    pack = _edv_groups("pack", (("GC-PPO", "gc", compare), ("Dual", "dual", compare)), "mode", None)
    groups = {
        key: [("TICK " + label, values) for label, values in tick[key]] + [("PACK " + label, values) for label, values in pack[key]]
        for key in ("E", "D", "V")
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), dpi=140)
    _panel(axes[0], groups["E"], "Success rate", "E  adapted policy on fixed kernel")
    _panel(axes[1], groups["D"], "Excess-delivery difference", "D  mutant adaptation minus fixed continuation")
    _panel(axes[2], groups["V"], "Discounted-return difference", "V  same policy, mutant minus fixed kernel")
    returns = [value for _, values in groups["V"] for value in values]
    axes[2].set_ylim(min(returns) - 0.02, max(returns) + 0.02)
    fig.suptitle("Adaptation 4096, natural reset, mode. Points are seeds 20–24. Seed bootstrap 95% interval; n=5, descriptive uncertainty.", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = OUT / "edv_tick_pack.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    craft_root = ROOT / "craft_remain_compare_v1"
    craft = _edv_groups(None, (("GC-PPO", "gc", craft_root), ("Dual", "dual", craft_root)), "sample", None)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3), dpi=140)
    _panel(axes[0], craft["E"], "Success rate", "E  adapted policy on fixed kernel")
    _panel(axes[1], craft["D"], "Exploit difference", "D  adaptation minus continuation")
    _panel(axes[2], craft["V"], "Return difference", "V  same policy, two kernels")
    returns = [value for _, values in craft["V"] for value in values]
    low, high = min(returns), max(returns)
    axes[2].set_ylim(low - 0.02, high + 0.02)
    fig.suptitle("CRAFT-REMAIN: sample, natural reset, adapt 4096; seed bootstrap 95% interval (n=5). Growth period 16.", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "edv_craft_sample.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("edv", path)


def _series(root, env, method, seed):
    success, exploit, gain = [], [], []
    for update in UPDATES:
        document = _curve(root, env, method, "mutant", seed, update)
        success.append(_success(_family(document, "fixed", "mode")))
        exploit.append(_exploit(_family(document, "mutant", "mode")))
        gain.append(_return(_family(document, "mutant", "mode")) - _return(_family(document, "fixed", "mode")))
    return np.asarray(success), np.asarray(exploit), np.asarray(gain)


def draw_curves():
    specs = (
        ("GC-PPO", ROOT / "dual_leo_compare_v1", "gc"),
        ("Dual", ROOT / "dual_leo_compare_v1", "dual"),
        ("BC-off", ROOT / "dual_adapt_bc_off_v1", "adapt_bc_off"),
    )
    x = np.asarray(UPDATES) * TRANSITIONS_PER_UPDATE / 1e6
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.2), dpi=140, sharex=True)
    metrics = ("Adapted policy on fixed kernel", "Mutant-kernel excess delivery", "Same-policy return difference")
    for column, env in enumerate(("tick", "pack")):
        collected = {label: [] for label, _, _ in specs}
        for label, root, method in specs:
            for seed in SEEDS:
                collected[label].append(_series(root, env, method, seed))
        for row in range(3):
            ax = axes[row, column]
            for label, _, _ in specs:
                stack = np.stack([item[row] for item in collected[label]])
                for line in stack:
                    ax.plot(x, line, color=COLORS[label], lw=0.7, alpha=0.45)
                    positive = np.flatnonzero(line > 0)
                    if row == 1 and positive.size:
                        ax.scatter([x[positive[0]]], [line[positive[0]]], s=12, facecolors="none", edgecolors=COLORS[label], zorder=3)
                ax.plot(x, stack.mean(axis=0), color=COLORS[label], lw=2.2, label=label)
            ax.set_title(f"{env.upper()}  {metrics[row]}", loc="left", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 2:
                ax.set_xlabel("Adaptation transitions (millions)")
            if column == 0:
                ax.set_ylabel(metrics[row])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Open circles on the excess-delivery row are the first stored checkpoint above zero, not the exact discovery time.", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT / "learning_curves.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("curves", path)


def _final_exploit_success(root, env, method, slice_name="mode"):
    exploit, success = [], []
    for seed in SEEDS:
        document = _curve(root, env, method, "mutant", seed, 4096)
        exploit.append(_exploit(_family(document, "mutant", slice_name)))
        success.append(_success(_family(document, "fixed", slice_name)))
    return exploit, success


def draw_causes():
    arms = (
        ("GC-PPO", ROOT / "dual_leo_compare_v1", "gc"),
        ("teacher only", ROOT / "teacher_imitation_split_v1", "teacher_only"),
        ("frozen imitation", ROOT / "teacher_imitation_split_v1", "frozen_imitation"),
        ("Dual", ROOT / "dual_leo_compare_v1", "dual"),
    )
    fig = plt.figure(figsize=(12.2, 7.2), dpi=140)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8])
    for column, env in enumerate(("tick", "pack")):
        ax = fig.add_subplot(grid[0, column])
        for index, (label, root, method) in enumerate(arms):
            exploit, success = _final_exploit_success(root, env, method)
            ax.scatter(np.full(5, index) - 0.08, exploit, s=26, c=COLORS[label], label="excess delivery" if column == 0 and index == 0 else None)
            ax.scatter(np.full(5, index) + 0.08, success, s=26, marker="D", c=COLORS[label], label="adapted policy on fixed" if column == 0 and index == 0 else None)
            ax.hlines(np.mean(exploit), index - 0.18, index - 0.02, colors=COLORS[label], lw=2)
            ax.hlines(np.mean(success), index + 0.02, index + 0.18, colors=COLORS[label], lw=2)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([label for label, _, _ in arms], rotation=15, ha="right")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"{env.upper()}  adapt 4096 mode", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if column == 0:
            ax.set_ylabel("Rate")
            ax.legend(frameon=False, fontsize=8)
    inset = fig.add_subplot(grid[1, 0])
    x = np.asarray(UPDATES) * TRANSITIONS_PER_UPDATE / 1e6
    success, exploit, _ = _series(ROOT / "teacher_imitation_split_v1", "tick", "teacher_only", 20)
    inset.plot(x, exploit, color=COLORS["teacher only"], lw=2, label="seed 20 excess delivery")
    inset.plot(x, success, color=COLORS["teacher only"], lw=2, ls="--", label="seed 20 normal success")
    inset.set_title("TICK teacher-only seed 20 history", loc="left", fontsize=10)
    inset.set_xlabel("Adaptation transitions (millions)")
    inset.legend(frameon=False, fontsize=8)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    paired_grid = grid[1, 1].subgridspec(1, 2, wspace=0.45)
    for column, env in enumerate(("tick", "pack")):
        paired = fig.add_subplot(paired_grid[column])
        on, _ = _final_exploit_success(ROOT / "dual_leo_compare_v1", env, "dual")
        off, _ = _final_exploit_success(ROOT / "dual_adapt_bc_off_v1", env, "adapt_bc_off")
        pile = {}
        for seed, left, right in zip(SEEDS, on, off):
            paired.plot([0, 1], [left, right], color="#888888", lw=0.8)
            paired.scatter([0, 1], [left, right], s=22, c="#222222")
            key = round(float(right), 4)
            slot = pile.get(key, 0)
            pile[key] = slot + 1
            paired.text(1.08, right + slot * 0.045, str(seed), fontsize=7, va="center")
        paired.set_xticks([0, 1])
        paired.set_xticklabels(["BC on", "BC off"])
        paired.set_xlim(-0.15, 1.4)
        paired.set_ylim(-0.05, 1.15)
        paired.set_title(env.upper(), loc="left", fontsize=10)
        paired.spines["top"].set_visible(False)
        paired.spines["right"].set_visible(False)
        if column == 0:
            paired.set_ylabel("Mutant-kernel excess delivery")
    fig.suptitle("Natural reset, mode. Top: full-history interventions. Bottom right: adaptation-only BC intervention.\nTeacher-only and GC-PPO use different RNG paths; this is not a paired teacher-effect estimate.", fontsize=9, y=0.01)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUT / "cause_split.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("causes", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_edv()
    draw_curves()
    draw_causes()
    (OUT / "sources.json").write_text(json.dumps(LOADED_SOURCES, indent=2) + "\n")


if __name__ == "__main__":
    main()
