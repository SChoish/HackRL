#!/usr/bin/env python3
"""Render the manuscript figure from the frozen teacher-matrix result JSON.

This script performs no rollout or learning.  It plots learner-seed estimates
without treating evaluation episodes as independent replications.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_RESULTS = Path(
    "docs/manifests/cross_environment_teacher_matrix_v1_results.json"
)
DEFAULT_PNG = Path(
    "docs/figures/cross_environment_teacher_matrix_v1_seed_effects.png"
)
DEFAULT_PDF = Path(
    "docs/figures/cross_environment_teacher_matrix_v1_seed_effects.pdf"
)
DEFAULT_CSV = Path(
    "docs/figures/cross_environment_teacher_matrix_v1_seed_effects.csv"
)
AGGREGATOR = Path(__file__).with_name(
    "aggregate_cross_environment_teacher_gpu_evaluation.py"
)
COMPARISONS = (
    ("tick_claim", "TICK", "D-task", "D-frozen", "common_setup"),
    ("craft_remain", "CRAFT", "D-task", "D-frozen", "path_check"),
    ("pack_restore", "PACK", "D-on", "D-off", "common_setup"),
)
COLORS = {
    "natural": "#2563eb",
    "opportunity": "#d97706",
    "left": "#059669",
    "right": "#6b7280",
}


def _load_aggregator():
    spec = importlib.util.spec_from_file_location("teacher_aggregate", AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(result: dict) -> list[dict]:
    aggregate = _load_aggregator()
    output = []
    for environment, label, left, right, opportunity in COMPARISONS:
        contrasts = {}
        for family_name, family in (
            ("natural", "natural_reset"),
            ("opportunity", opportunity),
        ):
            contrast = aggregate._find_contrast(
                result,
                environment=environment,
                left_condition=left,
                right_condition=right,
                policy_view="sample",
                family=family,
            )
            contrasts[family_name] = contrast
            for item in contrast["per_seed"]:
                output.extend(
                    [
                        {
                            "environment": environment,
                            "environment_label": label,
                            "comparison": f"{left} - {right}",
                            "seed": item["seed"],
                            "panel": "exploit_adoption_contrast",
                            "series": family_name,
                            "value": 100.0 * item["exploit_rate"],
                        },
                        {
                            "environment": environment,
                            "environment_label": label,
                            "comparison": f"{left} - {right}",
                            "seed": item["seed"],
                            "panel": "discounted_return_contrast",
                            "series": family_name,
                            "value": item["mean_discounted_return"],
                        },
                    ]
                )
        for condition, series in ((left, "left"), (right, "right")):
            matches = [
                row
                for row in result["per_policy_panel"]
                if row["environment"] == environment
                and row["condition"] == condition
                and row["trained_variant"] == "mutant"
                and row["kernel"] == "fixed"
                and row["policy_view"] == "mode"
                and row["family"] == "natural_reset"
            ]
            if sorted(row["seed"] for row in matches) != contrasts["natural"]["seeds"]:
                raise RuntimeError(
                    f"normal-performance seeds drifted for {environment} {condition}"
                )
            output.extend(
                {
                    "environment": environment,
                    "environment_label": label,
                    "comparison": f"{left} - {right}",
                    "seed": row["seed"],
                    "panel": "normal_success",
                    "series": series,
                    "value": row["success_rate"],
                }
                for row in matches
            )
    return sorted(
        output,
        key=lambda row: (
            row["panel"],
            row["environment"],
            row["seed"],
            row["series"],
        ),
    )


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _values(rows: list[dict], environment: str, panel: str, series: str) -> list[dict]:
    return [
        row
        for row in rows
        if row["environment"] == environment
        and row["panel"] == panel
        and row["series"] == series
    ]


def _paired_points(ax, rows, panel, series, offsets, scale=1.0):
    for index, (environment, label, left, right, _) in enumerate(COMPARISONS):
        left_rows = _values(rows, environment, panel, series[0])
        right_rows = _values(rows, environment, panel, series[1])
        left_by_seed = {row["seed"]: scale * row["value"] for row in left_rows}
        right_by_seed = {row["seed"]: scale * row["value"] for row in right_rows}
        if set(left_by_seed) != set(right_by_seed):
            raise RuntimeError(f"unpaired plotted seeds: {environment} {panel}")
        ordered_seeds = sorted(left_by_seed)
        for rank, seed in enumerate(ordered_seeds):
            jitter = (rank - (len(ordered_seeds) - 1) / 2.0) * 0.010
            xs = [index + offsets[0] + jitter, index + offsets[1] + jitter]
            ys = [left_by_seed[seed], right_by_seed[seed]]
            ax.plot(xs, ys, color="#cbd5e1", linewidth=0.7, zorder=1)
            ax.scatter(
                xs,
                ys,
                s=19,
                color=[COLORS[series[0]], COLORS[series[1]]],
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
        for offset, values in (
            (offsets[0], left_by_seed.values()),
            (offsets[1], right_by_seed.values()),
        ):
            mean = sum(values) / len(values)
            ax.scatter(
                [index + offset],
                [mean],
                marker="D",
                s=54,
                color="#111827",
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        count = len(left_by_seed)
        ax.text(
            index,
            -0.18,
            f"{label}\n{left} − {right}\nn={count}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
        )


def render(result: dict, png: Path, pdf: Path, csv_path: Path) -> list[dict]:
    rows = _rows(result)
    _atomic_csv(csv_path, rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.65), dpi=180)

    _paired_points(
        axes[0],
        rows,
        "exploit_adoption_contrast",
        ("natural", "opportunity"),
        (-0.15, 0.15),
    )
    axes[0].axhline(0, color="#111827", linewidth=0.7)
    axes[0].set_ylabel(r"Exploit-adoption contrast $\Delta U$ (pp)")
    axes[0].set_title(
        "a  Between-condition exploit adoption", loc="left", fontweight="bold"
    )
    axes[0].text(
        0.99,
        -0.08,
        "Direct exploitation rates are not shown",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#4b5563",
    )
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["natural"], label="Natural"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["opportunity"], label="Opportunity"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#111827", label="Seed mean"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )

    _paired_points(
        axes[1],
        rows,
        "normal_success",
        ("left", "right"),
        (-0.15, 0.15),
    )
    axes[1].set_ylim(-0.03, 1.07)
    axes[1].set_ylabel("Fixed-kernel success rate")
    axes[1].set_title("b  Normal-task preservation", loc="left", fontweight="bold")
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["left"], label="Left condition"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["right"], label="Right condition"),
        ],
        frameon=False,
        fontsize=7,
        loc="lower left",
    )

    _paired_points(
        axes[2],
        rows,
        "discounted_return_contrast",
        ("natural", "opportunity"),
        (-0.15, 0.15),
    )
    axes[2].axhline(0, color="#111827", linewidth=0.7)
    axes[2].set_ylabel(r"Discounted-return contrast $\Delta\Delta J$")
    axes[2].set_title("c  Task return", loc="left", fontweight="bold")

    for ax in axes:
        ax.set_xlim(-0.48, len(COMPARISONS) - 0.52)
        ax.set_xticks([])
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.31, wspace=0.39)
    for path in (png, pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=240)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "HackRL", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    import json

    result = json.loads(args.results.read_text())
    rows = render(result, args.png, args.pdf, args.csv)
    print(
        f"rendered {len(rows)} plotted seed values: {args.png}, {args.pdf}, {args.csv}"
    )


if __name__ == "__main__":
    main()
