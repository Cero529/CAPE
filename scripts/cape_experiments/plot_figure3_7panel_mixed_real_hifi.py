"""Create the complete seven-line-panel Figure 3 planning preview.

Evidence policy
---------------
* CAPE in panels (b) and (e) is read from the real five-seed HiFi histories.
* Panel (g) is a real five-seed CAPE HiFi per-task AUC retention plot.
* All baseline curves and the AV1M/heterogeneous stage profiles remain
  Table-II-constrained planning curves until their histories are available.

The figure is therefore deliberately watermarked as a mixed-evidence preview
and must not be presented as a fully measured baseline comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "cape_experiments" / "plot_predicted_figure3_tmm_7panel.py"
PAPER_DIR = (
    PROJECT_ROOT
    / "paper"
    / "cape"
    / "CAPE__Continual_Audio_Visual_Pattern_Experts_for_Open_World_Deepfake_Detection"
)
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "figure3" / "hifi" / "cape_grouped_clean"
DEFAULT_OUTPUT_DIR = PAPER_DIR / "figure3_mixed_real_hifi_preview"
OUTPUT_STEM = "figure3_tmm_7panel_hifi_cape_real_mixed_preview"

TASKS = [
    "generator:kling2.5",
    "generator:veo3.1",
    "generator:wan2.5",
    "generator:seedance1.0",
]
TASK_LABELS = ["Kling2.5", "Veo3.1", "Wan2.5", "Seedance1.0"]
TASK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
TASK_MARKERS = ["o", "s", "^", "D"]


def load_base_module():
    spec = importlib.util.spec_from_file_location("cape_figure3_planning_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_histories(result_root: Path) -> List[Tuple[Path, List[dict]]]:
    paths = sorted(result_root.glob("seed_*/history.json"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five histories under {result_root}, found {len(paths)}")
    runs = []
    for path in paths:
        history = json.loads(path.read_text(encoding="utf-8"))
        order = [str(row["current_task"]) for row in history]
        if order != TASKS:
            raise RuntimeError(f"Unexpected task order in {path}: {order}")
        if any(len(row.get("train", [])) != 20 for row in history):
            raise RuntimeError(f"Incomplete 20-epoch stage in {path}")
        runs.append((path, history))
    return runs


def stage_average(history: Sequence[dict], metric: str) -> np.ndarray:
    return np.asarray(
        [
            np.mean([float(item[metric]) for item in row["seen"].values()])
            for row in history
        ],
        dtype=float,
    )


def aggregate_stage_curve(
    runs: Sequence[Tuple[Path, List[dict]]], metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_seed = np.stack([stage_average(history, metric) for _, history in runs])
    return per_seed, per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)


def aggregate_retention(
    runs: Sequence[Tuple[Path, List[dict]]], metric: str = "auc"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((len(runs), len(TASKS), len(TASKS)), np.nan, dtype=float)
    for seed_index, (_, history) in enumerate(runs):
        for train_stage, row in enumerate(history):
            for task_index, task in enumerate(TASKS):
                if task in row["seen"]:
                    values[seed_index, train_stage, task_index] = float(
                        row["seen"][task][metric]
                    )
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.full(values.shape[1:], np.nan, dtype=float)
    sample_sd = np.full(values.shape[1:], np.nan, dtype=float)
    for stage in range(values.shape[1]):
        for task in range(values.shape[2]):
            cell = values[:, stage, task]
            cell = cell[np.isfinite(cell)]
            if len(cell):
                mean[stage, task] = float(cell.mean())
            if len(cell) >= 2:
                sample_sd[stage, task] = float(cell.std(ddof=1))
    return mean, sample_sd, count


def update_table2_with_real_hifi(base, table2, real_curves):
    """Replace only CAPE's HiFi aggregate cells in the in-memory Table II."""

    auc_per_seed, auc_mean, _ = real_curves["AUC"]
    ap_per_seed, ap_mean, _ = real_curves["AP"]
    real_results = {
        "AUC": (
            base.Result(float(auc_per_seed.mean(axis=1).mean()), float(auc_per_seed.mean(axis=1).std(ddof=1))),
            base.Result(float(auc_mean[-1]), float(auc_per_seed[:, -1].std(ddof=1))),
        ),
        "AP": (
            base.Result(float(ap_per_seed.mean(axis=1).mean()), float(ap_per_seed.mean(axis=1).std(ddof=1))),
            base.Result(float(ap_mean[-1]), float(ap_per_seed[:, -1].std(ddof=1))),
        ),
    }
    for metric, (trajectory, final) in real_results.items():
        table2[metric]["CAPE"][2] = trajectory
        table2[metric]["CAPE"][3] = final
    return real_results


def add_real_hifi_uncertainty(axis, mean: np.ndarray, sample_sd: np.ndarray) -> None:
    x = np.arange(1, 5)
    lower = np.clip(mean - sample_sd, 0.0, 1.0)
    upper = np.clip(mean + sample_sd, 0.0, 1.0)
    axis.fill_between(
        x,
        lower,
        upper,
        color="#0072B2",
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    axis.errorbar(
        x,
        mean,
        yerr=sample_sd,
        fmt="none",
        ecolor="#0072B2",
        elinewidth=0.65,
        capsize=1.6,
        capthick=0.65,
        zorder=6,
    )


def plot_retention_panel(
    base,
    axis,
    retention_mean: np.ndarray,
    retention_sd: np.ndarray,
    csv_rows: List[Dict[str, object]],
) -> None:
    for task_index, (label, color, marker) in enumerate(
        zip(TASK_LABELS, TASK_COLORS, TASK_MARKERS)
    ):
        stages = np.flatnonzero(np.isfinite(retention_mean[:, task_index]))
        x = stages + 1
        y = retention_mean[stages, task_index]
        sd = retention_sd[stages, task_index]
        axis.plot(
            x,
            y,
            color=color,
            marker=marker,
            linewidth=1.15,
            markersize=2.6,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=label,
            zorder=4,
        )
        if len(x) > 1:
            axis.fill_between(
                x,
                np.clip(y - sd, 0.0, 1.0),
                np.clip(y + sd, 0.0, 1.0),
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
        for stage, value, sample_sd in zip(x, y, sd):
            csv_rows.append(
                {
                    "panel": "g",
                    "record_type": "real_task_retention",
                    "evidence_type": "real_multiseed_history",
                    "protocol": "HiFi Generator-Incremental",
                    "metric": "AUC",
                    "method": "CAPE",
                    "evaluation_task": TASKS[task_index],
                    "stage": int(stage),
                    "value": f"{value:.9f}",
                    "sample_sd": f"{sample_sd:.9f}",
                    "n_seeds": 5,
                    "note": "real per-task AUC mean and sample SD from five histories",
                }
            )

    axis.set_title("(g) HiFi CAPE - AUC retention", loc="left", fontweight="bold", pad=3.0)
    axis.set_xlim(0.8, 4.2)
    axis.set_ylim(0.88, 1.005)
    axis.set_xticks([1, 2, 3, 4])
    axis.set_xlabel("Training stage completed")
    axis.set_ylabel("Per-task AUC")
    axis.grid(color="#D8D8D8", linewidth=0.45, alpha=0.9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.0, width=0.55, pad=1.5)
    axis.legend(
        loc="lower left",
        frameon=False,
        ncol=1,
        fontsize=4.3,
        labelspacing=0.20,
        handlelength=1.6,
        borderaxespad=0.15,
    )


def create_figure(base, table2, real_curves, retention_mean, retention_sd):
    fig = plt.figure(figsize=(7.16, 4.05))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.48,
        wspace=0.33,
    )
    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "c": fig.add_subplot(grid[0, 2]),
        "legend": fig.add_subplot(grid[0, 3]),
        "d": fig.add_subplot(grid[1, 0]),
        "e": fig.add_subplot(grid[1, 1]),
        "f": fig.add_subplot(grid[1, 2]),
        "g": fig.add_subplot(grid[1, 3]),
    }

    rows: List[Dict[str, object]] = []
    panels = [
        ("a", "AV1M - AUC", "AUC", 0, 3, base.AV1M_PROTOCOL),
        ("b", "HiFi - AUC", "AUC", 2, 4, base.HIFI_PROTOCOL),
        ("c", "Heterogeneous - AUC", "AUC", 4, 12, "Heterogeneous Long-Stream"),
        ("d", "AV1M - AP", "AP", 0, 3, base.AV1M_PROTOCOL),
        ("e", "HiFi - AP", "AP", 2, 4, base.HIFI_PROTOCOL),
        ("f", "Heterogeneous - AP", "AP", 4, 12, "Heterogeneous Long-Stream"),
    ]
    raw_stage_rows: List[Dict[str, object]] = []
    for key, title, metric, offset, n_stages, protocol in panels:
        axis = axes[key]
        base.plot_method_curves(
            axis,
            table2,
            metric,
            offset,
            n_stages,
            protocol,
            raw_stage_rows,
        )
        if key in {"b", "e"}:
            add_real_hifi_uncertainty(
                axis,
                real_curves[metric][1],
                real_curves[metric][2],
            )
        axis.set_title(f"({key}) {title}", loc="left", fontweight="bold", pad=3.0)
        axis.set_ylim(0.50, 1.005)
        ylabel = "Avg. AUC over seen stages" if key == "a" else None
        if key == "d":
            ylabel = "Avg. AP over seen stages"
        base.style_axis(axis, n_stages, ylabel)

    for row in raw_stage_rows:
        is_real = (
            row["protocol"] == base.HIFI_PROTOCOL
            and row["method"] == "CAPE"
            and row["record_type"] == "stage_metric"
        )
        rows.append(
            {
                "panel": {
                    (base.AV1M_PROTOCOL, "AUC"): "a",
                    (base.HIFI_PROTOCOL, "AUC"): "b",
                    ("Heterogeneous Long-Stream", "AUC"): "c",
                    (base.AV1M_PROTOCOL, "AP"): "d",
                    (base.HIFI_PROTOCOL, "AP"): "e",
                    ("Heterogeneous Long-Stream", "AP"): "f",
                }[(row["protocol"], row["metric"])],
                "record_type": row["record_type"],
                "evidence_type": (
                    "real_multiseed_history" if is_real else "table_constrained_planning"
                ),
                "protocol": row["protocol"],
                "metric": row["metric"],
                "method": row["method"],
                "evaluation_task": "",
                "stage": row["stage"],
                "value": row["predicted_value"],
                "sample_sd": (
                    f"{real_curves[row['metric']][2][int(row['stage']) - 1]:.9f}"
                    if is_real
                    else ""
                ),
                "n_seeds": 5 if is_real else "",
                "note": (
                    "real five-seed mean; no interpolation"
                    if is_real
                    else row["note"]
                ),
            }
        )

    plot_retention_panel(base, axes["g"], retention_mean, retention_sd, rows)

    legend_axis = axes["legend"]
    legend_axis.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=base.METHOD_COLORS[method],
            marker=base.METHOD_MARKERS[method],
            linestyle=base.METHOD_LINESTYLES[method],
            linewidth=1.75 if method == "CAPE" else 0.95,
            markersize=3.0 if method == "CAPE" else 2.4,
            label=method,
        )
        for method in base.DISPLAY_METHODS
    ]
    legend_axis.text(
        0.02,
        0.99,
        "Methods",
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        transform=legend_axis.transAxes,
    )
    legend_axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.92),
        frameon=False,
        ncol=1,
        handlelength=2.1,
        labelspacing=0.28,
        borderaxespad=0.0,
    )
    legend_axis.text(
        0.02,
        0.03,
        "Real evidence:\n"
        "(b),(e) CAPE HiFi\n"
        "(g) CAPE retention\n\n"
        "Other curves:\n"
        "Table-constrained plans",
        ha="left",
        va="bottom",
        fontsize=5.0,
        color="#444444",
        linespacing=1.15,
        transform=legend_axis.transAxes,
    )

    fig.text(
        0.50,
        0.50,
        "MIXED PREVIEW - HIFI CAPE REAL; OTHER CURVES PLANNED",
        ha="center",
        va="center",
        fontsize=14,
        color="#A33A2B",
        alpha=0.075,
        rotation=18,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.007,
        "Real five-seed HiFi CAPE data are used only in (b), (e), and (g); "
        "all remaining curves are Table-II-constrained planning profiles.",
        ha="center",
        va="bottom",
        fontsize=5.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.067, right=0.995, bottom=0.105, top=0.965)
    return fig, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "panel",
        "record_type",
        "evidence_type",
        "protocol",
        "metric",
        "method",
        "evaluation_task",
        "stage",
        "value",
        "sample_sd",
        "n_seeds",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot seven-panel Figure 3 with real HiFi CAPE.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result_root = args.result_root if args.result_root.is_absolute() else PROJECT_ROOT / args.result_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    base = load_base_module()
    base.configure_style()
    runs = load_histories(result_root)
    real_curves = {
        "AUC": aggregate_stage_curve(runs, "auc"),
        "AP": aggregate_stage_curve(runs, "ap"),
    }
    retention_mean, retention_sd, retention_count = aggregate_retention(runs, "auc")

    table2 = base.parse_table2(base.TEX_PATH.read_text(encoding="utf-8"))
    real_results = update_table2_with_real_hifi(base, table2, real_curves)
    original_constrained_curve = base.constrained_curve

    def mixed_curve(
        trajectory_mean: float,
        final_mean: float,
        n_stages: int,
        protocol: str,
        metric: str,
        method: str,
    ) -> np.ndarray:
        if protocol == base.HIFI_PROTOCOL and method == "CAPE":
            values = real_curves[metric][1].copy()
            if not np.isclose(values.mean(), trajectory_mean, atol=1e-12):
                raise AssertionError(f"Real {metric} trajectory mismatch")
            if not np.isclose(values[-1], final_mean, atol=1e-12):
                raise AssertionError(f"Real {metric} final mismatch")
            return values
        return original_constrained_curve(
            trajectory_mean,
            final_mean,
            n_stages,
            protocol,
            metric,
            method,
        )

    base.constrained_curve = mixed_curve
    fig, rows = create_figure(
        base,
        table2,
        real_curves,
        retention_mean,
        retention_sd,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{OUTPUT_STEM}.{extension}", dpi=600)
    plt.close(fig)
    write_csv(output_dir / f"{OUTPUT_STEM}.csv", rows)

    manifest = {
        "status": "mixed_evidence_planning_preview",
        "figure": "Figure 3 seven line panels",
        "script": str(Path(__file__).resolve()),
        "source_tex": str(base.TEX_PATH),
        "real_result_root": str(result_root),
        "real_history_sha256": [
            {"path": str(path), "sha256": sha256_file(path)} for path, _ in runs
        ],
        "real_panels": {
            "b": "CAPE HiFi stage-average AUC, mean ± sample SD over five seeds",
            "e": "CAPE HiFi stage-average AP, mean ± sample SD over five seeds",
            "g": "CAPE HiFi per-task AUC retention, mean ± sample SD over five seeds",
        },
        "planning_panels_or_series": (
            "All baseline series in (a)-(f), and all AV1M/heterogeneous stage "
            "profiles, remain Table-II-constrained planning curves."
        ),
        "real_hifi_aggregates": {
            metric: {
                "trajectory": {
                    "mean": real_results[metric][0].mean,
                    "sample_sd": real_results[metric][0].sd,
                },
                "final": {
                    "mean": real_results[metric][1].mean,
                    "sample_sd": real_results[metric][1].sd,
                },
                "stage_mean": real_curves[metric][1].tolist(),
                "stage_sample_sd": real_curves[metric][2].tolist(),
            }
            for metric in ("AUC", "AP")
        },
        "retention_observation_count": retention_count.tolist(),
        "warning": "Do not use as a fully measured baseline comparison.",
    }
    (output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Generated {output_dir / (OUTPUT_STEM + '.png')}")
    print(f"Generated {output_dir / (OUTPUT_STEM + '.pdf')}")
    print(f"Generated {output_dir / (OUTPUT_STEM + '.svg')}")
    print(f"Generated {output_dir / (OUTPUT_STEM + '.csv')}")
    print("Real panels: (b), (e), (g); all remaining series are planning curves.")


if __name__ == "__main__":
    main()
