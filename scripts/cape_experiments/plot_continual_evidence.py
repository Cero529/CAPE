"""Create the three CAPE continual-learning evidence figures.

The script never reconstructs or interpolates missing experimental values.
Every plotted cell is read from a real ``history.json`` produced by the
CAPE/baseline trainers.  Multiple histories assigned to the same method are
treated as independent seeds.

Required history schema
-----------------------
Each history is a list of post-stage records::

    {
      "current_task": "generator:kling2.5",
      "seen": {
        "generator:kling2.5": {
          "auc": 0.95,
          "ap": 0.94,
          "expert_mean_weight": [0.7, 0.3],
          "expert_top1_rate": [0.8, 0.2],
          "num_eval_samples": 62
        }
      }
    }

The expert fields are emitted by ``CAPEContinualTrainer.evaluate_task``.
Baseline histories need only contain AUC/AP.

Examples
--------
One CAPE run (useful only as a diagnostic preview)::

    python scripts/cape_experiments/plot_continual_evidence.py \
      --series "CAPE=results/cape_generator_e20_with_usage/history.json" \
      --matrix-method CAPE --routing-method CAPE

Five CAPE seeds plus baselines::

    python scripts/cape_experiments/plot_continual_evidence.py \
      --series "CAPE=results/cape_multiseed/seed_*/history.json" \
      --series "ER=results/baselines/er/seed_*/history.json" \
      --series "MoE-Adapters=results/baselines/moe_adapters/seed_*/history.json" \
      --matrix-method CAPE --routing-method CAPE \
      --output-dir paper/cape/continual_evidence_figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASK_LABELS = {
    "1": "Visual-only",
    "2": "Audio-only",
    "3": "Joint A-V",
    "generator:kling2.5": "Kling2.5",
    "generator:veo3.1": "Veo3.1",
    "generator:wan2.5": "Wan2.5",
    "generator:seedance1.0": "Seedance1.0",
    "faceswap": "FaceSwap",
    "fsgan": "FSGAN",
    "wav2lip": "Wav2Lip",
    "rtvc": "RTVC",
    "joint_av": "Joint A-V",
}

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "P", "X"]


def configure_style() -> None:
    """Use a compact IEEE-compatible and colorblind-safe style."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_history(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} is not a non-empty history list")
    for index, row in enumerate(history, start=1):
        if "current_task" not in row or "seen" not in row:
            raise ValueError(f"{path}: stage {index} lacks current_task/seen")
    return history


def expand_series(specs: Sequence[str]) -> "OrderedDict[str, List[Tuple[Path, List[dict]]]]":
    grouped: "OrderedDict[str, List[Tuple[Path, List[dict]]]]" = OrderedDict()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--series must be METHOD=GLOB, received: {spec}")
        method, pattern = spec.split("=", 1)
        method = method.strip()
        pattern_path = Path(pattern.strip())
        if not pattern_path.is_absolute():
            pattern = str(PROJECT_ROOT / pattern_path)
        paths = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if not paths:
            raise FileNotFoundError(f"No history matches {spec!r}")
        grouped.setdefault(method, [])
        known = {item[0].resolve() for item in grouped[method]}
        for path in paths:
            if path.is_dir():
                path = path / "history.json"
            if path.resolve() not in known:
                grouped[method].append((path, load_history(path)))
                known.add(path.resolve())
    return grouped


def task_order(history: Sequence[dict]) -> List[str]:
    return [str(row["current_task"]) for row in history]


def short_label(task: str) -> str:
    if task in TASK_LABELS:
        return TASK_LABELS[task]
    if task.startswith("generator:"):
        return task.split(":", 1)[1]
    return task.replace("_", " ")


def validate_same_protocol(runs: Sequence[Tuple[Path, List[dict]]], method: str) -> List[str]:
    reference = task_order(runs[0][1])
    for path, history in runs[1:]:
        current = task_order(history)
        if current != reference:
            raise ValueError(
                f"{method}: task order differs between seeds.\n"
                f"reference={reference}\n{path}={current}"
            )
    return reference


def stage_average(history: Sequence[dict], metric: str) -> np.ndarray:
    values = []
    for row in history:
        current = []
        for item in row.get("seen", {}).values():
            value = item.get(metric, float("nan"))
            if value is not None and np.isfinite(float(value)):
                current.append(float(value))
        values.append(float(np.mean(current)) if current else float("nan"))
    return np.asarray(values, dtype=float)


def performance_matrix(history: Sequence[dict], metric: str, tasks: Sequence[str]) -> np.ndarray:
    matrix = np.full((len(history), len(tasks)), np.nan, dtype=float)
    for train_index, row in enumerate(history):
        seen = row.get("seen", {})
        for eval_index, task in enumerate(tasks):
            if task in seen and metric in seen[task]:
                matrix[train_index, eval_index] = float(seen[task][metric])
    return matrix


def aggregate_mean_std(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.stack(arrays, axis=0)
    count = np.sum(np.isfinite(stacked), axis=0)
    mean = np.nanmean(stacked, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.nanstd(stacked, axis=0, ddof=1)
    std = np.where(count >= 2, std, np.nan)
    return mean, std, count


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}")
    plt.close(fig)


def write_stage_curve_csv(
    output_dir: Path,
    grouped: Mapping[str, Sequence[Tuple[Path, List[dict]]]],
) -> None:
    with (output_dir / "p0_1_stagewise_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "stage", "task", "metric", "mean", "sample_sd", "n_seeds"])
        for method, runs in grouped.items():
            tasks = validate_same_protocol(runs, method)
            for metric in ("auc", "ap"):
                mean, std, count = aggregate_mean_std([stage_average(h, metric) for _, h in runs])
                for index, task in enumerate(tasks):
                    writer.writerow(
                        [method, index + 1, task, metric, mean[index], std[index], int(count[index])]
                    )


def plot_stagewise_curves(
    grouped: Mapping[str, Sequence[Tuple[Path, List[dict]]]], output_dir: Path
) -> None:
    protocol_orders = [validate_same_protocol(runs, method) for method, runs in grouped.items()]
    reference = protocol_orders[0]
    if any(order != reference for order in protocol_orders[1:]):
        raise ValueError("All methods in the stage-wise curve must use the same task order")

    x = np.arange(1, len(reference) + 1)
    labels = [short_label(task) for task in reference]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), constrained_layout=False)
    for method_index, (method, runs) in enumerate(grouped.items()):
        color = COLORS[method_index % len(COLORS)]
        marker = MARKERS[method_index % len(MARKERS)]
        for axis, metric, panel in zip(axes, ("auc", "ap"), ("(a) AUC", "(b) AP")):
            mean, std, count = aggregate_mean_std([stage_average(h, metric) for _, h in runs])
            axis.plot(x, mean, color=color, marker=marker, label=f"{method} (n={len(runs)})")
            if np.any(np.isfinite(std)):
                axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
            axis.set_title(panel, loc="left", fontweight="bold")
            axis.set_ylabel(f"Average {metric.upper()} over seen stages")
            axis.set_xlabel("Training stage completed")
            axis.set_xticks(x, labels, rotation=28, ha="right")
            axis.set_ylim(0.5, 1.01)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(4, len(handles)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    save_figure(fig, output_dir, "fig_p0_1_stagewise_curves")
    write_stage_curve_csv(output_dir, grouped)


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, threshold: float = 0.75) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            color = "white" if value < threshold else "#111111"
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=6.7, color=color)


def plot_performance_matrix(
    method: str,
    runs: Sequence[Tuple[Path, List[dict]]],
    output_dir: Path,
) -> None:
    tasks = validate_same_protocol(runs, method)
    labels = [short_label(task) for task in tasks]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), constrained_layout=True)
    last_image = None
    for axis, metric, panel in zip(axes, ("auc", "ap"), ("(a) AUC matrix", "(b) AP matrix")):
        mean, std, count = aggregate_mean_std(
            [performance_matrix(history, metric, tasks) for _, history in runs]
        )
        masked = np.ma.masked_invalid(mean)
        cmap = mpl.colormaps["viridis"].copy()
        cmap.set_bad("#EEEEEE")
        last_image = axis.imshow(masked, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
        annotate_heatmap(axis, mean)
        axis.set_title(panel, loc="left", fontweight="bold")
        axis.set_xlabel("Evaluation stage")
        axis.set_ylabel("Training stage completed")
        axis.set_xticks(np.arange(len(tasks)), labels, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(tasks)), [f"S{i + 1}" for i in range(len(tasks))])
        # Outline the observed lower triangle and make missing future stages explicit.
        for index in range(len(tasks)):
            axis.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False,
                                     edgecolor="white", linewidth=1.1))
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.030, pad=0.02)
    colorbar.set_label("Performance")
    save_figure(fig, output_dir, "fig_p0_2_performance_matrix")

    for metric in ("auc", "ap"):
        mean, std, count = aggregate_mean_std(
            [performance_matrix(history, metric, tasks) for _, history in runs]
        )
        with (output_dir / f"p0_2_{metric}_matrix.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["train_stage", "eval_stage", "mean", "sample_sd", "n_seeds"])
            for row, train_task in enumerate(tasks):
                for column, eval_task in enumerate(tasks):
                    if np.isfinite(mean[row, column]):
                        writer.writerow(
                            [train_task, eval_task, mean[row, column], std[row, column], int(count[row, column])]
                        )


def routing_matrix(
    history: Sequence[dict], key: str, tasks: Sequence[str], num_experts: int
) -> Tuple[np.ndarray, np.ndarray]:
    final_seen = history[-1].get("seen", {})
    matrix = np.full((len(tasks), num_experts), np.nan, dtype=float)
    sample_count = np.zeros(len(tasks), dtype=float)
    for task_index, task in enumerate(tasks):
        metrics = final_seen.get(task, {})
        values = metrics.get(key)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        matrix[task_index, : len(values)] = values
        sample_count[task_index] = float(metrics.get("num_eval_samples", 0))
    return matrix, sample_count


def plot_expert_usage(
    method: str,
    runs: Sequence[Tuple[Path, List[dict]]],
    output_dir: Path,
    reuse_threshold: float,
) -> None:
    tasks = validate_same_protocol(runs, method)
    labels = [short_label(task) for task in tasks]
    num_experts = 0
    for path, history in runs:
        for metrics in history[-1].get("seen", {}).values():
            num_experts = max(num_experts, len(metrics.get("expert_mean_weight", [])))
    if num_experts == 0:
        paths = "\n".join(str(path) for path, _ in runs)
        raise ValueError(
            "Expert-routing fields are missing. Re-run CAPE with the updated "
            "CAPEContinualTrainer.evaluate_task before plotting P0-3. Histories:\n" + paths
        )

    soft_arrays, top1_arrays = [], []
    for _, history in runs:
        soft, _ = routing_matrix(history, "expert_mean_weight", tasks, num_experts)
        top1, _ = routing_matrix(history, "expert_top1_rate", tasks, num_experts)
        soft_arrays.append(soft)
        top1_arrays.append(top1)
    soft_mean, soft_std, soft_count = aggregate_mean_std(soft_arrays)
    top1_mean, top1_std, top1_count = aggregate_mean_std(top1_arrays)

    # Plot experts on rows and evaluation stages on columns.
    soft_plot = soft_mean.T
    top1_plot = top1_mean.T
    reuse_count = np.sum(top1_plot >= reuse_threshold, axis=1)
    experts = [f"E{i + 1}" for i in range(num_experts)]

    fig = plt.figure(figsize=(7.16, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.42])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    bar_axis = fig.add_subplot(grid[0, 2])
    images = []
    for axis, matrix, panel in zip(
        axes,
        (soft_plot, top1_plot),
        ("(a) Mean routing mass $w_j$", "(b) Top-1 selection frequency"),
    ):
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        images.append(image)
        for expert_index in range(num_experts):
            for task_index in range(len(tasks)):
                value = matrix[expert_index, task_index]
                if np.isfinite(value):
                    axis.text(
                        task_index,
                        expert_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        color="white" if value >= 0.55 else "#111111",
                    )
        axis.set_title(panel, loc="left", fontweight="bold")
        axis.set_xlabel("Evaluation stage")
        axis.set_ylabel("Pattern expert")
        axis.set_xticks(np.arange(len(tasks)), labels, rotation=32, ha="right")
        axis.set_yticks(np.arange(num_experts), experts)
    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.036, pad=0.02)
    colorbar.set_label("Usage")

    y = np.arange(num_experts)
    bar_axis.barh(y, reuse_count, color="#009E73", edgecolor="#1A1A1A", linewidth=0.5)
    bar_axis.set_title("(c) Reuse", loc="left", fontweight="bold")
    bar_axis.set_xlabel(f"Stages with\nTop-1 rate $\\geq${reuse_threshold:.2f}")
    bar_axis.set_yticks(y, experts)
    bar_axis.set_xlim(0, max(1, len(tasks)))
    bar_axis.invert_yaxis()
    bar_axis.spines[["top", "right"]].set_visible(False)
    bar_axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    for index, value in enumerate(reuse_count):
        bar_axis.text(value + 0.06, index, str(int(value)), va="center", fontsize=7.0)
    save_figure(fig, output_dir, "fig_p0_3_expert_usage")

    with (output_dir / "p0_3_expert_usage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task", "expert", "mean_weight", "weight_sd", "top1_rate", "top1_sd", "n_seeds"]
        )
        for task_index, task in enumerate(tasks):
            for expert_index in range(num_experts):
                writer.writerow(
                    [
                        task,
                        expert_index + 1,
                        soft_mean[task_index, expert_index],
                        soft_std[task_index, expert_index],
                        top1_mean[task_index, expert_index],
                        top1_std[task_index, expert_index],
                        int(top1_count[task_index, expert_index]),
                    ]
                )

    summary = {
        "method": method,
        "num_seeds": len(runs),
        "tasks": tasks,
        "num_experts": num_experts,
        "reuse_threshold": reuse_threshold,
        "expert_reuse_stage_count": {
            experts[index]: int(value) for index, value in enumerate(reuse_count)
        },
        "interpretation_guardrail": (
            "Cross-stage reuse is supported only when at least one expert has "
            "substantial usage on multiple stages and the matrix is not an identity-like diagonal."
        ),
    }
    with (output_dir / "p0_3_expert_reuse_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CAPE stage curves, continual matrices, and expert-usage evidence."
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="METHOD=GLOB",
        help="May be repeated. Histories with the same METHOD are independent seeds.",
    )
    parser.add_argument(
        "--matrix-method",
        default="CAPE",
        help="Method whose post-stage performance matrices are plotted.",
    )
    parser.add_argument(
        "--routing-method",
        default="CAPE",
        help="Method whose final expert-routing heatmap is plotted.",
    )
    parser.add_argument(
        "--reuse-threshold",
        type=float,
        default=0.10,
        help="Minimum per-stage top-1 selection rate counted as expert reuse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "cape" / "continual_evidence_figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    grouped = expand_series(args.series)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.matrix_method not in grouped:
        raise KeyError(f"--matrix-method {args.matrix_method!r} is not among {list(grouped)}")
    if args.routing_method not in grouped:
        raise KeyError(f"--routing-method {args.routing_method!r} is not among {list(grouped)}")

    plot_stagewise_curves(grouped, output_dir)
    plot_performance_matrix(args.matrix_method, grouped[args.matrix_method], output_dir)
    plot_expert_usage(
        args.routing_method,
        grouped[args.routing_method],
        output_dir,
        reuse_threshold=args.reuse_threshold,
    )
    print(f"Saved three figure families to: {output_dir}")


if __name__ == "__main__":
    main()
