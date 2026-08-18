"""Plot the real CAPE HiFi-AVDF panels (b) AUC and (e) AP.

All values are computed directly from complete ``history.json`` files.  The
line is the five-seed mean and the shaded band/error bars are the sample
standard deviation (ddof=1).  No interpolation or trajectory shaping is used.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TASKS = [
    "generator:kling2.5",
    "generator:veo3.1",
    "generator:wan2.5",
    "generator:seedance1.0",
]
TASK_LABELS = ["Kling2.5", "Veo3.1", "Wan2.5", "Seedance1.0"]
COLOR = "#0072B2"  # Okabe-Ito blue


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_histories(result_root: Path) -> List[Tuple[Path, List[dict]]]:
    paths = sorted(Path(path) for path in glob.glob(str(result_root / "seed_*" / "history.json")))
    if len(paths) != 5:
        raise RuntimeError(f"Expected exactly five seed histories, found {len(paths)} in {result_root}")
    runs = []
    for path in paths:
        history = json.loads(path.read_text(encoding="utf-8"))
        tasks = [str(row.get("current_task")) for row in history]
        if tasks != EXPECTED_TASKS:
            raise RuntimeError(f"Unexpected task order in {path}: {tasks}")
        if any(len(row.get("train", [])) != 20 for row in history):
            raise RuntimeError(f"Not every stage contains 20 epochs in {path}")
        runs.append((path, history))
    return runs


def stage_average(history: Sequence[dict], metric: str) -> np.ndarray:
    values = []
    for row in history:
        seen_values = [float(item[metric]) for item in row["seen"].values()]
        values.append(float(np.mean(seen_values)))
    return np.asarray(values, dtype=float)


def aggregate(
    runs: Sequence[Tuple[Path, List[dict]]], metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_seed = np.stack([stage_average(history, metric) for _, history in runs], axis=0)
    return per_seed, per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)


def style_axis(axis: plt.Axes, panel: str, metric: str) -> None:
    axis.set_title(f"{panel} HiFi-AVDF — {metric.upper()}", loc="left", fontweight="bold")
    axis.set_xlabel("Training stage completed")
    axis.set_ylabel(f"Average {metric.upper()} over seen stages")
    axis.set_xticks(np.arange(1, 5), TASK_LABELS, rotation=22, ha="right")
    axis.set_ylim(0.90, 1.005)
    axis.set_yticks(np.arange(0.90, 1.001, 0.02))
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.8)


def draw_metric(
    axis: plt.Axes,
    mean: np.ndarray,
    sample_sd: np.ndarray,
    panel: str,
    metric: str,
) -> None:
    x = np.arange(1, 5)
    lower = np.clip(mean - sample_sd, 0.0, 1.0)
    upper = np.clip(mean + sample_sd, 0.0, 1.0)
    axis.fill_between(x, lower, upper, color=COLOR, alpha=0.15, linewidth=0, zorder=1)
    axis.errorbar(
        x,
        mean,
        yerr=sample_sd,
        color=COLOR,
        marker="o",
        markersize=4.7,
        linewidth=1.8,
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        label="CAPE (mean ± sample SD, n=5)",
        zorder=3,
    )
    style_axis(axis, panel, metric)
    axis.legend(loc="lower left", frameon=False, handlelength=2.2)


def save_formats(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=600)
    plt.close(fig)


def write_data(
    output_dir: Path,
    runs: Sequence[Tuple[Path, List[dict]]],
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    aggregate_path = output_dir / "figure3_be_hifi_cape_real_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "metric", "stage", "task", "mean", "sample_sd", "n_seeds"])
        for metric, panel in (("auc", "b"), ("ap", "e")):
            _, mean, sample_sd = arrays[metric]
            for index, task in enumerate(EXPECTED_TASKS):
                writer.writerow(
                    [panel, metric, index + 1, task, mean[index], sample_sd[index], len(runs)]
                )

    seed_path = output_dir / "figure3_be_hifi_cape_real_per_seed.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "panel", "metric", "stage", "task", "value", "history_path"])
        for metric, panel in (("auc", "b"), ("ap", "e")):
            per_seed = arrays[metric][0]
            for run_index, (path, _) in enumerate(runs):
                seed = path.parent.name.replace("seed_", "")
                for stage_index, task in enumerate(EXPECTED_TASKS):
                    writer.writerow(
                        [
                            seed,
                            panel,
                            metric,
                            stage_index + 1,
                            task,
                            per_seed[run_index, stage_index],
                            str(path),
                        ]
                    )

    manifest = {
        "figure": "Figure 3 panels (b) and (e)",
        "method": "CAPE",
        "protocol": "HiFi-AVDF generator-incremental",
        "tasks": EXPECTED_TASKS,
        "num_seeds": len(runs),
        "uncertainty": "sample standard deviation across five seeds (ddof=1)",
        "interpolation": "none",
        "histories": [
            {"path": str(path), "sha256": sha256_file(path)} for path, _ in runs
        ],
        "metrics": {
            metric: {
                "mean": arrays[metric][1].tolist(),
                "sample_sd": arrays[metric][2].tolist(),
            }
            for metric in ("auc", "ap")
        },
    }
    (output_dir / "figure3_be_hifi_cape_real_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real Figure 3(b)/(e) CAPE HiFi curves.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "figure3" / "hifi" / "cape_grouped_clean",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figure3" / "hifi" / "figure3_be_final",
    )
    args = parser.parse_args()
    result_root = args.result_root if args.result_root.is_absolute() else PROJECT_ROOT / args.result_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    configure_style()
    runs = load_histories(result_root)
    arrays = {metric: aggregate(runs, metric) for metric in ("auc", "ap")}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_data(output_dir, runs, arrays)

    for metric, panel in (("auc", "(b)"), ("ap", "(e)")):
        _, mean, sample_sd = arrays[metric]
        fig, axis = plt.subplots(figsize=(3.48, 2.55))
        draw_metric(axis, mean, sample_sd, panel, metric)
        fig.tight_layout()
        save_formats(fig, output_dir, f"figure3_{panel[1]}_hifi_{metric}_cape_real")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.62))
    draw_metric(axes[0], arrays["auc"][1], arrays["auc"][2], "(b)", "auc")
    draw_metric(axes[1], arrays["ap"][1], arrays["ap"][2], "(e)", "ap")
    fig.tight_layout(w_pad=2.0)
    save_formats(fig, output_dir, "figure3_be_hifi_cape_real")

    print(f"Saved Figure 3(b)/(e) to {output_dir}")
    for metric in ("auc", "ap"):
        print(metric.upper(), "mean=", arrays[metric][1], "sample_sd=", arrays[metric][2])


if __name__ == "__main__":
    main()
