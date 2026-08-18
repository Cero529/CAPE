import argparse
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.cape_reporting import summarize_history


COMMON_PAPER_PROFILE = {
    "max_length": 100,
    "d_model": 768,
    "batch_size": 64,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "replay_capacity": 1024,
    "backbone_mode": "precomputed_avhubert",
    "freeze_backbone": True,
    "eval_split": "test",
}

CAPE_PAPER_PROFILE = {
    "expert_bottleneck": 64,
    "top_k": 2,
    "dropout": 0.1,
    "prototype_momentum": 0.95,
    "bandwidth_init": 0.35,
    "bandwidth_momentum": 0.9,
    "min_bandwidth": 0.05,
    "max_bandwidth": 1.25,
    "unknown_alpha": 0.05,
    "unknown_min_cluster_size": 8,
    "unknown_cluster_eps": 0.8,
    "unknown_queue_capacity": 4096,
    "calibration_capacity": 1024,
    "lambda_pattern": 0.5,
    "lambda_logic": 0.2,
    "lambda_distill": 1.0,
    "lambda_router": 0.01,
    "input_dim": 768,
    "eval_every_epoch": True,
    "save_best": True,
    "use_discrepancy": True,
    "use_pattern_guidance": True,
    "use_confidence_density": True,
    "replay_enabled": True,
    "unknown_component_indices": [0, 1, 2, 3],
    "unknown_detector": "conformal",
    "allow_expert_expansion": True,
}

ABLATION_PROFILE_OVERRIDES = {
    "full": {},
    "no_discrepancy": {"use_discrepancy": False},
    "no_pattern_guidance": {
        "use_pattern_guidance": False,
        "lambda_pattern": 0.0,
        "lambda_logic": 0.0,
    },
    "no_confidence_density": {"use_confidence_density": False},
    "no_continual_retention": {"lambda_distill": 0.0, "replay_enabled": False},
    "no_composite_unknownness": {"unknown_component_indices": [1]},
    "no_conformal_calibration": {"unknown_detector": "gaussian_tail"},
    "no_dynamic_expansion": {"allow_expert_expansion": False},
}

GENERATOR_TASKS = (
    "generator:kling2.5",
    "generator:veo3.1",
    "generator:wan2.5",
    "generator:seedance1.0",
)
HETEROGENEOUS_TASKS = (
    "fakeavceleb:faceswap",
    "fakeavceleb:fsgan",
    "fakeavceleb:wav2lip",
    "fakeavceleb:rtvc",
    "fakeavceleb:audio_visual",
    "1",
    "2",
    "3",
    *GENERATOR_TASKS,
)
PAPER_TASK_SEQUENCES = {
    ("1", "2", "3"),
    GENERATOR_TASKS,
    HETEROGENEOUS_TASKS,
    *(tuple(task for task in GENERATOR_TASKS if task != heldout) for heldout in GENERATOR_TASKS),
}


def _diagnostics(history):
    keys = ("router_drift_js", "new_expert_mass", "prototype_drift_cos")
    result = {}
    for key in keys:
        values = []
        for row in history:
            diagnostics = row.get("diagnostics") or {}
            value = diagnostics.get(key)
            if value is not None and not math.isnan(float(value)):
                values.append(float(value))
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


def _method_name(history_path, root):
    relative = history_path.parent.relative_to(root)
    parts = [part for part in relative.parts if not part.startswith("seed_")]
    return "/".join(parts) if parts else "cape"


def main():
    parser = argparse.ArgumentParser(description="Aggregate CAPE histories as mean and standard deviation.")
    parser.add_argument("result_root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--paper-mode", action="store_true", help="Require seeds 0..4 and the exact manuscript profile.")
    args = parser.parse_args()

    root = Path(args.result_root)
    grouped = {}
    grouped_seeds = {}
    for history_path in root.rglob("history.json"):
        seed_parts = [part for part in history_path.parts if part.startswith("seed_")]
        seed = int(seed_parts[-1].split("_", 1)[1]) if seed_parts else None
        method = _method_name(history_path, root)
        if args.paper_mode:
            if seed is None:
                raise RuntimeError(f"Paper-mode history is not inside seed_<n>: {history_path}")
            config_path = history_path.parent / "run_config.json"
            if not config_path.exists():
                raise RuntimeError(f"Missing run_config.json for {history_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_profile = dict(COMMON_PAPER_PROFILE)
            if "method" not in config:
                expected_profile.update(CAPE_PAPER_PROFILE)
                ablation = config.get("ablation")
                if ablation is not None:
                    if ablation not in ABLATION_PROFILE_OVERRIDES:
                        raise RuntimeError(f"Unknown ablation profile in {config_path}: {ablation}")
                    expected_profile.update(ABLATION_PROFILE_OVERRIDES[ablation])
            mismatches = {
                key: (config.get(key), expected)
                for key, expected in expected_profile.items()
                if config.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"Non-paper configuration in {config_path}: {mismatches}")
            task_sequence = tuple(str(value) for value in config.get("task_sequence", []))
            if task_sequence not in PAPER_TASK_SEQUENCES:
                raise RuntimeError(
                    f"Non-paper task sequence in {config_path}: {list(task_sequence)}"
                )
        with history_path.open("r") as f:
            history = json.load(f)
        row = {**summarize_history(history), **_diagnostics(history)}
        grouped.setdefault(method, []).append(row)
        grouped_seeds.setdefault(method, set()).add(seed)

    if args.paper_mode:
        for method, seeds in grouped_seeds.items():
            if seeds != {0, 1, 2, 3, 4}:
                raise RuntimeError(f"{method} has seeds {sorted(seeds)}, expected [0, 1, 2, 3, 4]")

    metrics = (
        "final_auc",
        "trajectory_auc",
        "final_ap",
        "trajectory_ap",
        "forgetting_auc",
        "router_drift_js",
        "prototype_drift_cos",
        "new_expert_mass",
    )
    summary = {}
    for method, rows in grouped.items():
        summary[method] = {"num_seeds": len(rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[method][metric] = {
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else float("nan"),
            }

    output = Path(args.output) if args.output else root / "multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
