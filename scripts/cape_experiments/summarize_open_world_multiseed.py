"""Aggregate the four-fold open-world protocol over independent seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_multiseed import (
    CAPE_PAPER_PROFILE,
    COMMON_PAPER_PROFILE,
    PAPER_TASK_SEQUENCES,
)


DISCOVERY_KEYS = ("unknown_auc", "unknown_ap", "fpr95", "delay", "ari", "fcr")
ADAPTATION_KEYS = ("post_adaptation_auc", "old_source_drop")


def _stats(values):
    values = np.asarray([float(value) for value in values], dtype=float)
    values = values[np.isfinite(values)]
    return {
        "mean": float(values.mean()) if values.size else float("nan"),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else float("nan"),
        "n": int(values.size),
    }


def _validate_fold_config(config_path):
    if not config_path.exists():
        raise RuntimeError(f"Missing run_config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {**COMMON_PAPER_PROFILE, **CAPE_PAPER_PROFILE}
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Non-paper configuration in {config_path}: {mismatches}")
    task_sequence = tuple(str(value) for value in config.get("task_sequence", []))
    if task_sequence not in PAPER_TASK_SEQUENCES or len(task_sequence) != 3:
        raise RuntimeError(f"Invalid leave-one-generator-out task sequence in {config_path}")


def _aggregate_sections(records, section, keys):
    return {key: _stats(record[section].get(key, float("nan")) for record in records) for key in keys}


def main():
    parser = argparse.ArgumentParser(description="Aggregate open-world results as mean and sample standard deviation.")
    parser.add_argument("result_root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--paper-mode", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_root)
    records = []
    seeds = set()
    for path in sorted(root.glob("seed_*/open_world_summary.json")):
        try:
            seed = int(path.parent.name.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Invalid seed directory: {path.parent}") from exc
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("seed", seed)) != seed:
            raise RuntimeError(f"Seed mismatch in {path}")
        if args.paper_mode:
            for fold in record.get("folds", []):
                generator = fold.get("heldout_generator")
                _validate_fold_config(path.parent / f"heldout_{generator}" / "run_config.json")
        seeds.add(seed)
        records.append(record)

    if not records:
        raise RuntimeError(f"No seed_*/open_world_summary.json files found below {root}")
    if args.paper_mode and seeds != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"Found seeds {sorted(seeds)}, expected [0, 1, 2, 3, 4]")

    fold_names = sorted(
        {fold["heldout_generator"] for record in records for fold in record.get("folds", [])}
    )
    per_fold = {}
    for name in fold_names:
        fold_records = []
        for record in records:
            matches = [fold for fold in record.get("folds", []) if fold.get("heldout_generator") == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one held-out {name} fold for seed {record.get('seed')}")
            fold_records.append(matches[0])
        per_fold[name] = {
            "discovery": _aggregate_sections(fold_records, "discovery", DISCOVERY_KEYS),
            "adaptation": _aggregate_sections(fold_records, "adaptation", ADAPTATION_KEYS),
        }

    summary = {
        "seeds": sorted(seeds),
        "num_seeds": len(records),
        "macro_discovery": _aggregate_sections(records, "macro_discovery", DISCOVERY_KEYS),
        "macro_adaptation": _aggregate_sections(records, "macro_adaptation", ADAPTATION_KEYS),
        "per_fold": per_fold,
    }
    output = Path(args.output) if args.output else root / "open_world_multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
