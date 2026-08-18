"""Audit whether checked-in metadata can instantiate every manuscript protocol."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AV1M_TASKS = ("1", "2", "3")
FAKEAVCELEB_TASKS = (
    "fakeavceleb:faceswap",
    "fakeavceleb:fsgan",
    "fakeavceleb:wav2lip",
    "fakeavceleb:rtvc",
    "fakeavceleb:audio_visual",
)
GENERATOR_TASKS = (
    "generator:kling2.5",
    "generator:veo3.1",
    "generator:wan2.5",
    "generator:seedance1.0",
)
MANUSCRIPT_HELDOUT_COUNTS = {
    "generator:kling2.5": 460,
    "generator:veo3.1": 628,
    "generator:wan2.5": 580,
    "generator:seedance1.0": 788,
}


def count_metadata(path):
    counts = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("dataset", "")),
                str(row.get("task_id", "")),
                str(row.get("split", "")),
                int(row.get("is_fake", 0)),
            )
            counts[key] += 1
    return counts


def task_split_counts(counts, dataset, task):
    return {
        split: {
            "real": counts[(dataset, task, split, 0)],
            "fake": counts[(dataset, task, split, 1)],
        }
        for split in ("train", "val", "test")
    }


def main():
    parser = argparse.ArgumentParser(description="Audit manuscript task/split coverage and held-out counts.")
    parser.add_argument("--metadata", default="data/cape_metadata.csv")
    parser.add_argument("--hifi-metadata", default="data/cape_metadata_hifi_grouped.csv")
    parser.add_argument("--hifi-manifest", default="data/cape_metadata_hifi_grouped_manifest.json")
    parser.add_argument("--report", default="results/paper_protocol_data_audit.json")
    args = parser.parse_args()

    metadata = (PROJECT_ROOT / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata)
    hifi_metadata = (
        (PROJECT_ROOT / args.hifi_metadata).resolve()
        if not Path(args.hifi_metadata).is_absolute()
        else Path(args.hifi_metadata)
    )
    manifest_path = (
        (PROJECT_ROOT / args.hifi_manifest).resolve()
        if not Path(args.hifi_manifest).is_absolute()
        else Path(args.hifi_manifest)
    )
    counts = count_metadata(metadata)
    hifi_counts = count_metadata(hifi_metadata)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    av1m = {task: task_split_counts(counts, "avdeepfake1m", task) for task in AV1M_TASKS}
    av1m_real = {
        split: counts[("avdeepfake1m", "real", split, 0)]
        for split in ("train", "val", "test")
    }
    fakeavceleb = {
        task: task_split_counts(counts, "fakeavceleb", task) for task in FAKEAVCELEB_TASKS
    }
    hifi = {task: task_split_counts(hifi_counts, "hifi_avdf", task) for task in GENERATOR_TASKS}

    checks = {}
    checks["av1m_has_official_test_rows"] = all(
        av1m[task]["test"]["fake"] > 0 and av1m_real["test"] > 0 for task in AV1M_TASKS
    )
    checks["fakeavceleb_five_stages_present"] = all(
        sum(values[label] for values in fakeavceleb[task].values() for label in ("real", "fake")) > 0
        for task in FAKEAVCELEB_TASKS
    )
    checks["hifi_all_splits_have_matched_labels"] = all(
        hifi[task][split]["real"] > 0 and hifi[task][split]["fake"] > 0
        for task in GENERATOR_TASKS
        for split in ("train", "val", "test")
    )
    checks["hifi_split_is_60_20_20"] = manifest.get("fractions") == {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    }

    heldout_counts = {
        task: sum(hifi[task][split]["fake"] for split in ("train", "val", "test"))
        for task in GENERATOR_TASKS
    }
    checks["heldout_fake_counts_match_manuscript"] = heldout_counts == MANUSCRIPT_HELDOUT_COUNTS

    report = {
        "metadata": str(metadata),
        "hifi_metadata": str(hifi_metadata),
        "checks": checks,
        "avdeepfake1m": {"tasks": av1m, "real": av1m_real},
        "fakeavceleb": fakeavceleb,
        "hifi_avdf": {
            "tasks": hifi,
            "actual_heldout_fake_counts": heldout_counts,
            "manuscript_heldout_counts": MANUSCRIPT_HELDOUT_COUNTS,
            "manifest_total_valid_rows": manifest.get("total_valid_rows"),
            "manifest_dropped_pairs": manifest.get("dropped_pairs", []),
        },
        "passed": all(checks.values()),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
