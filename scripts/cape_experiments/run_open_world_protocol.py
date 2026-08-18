import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_continual import CAPEContinualTrainer
from src.cape_datasets import CAPEContinualDataset, cape_collate
from src.cape_metrics import safe_ari, unknown_detection_metrics
from src.cape_unknown import UnknownQueue


GENERATORS = ("kling2.5", "veo3.1", "wan2.5", "seedance1.0")


class TaggedDataset(Dataset):
    def __init__(self, dataset, stream_unknown):
        self.dataset = dataset
        self.stream_unknown = int(stream_unknown)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["stream_unknown"] = torch.tensor(self.stream_unknown).long()
        return item


def _sample_indices(indices, count, rng):
    indices = list(indices)
    if not indices:
        raise RuntimeError("Cannot sample from an empty discovery stratum")
    replace = len(indices) < count
    return rng.choice(indices, size=count, replace=replace).tolist()


def build_discovery_loader(trainer, known_tasks, heldout_task, seed):
    unknown = CAPEContinualDataset(
        trainer.metadata_csv,
        split=None,
        max_length=trainer.max_length,
        task_id=heldout_task,
        include_real=False,
    )
    if not len(unknown):
        raise RuntimeError(f"No held-out fake samples found for {heldout_task}")

    known_parts = [
        CAPEContinualDataset(
            trainer.metadata_csv,
            split="test",
            max_length=trainer.max_length,
            task_id=task_id,
        )
        for task_id in known_tasks
    ]
    known = ConcatDataset(known_parts)
    real_indices, fake_indices = [], []
    offset = 0
    for part in known_parts:
        for local_index, row in enumerate(part.table):
            target = fake_indices if int(row.get("is_fake", 0)) else real_indices
            target.append(offset + local_index)
        offset += len(part)

    n_unknown = len(unknown)
    n_real = n_unknown // 2
    n_fake = n_unknown - n_real
    rng = np.random.default_rng(int(seed))
    selected_known = _sample_indices(real_indices, n_real, rng) + _sample_indices(fake_indices, n_fake, rng)
    known_subset = Subset(known, selected_known)

    tagged = ConcatDataset([TaggedDataset(unknown, 1), TaggedDataset(known_subset, 0)])
    order = rng.permutation(len(tagged)).tolist()
    stream = Subset(tagged, order)
    loader = DataLoader(stream, **trainer._loader_kwargs(shuffle=False))
    return loader, n_unknown


@torch.no_grad()
def run_discovery(trainer, loader, heldout_task, alpha, confirmation_size, purity_threshold):
    trainer.model.eval()
    queue = UnknownQueue(maxlen=trainer.unknown_queue.items.maxlen)
    y_unknown, scores = [], []
    observed = 0
    detection_delay = None

    for batch in loader:
        batch = trainer._move(batch)
        outputs = trainer.model(
            batch["video_features"],
            batch["audio_features"],
            valid_lengths=batch["valid_length"],
            backbone_features=batch.get("backbone_features"),
        )
        p_values = trainer.unknown_detector.p_value(outputs["unknown_score"])
        unknown_flags = batch["stream_unknown"].detach().cpu().numpy().astype(int)
        y_unknown.extend(unknown_flags.tolist())
        scores.extend(outputs["unknown_score"].detach().cpu().tolist())

        features = outputs["features"].detach().cpu()
        score_cpu = outputs["unknown_score"].detach().cpu()
        p_cpu = p_values.detach().cpu()
        for index in range(len(batch["sample_id"])):
            observed += 1
            if float(p_cpu[index]) >= float(alpha):
                continue
            queue.push(
                batch["sample_id"][index],
                features[index],
                score_cpu[index],
                meta={
                    "task_id": str(batch["task_id"][index]),
                    "is_unknown": int(unknown_flags[index]),
                    "stream_index": observed,
                    "p_value": float(p_cpu[index]),
                },
            )
            if detection_delay is None and queue.cluster_candidates(
                min_cluster_size=trainer.unknown_min_cluster_size,
                eps=trainer.unknown_cluster_eps,
            ):
                detection_delay = observed

    labels = queue.cluster_labels(
        min_cluster_size=trainer.unknown_min_cluster_size,
        eps=trainer.unknown_cluster_eps,
    )
    source_names = [str(item["meta"].get("task_id", "")) for item in queue.items]
    source_to_id = {name: index for index, name in enumerate(sorted(set(source_names)))}
    source_labels = [source_to_id[name] for name in source_names]
    ari = safe_ari(source_labels, labels)

    cluster_ids = sorted(label for label in set(labels.tolist()) if label != -1)
    false_candidates = 0
    confirmed_ids = []
    cluster_records = []
    for cluster_id in cluster_ids:
        indices = np.flatnonzero(labels == cluster_id).tolist()
        heldout = [
            queue.items[index]["sample_id"]
            for index in indices
            if str(queue.items[index]["meta"].get("task_id")) == heldout_task
            and int(queue.items[index]["meta"].get("is_unknown", 0)) == 1
        ]
        purity = len(heldout) / max(1, len(indices))
        confirmed = purity >= float(purity_threshold) and len(heldout) >= int(confirmation_size)
        if confirmed:
            confirmed_ids.extend(heldout)
        else:
            false_candidates += 1
        cluster_records.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(indices),
                "heldout_members": len(heldout),
                "heldout_purity": purity,
                "confirmed": confirmed,
            }
        )

    metrics = unknown_detection_metrics(y_unknown, scores)
    metrics.update(
        {
            "delay": detection_delay,
            "ari": ari,
            "fcr": false_candidates / len(cluster_ids) if cluster_ids else 1.0,
            "num_observed": observed,
            "num_rejected": len(queue),
            "num_clusters": len(cluster_ids),
            "clusters": cluster_records,
        }
    )
    return metrics, sorted(set(confirmed_ids))


def mean_seen_auc(trainer, task_ids):
    values = [trainer.evaluate_task("test", task_id=task_id)["auc"] for task_id in task_ids]
    clean = [value for value in values if value == value]
    return float(np.mean(clean)) if clean else float("nan")


def run_fold(args, heldout_generator):
    heldout_task = f"generator:{heldout_generator}"
    known_tasks = [f"generator:{name}" for name in GENERATORS if name != heldout_generator]
    fold_dir = os.path.join(args.output_dir, f"heldout_{heldout_generator}")
    trainer = CAPEContinualTrainer(
        metadata_csv=args.metadata,
        task_sequence=known_tasks,
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        replay_capacity=args.replay_capacity,
        seed=args.seed,
        device=args.device,
        output_dir=fold_dir,
        num_workers=args.num_workers,
        show_progress=not args.no_progress,
        eval_every_epoch=True,
        save_best=True,
        freeze_backbone=True,
        backbone_ckpt=args.backbone_ckpt or None,
        backbone_mode=args.backbone_mode,
        calibration_capacity=args.calibration_capacity,
        unknown_alpha=args.unknown_alpha,
        unknown_min_cluster_size=args.unknown_min_cluster_size,
        unknown_cluster_eps=args.unknown_cluster_eps,
        unknown_queue_capacity=args.unknown_queue_capacity,
        lambda_distill=1.0,
    )
    trainer.run()

    stream_loader, n_unknown = build_discovery_loader(
        trainer, known_tasks, heldout_task, seed=args.seed
    )
    discovery, candidates = run_discovery(
        trainer,
        stream_loader,
        heldout_task,
        alpha=args.unknown_alpha,
        confirmation_size=args.confirmation_size,
        purity_threshold=args.purity_threshold,
    )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    confirmation_ids = candidates[: args.confirmation_size]
    adaptation = {
        "confirmed": len(confirmation_ids) == args.confirmation_size,
        "confirmation_samples": len(confirmation_ids),
        "post_adaptation_auc": float("nan"),
        "old_source_drop": float("nan"),
    }
    if len(confirmation_ids) == args.confirmation_size:
        old_before = mean_seen_auc(trainer, known_tasks)
        old_model = copy.deepcopy(trainer.model).eval()
        confirmed_loader = trainer.loader(
            None,
            task_id=heldout_task,
            shuffle=True,
            sample_ids=confirmation_ids,
            confirmed_emerging=True,
        )
        trainer.train_task(
            f"confirmed:{heldout_generator}",
            old_model=old_model,
            seen_tasks=None,
            train_loader=confirmed_loader,
        )
        heldout_metrics = trainer.evaluate_task(
            "test",
            task_id=heldout_task,
            exclude_sample_ids=confirmation_ids,
        )
        old_after = mean_seen_auc(trainer, known_tasks)
        adaptation.update(
            {
                "post_adaptation_auc": heldout_metrics["auc"],
                "old_source_drop": old_before - old_after,
                "old_auc_before": old_before,
                "old_auc_after": old_after,
            }
        )

    result = {
        "heldout_generator": heldout_generator,
        "known_tasks": known_tasks,
        "n_unknown": n_unknown,
        "discovery": discovery,
        "adaptation": adaptation,
    }
    with open(os.path.join(fold_dir, "open_world_fold.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def macro_average(folds, section, keys):
    out = {}
    for key in keys:
        values = [float(fold[section].get(key, float("nan"))) for fold in folds]
        values = [value for value in values if value == value]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="CAPE four-fold open-world protocol.")
    parser.add_argument("--metadata", default="data/cape_metadata_hifi_grouped.csv")
    parser.add_argument("--output-dir", default="results/cape_open_world/seed_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d-hid", type=int, default=3072)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--replay-capacity", type=int, default=1024)
    parser.add_argument("--calibration-capacity", type=int, default=1024)
    parser.add_argument("--unknown-alpha", type=float, default=0.05)
    parser.add_argument("--unknown-min-cluster-size", type=int, default=8)
    parser.add_argument("--unknown-cluster-eps", type=float, default=0.8)
    parser.add_argument("--unknown-queue-capacity", type=int, default=4096)
    parser.add_argument("--confirmation-size", type=int, default=32)
    parser.add_argument("--purity-threshold", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone-mode", choices=["precomputed_avhubert", "legacy_dimodif"], default="precomputed_avhubert")
    parser.add_argument("--backbone-ckpt", default="")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    folds = [run_fold(args, generator) for generator in GENERATORS]
    summary = {
        "seed": args.seed,
        "folds": folds,
        "macro_discovery": macro_average(
            folds, "discovery", ("unknown_auc", "unknown_ap", "fpr95", "delay", "ari", "fcr")
        ),
        "macro_adaptation": macro_average(
            folds, "adaptation", ("post_adaptation_auc", "old_source_drop")
        ),
    }
    with open(os.path.join(args.output_dir, "open_world_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
