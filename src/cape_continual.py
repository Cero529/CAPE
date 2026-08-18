import copy
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cape_datasets import CAPEContinualDataset, cape_collate
from src.cape_memory import FeatureReservoir, ReplayBuffer
from src.cape_metrics import (
    binary_scores_from_av_logits,
    cosine_prototype_drift,
    fake_labels_from_av_targets,
    jensen_shannon_divergence,
    safe_ap,
    safe_auc,
)
from src.cape_models import CAPELoss, CAPEModel
from src.cape_perturbations import shift_temporal_features
from src.cape_unknown import ConformalUnknownDetector, UnknownQueue


class CAPEContinualTrainer:
    def __init__(
        self,
        metadata_csv,
        task_sequence,
        max_length=512,
        d_model=256,
        nhead=4,
        d_hid=512,
        nlayers=1,
        batch_size=64,
        epochs=20,
        lr=1e-4,
        weight_decay=1e-2,
        replay_capacity=1024,
        seed=0,
        device="cuda",
        output_dir="results/cape",
        eval_split="test",
        num_workers=0,
        pin_memory=None,
        prefetch_factor=2,
        persistent_workers=True,
        amp=False,
        max_train_batches=0,
        max_eval_batches=0,
        eval_every_epoch=False,
        save_best=False,
        best_metric="auc",
        show_progress=True,
        freeze_backbone=True,
        backbone_ckpt=None,
        use_confidence_density=True,
        expert_bottleneck=64,
        top_k=2,
        dropout=0.1,
        prototype_momentum=0.95,
        bandwidth_init=0.35,
        bandwidth_momentum=0.9,
        min_bandwidth=0.05,
        max_bandwidth=1.25,
        unknown_alpha=0.05,
        unknown_normalization_fraction=0.5,
        unknown_min_cluster_size=8,
        unknown_cluster_eps=0.8,
        unknown_queue_capacity=4096,
        calibration_capacity=1024,
        validation_calibration_fraction=0.5,
        lambda_pattern=0.5,
        lambda_logic=0.2,
        lambda_distill=1.0,
        lambda_router=0.01,
        use_discrepancy=True,
        use_pattern_guidance=True,
        unknown_component_indices=None,
        allow_expert_expansion=True,
        backbone_mode="legacy_dimodif",
        input_dim=768,
    ):
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.metadata_csv = metadata_csv
        self.task_sequence = [str(task) for task in task_sequence]
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = float(weight_decay)
        self.device = device
        self.output_dir = output_dir
        self.eval_split = eval_split
        self.num_workers = num_workers
        self.pin_memory = device.startswith("cuda") if pin_memory is None else pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.amp = amp and device.startswith("cuda")
        self.max_train_batches = max_train_batches
        self.max_eval_batches = max_eval_batches
        self.eval_every_epoch = eval_every_epoch
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_final = None
        self.show_progress = show_progress
        self.unknown_alpha = unknown_alpha
        self.unknown_normalization_fraction = float(unknown_normalization_fraction)
        if not 0.0 < self.unknown_normalization_fraction < 1.0:
            raise ValueError("unknown_normalization_fraction must lie strictly between 0 and 1")
        self.unknown_min_cluster_size = unknown_min_cluster_size
        self.unknown_cluster_eps = unknown_cluster_eps
        self.validation_calibration_fraction = float(validation_calibration_fraction)
        if not 0.0 < self.validation_calibration_fraction < 1.0:
            raise ValueError("validation_calibration_fraction must lie strictly between 0 and 1")
        calibration_capacity = int(calibration_capacity)
        if calibration_capacity < 2:
            raise ValueError("calibration_capacity must be at least 2")
        self.calibration_capacity = calibration_capacity
        self.allow_expert_expansion = bool(allow_expert_expansion)
        os.makedirs(output_dir, exist_ok=True)
        self.model = CAPEModel(
            max_length=max_length,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            device=device,
            freeze_backbone=freeze_backbone,
            backbone_ckpt=backbone_ckpt,
            use_confidence_density=use_confidence_density,
            expert_bottleneck=expert_bottleneck,
            top_k=top_k,
            dropout=dropout,
            prototype_momentum=prototype_momentum,
            bandwidth_init=bandwidth_init,
            bandwidth_momentum=bandwidth_momentum,
            min_bandwidth=min_bandwidth,
            max_bandwidth=max_bandwidth,
            use_discrepancy=use_discrepancy,
            use_pattern_guidance=use_pattern_guidance,
            unknown_component_indices=unknown_component_indices,
            backbone_mode=backbone_mode,
            input_dim=input_dim,
        ).to(device)
        self.criterion = CAPELoss(
            lambda_pattern=lambda_pattern,
            lambda_logic=lambda_logic,
            lambda_distill=lambda_distill,
            lambda_router=lambda_router,
        )
        self.replay = ReplayBuffer(capacity=replay_capacity, seed=self.seed)
        normalization_capacity = calibration_capacity // 2
        conformal_capacity = calibration_capacity - normalization_capacity
        self.normalization_reservoir = FeatureReservoir(normalization_capacity, seed=self.seed + 1009)
        self.conformal_reservoir = FeatureReservoir(conformal_capacity, seed=self.seed + 2027)
        self.calibration_rng = random.Random(self.seed + 4099)
        self.calibration_assignments = {}
        self.calibration_tasks = set()
        self.unknown_detector = ConformalUnknownDetector()
        self.unknown_queue = UnknownQueue(maxlen=unknown_queue_capacity)
        self.history = []
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.run_config = {
            "seed": self.seed,
            "metadata_csv": metadata_csv,
            "task_sequence": self.task_sequence,
            "eval_split": self.eval_split,
            "device": self.device,
            "max_length": max_length,
            "d_model": d_model,
            "nhead": nhead,
            "d_hid": d_hid,
            "nlayers": nlayers,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": self.weight_decay,
            "replay_capacity": replay_capacity,
            "replay_enabled": True,
            "expert_bottleneck": expert_bottleneck,
            "top_k": top_k,
            "dropout": dropout,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "amp": self.amp,
            "max_train_batches": self.max_train_batches,
            "max_eval_batches": self.max_eval_batches,
            "eval_every_epoch": self.eval_every_epoch,
            "save_best": self.save_best,
            "best_metric": self.best_metric,
            "freeze_backbone": freeze_backbone,
            "backbone_ckpt": backbone_ckpt,
            "use_confidence_density": use_confidence_density,
            "prototype_momentum": prototype_momentum,
            "bandwidth_init": bandwidth_init,
            "bandwidth_momentum": bandwidth_momentum,
            "min_bandwidth": min_bandwidth,
            "max_bandwidth": max_bandwidth,
            "unknown_alpha": unknown_alpha,
            "unknown_normalization_fraction": unknown_normalization_fraction,
            "unknown_min_cluster_size": unknown_min_cluster_size,
            "unknown_cluster_eps": unknown_cluster_eps,
            "unknown_queue_capacity": unknown_queue_capacity,
            "calibration_capacity": calibration_capacity,
            "validation_calibration_fraction": validation_calibration_fraction,
            "lambda_pattern": lambda_pattern,
            "lambda_logic": lambda_logic,
            "lambda_distill": lambda_distill,
            "lambda_router": lambda_router,
            "use_discrepancy": use_discrepancy,
            "use_pattern_guidance": use_pattern_guidance,
            "unknown_component_indices": list(range(4)) if unknown_component_indices is None else list(unknown_component_indices),
            "unknown_detector": "conformal",
            "allow_expert_expansion": allow_expert_expansion,
            "backbone_mode": backbone_mode,
            "input_dim": input_dim,
        }
        with open(os.path.join(output_dir, "run_config.json"), "w") as f:
            json.dump(self.run_config, f, indent=2)

    def _loader_kwargs(self, shuffle):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "collate_fn": cape_collate,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if shuffle:
            kwargs["generator"] = torch.Generator().manual_seed(self.seed)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs

    def loader(
        self,
        split,
        task_id=None,
        shuffle=False,
        validation_role=None,
        sample_ids=None,
        exclude_sample_ids=None,
        confirmed_emerging=False,
    ):
        if validation_role is None and split == "val":
            validation_role = "checkpoint"
        dataset = CAPEContinualDataset(
            self.metadata_csv,
            split=split,
            max_length=self.max_length,
            task_id=task_id,
            validation_role=validation_role,
            calibration_fraction=self.validation_calibration_fraction,
            partition_seed=self.seed,
            sample_ids=sample_ids,
            exclude_sample_ids=exclude_sample_ids,
            confirmed_emerging=confirmed_emerging,
            require_backbone_features=self.model.backbone_mode == "precomputed_avhubert",
            require_explicit_valid_length=self.model.backbone_mode == "precomputed_avhubert",
        )
        if len(dataset) == 0:
            raise RuntimeError(
                f"No samples for split={split!r}, task_id={task_id!r}, "
                f"validation_role={validation_role!r} in {self.metadata_csv}"
            )
        return DataLoader(dataset, **self._loader_kwargs(shuffle))

    def _move(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device, non_blocking=self.pin_memory) if torch.is_tensor(value) else value
        return moved

    def _average_seen_metric(self, seen, metric):
        values = [metrics.get(metric, float("nan")) for metrics in seen.values()]
        values = [value for value in values if value == value]
        return sum(values) / len(values) if values else float("nan")

    def _save_best_if_needed(self, seen, current_task, epoch):
        if not self.save_best or current_task != self.task_sequence[-1]:
            return
        value = self._average_seen_metric(seen, self.best_metric)
        if value != value:
            return
        if self.best_final is not None and value <= self.best_final.get(self.best_metric, float("-inf")):
            return
        self.best_final = {
            self.best_metric: value,
            "current_task": current_task,
            "epoch": epoch,
            "seen": seen,
        }
        best_path = os.path.join(self.output_dir, "best_final.json")
        ckpt_path = os.path.join(self.output_dir, "best_final.pt")
        with open(best_path, "w") as f:
            json.dump(self.best_final, f, indent=2)
        torch.save({"model": self.model.state_dict(), "best": self.best_final}, ckpt_path)

    def _restore_best_final_if_available(self):
        if not self.save_best:
            return False
        ckpt_path = os.path.join(self.output_dir, "best_final.pt")
        if not os.path.exists(ckpt_path):
            return False
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        return True

    def train_task(self, task_id, old_model=None, seen_tasks=None, train_loader=None):
        if self.allow_expert_expansion and self.model.num_experts < len(self.history) + 1:
            self.model.add_expert()
        self.model.freeze_old_experts(keep_last_trainable=True)
        optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if old_model is not None:
            old_model = old_model.to(self.device).eval()
        train_loader = train_loader or self.loader("train", task_id=task_id, shuffle=True)
        self.model.train()
        epoch_summaries = []
        task_index = len(self.history) + 1
        total_tasks = len(self.task_sequence)
        if self.show_progress:
            tqdm.write(
                f"[CAPE] Train task {task_index}/{total_tasks}: {task_id} "
                f"({len(train_loader.dataset)} samples, {len(train_loader)} batches)"
            )
        for epoch in range(self.epochs):
            running = {
                "loss": 0.0,
                "det": 0.0,
                "pattern": 0.0,
                "logic": 0.0,
                "distill": 0.0,
                "router": 0.0,
            }
            progress = tqdm(
                train_loader,
                desc=f"Task {task_index}/{total_tasks} {task_id} | epoch {epoch + 1}/{self.epochs}",
                leave=True,
                dynamic_ncols=True,
                ascii=True,
                disable=not self.show_progress,
            )
            for step, batch in enumerate(progress, start=1):
                if self.max_train_batches and step > self.max_train_batches:
                    break
                batch = self._move(batch)
                current_batch_for_replay = {
                    key: (value.detach().cpu() if torch.is_tensor(value) else list(value))
                    for key, value in batch.items()
                    if torch.is_tensor(value) or isinstance(value, list)
                }
                replay_batch = self.replay.sample(self.batch_size // 2)
                if replay_batch is not None:
                    replay_batch = self._move(replay_batch)
                    for key in (
                        "video_features",
                        "audio_features",
                        "valid_length",
                        "labels",
                        "has_fake_period",
                        "unknown_flag",
                        "pattern_availability",
                        "backbone_features",
                    ):
                        if torch.is_tensor(batch.get(key)) and torch.is_tensor(replay_batch.get(key)):
                            batch[key] = torch.cat([batch[key], replay_batch[key]], dim=0)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    outputs = self.model(
                        batch["video_features"],
                        batch["audio_features"],
                        valid_lengths=batch["valid_length"],
                        backbone_features=batch.get("backbone_features"),
                    )
                    old_outputs = None
                    if old_model is not None:
                        with torch.no_grad():
                            old_outputs = old_model(
                                batch["video_features"],
                                batch["audio_features"],
                                valid_lengths=batch["valid_length"],
                                backbone_features=batch.get("backbone_features"),
                            )
                    losses = self.criterion(
                        outputs,
                        batch["labels"],
                        old_outputs=old_outputs,
                        has_fake_period=batch["has_fake_period"],
                        unknown_flag=batch["unknown_flag"],
                        pattern_availability=batch["pattern_availability"],
                    )
                optimizer.zero_grad(set_to_none=True)
                if self.amp:
                    self.scaler.scale(losses["loss"]).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    losses["loss"].backward()
                    optimizer.step()
                self.model.update_prototypes(
                    outputs["features"].detach().float(),
                    outputs.get("confidence_weights", outputs["router_probs"]).detach(),
                )
                self.replay.add_batch(current_batch_for_replay)
                for key in running:
                    running[key] += float(losses[key].detach().cpu())
                progress.set_postfix(
                    loss=running["loss"] / step,
                    det=running["det"] / step,
                    distill=running["distill"] / step,
                )
            denom = min(len(train_loader), self.max_train_batches) if self.max_train_batches else len(train_loader)
            epoch_record = {key: value / max(1, denom) for key, value in running.items()}
            if self.eval_every_epoch and seen_tasks:
                epoch_record["seen"] = {
                    seen_task: self.evaluate_task("val", task_id=seen_task, show_progress=False)
                    for seen_task in seen_tasks
                }
                epoch_record[f"avg_{self.best_metric}"] = self._average_seen_metric(
                    epoch_record["seen"], self.best_metric
                )
                self._save_best_if_needed(epoch_record["seen"], task_id, epoch + 1)
                if self.show_progress:
                    tqdm.write(
                        f"[CAPE] Epoch eval task={task_id} epoch={epoch + 1}: "
                        f"avg_{self.best_metric}={epoch_record[f'avg_{self.best_metric}']:.4f}"
                    )
            epoch_summaries.append(epoch_record)
        return epoch_summaries

    @torch.no_grad()
    def evaluate_task(
        self,
        split,
        task_id=None,
        show_progress=False,
        audio_shift_steps=0,
        sample_ids=None,
        exclude_sample_ids=None,
    ):
        self.model.eval()
        labels, scores, unknown_scores = [], [], []
        # Accumulate the *actual* confidence-density routing weights w used
        # by the expert bank.  Keeping both soft mass and top-1 frequency
        # makes it possible to audit whether experts are reused across
        # stages instead of silently behaving as one expert per stage.
        expert_weight_sum = None
        expert_top1_count = None
        expert_sample_count = 0
        loader = self.loader(
            split,
            task_id=task_id,
            sample_ids=sample_ids,
            exclude_sample_ids=exclude_sample_ids,
        )
        progress = tqdm(
            loader,
            desc=f"Eval {split} {task_id}",
            leave=False,
            dynamic_ncols=True,
            ascii=True,
            disable=not (self.show_progress and show_progress),
        )
        for step, batch in enumerate(progress, start=1):
            if self.max_eval_batches and step > self.max_eval_batches:
                break
            batch = self._move(batch)
            audio_features = shift_temporal_features(batch["audio_features"], audio_shift_steps)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                outputs = self.model(
                    batch["video_features"],
                    audio_features,
                    valid_lengths=batch["valid_length"],
                    backbone_features=batch.get("backbone_features"),
                )
            labels.extend(fake_labels_from_av_targets(batch["labels"]).tolist())
            scores.extend(binary_scores_from_av_logits(outputs["fake_logits"]).tolist())
            unknown_scores.extend(outputs["unknown_score"].detach().cpu().tolist())
            weights = outputs.get("confidence_weights", outputs["router_probs"]).detach().float().cpu()
            if expert_weight_sum is None:
                expert_weight_sum = torch.zeros(weights.shape[-1], dtype=torch.float64)
                expert_top1_count = torch.zeros(weights.shape[-1], dtype=torch.float64)
            expert_weight_sum += weights.sum(dim=0, dtype=torch.float64)
            expert_top1_count += torch.bincount(
                weights.argmax(dim=-1), minlength=weights.shape[-1]
            ).to(dtype=torch.float64)
            expert_sample_count += int(weights.shape[0])
        if expert_sample_count:
            expert_mean_weight = (expert_weight_sum / expert_sample_count).tolist()
            expert_top1_rate = (expert_top1_count / expert_sample_count).tolist()
        else:
            expert_mean_weight = []
            expert_top1_rate = []
        return {
            "auc": safe_auc(labels, scores),
            "ap": safe_ap(labels, scores),
            "unknown_score_mean": float(torch.tensor(unknown_scores).mean()) if unknown_scores else float("nan"),
            "expert_mean_weight": expert_mean_weight,
            "expert_top1_rate": expert_top1_rate,
            "num_eval_samples": expert_sample_count,
        }

    @torch.no_grad()
    def update_calibration_reservoir(self, task_id):
        """Insert one admitted stage's disjoint validation features once."""
        task_id = str(task_id)
        if task_id in self.calibration_tasks:
            return
        self.model.eval()
        loader = self.loader(
            "val",
            task_id=task_id,
            validation_role="calibration",
        )
        for batch in loader:
            for idx, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                pool = self.calibration_assignments.get(sample_id)
                if pool is None:
                    if len(self.normalization_reservoir) == 0 and len(self.conformal_reservoir) > 0:
                        pool = "normalization"
                    elif len(self.conformal_reservoir) == 0 and len(self.normalization_reservoir) > 0:
                        pool = "conformal"
                    else:
                        pool = "normalization" if self.calibration_rng.random() < 0.5 else "conformal"
                    self.calibration_assignments[sample_id] = pool
                reservoir = (
                    self.normalization_reservoir if pool == "normalization" else self.conformal_reservoir
                )
                reservoir.add(
                    sample_id,
                    video_features=batch["video_features"][idx],
                    audio_features=batch["audio_features"][idx],
                    valid_length=batch["valid_length"][idx],
                    **(
                        {"backbone_features": batch["backbone_features"][idx]}
                        if torch.is_tensor(batch.get("backbone_features"))
                        else {}
                    ),
                )
        self.calibration_tasks.add(task_id)

    @torch.no_grad()
    def fit_unknown_calibration(self):
        self.model.eval()
        if not self.normalization_reservoir.items or not self.conformal_reservoir.items:
            raise RuntimeError(
                "Both fixed calibration reservoirs must be non-empty; call "
                "update_calibration_reservoir after admitting a stage"
            )

        normalizer_components = []
        for batch in self.normalization_reservoir.batches(self.batch_size):
            batch = self._move(batch)
            outputs = self.model(
                batch["video_features"],
                batch["audio_features"],
                valid_lengths=batch["valid_length"],
                backbone_features=batch.get("backbone_features"),
            )
            normalizer_components.append(outputs["unknown_components"].detach().float().cpu())
        normalizer_components = torch.cat(normalizer_components, dim=0)
        self.model.fit_unknown_normalizer(normalizer_components)

        calibration_components = []
        for batch in self.conformal_reservoir.batches(self.batch_size):
            batch = self._move(batch)
            outputs = self.model(
                batch["video_features"],
                batch["audio_features"],
                valid_lengths=batch["valid_length"],
                backbone_features=batch.get("backbone_features"),
            )
            calibration_components.append(outputs["unknown_components"].detach().float().cpu())
        calibration_components = torch.cat(calibration_components, dim=0)
        calibration_scores = self.model.aggregate_unknown_components(calibration_components)
        self.unknown_detector.fit(calibration_scores)

    @torch.no_grad()
    def collect_unknown_candidates(self, split, task_ids, alpha=None):
        if self.unknown_detector.calibration_scores is None:
            self.fit_unknown_calibration()
        alpha = self.unknown_alpha if alpha is None else alpha
        task_ids = [str(task_id) for task_id in task_ids]
        pushed = 0
        observed = 0
        detection_delay = None
        for task_id in task_ids:
            for batch in self.loader(split, task_id=task_id):
                batch = self._move(batch)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    outputs = self.model(
                        batch["video_features"],
                        batch["audio_features"],
                        valid_lengths=batch["valid_length"],
                        backbone_features=batch.get("backbone_features"),
                    )
                p_values = self.unknown_detector.p_value(outputs["unknown_score"]).to(self.device)
                mask = p_values < alpha
                features = outputs["features"].detach().cpu()
                scores = outputs["unknown_score"].detach().cpu()
                p_cpu = p_values.detach().cpu()
                mask_cpu = mask.detach().cpu()
                for idx in range(mask_cpu.numel()):
                    observed += 1
                    if not bool(mask_cpu[idx]):
                        continue
                    self.unknown_queue.push(
                        batch["sample_id"][idx],
                        features[idx],
                        scores[idx],
                        meta={
                            "task_id": task_id,
                            "p_value": float(p_cpu[idx]),
                            "stream_index": observed,
                        },
                    )
                    pushed += 1
                    # Replay DBSCAN on the current queue prefix only until the
                    # first non-noise candidate cluster appears.  The stream
                    # index counts every observed sample, not only rejections.
                    if detection_delay is None:
                        prefix_clusters = self.unknown_queue.cluster_candidates(
                            min_cluster_size=self.unknown_min_cluster_size,
                            eps=self.unknown_cluster_eps,
                        )
                        if prefix_clusters:
                            detection_delay = observed
        clusters = self.unknown_queue.cluster_candidates(
            min_cluster_size=self.unknown_min_cluster_size,
            eps=self.unknown_cluster_eps,
        )
        cluster_records = []
        for cluster_id, cluster in enumerate(clusters):
            cluster_records.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(cluster),
                    "sample_ids": [item["sample_id"] for item in cluster],
                    "mean_unknown_score": float(np.mean([item["score"] for item in cluster])),
                    "source_task_ids": sorted(
                        {str(item.get("meta", {}).get("task_id", "")) for item in cluster}
                    ),
                }
            )
        return {
            "num_candidates": pushed,
            "queue_size": len(self.unknown_queue),
            "num_clusters": len(clusters),
            "cluster_sizes": [len(cluster) for cluster in clusters],
            "clusters": cluster_records,
            "num_observed": observed,
            "detection_delay": detection_delay,
        }

    @torch.no_grad()
    def evaluate_drift(self, old_model, task_ids):
        if old_model is None:
            return {
                "router_drift_js": float("nan"),
                "new_expert_mass": float("nan"),
                "prototype_drift_cos": float("nan"),
            }
        self.model.eval()
        old_model = old_model.to(self.device).eval()
        old_experts = old_model.num_experts
        router_drift, new_expert_mass = [], []
        for task_id in task_ids:
            for batch in self.loader("val", task_id=task_id):
                batch = self._move(batch)
                old_outputs = old_model(
                    batch["video_features"],
                    batch["audio_features"],
                    valid_lengths=batch["valid_length"],
                    backbone_features=batch.get("backbone_features"),
                )
                new_outputs = self.model(
                    batch["video_features"],
                    batch["audio_features"],
                    valid_lengths=batch["valid_length"],
                    backbone_features=batch.get("backbone_features"),
                )
                old_weights = old_outputs["confidence_weights"].detach().cpu().numpy()
                new_weights = new_outputs["confidence_weights"].detach().cpu().numpy()
                retained = new_weights[:, :old_experts]
                retained_mass = retained.sum(axis=-1)
                retained = retained / np.clip(retained_mass[:, None], 1e-8, None)
                router_drift.extend(jensen_shannon_divergence(old_weights, retained).tolist())
                new_expert_mass.extend((1.0 - retained_mass).tolist())

        old_counts = old_model.prototype_counts[:old_experts].detach().cpu().numpy()
        new_counts = self.model.prototype_counts[:old_experts].detach().cpu().numpy()
        active = (old_counts > 0) & (new_counts > 0)
        prototype_drift = cosine_prototype_drift(
            old_model.expert_prototypes[:old_experts].detach().cpu().numpy(),
            self.model.expert_prototypes[:old_experts].detach().cpu().numpy(),
            active_mask=active,
        )
        return {
            "router_drift_js": float(np.mean(router_drift)) if router_drift else float("nan"),
            "new_expert_mass": float(np.mean(new_expert_mass)) if new_expert_mass else float("nan"),
            "prototype_drift_cos": prototype_drift,
        }

    def run(self):
        for task_id in self.task_sequence:
            old_model = copy.deepcopy(self.model).eval() if self.history else None
            seen_tasks = self.task_sequence[: len(self.history) + 1]
            train_summary = self.train_task(task_id, old_model=old_model, seen_tasks=seen_tasks)
            if task_id == self.task_sequence[-1]:
                self._restore_best_final_if_available()
            previous_tasks = self.task_sequence[: len(self.history)]
            diagnostics = self.evaluate_drift(old_model, previous_tasks) if previous_tasks else None
            self.update_calibration_reservoir(task_id)
            self.fit_unknown_calibration()
            row = {
                "current_task": task_id,
                "train": train_summary,
                "seen": {},
                "diagnostics": diagnostics,
            }
            for seen_task in self.task_sequence[: len(self.history) + 1]:
                row["seen"][seen_task] = self.evaluate_task(self.eval_split, task_id=seen_task, show_progress=True)
            if self.show_progress:
                tqdm.write(f"[CAPE] Evaluation after {task_id}: {row['seen']}")
            self.history.append(row)
            with open(os.path.join(self.output_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
        return self.history
