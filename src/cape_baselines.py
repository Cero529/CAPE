import copy
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.cape_datasets import CAPEContinualDataset, cape_collate
from src.cape_memory import ReplayBuffer
from src.cape_metrics import binary_scores_from_av_logits, fake_labels_from_av_targets, safe_ap, safe_auc
from src.cape_models import PrecomputedAVHubertBackbone
from src.models import AVDeepFakeDetector


class LogitReplayBuffer(ReplayBuffer):
    """Replay buffer that can also store teacher logits for DER-style baselines."""

    def add_batch(self, batch, logits=None):
        stored = dict(batch)
        if logits is not None:
            stored["stored_logits"] = logits.detach().cpu()
        super().add_batch(stored)


class AVContinualBaseline(nn.Module):
    """Task-free audio-visual detector used by baseline continual learners."""

    def __init__(
        self,
        max_length,
        d_model=256,
        nhead=4,
        d_hid=512,
        nlayers=1,
        dropout=0.1,
        freeze_backbone=False,
        backbone_ckpt=None,
        backbone_mode="legacy_dimodif",
        device="cpu",
    ):
        super().__init__()
        self.backbone_mode = str(backbone_mode)
        if self.backbone_mode == "legacy_dimodif":
            self.backbone = AVDeepFakeDetector(
                task="dfd",
                max_length=max_length,
                d_model=d_model,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=nlayers,
                dropout=dropout,
                feature_pyramid=False,
                device=device,
            )
            self.classification = None
        elif self.backbone_mode == "precomputed_avhubert":
            if int(d_model) != 768:
                raise ValueError("The paper baseline profile requires d_model=768")
            self.backbone = PrecomputedAVHubertBackbone(output_dim=d_model)
            self.classification = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
            )
        else:
            raise ValueError("backbone_mode must be 'legacy_dimodif' or 'precomputed_avhubert'")
        self.backbone_load_report = None
        if backbone_ckpt and self.backbone_mode == "legacy_dimodif":
            self.backbone_load_report = self.load_backbone_checkpoint(backbone_ckpt, map_location=device)
        elif backbone_ckpt:
            raise ValueError("precomputed_avhubert does not load a detector checkpoint")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Keep the detector head trainable while freezing the shared
            # audio-visual encoder.  Freezing every parameter leaves the
            # continual baselines with an empty optimizer and is not the
            # intended matched-backbone protocol.
            if self.backbone_mode == "legacy_dimodif":
                for param in self.backbone.classification.parameters():
                    param.requires_grad = True

    def forward(self, video_features, audio_features, valid_lengths=None, backbone_features=None):
        if self.backbone_mode == "legacy_dimodif":
            logits, features = self.backbone([video_features, audio_features])
        else:
            features = self.backbone(backbone_features, valid_lengths=valid_lengths)
            logits = self.classification(features)
        return {"fake_logits": logits, "features": features}

    def load_backbone_checkpoint(self, ckpt_path, map_location="cpu"):
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        state = checkpoint.get("model", checkpoint)
        current = self.backbone.state_dict()
        matched = {
            key: value
            for key, value in state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        skipped = sorted(set(state.keys()) - set(matched.keys()))
        self.backbone.load_state_dict(matched, strict=False)
        return {
            "path": str(ckpt_path),
            "matched": len(matched),
            "skipped": len(skipped),
            "skipped_examples": skipped[:10],
        }


class EWCState:
    def __init__(self):
        self.means = {}
        self.fisher = {}

    def empty(self):
        return len(self.means) == 0


class ContinualBaselineTrainer:
    """Unified trainer for real continual-learning baselines.

    Supported methods:
    - seq_ft: sequential fine-tuning.
    - er: exemplar replay.
    - lwf: Learning without Forgetting-style response distillation.
    - der: replay with stored logits.
    - derpp: replay with stored logits plus supervised replay.
    - ewc: diagonal Fisher regularization.
    - joint: offline joint reference, trained once on all task data.
    - single_task: task oracle, one independent model per task.
    """

    def __init__(
        self,
        metadata_csv,
        task_sequence,
        method="seq_ft",
        max_length=512,
        d_model=256,
        nhead=4,
        d_hid=512,
        nlayers=1,
        batch_size=16,
        epochs=1,
        lr=1e-4,
        weight_decay=1e-2,
        replay_capacity=1024,
        seed=0,
        device="cuda",
        output_dir="results/baselines",
        eval_split="test",
        num_workers=0,
        pin_memory=None,
        prefetch_factor=2,
        persistent_workers=True,
        amp=False,
        freeze_backbone=False,
        backbone_ckpt=None,
        backbone_mode="legacy_dimodif",
        distill_weight=0.5,
        der_weight=0.5,
        ewc_weight=100.0,
        show_progress=True,
    ):
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.metadata_csv = metadata_csv
        self.task_sequence = [str(task) for task in task_sequence]
        self.method = str(method).lower()
        if self.method not in {"seq_ft", "er", "lwf", "der", "derpp", "ewc", "joint", "single_task"}:
            raise ValueError(f"Unsupported baseline method: {self.method}")
        self.max_length = max_length
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
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
        self.freeze_backbone = freeze_backbone
        self.backbone_ckpt = backbone_ckpt
        self.backbone_mode = str(backbone_mode)
        self.distill_weight = distill_weight
        self.der_weight = der_weight
        self.ewc_weight = ewc_weight
        self.show_progress = show_progress
        os.makedirs(output_dir, exist_ok=True)
        self.model = self._new_model()
        self.det_loss = nn.BCEWithLogitsLoss()
        self.replay = LogitReplayBuffer(capacity=replay_capacity, seed=self.seed)
        self.ewc = EWCState()
        self.history = []
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.run_config = {
            "seed": self.seed,
            "method": self.method,
            "metadata_csv": metadata_csv,
            "task_sequence": self.task_sequence,
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
            "eval_split": eval_split,
            "backbone_mode": self.backbone_mode,
            "backbone_ckpt": backbone_ckpt,
            "freeze_backbone": freeze_backbone,
            "distill_weight": distill_weight,
            "der_weight": der_weight,
            "ewc_weight": ewc_weight,
        }
        with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
            json.dump(self.run_config, handle, indent=2)

    def _new_model(self):
        return AVContinualBaseline(
            max_length=self.max_length,
            d_model=self.d_model,
            nhead=self.nhead,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            freeze_backbone=self.freeze_backbone,
            backbone_ckpt=self.backbone_ckpt,
            backbone_mode=self.backbone_mode,
            device=self.device,
        ).to(self.device)

    def dataset(self, split, task_id=None):
        return CAPEContinualDataset(
            self.metadata_csv,
            split=split,
            max_length=self.max_length,
            task_id=task_id,
            require_backbone_features=self.backbone_mode == "precomputed_avhubert",
            require_explicit_valid_length=self.backbone_mode == "precomputed_avhubert",
        )

    def loader(self, split, task_id=None, shuffle=False, dataset=None):
        dataset = dataset if dataset is not None else self.dataset(split, task_id)
        if len(dataset) == 0:
            raise RuntimeError(
                f"No samples for split={split!r}, task_id={task_id!r} in {self.metadata_csv}"
            )
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "collate_fn": cape_collate,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        if shuffle:
            kwargs["generator"] = torch.Generator().manual_seed(self.seed)
        return DataLoader(dataset, **kwargs)

    def _move(self, batch):
        return {
            key: (value.to(self.device, non_blocking=self.pin_memory) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    def _batch_for_replay(self, batch):
        return {
            key: (value.detach().cpu() if torch.is_tensor(value) else list(value))
            for key, value in batch.items()
            if torch.is_tensor(value) or isinstance(value, list)
        }

    def _loss(self, batch, old_model=None, replay_batch=None):
        outputs = self.model(
            batch["video_features"],
            batch["audio_features"],
            valid_lengths=batch.get("valid_length"),
            backbone_features=batch.get("backbone_features"),
        )
        loss = self.det_loss(outputs["fake_logits"], batch["labels"].float())
        logs = {"loss": loss, "det": loss.detach()}

        if old_model is not None and self.method == "lwf":
            with torch.no_grad():
                old_outputs = old_model(
                    batch["video_features"],
                    batch["audio_features"],
                    valid_lengths=batch.get("valid_length"),
                    backbone_features=batch.get("backbone_features"),
                )
            distill = nn.functional.mse_loss(
                torch.sigmoid(outputs["fake_logits"]),
                torch.sigmoid(old_outputs["fake_logits"]).detach(),
            )
            loss = loss + self.distill_weight * distill
            logs["distill"] = distill.detach()

        if replay_batch is not None and self.method in {"der", "derpp"} and "stored_logits" in replay_batch:
            replay_outputs = self.model(
                replay_batch["video_features"],
                replay_batch["audio_features"],
                valid_lengths=replay_batch.get("valid_length"),
                backbone_features=replay_batch.get("backbone_features"),
            )
            der = nn.functional.mse_loss(replay_outputs["fake_logits"], replay_batch["stored_logits"].float())
            loss = loss + self.der_weight * der
            logs["der"] = der.detach()

        if self.method == "ewc" and not self.ewc.empty():
            penalty = outputs["fake_logits"].new_tensor(0.0)
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in self.ewc.means:
                    continue
                penalty = penalty + (self.ewc.fisher[name] * (param - self.ewc.means[name]).pow(2)).sum()
            loss = loss + self.ewc_weight * penalty
            logs["ewc"] = penalty.detach()

        logs["loss"] = loss
        return logs, outputs

    def train_on_loader(self, loader, task_id, old_model=None):
        optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        epoch_summaries = []
        for epoch in range(self.epochs):
            running = {}
            progress = tqdm(
                loader,
                desc=f"{self.method} {task_id} | epoch {epoch + 1}/{self.epochs}",
                leave=True,
                dynamic_ncols=True,
                ascii=True,
                disable=not self.show_progress,
            )
            for step, batch in enumerate(progress, start=1):
                batch = self._move(batch)
                current_batch = self._batch_for_replay(batch)
                current_size = batch["labels"].shape[0]
                replay_batch = None
                if self.method in {"er", "der", "derpp"}:
                    replay_batch = self.replay.sample(self.batch_size // 2)
                    if replay_batch is not None:
                        replay_batch = self._move(replay_batch)
                        if self.method in {"er", "derpp"}:
                            for key in (
                                "video_features",
                                "audio_features",
                                "valid_length",
                                "labels",
                                "has_fake_period",
                                "backbone_features",
                            ):
                                if torch.is_tensor(batch.get(key)) and torch.is_tensor(replay_batch.get(key)):
                                    batch[key] = torch.cat([batch[key], replay_batch[key]], dim=0)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    logs, outputs = self._loss(batch, old_model=old_model, replay_batch=replay_batch)
                optimizer.zero_grad(set_to_none=True)
                if self.amp:
                    self.scaler.scale(logs["loss"]).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logs["loss"].backward()
                    optimizer.step()

                if self.method in {"er", "der", "derpp"}:
                    self.replay.add_batch(
                        current_batch,
                        # DER stores the response observed when the sample first
                        # enters training, before the optimizer mutates the model.
                        logits=(
                            outputs["fake_logits"][:current_size]
                            if self.method in {"der", "derpp"}
                            else None
                        ),
                    )

                for key, value in logs.items():
                    running[key] = running.get(key, 0.0) + float(value.detach().cpu())
                progress.set_postfix(loss=running["loss"] / step, det=running.get("det", 0.0) / step)
            epoch_summaries.append({key: value / max(1, len(loader)) for key, value in running.items()})
        return epoch_summaries

    @torch.no_grad()
    def evaluate_task(self, split, task_id=None):
        self.model.eval()
        labels, scores = [], []
        for batch in self.loader(split, task_id=task_id):
            batch = self._move(batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                outputs = self.model(
                    batch["video_features"],
                    batch["audio_features"],
                    valid_lengths=batch.get("valid_length"),
                    backbone_features=batch.get("backbone_features"),
                )
            labels.extend(fake_labels_from_av_targets(batch["labels"]).tolist())
            scores.extend(binary_scores_from_av_logits(outputs["fake_logits"]).tolist())
        return {"auc": safe_auc(labels, scores), "ap": safe_ap(labels, scores)}

    def estimate_ewc(self, task_id, max_batches=32):
        self.model.eval()
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        loader = self.loader("train", task_id=task_id, shuffle=True)
        seen = 0
        for batch in loader:
            batch = self._move(batch)
            outputs = self.model(
                batch["video_features"],
                batch["audio_features"],
                valid_lengths=batch.get("valid_length"),
                backbone_features=batch.get("backbone_features"),
            )
            loss = self.det_loss(outputs["fake_logits"], batch["labels"].float())
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            seen += 1
            if seen >= max_batches:
                break
        denom = max(1, seen)
        self.ewc.means = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.ewc.fisher = {name: value / denom for name, value in fisher.items()}

    def _write_history(self):
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        if self.history:
            with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
                json.dump({"final_seen": self.history[-1].get("seen", {})}, f, indent=2)

    def run_sequential(self):
        for task_id in self.task_sequence:
            old_model = copy.deepcopy(self.model).eval() if self.method == "lwf" and self.history else None
            train_loader = self.loader("train", task_id=task_id, shuffle=True)
            train_summary = self.train_on_loader(train_loader, task_id=task_id, old_model=old_model)
            if self.method == "ewc":
                self.estimate_ewc(task_id)
            row = {"current_task": task_id, "train": train_summary, "seen": {}}
            for seen_task in self.task_sequence[: len(self.history) + 1]:
                row["seen"][seen_task] = self.evaluate_task(self.eval_split, task_id=seen_task)
            self.history.append(row)
            self._write_history()
        return self.history

    def run_joint(self):
        datasets = [self.dataset("train", task_id=task_id) for task_id in self.task_sequence]
        train_loader = self.loader("train", dataset=ConcatDataset(datasets), shuffle=True)
        train_summary = self.train_on_loader(train_loader, task_id="joint")
        row = {"current_task": "joint", "train": train_summary, "seen": {}}
        for task_id in self.task_sequence:
            row["seen"][task_id] = self.evaluate_task(self.eval_split, task_id=task_id)
        self.history = [row]
        self._write_history()
        return self.history

    def run_single_task(self):
        rows = []
        for task_id in self.task_sequence:
            self.model = self._new_model()
            train_loader = self.loader("train", task_id=task_id, shuffle=True)
            train_summary = self.train_on_loader(train_loader, task_id=task_id)
            rows.append(
                {
                    "current_task": task_id,
                    "train": train_summary,
                    "seen": {task_id: self.evaluate_task(self.eval_split, task_id=task_id)},
                }
            )
        self.history = rows
        self._write_history()
        return self.history

    def run(self):
        if self.method == "joint":
            return self.run_joint()
        if self.method == "single_task":
            return self.run_single_task()
        if self.method not in {"seq_ft", "er", "lwf", "der", "derpp", "ewc"}:
            raise ValueError(f"Unsupported baseline method: {self.method}")
        return self.run_sequential()
