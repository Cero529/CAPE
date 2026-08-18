import os
import shutil
import tempfile
import csv
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_datasets import CAPEContinualDataset
from src.cape_logic import pattern_supervision_loss
from src.cape_memory import ReplayBuffer
from src.cape_models import CAPELoss, CAPEModel
from src.cape_perturbations import shift_temporal_features
from src.cape_unknown import ConformalUnknownDetector, UnknownQueue


def make_synthetic_metadata(root, n=18, max_length=12):
    rows = []
    for idx in range(n):
        task_id = 1 + (idx % 3)
        video_target, audio_target = [(1, 0), (0, 1), (1, 1)][idx % 3]
        if idx % 5 == 0:
            video_target, audio_target, task_id = 0, 0, "real"
        feature_path = os.path.join(root, f"sample_{idx}.npz")
        length = max(4, max_length - (idx % 4))
        base = np.random.randn(length, 768).astype("float32")
        video = base + video_target * 0.6
        audio = base - audio_target * 0.6
        np.savez(feature_path, video_features=video, audio_features=audio)
        split = "train" if idx < 12 else ("val" if idx < 15 else "test")
        rows.append(
            {
                "sample_id": idx,
                "dataset": "avdeepfake1m",
                "feature_path": feature_path,
                "split": split,
                "video_target": video_target,
                "audio_target": audio_target,
                "is_fake": int(video_target or audio_target),
                "pattern_id": 0 if not (video_target or audio_target) else task_id,
                "task_id": task_id,
                "generator": "none",
                "fake_periods": "" if idx == 3 else "[]",
            }
        )
    metadata = os.path.join(root, "cape_metadata.csv")
    with open(metadata, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def main():
    torch.manual_seed(7)
    np.random.seed(7)
    tmp = tempfile.mkdtemp(prefix="cape_smoke_")
    try:
        metadata = make_synthetic_metadata(tmp)
        dataset = CAPEContinualDataset(metadata, split="train", max_length=12, task_id=1)
        assert len(dataset) > 0
        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        video = torch.stack([x["video_features"] for x in batch])
        audio = torch.stack([x["audio_features"] for x in batch])
        valid_lengths = torch.stack([x["valid_length"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        has_fake_period = torch.stack([x["has_fake_period"] for x in batch])
        unknown_flag = torch.stack([x["unknown_flag"] for x in batch])
        pattern_availability = torch.stack([x["pattern_availability"] for x in batch])
        assert (pattern_availability[:, 3] == 0).any(), "missing temporal labels must remain unavailable"

        model = CAPEModel(max_length=12, d_model=32, nhead=4, d_hid=64, nlayers=1, freeze_backbone=True, device="cpu")
        criterion = CAPELoss(lambda_distill=0.0)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)

        model.discrepancy_encoder.eval()
        video_with_noisy_padding = video.clone()
        audio_with_noisy_padding = audio.clone()
        for idx, length in enumerate(valid_lengths.tolist()):
            video_with_noisy_padding[idx, length:] = torch.randn_like(video_with_noisy_padding[idx, length:])
            audio_with_noisy_padding[idx, length:] = torch.randn_like(audio_with_noisy_padding[idx, length:])
        with torch.no_grad():
            clean_discrepancy = model.discrepancy_encoder(video, audio, valid_lengths=valid_lengths)
            noisy_discrepancy = model.discrepancy_encoder(
                video_with_noisy_padding,
                audio_with_noisy_padding,
                valid_lengths=valid_lengths,
            )
        assert torch.allclose(clean_discrepancy, noisy_discrepancy, atol=1e-6)
        model.discrepancy_encoder.train()

        losses = []
        for _ in range(8):
            outputs = model(video, audio, valid_lengths=valid_lengths)
            loss_dict = criterion(
                outputs,
                labels,
                has_fake_period=has_fake_period,
                unknown_flag=unknown_flag,
                pattern_availability=pattern_availability,
            )
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            model.update_prototypes(outputs["features"].detach(), outputs["router_probs"].detach())
            losses.append(float(loss_dict["loss"].detach()))

        assert torch.isfinite(torch.tensor(losses)).all()
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
        assert model.router.net[-1].in_features == 32 + model.num_patterns

        masked_logits = torch.zeros(1, 5)
        masked_targets = torch.zeros(1, 5)
        availability = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])
        base_pattern_loss = pattern_supervision_loss(masked_logits, masked_targets, availability)
        masked_logits[:, 3] = 100.0
        changed_pattern_loss = pattern_supervision_loss(masked_logits, masked_targets, availability)
        assert torch.allclose(base_pattern_loss, changed_pattern_loss)

        sequence = torch.arange(5.0).view(1, 5, 1)
        shifted = shift_temporal_features(sequence, 2)
        assert shifted.flatten().tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]

        replay = ReplayBuffer(capacity=3, seed=7)
        for index in range(10):
            replay.add_batch({"labels": torch.tensor([[float(index), 0.0]])})
        assert len(replay) == 3 and replay.seen == 10

        old_experts = model.num_experts
        model.add_expert()
        assert model.num_experts == old_experts + 1
        outputs = model(video, audio, valid_lengths=valid_lengths)
        assert outputs["router_probs"].shape[-1] == model.num_experts
        assert outputs["pattern_probs"].shape[-1] == model.num_patterns
        assert outputs["unknown_components"].shape == (video.shape[0], 4)
        positive_logits = torch.full_like(outputs["fake_logits"], 3.0)
        negative_logits = -positive_logits
        positive_energy = model.unknown_score_components(
            outputs["features"],
            outputs["confidence_weights"],
            positive_logits,
            outputs["expert_density"],
        )[:, 1]
        negative_energy = model.unknown_score_components(
            outputs["features"],
            outputs["confidence_weights"],
            negative_logits,
            outputs["expert_density"],
        )[:, 1]
        assert torch.allclose(positive_energy, negative_energy, atol=1e-6)
        pattern_to_router_grad = torch.autograd.grad(
            outputs["router_logits"].sum(),
            outputs["pattern_logits"],
            retain_graph=True,
        )[0]
        assert pattern_to_router_grad.abs().sum() > 0

        assert model.prune_expert(model.num_experts - 1)
        assert model.num_experts == old_experts
        model.add_expert()
        model.add_expert()
        assert model.num_experts == old_experts + 2
        assert model.merge_experts(0, model.num_experts - 1)
        assert model.num_experts == old_experts + 1
        outputs = model(video, audio, valid_lengths=valid_lengths)
        assert outputs["router_probs"].shape[-1] == model.num_experts

        sparse_model = CAPEModel(
            max_length=12,
            d_model=32,
            nhead=4,
            d_hid=64,
            nlayers=1,
            num_experts=3,
            top_k=1,
            freeze_backbone=True,
            device="cpu",
        )
        with torch.no_grad():
            sparse_model.router.net[-1].weight.zero_()
            sparse_model.router.net[-1].bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
        call_counts = [0, 0, 0]
        hooks = []
        for idx, expert in enumerate(sparse_model.expert_bank.experts):
            hooks.append(
                expert.register_forward_hook(
                    lambda module, inputs, output, idx=idx: call_counts.__setitem__(
                        idx, call_counts[idx] + 1
                    )
                )
            )
        sparse_model.eval()
        with torch.no_grad():
            sparse_model(video, audio, valid_lengths=valid_lengths)
        for hook in hooks:
            hook.remove()
        assert call_counts == [1, 0, 0], f"inactive experts were evaluated: {call_counts}"

        detector = ConformalUnknownDetector()
        model.fit_unknown_normalizer(outputs["unknown_components"].detach())
        normalized_scores = model.aggregate_unknown_components(outputs["unknown_components"].detach())
        assert torch.isfinite(normalized_scores).all()
        assert normalized_scores.mean().abs() < 1e-5
        detector.fit(normalized_scores)
        p_values = detector.p_value(normalized_scores)
        assert p_values.shape == outputs["unknown_score"].shape

        queue = UnknownQueue(maxlen=16)
        for index, feature in enumerate(
            (
                torch.tensor([1.0, 0.00]),
                torch.tensor([1.0, 0.02]),
                torch.tensor([0.99, -0.01]),
                torch.tensor([1.0, 0.01]),
                torch.tensor([-1.0, 0.0]),
            )
        ):
            queue.push(str(index), feature, score=float(index), meta={"task_id": "synthetic"})
        clusters = queue.cluster_candidates(min_cluster_size=3, eps=0.1)
        assert len(clusters) == 1 and len(clusters[0]) == 4
        print("CAPE smoke test passed")
        print(
            {
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "num_experts": model.num_experts,
                "sparse_calls": call_counts,
            }
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
