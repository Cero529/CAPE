import torch
from torch import nn
import torch.nn.functional as F

from src.cape_experts import PatternExpertBank
from src.cape_logic import build_weak_pattern_targets, logic_consistency_loss, pattern_supervision_loss
from src.cape_router import DiscrepancyEncoder, DiscrepancyRouter, router_entropy
from src.models import AVDeepFakeDetector


class PrecomputedAVHubertBackbone(nn.Module):
    """Consume a pooled vector or pool a frozen AV-HuBERT final sequence.

    CAPE trains only on frozen features. The extraction job must therefore
    store ``h_bb`` or its final multimodal sequence as ``backbone_features``
    in each NPZ file; silently substituting a different backbone is prohibited.
    """

    def __init__(self, output_dim=768):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, backbone_features, valid_lengths=None):
        if backbone_features is None:
            raise ValueError(
                "backbone_mode='precomputed_avhubert' requires NPZ key "
                "'backbone_features' produced by the frozen AV-HuBERT extractor"
            )
        if backbone_features.ndim == 2:
            pooled = backbone_features
        elif backbone_features.ndim == 3:
            if valid_lengths is None:
                pooled = backbone_features.mean(dim=1)
            else:
                lengths = torch.as_tensor(valid_lengths, device=backbone_features.device).long().flatten()
                lengths = lengths.clamp(min=1, max=backbone_features.shape[1])
                mask = (
                    torch.arange(backbone_features.shape[1], device=backbone_features.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                weights = mask.to(backbone_features.dtype).unsqueeze(-1)
                pooled = (backbone_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(
                "backbone_features must have shape [B, D] or [B, T, D], "
                f"got {tuple(backbone_features.shape)}"
            )
        if pooled.shape[-1] != self.output_dim:
            raise ValueError(
                f"AV-HuBERT backbone dimension must be {self.output_dim}, got {pooled.shape[-1]}"
            )
        return pooled


class CAPEModel(nn.Module):
    """CAPE wrapper around the existing audio-visual detector backbone."""

    def __init__(
        self,
        max_length,
        d_model=256,
        nhead=4,
        d_hid=512,
        nlayers=1,
        num_experts=1,
        num_patterns=5,
        expert_bottleneck=64,
        top_k=2,
        dropout=0.1,
        freeze_backbone=True,
        backbone_ckpt=None,
        use_confidence_density=True,
        bandwidth_init=0.35,
        bandwidth_momentum=0.9,
        prototype_momentum=0.95,
        min_bandwidth=0.05,
        max_bandwidth=1.25,
        use_discrepancy=True,
        use_pattern_guidance=True,
        unknown_component_indices=None,
        backbone_mode="legacy_dimodif",
        input_dim=768,
        device="cpu",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_patterns = num_patterns
        self.use_confidence_density = use_confidence_density
        self.use_discrepancy = bool(use_discrepancy)
        self.use_pattern_guidance = bool(use_pattern_guidance)
        self.unknown_component_indices = tuple(
            range(4) if unknown_component_indices is None else unknown_component_indices
        )
        if not self.unknown_component_indices or any(
            index not in range(4) for index in self.unknown_component_indices
        ):
            raise ValueError("unknown_component_indices must select one or more values from 0..3")
        self.backbone_mode = str(backbone_mode)
        self.bandwidth_momentum = bandwidth_momentum
        self.bandwidth_init = bandwidth_init
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
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
        elif self.backbone_mode == "precomputed_avhubert":
            if int(d_model) != 768 or int(input_dim) != 768:
                raise ValueError("The paper AV-HuBERT profile requires input_dim=d_model=768")
            self.backbone = PrecomputedAVHubertBackbone(output_dim=d_model)
        else:
            raise ValueError("backbone_mode must be 'legacy_dimodif' or 'precomputed_avhubert'")
        self.backbone_load_report = None
        if backbone_ckpt and self.backbone_mode == "legacy_dimodif":
            self.backbone_load_report = self.load_backbone_checkpoint(backbone_ckpt, map_location=device)
            if freeze_backbone and self.backbone_load_report["coverage"] < 0.9:
                raise RuntimeError(
                    "Refusing to freeze a mostly unmatched backbone checkpoint: "
                    f"coverage={self.backbone_load_report['coverage']:.3f}"
                )
        elif backbone_ckpt:
            raise ValueError(
                "precomputed_avhubert consumes frozen features and does not load a detector checkpoint"
            )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.discrepancy_encoder = DiscrepancyEncoder(input_dim=input_dim, d_model=d_model, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pattern_head = nn.Linear(d_model, num_patterns)
        self.router = DiscrepancyRouter(
            d_model=d_model,
            num_experts=num_experts,
            num_patterns=num_patterns,
            top_k=top_k,
            dropout=dropout,
        )
        self.expert_bank = PatternExpertBank(
            d_model=d_model,
            num_experts=num_experts,
            bottleneck=expert_bottleneck,
            dropout=dropout,
        )
        self.det_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.prototype_momentum = prototype_momentum
        self.register_buffer("expert_prototypes", torch.zeros(num_experts, d_model))
        self.register_buffer("prototype_counts", torch.zeros(num_experts))
        self.register_buffer("expert_bandwidths", torch.full((num_experts,), float(bandwidth_init)))
        self.register_buffer("unknown_component_mean", torch.zeros(4))
        self.register_buffer("unknown_component_std", torch.ones(4))
        self.register_buffer("unknown_normalizer_fitted", torch.tensor(False, dtype=torch.bool))

    @property
    def num_experts(self):
        return self.expert_bank.num_experts

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
            "expected": len(current),
            "coverage": len(matched) / max(1, len(current)),
            "skipped": len(skipped),
            "skipped_examples": skipped[:10],
        }

    def add_expert(self, init_from=None):
        idx = self.expert_bank.add_expert(init_from=init_from)
        self.router.expand(self.expert_bank.num_experts)
        self.expert_prototypes = torch.cat(
            [self.expert_prototypes, self.expert_prototypes.new_zeros(1, self.expert_prototypes.shape[-1])],
            dim=0,
        )
        self.prototype_counts = torch.cat([self.prototype_counts, self.prototype_counts.new_zeros(1)])
        self.expert_bandwidths = torch.cat(
            [self.expert_bandwidths, self.expert_bandwidths.new_full((1,), float(self.bandwidth_init))]
        )
        return idx

    def freeze_old_experts(self, keep_last_trainable=True):
        self.expert_bank.freeze_old_experts(keep_last_trainable=keep_last_trainable)

    def prune_expert(self, expert_idx):
        if self.num_experts <= 1:
            return False
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(f"expert_idx={expert_idx} is outside [0, {self.num_experts})")
        self.expert_bank.prune_expert(expert_idx)
        self.router.prune(expert_idx)
        keep = [idx for idx in range(self.prototype_counts.numel()) if idx != expert_idx]
        self.expert_prototypes = self.expert_prototypes[keep]
        self.prototype_counts = self.prototype_counts[keep]
        self.expert_bandwidths = self.expert_bandwidths[keep]
        return True

    def merge_experts(self, keep_idx, drop_idx):
        if keep_idx == drop_idx:
            return False
        if not 0 <= keep_idx < self.num_experts or not 0 <= drop_idx < self.num_experts:
            raise IndexError("expert merge indices are outside the expert bank")
        keep_count = self.prototype_counts[keep_idx]
        drop_count = self.prototype_counts[drop_idx]
        total_count = keep_count + drop_count
        if total_count > 0:
            merged_prototype = (
                keep_count * self.expert_prototypes[keep_idx]
                + drop_count * self.expert_prototypes[drop_idx]
            ) / total_count
            merged_bandwidth = (
                keep_count * self.expert_bandwidths[keep_idx]
                + drop_count * self.expert_bandwidths[drop_idx]
            ) / total_count
        else:
            merged_prototype = 0.5 * (
                self.expert_prototypes[keep_idx] + self.expert_prototypes[drop_idx]
            )
            merged_bandwidth = 0.5 * (
                self.expert_bandwidths[keep_idx] + self.expert_bandwidths[drop_idx]
            )
        self.expert_prototypes[keep_idx] = merged_prototype
        self.prototype_counts[keep_idx] = total_count
        self.expert_bandwidths[keep_idx] = merged_bandwidth
        self.expert_bank.merge_experts(keep_idx, drop_idx)
        self.router.merge(keep_idx, drop_idx)
        keep = [idx for idx in range(self.prototype_counts.numel()) if idx != drop_idx]
        self.expert_prototypes = self.expert_prototypes[keep]
        self.prototype_counts = self.prototype_counts[keep]
        self.expert_bandwidths = self.expert_bandwidths[keep]
        return True

    def prune_inactive_experts(self, min_usage=0.01):
        stats = self.expert_bank.expert_usage_stats()
        pruned = []
        for idx in reversed(range(self.num_experts)):
            if self.num_experts > 1 and stats[idx].item() < min_usage:
                self.prune_expert(idx)
                pruned.append(idx)
        return list(reversed(pruned))

    def forward(self, video_features, audio_features, valid_lengths=None, backbone_features=None):
        if self.backbone_mode == "legacy_dimodif":
            backbone_logits, backbone_repr = self.backbone([video_features, audio_features])
        else:
            backbone_logits = None
            backbone_repr = self.backbone(backbone_features, valid_lengths=valid_lengths)
        if self.use_discrepancy:
            discrepancy = self.discrepancy_encoder(
                video_features, audio_features, valid_lengths=valid_lengths
            )
        else:
            discrepancy = backbone_repr.new_zeros(backbone_repr.shape[0], self.d_model)
        features = self.fusion(torch.cat([backbone_repr, discrepancy], dim=-1))
        pattern_logits = self.pattern_head(features)
        pattern_probs = torch.sigmoid(pattern_logits)
        routing_patterns = pattern_probs if self.use_pattern_guidance else torch.zeros_like(pattern_probs)
        router_probs, router_logits = self.router(features, routing_patterns)
        expert_density = self.expert_confidence_density(features)
        confidence_weights = self.combine_router_and_density(router_probs, expert_density)
        expert_features = self.expert_bank(features, confidence_weights)
        fake_logits = self.det_head(expert_features)
        unknown_components = self.unknown_score_components(
            features, confidence_weights, fake_logits, expert_density
        )
        unknown_score = self.aggregate_unknown_components(unknown_components)
        return {
            "fake_logits": fake_logits,
            "backbone_logits": backbone_logits,
            "pattern_logits": pattern_logits,
            "pattern_probs": pattern_probs,
            "router_probs": router_probs,
            "router_logits": router_logits,
            "expert_density": expert_density,
            "confidence_weights": confidence_weights,
            "features": features,
            "unknown_components": unknown_components,
            "unknown_score": unknown_score,
        }

    def combine_router_and_density(self, router_probs, expert_density):
        if not self.use_confidence_density:
            return router_probs
        weights = router_probs * expert_density.clamp_min(1e-8)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def expert_confidence_density(self, features):
        if self.prototype_counts.sum() <= 0:
            return features.new_ones(features.shape[0], self.num_experts)
        prototypes = F.normalize(self.expert_prototypes, dim=-1)
        feat = F.normalize(features, dim=-1)
        distances = 1.0 - torch.matmul(feat, prototypes.T)
        bandwidths = self.expert_bandwidths.clamp(self.min_bandwidth, self.max_bandwidth)
        density = torch.exp(-(distances.pow(2)) / (2.0 * bandwidths.unsqueeze(0).pow(2)))
        # Uninitialized experts should still be reachable through the router
        # when a new task starts, so use a neutral density for them.
        inactive = (self.prototype_counts <= 0).unsqueeze(0)
        return torch.where(inactive, torch.ones_like(density), density)

    def unknown_score_components(self, features, router_probs, fake_logits, expert_density=None):
        entropy = router_entropy(router_probs)
        # Centered Bernoulli energy is high near the decision boundary and low
        # for confident real or fake predictions. Applying standard multiclass
        # energy directly to two independent fake logits would incorrectly make
        # confident-real samples appear unknown.
        binary_logits = torch.stack((-0.5 * fake_logits, 0.5 * fake_logits), dim=-1)
        energy = -torch.logsumexp(binary_logits, dim=-1).mean(dim=-1)
        if self.prototype_counts.sum() > 0:
            active = self.prototype_counts > 0
            prototypes = F.normalize(self.expert_prototypes[active], dim=-1)
            feat = F.normalize(features, dim=-1)
            distance = 1.0 - torch.matmul(feat, prototypes.T).max(dim=-1).values
            if expert_density is None:
                expert_density = self.expert_confidence_density(features)
            modal_density = expert_density[:, active].max(dim=-1).values.clamp_min(1e-8)
            modal_surprise = -modal_density.log()
        else:
            distance = torch.zeros_like(entropy)
            modal_surprise = torch.zeros_like(entropy)
        return torch.stack((entropy, energy, distance, modal_surprise), dim=-1)

    def aggregate_unknown_components(self, components):
        components = torch.as_tensor(components).float()
        if bool(self.unknown_normalizer_fitted.item()):
            mean = self.unknown_component_mean.to(components.device)
            std = self.unknown_component_std.to(components.device)
            components = (components - mean) / std.clamp_min(1e-6)
        selected = components[..., list(self.unknown_component_indices)]
        return selected.sum(dim=-1)

    def unknown_score(self, features, router_probs, fake_logits, expert_density=None):
        components = self.unknown_score_components(features, router_probs, fake_logits, expert_density)
        return self.aggregate_unknown_components(components)

    @torch.no_grad()
    def fit_unknown_normalizer(self, components):
        components = torch.as_tensor(components).float()
        if components.ndim != 2 or components.shape[-1] != 4 or components.shape[0] == 0:
            raise ValueError("unknown components must have shape [N, 4] with N > 0")
        mean = components.mean(dim=0).to(self.unknown_component_mean.device)
        std = components.std(dim=0, unbiased=False).clamp_min(1e-6).to(self.unknown_component_std.device)
        self.unknown_component_mean.copy_(mean)
        self.unknown_component_std.copy_(std)
        self.unknown_normalizer_fitted.fill_(True)

    @torch.no_grad()
    def update_prototypes(self, features, router_probs):
        for idx in range(self.num_experts):
            weights = router_probs[:, idx].detach().clamp_min(0.0)
            if weights.sum() <= 1e-8:
                continue
            if self.prototype_counts[idx] == 0:
                value = (features * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1e-8)
                self.expert_prototypes[idx] = value
            else:
                proto = self.expert_prototypes[idx].unsqueeze(0)
                distance = 1.0 - torch.sum(F.normalize(features, dim=-1) * F.normalize(proto, dim=-1), dim=-1)
                bandwidth = self.expert_bandwidths[idx].clamp(self.min_bandwidth, self.max_bandwidth)
                kernel = torch.exp(-(distance.pow(2)) / (2.0 * bandwidth.pow(2)))
                mode_weights = weights * kernel
                if mode_weights.sum() <= 1e-8:
                    continue
                value = (features * mode_weights.unsqueeze(-1)).sum(dim=0) / mode_weights.sum().clamp_min(1e-8)
                self.expert_prototypes[idx] = (
                    self.prototype_momentum * self.expert_prototypes[idx]
                    + (1.0 - self.prototype_momentum) * value
                )
                bw_value = torch.sqrt((mode_weights * distance.pow(2)).sum() / mode_weights.sum().clamp_min(1e-8))
                self.expert_bandwidths[idx] = (
                    self.bandwidth_momentum * self.expert_bandwidths[idx]
                    + (1.0 - self.bandwidth_momentum) * bw_value.clamp(self.min_bandwidth, self.max_bandwidth)
                )
            self.prototype_counts[idx] += weights.sum()


class CAPELoss(nn.Module):
    def __init__(self, lambda_pattern=0.5, lambda_logic=0.2, lambda_distill=1.0, lambda_router=0.01):
        super().__init__()
        self.lambda_pattern = lambda_pattern
        self.lambda_logic = lambda_logic
        self.lambda_distill = lambda_distill
        self.lambda_router = lambda_router
        self.det_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs,
        labels,
        old_outputs=None,
        has_fake_period=None,
        unknown_flag=None,
        pattern_availability=None,
    ):
        fake_logits = outputs["fake_logits"]
        det = self.det_loss(fake_logits, labels.float())
        weak_patterns = build_weak_pattern_targets(
            labels[:, 0],
            labels[:, 1],
            has_fake_period=has_fake_period,
            unknown_flag=unknown_flag,
        )
        pattern = pattern_supervision_loss(
            outputs["pattern_logits"],
            weak_patterns,
            availability=pattern_availability,
        )
        logic = logic_consistency_loss(fake_logits, outputs["pattern_logits"], labels)

        distill = fake_logits.new_tensor(0.0)
        if old_outputs is not None:
            distill = F.mse_loss(torch.sigmoid(fake_logits), torch.sigmoid(old_outputs["fake_logits"]).detach())

        router_probs = outputs["router_probs"].clamp_min(1e-8)
        usage = router_probs.mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
        uniform = torch.full_like(usage, 1.0 / usage.numel())
        router = torch.sum(usage * (usage.log() - uniform.log()))

        total = (
            det
            + self.lambda_pattern * pattern
            + self.lambda_logic * logic
            + self.lambda_distill * distill
            + self.lambda_router * router
        )
        return {
            "loss": total,
            "det": det.detach(),
            "pattern": pattern.detach(),
            "logic": logic.detach(),
            "distill": distill.detach(),
            "router": router.detach(),
        }
