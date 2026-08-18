import torch
from torch import nn
import torch.nn.functional as F


class DiscrepancyEncoder(nn.Module):
    """Encode mask-aware discrepancy statistics in a learned common space."""

    def __init__(self, input_dim=768, d_model=256, dropout=0.1):
        super().__init__()
        self.video_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _masked_moments(features, valid_lengths=None):
        if valid_lengths is None:
            return features.mean(dim=1), features.var(dim=1, unbiased=False)

        lengths = torch.as_tensor(valid_lengths, device=features.device).long().flatten()
        if lengths.numel() != features.shape[0]:
            raise ValueError(
                f"valid_lengths has {lengths.numel()} entries for a batch of {features.shape[0]}"
            )
        lengths = lengths.clamp(min=0, max=features.shape[1])
        mask = torch.arange(features.shape[1], device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
        weights = mask.to(dtype=features.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        mean = (features * weights).sum(dim=1) / denom
        variance = ((features - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1) / denom
        return mean, variance

    def forward(self, video_features, audio_features, valid_lengths=None):
        if video_features.shape[:2] != audio_features.shape[:2]:
            raise ValueError(
                "video_features and audio_features must share batch and temporal dimensions"
            )
        video_aligned = self.video_projection(video_features)
        audio_aligned = self.audio_projection(audio_features)
        diff = video_aligned - audio_aligned
        abs_diff = diff.abs()
        video_mean, _ = self._masked_moments(video_aligned, valid_lengths)
        audio_mean, _ = self._masked_moments(audio_aligned, valid_lengths)
        diff_mean, diff_variance = self._masked_moments(abs_diff, valid_lengths)
        stats = torch.cat(
            [
                video_mean,
                audio_mean,
                diff_mean,
                diff_variance,
            ],
            dim=-1,
        )
        return self.net(stats)


class DiscrepancyRouter(nn.Module):
    def __init__(self, d_model, num_experts, num_patterns=5, top_k=2, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_patterns = num_patterns
        input_dim = d_model + num_patterns
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_experts),
        )

    def expand(self, num_experts):
        if num_experts <= self.num_experts:
            return
        old = self.net[-1]
        new = nn.Linear(old.in_features, num_experts).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[: self.num_experts].copy_(old.weight)
            new.bias[: self.num_experts].copy_(old.bias)
            nn.init.xavier_uniform_(new.weight[self.num_experts :])
            new.bias[self.num_experts :].zero_()
        self.net[-1] = new
        self.num_experts = num_experts

    def prune(self, expert_idx):
        if self.num_experts <= 1:
            return False
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(f"expert_idx={expert_idx} is outside [0, {self.num_experts})")
        old = self.net[-1]
        keep = [idx for idx in range(self.num_experts) if idx != expert_idx]
        new = nn.Linear(old.in_features, self.num_experts - 1).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        with torch.no_grad():
            new.weight.copy_(old.weight[keep])
            new.bias.copy_(old.bias[keep])
        self.net[-1] = new
        self.num_experts -= 1
        return True

    def merge(self, keep_idx, drop_idx):
        if keep_idx == drop_idx:
            return False
        if not 0 <= keep_idx < self.num_experts or not 0 <= drop_idx < self.num_experts:
            raise IndexError("expert merge indices are outside the router output range")
        output = self.net[-1]
        with torch.no_grad():
            output.weight[keep_idx].copy_(0.5 * (output.weight[keep_idx] + output.weight[drop_idx]))
            output.bias[keep_idx].copy_(0.5 * (output.bias[keep_idx] + output.bias[drop_idx]))
        return self.prune(drop_idx)

    def forward(self, x, pattern_probs):
        if pattern_probs.shape[0] != x.shape[0] or pattern_probs.shape[-1] != self.num_patterns:
            raise ValueError(
                f"pattern_probs must have shape [batch, {self.num_patterns}], "
                f"got {tuple(pattern_probs.shape)}"
            )
        routing_input = torch.cat([x, pattern_probs], dim=-1)
        logits = self.net(routing_input)
        if self.top_k is None or self.top_k >= logits.shape[-1]:
            return F.softmax(logits, dim=-1), logits
        values, indexes = torch.topk(logits, k=self.top_k, dim=-1)
        sparse = torch.full_like(logits, float("-inf"))
        sparse.scatter_(dim=-1, index=indexes, src=values)
        return F.softmax(sparse, dim=-1), logits


def router_entropy(router_probs):
    probs = router_probs.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)
