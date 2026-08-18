import copy

import torch
from torch import nn


class PatternExpert(nn.Module):
    """Small residual bottleneck expert for one discovered forgery pattern."""

    def __init__(self, d_model, bottleneck=64, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )

    def forward(self, x):
        return x + self.adapter(x)


class PatternExpertBank(nn.Module):
    """Dynamic expert bank with freeze, add, merge, prune, and usage stats."""

    def __init__(self, d_model, num_experts=1, bottleneck=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.experts = nn.ModuleList(
            [PatternExpert(d_model, bottleneck=bottleneck, dropout=dropout) for _ in range(num_experts)]
        )
        self.register_buffer("usage", torch.zeros(num_experts))

    @property
    def num_experts(self):
        return len(self.experts)

    def add_expert(self, init_from=None):
        if init_from is None:
            expert = PatternExpert(self.d_model, bottleneck=self.bottleneck, dropout=self.dropout)
        else:
            expert = copy.deepcopy(self.experts[init_from])
        device = self.usage.device
        dtype = self.usage.dtype
        if self.num_experts > 0:
            ref = next(self.experts[0].parameters(), None)
            if ref is not None:
                device = ref.device
                dtype = ref.dtype
        expert = expert.to(device=device, dtype=dtype)
        self.experts.append(expert)
        self.usage = torch.cat([self.usage, self.usage.new_zeros(1)])
        return self.num_experts - 1

    def freeze_old_experts(self, keep_last_trainable=True):
        last_idx = self.num_experts - 1
        for idx, expert in enumerate(self.experts):
            trainable = (idx == last_idx) if keep_last_trainable else False
            for param in expert.parameters():
                param.requires_grad = trainable

    def forward(self, x, router_probs):
        if router_probs.shape[-1] != self.num_experts:
            raise ValueError(f"router has {router_probs.shape[-1]} experts, bank has {self.num_experts}")
        output = x.new_zeros(x.shape)
        for idx, expert in enumerate(self.experts):
            active_idx = torch.nonzero(router_probs[:, idx] > 0, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                continue
            active_x = x.index_select(0, active_idx)
            active_weights = router_probs.index_select(0, active_idx)[:, idx].unsqueeze(-1)
            contribution = expert(active_x) * active_weights
            output = output.index_add(0, active_idx, contribution)
        with torch.no_grad():
            self.usage += router_probs.detach().sum(dim=0).to(self.usage.device)
        return output

    def expert_usage_stats(self):
        total = self.usage.sum().clamp_min(1.0)
        return (self.usage / total).detach().cpu()

    def merge_experts(self, keep_idx, drop_idx):
        if keep_idx == drop_idx:
            return
        keep_state = self.experts[keep_idx].state_dict()
        drop_state = self.experts[drop_idx].state_dict()
        merged = {key: 0.5 * keep_state[key] + 0.5 * drop_state[key] for key in keep_state}
        self.experts[keep_idx].load_state_dict(merged)
        self.usage[keep_idx] += self.usage[drop_idx]
        self.prune_expert(drop_idx)

    def prune_expert(self, idx):
        if self.num_experts <= 1:
            return
        self.experts = nn.ModuleList([expert for i, expert in enumerate(self.experts) if i != idx])
        keep = [i for i in range(self.usage.numel()) if i != idx]
        self.usage = self.usage[keep]

    def prune_inactive(self, min_usage=0.01):
        stats = self.expert_usage_stats()
        for idx in reversed(range(self.num_experts)):
            if self.num_experts > 1 and stats[idx].item() < min_usage:
                self.prune_expert(idx)
