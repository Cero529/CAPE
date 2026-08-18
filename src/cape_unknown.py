from collections import deque
import math

import numpy as np
import torch


class ScoreNormalizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, scores):
        scores = torch.as_tensor(scores).float()
        self.mean = scores.mean()
        self.std = scores.std(unbiased=False).clamp_min(self.eps)

    def normalize(self, scores):
        if self.mean is None:
            return torch.as_tensor(scores).float()
        return (torch.as_tensor(scores).float() - self.mean) / self.std


class ConformalUnknownDetector:
    """Calibration-based unknown detector with no hand-tuned raw threshold."""

    def __init__(self):
        self.calibration_scores = None

    def fit(self, calibration_scores):
        scores = torch.as_tensor(calibration_scores).float().flatten()
        if scores.numel() == 0:
            raise ValueError("calibration_scores must be non-empty")
        self.calibration_scores = scores.detach().cpu()

    def p_value(self, score):
        if self.calibration_scores is None:
            raise RuntimeError("ConformalUnknownDetector.fit must be called first")
        score = torch.as_tensor(score).float().detach().cpu()
        flat = score.flatten()
        n = self.calibration_scores.numel()
        # Large score means more unknown-like. The upper-tail rank is the
        # conformal p-value for compatibility with the seen distribution.
        counts = (self.calibration_scores.unsqueeze(0) >= flat.unsqueeze(1)).sum(dim=1)
        return ((counts.float() + 1.0) / (n + 1.0)).reshape(score.shape)

    def is_unknown(self, score, alpha=0.05):
        return self.p_value(score) < alpha


class GaussianTailUnknownDetector:
    """Non-conformal ablation using a Gaussian upper-tail approximation."""

    def __init__(self, eps=1e-6):
        self.eps = float(eps)
        self.calibration_scores = None
        self.mean = None
        self.std = None

    def fit(self, calibration_scores):
        scores = torch.as_tensor(calibration_scores).float().flatten()
        if scores.numel() == 0:
            raise ValueError("calibration_scores must be non-empty")
        self.calibration_scores = scores.detach().cpu()
        self.mean = float(scores.mean())
        self.std = float(scores.std(unbiased=False).clamp_min(self.eps))

    def p_value(self, score):
        if self.mean is None:
            raise RuntimeError("GaussianTailUnknownDetector.fit must be called first")
        score = torch.as_tensor(score).float()
        z = (score - self.mean) / self.std
        values = [0.5 * math.erfc(float(value) / math.sqrt(2.0)) for value in z.detach().cpu().flatten()]
        return torch.tensor(values, dtype=torch.float32).reshape(score.shape)

    def is_unknown(self, score, alpha=0.05):
        return self.p_value(score) < alpha


class UnknownQueue:
    """Bounded FIFO queue; appending at capacity evicts the oldest item."""

    def __init__(self, maxlen=4096):
        self.items = deque(maxlen=maxlen)

    def push(self, sample_id, feature, score, meta=None):
        feature = torch.as_tensor(feature).detach().cpu().float()
        feature = feature / feature.norm().clamp_min(1e-8)
        self.items.append(
            {
                "sample_id": sample_id,
                "feature": feature,
                "score": float(torch.as_tensor(score).detach().cpu().item()),
                "meta": meta or {},
            }
        )

    def __len__(self):
        return len(self.items)

    def features(self):
        if not self.items:
            return torch.empty(0)
        return torch.stack([item["feature"].float() for item in self.items])

    def cluster_candidates(self, min_cluster_size=8, eps=0.8):
        if len(self.items) < min_cluster_size:
            return []
        labels = self.cluster_labels(min_cluster_size=min_cluster_size, eps=eps)
        clusters = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            idx = np.where(labels == label)[0].tolist()
            clusters.append([self.items[i] for i in idx])
        return clusters

    def cluster_labels(self, min_cluster_size=8, eps=0.8):
        if len(self.items) == 0:
            return np.empty((0,), dtype=int)
        if len(self.items) < min_cluster_size:
            return np.full((len(self.items),), -1, dtype=int)
        from sklearn.cluster import DBSCAN

        features = self.features().float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return DBSCAN(eps=eps, min_samples=min_cluster_size).fit_predict(features.numpy())
