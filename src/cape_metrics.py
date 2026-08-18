import numpy as np

try:
    from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score, roc_curve
except ModuleNotFoundError:  # pragma: no cover - lightweight smoke-test fallback
    adjusted_rand_score = None
    average_precision_score = None
    roc_auc_score = None
    roc_curve = None


def binary_scores_from_av_logits(logits):
    probs = logits.sigmoid().detach().cpu().numpy()
    return probs.max(axis=-1)


def fake_labels_from_av_targets(labels):
    return labels.detach().cpu().numpy().max(axis=-1)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(y_true, y_score))
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / total)


def safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if average_precision_score is not None:
        return float(average_precision_score(y_true, y_score))
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    denom = max(1, np.sum(y_sorted == 1))
    return float(np.sum(precision[y_sorted == 1]) / denom)


def forgetting_score(history):
    """Mean max-past minus final performance over tasks."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    final = values[-1]
    past_best = np.nanmax(values[:-1], axis=0)
    return float(np.nanmean(np.maximum(0.0, past_best - final)))


def unknown_detection_metrics(y_unknown, scores):
    return {
        "unknown_auc": safe_auc(y_unknown, scores),
        "unknown_ap": safe_ap(y_unknown, scores),
        "fpr95": fpr_at_tpr(y_unknown, scores, target_tpr=0.95),
    }


def fpr_at_tpr(y_true, y_score, target_tpr=0.95):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if roc_curve is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        valid = np.flatnonzero(tpr >= float(target_tpr))
        return float(fpr[valid[0]]) if valid.size else 1.0

    positives = max(1, int((y_true == 1).sum()))
    negatives = max(1, int((y_true == 0).sum()))
    order = np.argsort(-y_score, kind="stable")
    tp = fp = 0
    for index in order:
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
        if tp / positives >= float(target_tpr):
            return float(fp / negatives)
    return 1.0


def safe_ari(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        return float("nan")
    if adjusted_rand_score is None:
        return float("nan")
    return float(adjusted_rand_score(y_true, y_pred))


def jensen_shannon_divergence(p, q, eps=1e-8):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(midpoint)), axis=-1)
    kl_qm = np.sum(q * (np.log(q) - np.log(midpoint)), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def cosine_prototype_drift(old_prototypes, new_prototypes, active_mask=None, eps=1e-8):
    old = np.asarray(old_prototypes, dtype=float)
    new = np.asarray(new_prototypes, dtype=float)
    if old.shape != new.shape:
        raise ValueError(f"prototype shapes differ: {old.shape} versus {new.shape}")
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        old = old[active_mask]
        new = new[active_mask]
    if old.size == 0:
        return float("nan")
    old = old / np.clip(np.linalg.norm(old, axis=-1, keepdims=True), eps, None)
    new = new / np.clip(np.linalg.norm(new, axis=-1, keepdims=True), eps, None)
    return float(np.mean(1.0 - np.sum(old * new, axis=-1)))
