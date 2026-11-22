import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def padded(classes: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    full = np.zeros((len(proba), len(labels)))  # (n_q, c), labels sorted
    # a class absent from the fit keeps exactly 0
    full[:, np.searchsorted(labels, classes)] = np.maximum(proba, 1e-300)
    return full / full.sum(1, keepdims=True)


# ECE = sum_b (|B_b| / n_q) |acc(B_b) - conf(B_b)| over equal-width bins of the top probability
def calibration_error(proba: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    confidence = proba.max(1)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def scores(classes: np.ndarray, proba: np.ndarray, y_query: np.ndarray,
           labels: np.ndarray) -> dict[str, float]:
    full = np.zeros((len(y_query), len(labels)))  # (n_q, c)
    full[:, np.searchsorted(labels, classes)] = proba
    full = full / full.sum(1, keepdims=True)
    correct = labels[full.argmax(1)] == y_query
    if not set(labels) <= set(y_query):
        auroc = float("nan")  # a class missing from the query
    elif len(labels) == 2:
        auroc = float(roc_auc_score(y_query == labels[1], full[:, 1]))  # positive: labels[1]
    else:
        # one-vs-rest macro AUROC
        auroc = float(roc_auc_score(y_query, full, multi_class="ovr", labels=labels))
    return {"accuracy": float(correct.mean()),
            "auroc": auroc,
            "log_loss": float(log_loss(y_query, np.clip(full, 1e-12, 1), labels=labels)),
            "ece": calibration_error(full, correct)}
