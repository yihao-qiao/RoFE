# PCA protocol of Zhang et al.: per-column z-score, PCA on the labeled rows, TabPFN or a head
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from ..experiments.common import base_record, labeled_sets, settings, stream
from ..metrics import padded, scores
from ..pca import support_pca
from ..pfn.tabpfn import table_codes, write_export
from ..rotation import rotate
from ..table import encoded
from ..tasks import Fold

DIMS = (8, 16, 32, 48, 64, 96, "all")  # their grid of d
ROTATED = (8, 16, 32, 48, 64, 96)  # d with a four-rotation variant, not in their protocol
ROTATIONS = 4
SEED_STREAM = 9900
MAX_FEATURES = 2000  # their cap on d
EPS = 1e-8  # floor of the column std
HEADS = ("knn", "logreg", "linear_svm", "mlp", "random_forest", "xgboost")
SVM_MAX_ITER = 1_000_000  # libsvm cap; their heads run uncapped
ANCHOR = "support"  # the CoMET interface's labeled-row PCA, fitted without pretraining limits


# k = min(d, max(1, min(n - 1, p)), MAX_FEATURES), d = "all" without the first bound
def effective_dim(d: int | str, n: int, width: int) -> int:
    top = max(1, min(n - 1, width))
    k = top if d == "all" else min(int(d), top)
    return min(k, MAX_FEATURES)


def dims_for(n: int, width: int) -> dict[str, int]:
    return {str(d): effective_dim(d, n, width) for d in DIMS}


def zpca(x: np.ndarray, rows: np.ndarray, used: np.ndarray, k: int) -> np.ndarray:
    train = x[rows]
    mu = train.mean(axis=0, keepdims=True)  # (1, p)
    sigma = np.maximum(train.std(axis=0, keepdims=True), EPS)  # ddof 0
    pca = PCA(n_components=k, whiten=False, svd_solver="auto", random_state=0)
    pca.fit((train - mu) / sigma)  # z-score, then unwhitened PCA on the labeled rows
    return pca.transform((x[used] - mu) / sigma).astype(np.float32)  # (len(used), k)


def rotation_seeds(fold: int, shots: int, repeat: int, stream_id: int, k: int) -> list[int]:
    rng = np.random.default_rng([SEED_STREAM, fold, shots, repeat, stream_id, k])
    return [int(s) for s in rng.integers(2 ** 31, size=ROTATIONS)]


def export(fold: Fold, cfg: dict, ctx, directory: Path) -> None:
    # cfg["anchor"]: one more fit on the CoMET interface's PCA (method comet), without
    # TabPFN's pretraining limits, as in the interface runs it pairs with
    codes, categorical = table_codes(fold.table, fold.categorical, fold.pool)
    for draw in labeled_sets(fold, cfg):
        used = np.concatenate([draw.rows, fold.query])
        for encoder in cfg["encoders"]:
            x = fold.embedding(encoder)
            mapping = dims_for(len(draw.rows), x.shape[1])
            rotated = {mapping[str(d)] for d in ROTATED}
            blocks = {"table": codes[used].astype(np.float32), "categorical": categorical}
            for k in sorted(set(mapping.values())):
                blocks[f"z{k}"] = zpca(x, draw.rows, used, k)
                if k in rotated:
                    seeds = rotation_seeds(fold.fold, draw.shots, draw.repeat, stream(encoder), k)
                    for r, seed in enumerate(seeds):
                        blocks[f"z{k}_rot{r}"] = rotate(blocks[f"z{k}"], seed)
            # (job, block, seed): one fit per block, random_state = fold
            jobs = [(name, name, fold.fold) for name in blocks
                    if name not in ("table", "categorical")]
            methods = {}
            for d in DIMS:
                k = mapping[str(d)]
                methods[f"zpca{d}"] = [f"z{k}"]
                if d in ROTATED:
                    methods[f"zpca{d}_rot_x4"] = [f"z{k}_rot{r}" for r in range(ROTATIONS)]
            meta = {"labels": fold.labels, "rows": draw.rows, "query": fold.query,
                    "y_rows": fold.y[draw.rows], "y_query": fold.y[fold.query],
                    "fold": fold.fold, "shots": draw.shots, "repeat": draw.repeat,
                    "encoder": encoder, "dims": json.dumps(mapping)}
            if cfg.get("anchor"):
                support = support_pca(x, draw.rows)[0]  # (N, min(n - 1, 256))
                blocks[ANCHOR] = support[used]
                jobs.append((ANCHOR, ANCHOR, fold.fold))
                methods["comet"] = [ANCHOR]
                meta |= {"support_width": int(support.shape[1]),
                         "unlimited_jobs": np.array([ANCHOR])}
            write_export(directory, f"f{fold.fold}_{encoder}_s{draw.shots}_r{draw.repeat}",
                         meta, blocks, jobs, methods)


def score(task: str, directory: Path, cfg: dict, ctx) -> Iterator[dict]:
    exports, predictions = Path(directory) / "exports", Path(directory) / "predictions"
    wanted = cfg.get("settings", ("embedding", "joint"))
    for path in sorted(exports.glob("*_meta.npz")):
        stem = path.name[:-len("_meta.npz")]
        with np.load(path) as m, np.load(predictions / f"{stem}.npz") as p:
            meta, saved = dict(m), dict(p)
        labels, mapping = meta["labels"], json.loads(str(meta["dims"]))
        for setting in (s for s in ("embedding", "joint") if s in wanted):
            def proba(job: str, setting: str = setting) -> np.ndarray:
                return padded(saved[f"{setting}_{job}_classes"],
                              saved[f"{setting}_{job}_proba"].astype(np.float64), labels)

            for method, jobs in zip(meta["method_names"], meta["method_jobs"]):
                method = str(method)
                if method == "comet":
                    d, used = ANCHOR, int(meta["support_width"])
                else:
                    d = method[len("zpca"):].removesuffix("_rot_x4")
                    used = int(mapping[d])
                # mean over the method's jobs: one fit, or the four rotations
                p = np.mean([proba(j) for j in str(jobs).split(",")], axis=0)  # (n_q, c)
                yield {"task": task, "fold": int(meta["fold"]), "shots": int(meta["shots"]),
                       "repeat": int(meta["repeat"]), "n_labeled": int(len(meta["rows"])),
                       "setting": setting, "encoder": str(meta["encoder"]), "method": method,
                       "d_req": d, "d_used": used, **scores(labels, p, meta["y_query"], labels)}


# their heads with the settings of their experiments, seed = fold
def make_head(name: str, shots: int, seed: int, threads: int):
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=min(5, shots), weights="distance",
                                    metric="cosine")
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "linear_svm":
        from sklearn.svm import SVC

        return SVC(kernel="linear", probability=True, random_state=seed, max_iter=SVM_MAX_ITER)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=seed)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=threads,
                                      random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=threads,
                             random_state=seed)
    raise ValueError(f"Unknown head {name}")


def svm_status(svc) -> dict:
    if svc.fit_status_ not in (0, 1):
        raise ValueError(f"Unexpected libsvm fit status {svc.fit_status_}")
    # svm_capped: some libsvm solve hit the cap, Platt fits included;
    # svm_final_capped: a final one-vs-one solve did, so predict is affected
    most = int(np.max(svc.n_iter_))
    return {"svm_capped": bool(svc.fit_status_ == 1),
            "svm_final_capped": bool(svc.max_iter != -1 and most >= svc.max_iter),
            "svm_max_n_iter": most}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def head_outputs(name: str, X: np.ndarray, rows: np.ndarray, query: np.ndarray, y: np.ndarray,
                 labels: np.ndarray, shots: int, seed: int, threads: int) -> list[tuple]:
    codes = np.searchsorted(labels, y[rows])
    if not np.array_equal(np.unique(codes), np.arange(len(labels))):
        raise ValueError("Every class must occur among the labeled rows")
    model = make_head(name, shots, seed, threads).fit(X[rows], codes)
    fitted = np.asarray(model.classes_, int)
    predicted = np.asarray(model.predict(X[query]), int)

    def full(proba: np.ndarray) -> np.ndarray:
        out = np.zeros((len(query), len(labels)))  # (n_q, c)
        out[:, fitted] = proba
        return out

    extra = svm_status(model) if name == "linear_svm" else {}
    outputs = [(name, full(np.asarray(model.predict_proba(X[query]), np.float64)), predicted,
                extra)]
    if name == "linear_svm":  # softmax of the decision function as a second record
        z = np.asarray(model.decision_function(X[query]), np.float64)
        if z.ndim == 1:
            z = np.stack([-z, z], axis=1)  # binary: (n_q,) -> (n_q, 2)
        outputs.append(("linear_svm_decision", full(softmax(z)), predicted, extra))
    return outputs


def run_heads(fold: Fold, cfg: dict, ctx) -> Iterator[dict]:
    heads = cfg.get("methods", HEADS)
    threads = getattr(ctx, "threads", 1)
    query, labels = fold.query, fold.labels
    for draw in labeled_sets(fold, cfg):
        used = np.concatenate([draw.rows, query])
        dense = (encoded(fold.table, fold.categorical, draw.rows, one_hot=True)
                 if fold.table.shape[1] else None)
        for encoder in cfg["encoders"]:
            x = fold.embedding(encoder)
            mapping = dims_for(len(draw.rows), x.shape[1])
            for k in sorted(set(mapping.values())):
                z = np.zeros((len(fold.y), k), np.float32)  # (N, k), filled on labeled + query
                z[used] = zpca(x, draw.rows, used, k)
                for setting in settings(fold, cfg):
                    X = z if setting == "embedding" else np.hstack([dense, z])
                    for head in heads:
                        for name, proba, predicted, extra in head_outputs(
                                head, X, draw.rows, query, fold.y, labels, draw.shots,
                                fold.fold, threads):
                            values = scores(labels, proba, fold.y[query], labels)
                            # accuracy from predict, as in their code; the argmax of the
                            # probabilities is kept as accuracy_proba
                            values["accuracy_proba"] = values["accuracy"]
                            values["accuracy"] = float((labels[predicted] == fold.y[query]).mean())
                            for d in (d for d in DIMS if mapping[str(d)] == k):
                                yield {**base_record(fold, draw, setting, encoder),
                                       "method": f"zpca{d}", "head": name, "d_req": str(d),
                                       "d_used": k, **values, **extra}
