import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HEADS = ("logreg_cv", "knn", "linear_svm", "mlp")
PROBE_CS = (0.01, 0.1, 1.0, 10.0)  # inverse L2 strengths of the linear probe
SVM_MAX_ITER = 1_000_000  # libsvm iteration cap; sklearn's default -1 has none


def make_head(name: str, rows: int, seed: int) -> Pipeline:
    if name == "logreg_cv":
        # linear probe, C chosen by 3-fold cross-validated log-loss
        model = LogisticRegressionCV(Cs=list(PROBE_CS), cv=3, max_iter=2000,
                                     scoring="neg_log_loss", random_state=seed)
    elif name == "knn":
        model = KNeighborsClassifier(n_neighbors=min(5, rows), weights="distance",
                                     metric="cosine")
    elif name == "linear_svm":
        # probabilities by Platt scaling
        model = SVC(kernel="linear", probability=True, random_state=seed, max_iter=SVM_MAX_ITER)
    elif name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"Unknown head {name}")
    return make_pipeline(StandardScaler(), model)


def head_inputs(embedding: np.ndarray, dense_table: np.ndarray | None) -> np.ndarray:
    x = np.asarray(embedding, np.float64)
    return x if dense_table is None else np.hstack([dense_table, x])  # [table, raw embedding]


def svm_status(svc: SVC) -> dict:
    if svc.fit_status_ not in (0, 1):
        raise ValueError(f"Unexpected libsvm fit status {svc.fit_status_}")
    most = int(np.max(svc.n_iter_))  # n_iter_: one entry per final one-vs-one model
    return {"svm_capped": bool(svc.fit_status_ == 1),  # some libsvm solve stopped at the cap
            "svm_final_capped": bool(svc.max_iter != -1 and most >= svc.max_iter),
            "svm_max_n_iter": most}
