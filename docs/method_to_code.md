# From the paper to the code

This file maps each part of the paper that this package covers to its modules.

## 1. Algorithm 1

| Step | Paper | Code |
|---|---|---|
| Lines 1-4 | mean of the labeled rows, SVD, w = min(n - 1, 256), scores z = V_w^T (x - mu) of labeled and query rows (Eq. (1)) | `rofe.pca.support_pca` and `rofe.pca.pca_width` |
| Lines 5-7 | R_k from the QR decomposition of a Gaussian matrix, signs of diag(B) fixed | `rofe.rotation.haar_rotation` |
| Line 8 | rotated scores Z R_k followed by the table columns | `rofe.rotation.rotate`, `rofe.method.pfn_frame` |
| Line 9 | one PFN pass per rotation, the labeled rows as context | `rofe.pfn.tabicl.TabICL.fit_predict`; TabPFN through `rofe.pfn.tabpfn.write_export`, `rofe.pfn.tabpfn_predict`, `rofe.pfn.tabpfn.score_exports` |
| Line 11 | average of the K probability vectors | `rofe.method.RoFE.predict_proba`, `rofe.experiments.interface.run` (method `rofe`) |
| Seeds | rotations drawn once per labeled set and shared by every query row | `rofe.rotation.rotation_seeds` via `rofe.method.labeled_scores` |
| Several images per row | views concatenated before one PCA and one rotation | `rofe.features.bank.FeatureBank.rows`, `rofe.tasks.Fold.embedding` |
| Several embeddings | one PCA and one set of rotations per embedding | `rofe.method.RoFE` (a list of embedding blocks) |
| CoMET | the same PCA and PFN without rotation | `RoFE(..., rotate=False)`, method `comet` in `rofe.experiments.interface` |

## 2. Section 4 (Method) and Appendix B

| Paper | Code |
|---|---|
| Hadamard transform of the scores padded to m = 2^ceil(log2 w) columns | `rofe.mechanism.hadamard_basis` |
| Givens rotations to equal column variances | `rofe.mechanism.givens_equal_variance`, `rofe.mechanism.givens_basis` |
| Rotations within the strong or within the weak directions only, R = diag(R_A, I) or diag(I, R_B) | `rofe.mechanism.partial_rotation`; the split at the Gavish-Donoho rank `rofe.pca.gavish_donoho_rank` |

## 3. Section 5 (Experiments) and Appendices C and D

| Paper | Code |
|---|---|
| Labeled rows: 10, 25, 50, 100, 250, 500 per class, three draws, and the whole pool | `rofe.draws.draws`, `rofe.draws.subsample`, `rofe.draws.many_draws` |
| Ten classes per fold for tasks with more | `rofe.data.classes.class_draw` |
| Metrics: AUROC (one-vs-rest macro average), accuracy, log-loss, ECE | `rofe.metrics.scores` |
| Task mean over folds and draws, mean over tasks, 95% bootstrap interval over tasks | `scripts/tables_common.py` (`task_mean`, `level`, `diff`, `stat`), used by `scripts/make_tables.py` |
| Linear probe, kNN, linear SVM, MLP | `rofe.heads.make_head`, `rofe.experiments.heads` |
| Per-column standardization against one global scale (scaling table) | `rofe.scaling`, `rofe.experiments.scaling` |
| Change of basis (figure and appendix table) | `rofe.experiments.basis` |
| TabPFN v2, 2.5 and 3.5 | `rofe.pfn.tabpfn`, `rofe.pfn.tabpfn_predict` |
| CatBoost, MultiModalPFN, PCA of Zhang et al. | `rofe.baselines.catboost`, `rofe.baselines.mmpfn`, `rofe.baselines.mmpfn_run`, `rofe.baselines.zhang` |
