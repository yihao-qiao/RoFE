# Third-party code, weights and baseline settings

This file lists the pinned upstream code and weights, their licences, and the settings of each baseline. The run
commands of the baselines are in `docs/reproducing.md`.

## 1. PFNs

- **TabICLv2** (BSD-3-Clause). It is installed from soda-inria/tabicl at commit
  `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`. The checkpoint `tabicl-classifier-v2-20260212.ckpt` comes from the
  Hugging Face repository `jingang/TabICL` (SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`).
- **TabPFN 2.5 and 3.5**. They use `tabpfn` 9.2.0 with the classifiers `tabpfn-v2.5-classifier-v2.5_default.ckpt` and
  `tabpfn-v3.5-20260909.safetensors`. Their SHA-256 values are in `rofe/pfn/tabpfn_predict.py` and are checked before
  every prediction run. Their weights are under the TabPFN Non-Commercial Licence and must be obtained by the user,
  for example through the `tabpfn` package after accepting the licence. They run with the defaults of
  `create_default_for_version`, eight members.
- **TabPFN v2**. It is the copy shipped with MultiModalPFN, with four members. Place `tabpfn-v2-classifier.ckpt` from
  `Prior-Labs/TabPFN-v2-clf` at revision `f851f2a3c941544733b712d8c0f96dfae9b28862` in
  `MultiModalPFN/mmpfn/parameters/`.

`rofe.pfn.tabpfn_predict` runs all three with TabPFN's pretraining limits lifted (`ignore_pretraining_limits=True`).

## 2. Linear probe and other heads

The heads use scikit-learn. The probe standardizes the inputs. It selects the inverse regularization among 0.01, 0.1, 1
and 10 by three-fold cross-validation of the log-loss.

## 3. CatBoost

CatBoost 1.2.8 (Apache-2.0) runs with MulTaBench's defaults, in `envs/catboost`:

- 2000 iterations, early stopping after 50 rounds with `od_pval=0.001`, and `random_state=42`;
- native categorical columns and TabSTAR's validation split;
- each image view standardized and reduced by PCA to 30 components;
- text embeddings reduced by PCA without scaling.

## 4. MultiModalPFN

MultiModalPFN (Prior Labs License 1.1) is not included.

- It fine-tunes with the authors' settings: learning rate 1e-5, batch size 1, 100 steps, log-loss validation, frozen
  input, and four inference members.
- The head configuration of each task is read from `configs/mmpfn_selection.json`. It is the candidate that the
  authors' code selects from its 39-candidate grid: the best mean query accuracy over the five folds on the whole
  labeled pool.

## 5. PCA of Zhang et al.

This is a port of the protocol. Each column is z-scored, and a PCA is fitted on the labeled rows with d in {8, 16, 32,
48, 64, 96, all}. TabPFN 2.5 or 3.5 runs with its defaults. The heads are kNN, logistic regression, linear SVM, MLP,
random forest and XGBoost.

## 6. MulTaBench

MulTaBench has no licence file at the pinned commit. It is used through its official loader in the user's clone. No
code of it is included.

## 7. Encoders

- **DINOv3** (DINOv3 License, weights from Meta). The code is the `facebookresearch/dinov3` clone at the commit in
  `docs/data.md`.
- **SigLIP so400m** (Apache-2.0), **ELECTRA-base** (Apache-2.0) and **E5-base-v2** (MIT). Each comes from its pinned
  Hugging Face commit:
  - `google/siglip-so400m-patch14-384` `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` (`rofe/features/siglip.py`)
  - `google/electra-base-discriminator` `1ae76a97c7e84a4e640876a07453fccd636f0667` (`rofe/features/text.py`)
  - `intfloat/e5-base-v2` `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` (`rofe/features/text.py`)

## 8. Figure in the README

`assets/overview.png` shows labeled rows of the floral-bouquet task of MulTaBench (Kaggle, Apache 2.0).
