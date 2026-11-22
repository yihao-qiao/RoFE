# Reproducing the tables and figures

This file lists how each table and figure in this package is rebuilt. Prepare the data first (`docs/data.md`).

## 1. Environments

Some models need their own environment. Each one is defined in `envs/` with the versions used for the paper:

| Environment | Used for | Definition |
|---|---|---|
| main | RoFE, CoMET, TabICLv2, heads, Zhang et al. heads, tables | `pyproject.toml` |
| `envs/tabpfn` | TabPFN 2.5 and 3.5 predictions | `envs/tabpfn/pyproject.toml` |
| `envs/mmpfn` | TabPFN v2 predictions and MultiModalPFN (Python 3.10.16) | `envs/mmpfn/requirements.txt` |
| `envs/catboost` | CatBoost | `envs/catboost/pyproject.toml` |
| `envs/multabench` | MulTaBench data preparation | `envs/multabench/pyproject.toml` |
| `envs/dinov3` | DINOv3 embeddings | `envs/dinov3/pyproject.toml` |
| `envs/features` | SigLIP, ELECTRA and E5 embeddings | `envs/features/pyproject.toml` |

Each of them is created from the package root, and this package is then installed into it without dependencies:

```bash
uv sync --project envs/tabpfn                # the same for catboost, multabench, dinov3 and features
source envs/tabpfn/.venv/bin/activate
uv pip install --no-deps -e .

uv venv --python 3.10.16 envs/mmpfn/.venv
source envs/mmpfn/.venv/bin/activate
uv pip install -r envs/mmpfn/requirements.txt
uv pip install --no-deps -e .
```

## 2. Runs and tables

Each configuration in `configs/` runs one experiment on all its tasks and writes `runs/<config>/<task>/records.jsonl`.
`--tasks` runs a subset, for example one task per job.

Every record holds the task, fold, level (`shots`, 0 for the whole pool) and draw (`repeat`). It also holds the number
of labeled rows, the setting (`embedding` or `joint`), the encoder, the method and the PFN. The metrics are accuracy,
AUROC, log-loss and ECE.

```bash
python scripts/run.py configs/<config>.yaml --data DATA --weights weights --out runs
python scripts/make_tables.py --runs runs --out tables --data DATA
```

`scripts/make_tables.py` averages over the folds and draws of each task and then over the tasks. It adds 95%
percentile bootstrap intervals over tasks (10,000 resamples) and writes `tables/`. It writes the main-text outputs by
default. `--appendix` adds the others, and `--only <name> ...` selects some.

## 3. Configurations, outputs and cost

| Table or figure | Paper | Configurations | Output | Rough cost |
|---|---|---|---|---|
| Tasks by source (task summary table) | App. C | prepared data | `task_summary.tex` | seconds |
| Main table, image tasks | Tab. 1 (left) | `table2_image`, `probe_image` | `main_image.tex` | 10-100 labels: about 0.5 GPU-h per task; 250, 500 and all: 1-4 GPU-h per task; probe: CPU, up to a day for the largest pools |
| Main table, text tasks | Tab. 1 (right) | `table3_text`, `probe_text` | `main_text.tex` | as for the image tasks |
| Gain for each encoder (figure) | App. D.1 | `table2_image`, `table3_text` | `fig3_gain_by_encoder.csv` | from the two main tables |
| Other PFNs (short and full tables) | Tab. 2; App. D.2 | `pfns_tabpfn_{v2,v25,v35}_{image,text}` and the two main tables | `pfns_short.tex`, `pfns_full.tex` | 0.3-1 GPU-h per task and PFN |
| Change of basis (figure and appendix table) | Fig. 4; App. D.8 | `fig4_basis` | `fig4_basis.csv`, `basis.tex` | 1-2 GPU-h per task |
| Per-column standardization and global scaling (table) | Tab. 3 | `table4_scaling_image`, `table4_scaling_text` | `scaling.tex` | 0.2-0.5 GPU-h per task |
| Gain by number of labeled rows (figure) | Fig. 5 | `table2_image`, `table3_text` | `fig5_fade.csv` | from the two main tables |
| Other heads | App. D.4 | `heads_image`, `heads_text` | `heads.tex` | CPU, under an hour per task |
| Trained baselines | App. D.5 | `trained_catboost_{image,text}`, `trained_mmpfn` | `trained.tex` | CatBoost: CPU, up to 4 h per task and fold; MultiModalPFN: up to 4 GPU-h per task and fold |
| PCA of Zhang et al. | App. D.6 | `zhang_sweep`, `zhang_sweep_v35`, `zhang_heads` | `zhang.tex` | 0.5-2 GPU-h per task |

## 4. Seeds and hardware

- Times are for one NVIDIA A100 or L40S and vary with the task size.
- TabICLv2 runs with eight members and `random_state` equal to the fold.
- RoFE draws its rotations from fixed seeds (`rofe.rotation.rotation_seeds`).

## 5. Runs in three stages

PFNs and baselines with their own environment run in three stages: `--stage export` (main environment), the
prediction command, and `--stage score` (main environment).

### TabPFN 2.5, 3.5 and v2

```bash
python scripts/run.py configs/pfns_tabpfn_v25_image.yaml --data DATA --stage export
python -m rofe.pfn.tabpfn_predict --version v2.5 --checkpoint weights/tabpfn-v2.5-classifier-v2.5_default.ckpt \
    --exports runs/pfns_tabpfn_v25_image/<TASK>/exports --output runs/pfns_tabpfn_v25_image/<TASK>/predictions  # envs/tabpfn
python scripts/run.py configs/pfns_tabpfn_v25_image.yaml --data DATA --stage score
```

TabPFN v2 uses `--version v2 --mmpfn-source MultiModalPFN --checkpoint MultiModalPFN/mmpfn/parameters/tabpfn-v2-classifier.ckpt` in `envs/mmpfn`.

### CatBoost

```bash
python scripts/run.py configs/trained_catboost_image.yaml --data DATA --stage export
python -m rofe.baselines.catboost --data DATA --task <TASK> --exports runs/trained_catboost_image/<TASK>/exports \
    --output runs/trained_catboost_image/<TASK>/predictions --threads 8 --refit     # envs/catboost
python scripts/run.py configs/trained_catboost_image.yaml --data DATA --stage score
```

`--refit` adds a model fitted on all labeled rows with the early-stopped tree count and learning rate. The settings are
in `docs/third_party.md`.

### MultiModalPFN

```bash
python scripts/run.py configs/trained_mmpfn.yaml --data DATA --stage export
python rofe/baselines/mmpfn_run.py --data DATA --task <TASK> --exports runs/trained_mmpfn/<TASK>/exports \
    --output runs/trained_mmpfn/<TASK>/predictions --mmpfn-source MultiModalPFN \
    --selection configs/mmpfn_selection.json --inputs image_table                    # envs/mmpfn; again with --inputs image_only
python scripts/run.py configs/trained_mmpfn.yaml --data DATA --stage score
```

### PCA of Zhang et al.

```bash
python scripts/run.py configs/zhang_sweep.yaml --data DATA --stage export
python -m rofe.baselines.zhang_predict --version v2.5 --checkpoint weights/tabpfn-v2.5-classifier-v2.5_default.ckpt \
    --exports runs/zhang_sweep/<TASK>/exports --output runs/zhang_sweep/<TASK>/predictions  # envs/tabpfn
python scripts/run.py configs/zhang_sweep.yaml --data DATA --stage score
```

`zhang_sweep_v35` uses `--version v3.5`.
