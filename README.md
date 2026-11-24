# RoFE: Rotated Frozen Embeddings for Low-Data Multimodal Tabular Learning

[![arXiv](https://img.shields.io/badge/-arXiv%202611.31049-lightgrey?style=social&logo=arxiv)](https://arxiv.org/abs/2611.31049)
[![Project Page](https://img.shields.io/badge/-Project%20Page-lightgrey?style=social&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiI%2BPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwwKSBzY2FsZSgxLjApIiBmaWxsPSJub25lIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMS40IiBzdHJva2UtbGluZWNhcD0icm91bmQiPjxjaXJjbGUgY3g9IjgiIGN5PSI4IiByPSI2LjYiLz48ZWxsaXBzZSBjeD0iOCIgY3k9IjgiIHJ4PSIyLjkiIHJ5PSI2LjYiLz48cGF0aCBkPSJNMS40IDhIMTQuNk0yLjYgNC42SDEzLjRNMi42IDExLjRIMTMuNCIvPjwvZz48L3N2Zz4%3D)](https://yihao-qiao.github.io/RoFE/)

This is the implementation of RoFE and of the experiments listed in `docs/reproducing.md`. The core method is located
in `rofe/method.py`.

<p align="center"><img src="assets/overview.png" width="860" alt="Overview of RoFE"></p>

## Install

The main environment (Python 3.12) runs RoFE, CoMET, TabICLv2, the heads and the table scripts:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[test,data]"
python scripts/download_weights.py --weights weights      # TabICLv2 classifier checkpoint
```

TabPFN, MultiModalPFN, CatBoost, MulTaBench and the encoders have their own environments in `envs/`;
`docs/reproducing.md` gives the commands.

## Quick start

```python
from rofe import RoFE, make_pfn

pfn = make_pfn("tabicl", checkpoint="weights/tabicl-classifier-v2-20260212.ckpt")
model = RoFE(pfn, n_rotations=4).fit(x_labeled, y_labeled, table_labeled)  # embeddings (n, p)
proba = model.predict_proba(x_query, table_query)                         # (n_q, classes)
comet = RoFE(pfn, rotate=False).fit(x_labeled, y_labeled, table_labeled)  # the CoMET interface
```

One prepared task with ten labels per class (one GPU, a few minutes), and the tests on synthetic data (CPU):

```bash
python scripts/run.py configs/quickstart.yaml --data DATA --weights weights --out runs
pytest
```

## Data

`docs/data.md` gives the commands that clone MulTaBench, MultiModalPFN and DINOv3 at the pinned commits, prepare the
tasks in `DATA/tasks/` and extract the embeddings to `DATA/features/`. Every dataset and weight file is obtained by the
user under its own licence.

## Reproducing the tables

Each configuration in `configs/` runs one experiment, and `scripts/make_tables.py` writes the tables:

```bash
python scripts/run.py configs/<config>.yaml --data DATA --weights weights --out runs
python scripts/make_tables.py --runs runs --out tables --data DATA
```

`docs/reproducing.md` lists the configuration, output and cost of each table and figure and the staged commands of
the models in their own environments. `docs/method_to_code.md` maps the paper to the code.

## Citation

```bibtex
@article{qiao2026rofe,
  title   = {RoFE: Rotated Frozen Embeddings for Low-Data Multimodal Tabular Learning},
  author  = {Qiao, Yihao and Ma, Jiajian and Rui, Wushuang and Jha, Abhishek and Darji, Aarti and Shen, Yiqiu},
  journal = {arXiv preprint arXiv:2611.31049},
  year    = {2026}
}
```

## License

The code is released under the MIT License (`LICENSE`). Upstream code and model weights keep their own licences,
listed in `docs/third_party.md`.
