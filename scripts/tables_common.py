import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BOOT_N, BOOT_SEED = 10_000, 20261104
FEW = [10, 25, 50, 100]
ALL_LEVELS = [10, 25, 50, 100, 250, 500, 0]  # labeled rows per class; 0: whole pool
IMAGE_ENCODERS = (("dinov3", "DINOv3"), ("siglip", "SigLIP"))
TEXT_ENCODERS = (("electra", "ELECTRA"), ("e5", "E5"))
SETTINGS = (("embedding", "only"), ("joint", "+ table"))
# {(task, setting, encoder, shots, method): {(fold, repeat): primary}}
Cells = dict[tuple, dict[tuple, float]]


# ---------------------------------------------------------------- records and aggregation
def load(runs: Path, config: str, metric: str | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted((runs / config).glob("*/records*.jsonl")):
        if path.with_name(path.name + ".complete").is_file():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric = metric or ("auroc" if "auroc" in df else "r2")
    df["primary"] = 100 * df[metric].astype(float)
    if "head" in df:  # heads on the PCA of Zhang et al.: one method per head and d
        df["method"] = np.where(df["head"].notna(), df["head"].astype(str) + "|" + df["method"],
                                df["method"])
    return df


def cells(df: pd.DataFrame) -> Cells:
    out: Cells = defaultdict(dict)
    if df.empty:
        return out
    for r in df.itertuples(index=False):
        out[(r.task, r.setting, r.encoder, int(r.shots), r.method)][(r.fold, r.repeat)] = r.primary
    return out


def task_mean(c: Cells, key: tuple, keys=None) -> float | None:
    v = c.get(key)
    if not v:
        return None
    ks = v.keys() if keys is None else keys
    if any(k not in v for k in ks):
        return None
    vals = [v[k] for k in ks if v[k] == v[k]]  # drops NaN AUROC, a class missing from the query
    return float(np.mean(vals)) if vals else None


def stat(d) -> dict:
    d = np.asarray(d, float)
    rng = np.random.default_rng(BOOT_SEED)
    # percentile bootstrap: BOOT_N resamples of the tasks with replacement, 95% interval
    means = d[rng.integers(0, len(d), size=(BOOT_N, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(d.mean()), "wins": int((d > 0).sum()), "n": len(d),
            "ci": [float(lo), float(hi)]}


def level(c: Cells, tasks, setting: str, encoder: str, shots: int, method: str) -> float | None:
    vals = [task_mean(c, (t, setting, encoder, shots, method)) for t in tasks]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# per task A - B on the same (fold, repeat) keys; sorted tasks fix the bootstrap draws
def task_diffs(ca: Cells, ma: str, cb: Cells, mb: str, tasks, setting: str, encoder: str,
               shots: int) -> dict[str, float]:
    out = {}
    for t in sorted(tasks):
        x = ca.get((t, setting, encoder, shots, ma))
        y = cb.get((t, setting, encoder, shots, mb))
        if not x or not y or set(x) != set(y):
            continue
        a = task_mean(ca, (t, setting, encoder, shots, ma), x.keys())
        b = task_mean(cb, (t, setting, encoder, shots, mb), x.keys())
        if a is not None and b is not None:
            out[t] = a - b
    return out


def diff(ca: Cells, ma: str, cb: Cells, mb: str, tasks, setting: str, encoder: str,
         shots: int) -> dict | None:
    d = task_diffs(ca, ma, cb, mb, tasks, setting, encoder, shots)
    return stat(list(d.values())) if d else None


# ---------------------------------------------------------------- formatting
def num(v: float | None) -> str:
    return "--" if v is None else f"{v:.1f}"


def signed(d: dict | None, interval: bool = False) -> str:
    if d is None:
        return "--"
    text = ("+" if round(d["mean"], 1) > 0 else "") + f"{d['mean']:.1f}"
    if interval:  # ± half-width of the 95% interval
        text += rf" {{\scriptsize$\pm${(d['ci'][1] - d['ci'][0]) / 2:.1f}}}"
    return text


# booktabs tabular, one section row per block; a row is (*labels, values)
def latex_table(cols: list[int], sections: list[tuple[str, list[tuple]]],
                lead: tuple[str, ...] = ("Method",)) -> str:
    heads = ["All" if k == 0 else str(k) for k in cols]
    width = len(lead) + len(cols)
    lines = [r"\begin{tabular}{" + "l" * len(lead) + "c" * len(cols) + "}", r"\toprule",
             " & ".join([*lead, *heads]) + r" \\", r"\midrule"]
    for i, (title, rows) in enumerate(sections):
        lines.append(rf"\multicolumn{{{width}}}{{l}}{{\textit{{{title}}}}} \\")
        lines += [" & ".join([*row[:-1], *row[-1]]) + r" \\" for row in rows]
        lines.append(r"\midrule" if i < len(sections) - 1 else r"\bottomrule")
    return "\n".join([*lines, r"\end{tabular}"]) + "\n"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for r in rows:
            writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in r])


def interval_cells(d: dict | None) -> list:
    if d is None:
        return [float("nan")] * 3 + [0, 0]
    return [d["mean"], d["ci"][0], d["ci"][1], d["wins"], d["n"]]
