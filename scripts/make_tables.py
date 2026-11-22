import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tables_common import (  # noqa: E402
    ALL_LEVELS,
    FEW,
    IMAGE_ENCODERS,
    SETTINGS,
    TEXT_ENCODERS,
    Cells,
    cells,
    diff,
    interval_cells,
    latex_table,
    level,
    load,
    num,
    signed,
    stat,
    task_diffs,
    task_mean,
    write_csv,
)

from rofe.tasks import (  # noqa: E402
    FOLDS,
    IMAGE_TASKS,
    MMPFN_IMAGE,
    MULTABENCH_IMAGE,
    MULTABENCH_TEXT,
    RELEASED_IMAGE,
    RELEASED_TEXT,
    TEXT_TASKS,
    load_fold,
)

FADE_BINS = [0, 50, 100, 257, 500, 1000, 3000, 1e9]  # n, labeled rows of a draw; bins (a, b]
FADE_LABELS = ["<=50", "51-100", "101-257", "258-500", "501-1000", "1001-3000", ">3000"]
BASIS = (("comet_x4", "4 PFN seeds, no rotation"), ("rot_strong", "Within strong directions"),
         ("rot_weak", "Within weak directions"), ("hadamard", "Hadamard"),
         ("givens", "Givens"), ("rot1", "One rotation"), ("rofe", "RoFE"))
# (row, config prefix); TabICLv2 reads the main image or text configuration
PFN_CONFIGS = (("TabICLv2", None), ("TabPFN v2", "pfns_tabpfn_v2"),
               ("TabPFN 2.5", "pfns_tabpfn_v25"), ("TabPFN 3.5", "pfns_tabpfn_v35"))
# (Rotated, Standardized, method); _glob: one global scale for every column
SCALING = (("--", "--", "pc_glob"), ("--", "yes", "pc_std"), ("yes", "--", "rot1_glob"),
           ("yes", "yes", "rot1_std"))

MAIN_OUTPUTS = ("main_image", "main_text", "gain", "pfns", "basis", "scaling", "fade",
                "task_summary")


# ---------------------------------------------------------------- tables
class Paper:
    def __init__(self, runs: Path, out: Path):
        self.runs, self.out = runs, out
        self.cache: dict[tuple, Cells] = {}
        out.mkdir(parents=True, exist_ok=True)

    # primary metric AUROC, R^2 for regression, unless `metric` is given
    def c(self, config: str, metric: str | None = None) -> Cells:
        if (config, metric) not in self.cache:
            self.cache[(config, metric)] = cells(load(self.runs, config, metric))
        return self.cache[(config, metric)]

    def write(self, name: str, text: str) -> None:
        (self.out / name).write_text(text)
        print("wrote", self.out / name)

    def main_results(self, name: str, interface: str, probe: str, tasks, encoders,
                     noun: str) -> None:
        ci, cp = self.c(interface), self.c(probe)
        sections = []
        for encoder, label in encoders:
            for setting, suffix in SETTINGS:
                rows = [(r"Linear probe", [num(level(cp, tasks, setting, encoder, k, "logreg_cv"))
                                           for k in ALL_LEVELS]),
                        ("CoMET", [num(level(ci, tasks, setting, encoder, k, "comet"))
                                   for k in ALL_LEVELS]),
                        ("RoFE", [num(level(ci, tasks, setting, encoder, k, "rofe"))
                                  for k in ALL_LEVELS]),
                        # Δ = RoFE - CoMET
                        (r"$\Delta$", [signed(diff(ci, "rofe", ci, "comet", tasks, setting,
                                                   encoder, k)) for k in ALL_LEVELS])]
                sections.append((f"{noun} {suffix}, {label}", rows))
        self.write(f"{name}.tex", latex_table(ALL_LEVELS, sections))

    # RoFE - CoMET per setting, encoder and level, 95% intervals over tasks
    def gain_figure(self) -> None:
        header = ["setting", "encoder", "level", "mean", "ci_lo", "ci_hi", "wins", "n"]
        rows = []
        for config, tasks, encoders in (("table2_image", IMAGE_TASKS, IMAGE_ENCODERS),
                                        ("table3_text", TEXT_TASKS, TEXT_ENCODERS)):
            c = self.c(config)
            for setting, _ in SETTINGS:
                for encoder, _ in encoders:
                    for k in ALL_LEVELS:
                        d = diff(c, "rofe", c, "comet", tasks, setting, encoder, k)
                        rows.append([setting, encoder, k, *interval_cells(d)])
        write_csv(self.out / "fig3_gain_by_encoder.csv", header, rows)

    # RoFE - CoMET with the PFN held fixed; the short table: embedding alone, DINOv3 and ELECTRA
    def pfns(self) -> None:
        full, short = [], []
        for modality, tasks, encoders, main in (
                ("image", IMAGE_TASKS, IMAGE_ENCODERS, "table2_image"),
                ("text", TEXT_TASKS, TEXT_ENCODERS, "table3_text")):
            noun = "Image" if modality == "image" else "Text"
            for encoder, label in encoders:
                for setting, suffix in SETTINGS:
                    rows, plain = [], []
                    for name, config in PFN_CONFIGS:
                        c = self.c(main if config is None else f"{config}_{modality}")
                        ds = [diff(c, "rofe", c, "comet", tasks, setting, encoder, k)
                              for k in FEW]
                        rows.append((name, [signed(d, interval=True) for d in ds]))
                        plain.append((name, [signed(d) for d in ds]))
                    title = f"{noun} {suffix}, {label}"
                    full.append((title, rows))
                    if setting == "embedding" and encoder in ("dinov3", "electra"):
                        short.append((title, plain))
        self.write("pfns_full.tex", latex_table(FEW, full, lead=("PFN",)))
        self.write("pfns_short.tex", latex_table(FEW, short, lead=("PFN",)))

    # gain over CoMET of each change of basis, embedding alone, TabICLv2
    def basis(self) -> None:
        c = self.c("fig4_basis")
        header = ["method", "encoder", "level", "mean", "ci_lo", "ci_hi", "wins", "n"]
        rows = []
        for method, _ in BASIS:
            for encoder, _ in IMAGE_ENCODERS:
                for k in FEW:
                    d = diff(c, method, c, "comet", IMAGE_TASKS, "embedding", encoder, k)
                    rows.append([method, encoder, k, *interval_cells(d)])
        write_csv(self.out / "fig4_basis.csv", header, rows)
        sections = []
        for encoder, label in IMAGE_ENCODERS:
            sections.append((f"TabICLv2, image only, {label}", [
                (name, [signed(diff(c, m, c, "comet", IMAGE_TASKS, "embedding", encoder, k),
                               interval=True) for k in FEW]) for m, name in BASIS]))
        self.write("basis.tex", latex_table(FEW, sections, lead=("Change of basis",)))

    # one global scale against per-column standardization, principal and rotated axes
    def scaling(self) -> None:
        sections = []
        for config, tasks, encoders, noun in (
                ("table4_scaling_image", IMAGE_TASKS, IMAGE_ENCODERS, "Image only"),
                ("table4_scaling_text", TEXT_TASKS, TEXT_ENCODERS, "Text only")):
            c = self.c(config)
            for encoder, label in encoders:
                rows = [(rot, std, [num(level(c, tasks, "embedding", encoder, k, m))
                                    for k in FEW]) for rot, std, m in SCALING]
                sections.append((f"{noun}, {label}", rows))
        self.write("scaling.tex", latex_table(FEW, sections, lead=("Rotated", "Standardized")))

    # RoFE - CoMET binned by n, the number of labeled rows
    def fade(self) -> None:
        header = ["setting", "encoder", "bin", "mean", "ci_lo", "ci_hi", "wins", "n"]
        rows = []
        for config, tasks, encoders in (("table2_image", IMAGE_TASKS, IMAGE_ENCODERS),
                                        ("table3_text", TEXT_TASKS, TEXT_ENCODERS)):
            df = load(self.runs, config)
            if df.empty:
                continue
            c = cells(df)
            meta = df[df.method.isin(["comet", "rofe"])].groupby(
                ["task", "setting", "encoder", "shots"]).agg(
                n=("n_labeled", "mean"), cols=("width", "mean")).reset_index()
            for setting, _ in SETTINGS:
                for encoder, _ in encoders:
                    found = []
                    part = meta[(meta.setting == setting) & (meta.encoder == encoder)]
                    for r in part.itertuples():
                        if r.shots == 5 or r.task not in tasks:
                            continue
                        d = task_diffs(c, "rofe", c, "comet", [r.task], setting, encoder,
                                       int(r.shots))
                        if d:
                            # full: w < 256, the PCA keeps every direction of the labeled rows
                            found.append({"task": r.task, "n": r.n, "full": r.cols < 255.5,
                                          "d": d[r.task]})
                    if not found:
                        continue
                    frame = pd.DataFrame(found)
                    frame["bin"] = pd.cut(frame.n, bins=FADE_BINS, labels=FADE_LABELS, right=True)
                    for b, g in frame.groupby("bin", observed=True):  # each task once per bin
                        d = stat(g.groupby("task").d.mean().to_numpy())
                        rows.append([setting, encoder, b, *interval_cells(d)])
                    both = frame.groupby("task").filter(lambda g: g.full.any() and (~g.full).any())
                    if not both.empty:  # within a task: all n - 1 directions minus truncated
                        gap = both.groupby("task").apply(
                            lambda g: g[g.full].d.mean() - g[~g.full].d.mean())
                        rows.append([setting, encoder, "full-truncated",
                                     *interval_cells(stat(gap.to_numpy()))])
        write_csv(self.out / "fig5_fade.csv", header, rows)

    def heads(self) -> None:
        sections = []
        for heads, main, probe, tasks, encoders, noun in (
                ("heads_image", "table2_image", "probe_image", IMAGE_TASKS, IMAGE_ENCODERS,
                 "Image"),
                ("heads_text", "table3_text", "probe_text", TEXT_TASKS, TEXT_ENCODERS, "Text")):
            ch, cm, cp = self.c(heads), self.c(main), self.c(probe)
            for encoder, label in encoders:
                for setting, suffix in SETTINGS:
                    rows = [(name, [num(level(ch, tasks, setting, encoder, k, m)) for k in FEW])
                            for m, name in (("knn", "kNN"), ("linear_svm", "Linear SVM"),
                                            ("mlp", "MLP"))]
                    rows.append(("Linear probe", [num(level(cp, tasks, setting, encoder, k,
                                                            "logreg_cv")) for k in FEW]))
                    rows.append(("RoFE", [num(level(cm, tasks, setting, encoder, k, "rofe"))
                                          for k in FEW]))
                    sections.append((f"{noun} {suffix}, {label}", rows))
        self.write("heads.tex", latex_table(FEW, sections))

    # RoFE against MultiModalPFN and CatBoost on the same labeled rows
    def trained(self) -> None:
        sections = []
        for main, catboost, tasks, encoders, noun in (
                ("table2_image", "trained_catboost_image", IMAGE_TASKS, IMAGE_ENCODERS, "Image"),
                ("table3_text", "trained_catboost_text", TEXT_TASKS, TEXT_ENCODERS, "Text")):
            cm, cc, mm = self.c(main), self.c(catboost), self.c("trained_mmpfn")
            for encoder, label in encoders:
                for setting, suffix in SETTINGS:
                    rows = []
                    if encoder == "dinov3":  # MultiModalPFN runs on DINOv3 image tasks
                        rows.append(("RoFE - MultiModalPFN", [signed(
                            diff(cm, "rofe", mm, "mmpfn", tasks, setting, encoder, k), True)
                            for k in ALL_LEVELS]))
                    rows.append(("RoFE - CatBoost", [signed(
                        diff(cm, "rofe", cc, "catboost", tasks, setting, encoder, k), True)
                        for k in ALL_LEVELS]))
                    rows.append(("RoFE - CatBoost refit", [signed(
                        diff(cm, "rofe", cc, "catboost_refit", tasks, setting, encoder, k), True)
                        for k in ALL_LEVELS]))
                    sections.append((f"{noun} {suffix}, {label}", rows))
        self.write("trained.tex", latex_table(ALL_LEVELS, sections, lead=("Difference",)))

    # PCA of Zhang et al.: best fixed d, best PFN or head and d, best per task, RoFE
    def zhang(self) -> None:
        s25, s35, hd = self.c("zhang_sweep"), self.c("zhang_sweep_v35"), self.c("zhang_heads")
        dims = ("8", "16", "32", "48", "64", "96", "all")
        fixed = {f"tabpfn|zpca{d}": (s25, f"zpca{d}") for d in dims}
        every = {**fixed, **{f"tabpfn35|zpca{d}": (s35, f"zpca{d}") for d in dims},
                 **{key[4]: (hd, key[4]) for key in hd if "|" in key[4]}}
        ours = (("RoFE, TabPFN 2.5", self.c("pfns_tabpfn_v25_image")),
                ("RoFE, TabICLv2", self.c("table2_image")),
                ("RoFE, TabPFN 3.5", self.c("pfns_tabpfn_v35_image")))

        def values(cc, m, setting, encoder, k):
            return [task_mean(cc, (t, setting, encoder, k, m)) for t in sorted(IMAGE_TASKS)]

        # candidate with the highest mean over tasks, among those defined on every task
        def best(candidates, setting, encoder, k):
            found = {}
            for name, (cc, m) in candidates.items():
                v = values(cc, m, setting, encoder, k)
                if all(x is not None for x in v):
                    found[name] = v
            return max(found.values(), key=np.mean) if found else None

        def mean(v):
            return None if v is None or any(x is None for x in v) else float(np.mean(v))

        def gap(a, b):
            ok = a is not None and b is not None and None not in a and None not in b
            return stat(np.subtract(a, b)) if ok else None

        sections = []
        for encoder, label in IMAGE_ENCODERS:
            for setting, suffix in SETTINGS:
                cols = {name: [] for name in (
                    "CoMET, TabPFN 2.5", "Best fixed $d$, TabPFN 2.5", "Best PFN or head and $d$",
                    "Best per task", *(n for n, _ in ours), r"$\Delta$ best fixed $d$",
                    r"$\Delta$ best PFN or head")}
                for k in FEW:
                    a, c = best(fixed, setting, encoder, k), best(every, setting, encoder, k)
                    # best per task: maximum over every candidate
                    per_task = [[x for x in (task_mean(cc, (t, setting, encoder, k, m))
                                             for cc, m in every.values()) if x is not None]
                                for t in sorted(IMAGE_TASKS)]
                    oracle = [max(v) for v in per_task] if all(per_task) else None
                    mine = {n: values(cc, "rofe", setting, encoder, k) for n, cc in ours}
                    cols["CoMET, TabPFN 2.5"].append(num(level(
                        ours[0][1], IMAGE_TASKS, setting, encoder, k, "comet")))
                    cols["Best fixed $d$, TabPFN 2.5"].append(num(mean(a)))
                    cols["Best PFN or head and $d$"].append(num(mean(c)))
                    cols["Best per task"].append(num(mean(oracle)))
                    for n, _ in ours:
                        cols[n].append(num(mean(mine[n])))
                    cols[r"$\Delta$ best fixed $d$"].append(
                        signed(gap(mine["RoFE, TabPFN 2.5"], a), True))
                    cols[r"$\Delta$ best PFN or head"].append(
                        signed(gap(mine["RoFE, TabPFN 3.5"], c), True))
                sections.append((f"Image {suffix}, {label}", list(cols.items())))
        self.write("zhang.tex", latex_table(FEW, sections))


# task summary: ranges over the tasks of each source, fold 0
def task_summary(data: Path, out: Path) -> None:
    groups = (("Image", "MulTaBench", MULTABENCH_IMAGE), ("Image", "Released", RELEASED_IMAGE),
              ("Image", "MultiModalPFN", MMPFN_IMAGE), ("Text", "MulTaBench", MULTABENCH_TEXT),
              ("Text", "Released", RELEASED_TEXT))
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Modality & Source & Tasks & Classes & Pool & Query & Columns \\", r"\midrule"]
    for kind, source, tasks in groups:
        stats = []
        for task in tasks:
            fold = load_fold(data, task, FOLDS[0])
            stats.append((len(fold.labels), len(fold.pool), len(fold.query), fold.table.shape[1]))
        # min--max, one value when they are equal
        spans = ["--".join(dict.fromkeys(f"{v:,}" for v in (min(s), max(s))))
                 for s in zip(*stats, strict=True)]
        lines.append(" & ".join([kind, source, str(len(tasks)), *spans]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "task_summary.tex").write_text("\n".join(lines) + "\n")
    print("wrote", out / "task_summary.tex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tables and figure data from the run records")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("tables"))
    parser.add_argument("--data", type=Path, help="prepared tasks, for the task summary")
    parser.add_argument("--only", nargs="+", help="a subset of the outputs")
    parser.add_argument("--appendix", action="store_true", help="also the appendix outputs")
    args = parser.parse_args()
    paper = Paper(args.runs, args.out)
    outputs = {
        "main_image": lambda: paper.main_results("main_image", "table2_image", "probe_image",
                                                 IMAGE_TASKS, IMAGE_ENCODERS, "Image"),
        "main_text": lambda: paper.main_results("main_text", "table3_text", "probe_text",
                                                TEXT_TASKS, TEXT_ENCODERS, "Text"),
        "gain": paper.gain_figure, "pfns": paper.pfns, "basis": paper.basis,
        "scaling": paper.scaling, "fade": paper.fade, "heads": paper.heads,
        "trained": paper.trained, "zhang": paper.zhang,
    }
    if args.data is not None:
        outputs["task_summary"] = lambda: task_summary(args.data, args.out)
    names = args.only or [n for n in outputs if args.appendix or n in MAIN_OUTPUTS]
    for name in names:
        if name not in outputs:
            parser.error(f"Unknown output {name} (task_summary needs --data)")
        outputs[name]()


if __name__ == "__main__":
    main()
