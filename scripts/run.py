import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rofe.experiments import Context, get_experiment  # noqa: E402
from rofe.tasks import FOLDS, load_fold, resolve_tasks  # noqa: E402

STAGES = ("run", "export", "score")  # export, then score: PFNs predicted in their own environment


def build_pfn(name: str | None, weights: Path):
    if name is None:
        return None
    from rofe.pfn import make_pfn

    if name == "tabicl":
        import torch

        from rofe.features.bank import sha256_file
        from rofe.pfn.tabicl import CHECKPOINT, CHECKPOINT_SHA256

        checkpoint = weights / CHECKPOINT[1]
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise SystemExit(f"{checkpoint} is not the pinned TabICLv2 checkpoint")
        torch.set_num_threads(1)
        return make_pfn("tabicl", checkpoint=checkpoint)
    return make_pfn(name)


def write_records(path: Path, records, pfn: str | None) -> int:
    count = 0
    with path.open("w") as log:
        for record in records:
            log.write(json.dumps({**record, "pfn": pfn}) + "\n")
            log.flush()
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configuration of the paper")
    parser.add_argument("config", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="prepared tasks and features")
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--tasks", nargs="+", help="a subset of the configuration's tasks")
    parser.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    parser.add_argument("--stage", choices=STAGES, default="run")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    experiment = get_experiment(cfg["experiment"])
    tasks = resolve_tasks(cfg["tasks"])
    if args.tasks:
        unknown = set(args.tasks) - set(tasks)
        if unknown:
            parser.error(f"Not in {args.config.name}: {sorted(unknown)}")
        tasks = [t for t in tasks if t in args.tasks]
    if args.stage != "run" and getattr(experiment, args.stage) is None:
        parser.error(f"{cfg['experiment']} has no {args.stage} stage")
    pfn = build_pfn(cfg.get("pfn"), args.weights) if args.stage == "run" else None
    if args.stage == "run" and pfn is not None and not pfn.in_process:
        parser.error(f"{cfg['pfn']} runs through the export and score stages")
    ctx = Context(args.data, args.weights, pfn, args.threads, cfg.get("options", {}))
    root = args.out / args.config.stem
    for task in tasks:
        directory = root / task
        directory.mkdir(parents=True, exist_ok=True)
        if args.stage == "export":
            for fold in args.folds:
                experiment.export(load_fold(args.data, task, fold), cfg, ctx,
                                  directory / "exports")
            (directory / "exports" / "EXPORTED").write_text("exported\n")
            print(json.dumps({"task": task, "exports": str(directory / "exports")}), flush=True)
            continue
        if args.stage == "score":
            records = experiment.score(task, directory, cfg, ctx)
        else:
            records = (record for fold in args.folds
                       for record in experiment.run(load_fold(args.data, task, fold), cfg, ctx))
        # a subset of the folds writes its own file; .complete marks a finished file
        name = "records.jsonl" if args.folds == list(FOLDS) else \
            f"records_folds{''.join(map(str, args.folds))}.jsonl"
        count = write_records(directory / name, records, cfg.get("pfn"))
        (directory / f"{name}.complete").write_text(f"{count} records\n")
        print(json.dumps({"task": task, "records": count}), flush=True)


if __name__ == "__main__":
    main()
