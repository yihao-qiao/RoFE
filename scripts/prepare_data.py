import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rofe.tasks import (  # noqa: E402
    MMPFN_IMAGE,
    MULTABENCH_IMAGE,
    RELEASED_IMAGE,
    TEXT_TASKS,
)

MULTABENCH_IMAGE_TASKS = (*MULTABENCH_IMAGE, *RELEASED_IMAGE)  # curated and released
MULTABENCH_SETS = {"image": MULTABENCH_IMAGE_TASKS, "text": TEXT_TASKS,
                   "all": (*MULTABENCH_IMAGE_TASKS, *TEXT_TASKS)}


def multabench(args: argparse.Namespace) -> None:
    if not os.environ.get("KAGGLEHUB_CACHE"):
        raise SystemExit("Set KAGGLEHUB_CACHE to the directory for the MulTaBench downloads")
    if args.all_classes and not args.task:
        raise SystemExit("--all-classes prepares one task at a time (--task)")
    if args.task:
        from rofe.data.multabench import prepare_image, prepare_text

        prepare = prepare_text if args.task in TEXT_TASKS else prepare_image
        audit = prepare(args.task, args.source, args.data, all_classes=args.all_classes)
        print(json.dumps({"task": args.task, "splits": audit["splits"]}))
        return
    failed = []
    for task in MULTABENCH_SETS[args.set]:
        if (args.data / "tasks" / task / "audit.json").exists():
            continue
        # one process per task: a failed task does not stop the set
        done = subprocess.run([sys.executable, __file__, "multabench", "--task", task,
                               "--source", str(args.source), "--data", str(args.data)])
        if done.returncode:
            failed.append(task)
    if failed:
        raise SystemExit(f"Failed tasks: {failed}")


def mmpfn(args: argparse.Namespace) -> None:
    from rofe.data.mmpfn_tasks import fetch_pad_metadata, prepare

    if args.task == "pad_ufes_20":
        fetch_pad_metadata(args.root)
    audit = prepare(args.task, args.root, args.data, args.cbis_policy, args.mmpfn_source)
    print(json.dumps({"task": args.task, "splits": audit["splits"]}))


def download(args: argparse.Namespace) -> None:
    from rofe.data.mmpfn_tasks import download as fetch

    print(fetch(args.dataset, args.output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the task manifests of the paper")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("multabench", help="MulTaBench image and text tasks")
    which = command.add_mutually_exclusive_group(required=True)
    which.add_argument("--task", choices=MULTABENCH_SETS["all"])
    which.add_argument("--set", choices=sorted(MULTABENCH_SETS))
    command.add_argument("--source", type=Path, required=True, help="MulTaBench clone")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--all-classes", action="store_true",
                         help="keep every class (Hearthstone, Spotify, wine); use its own --data")
    command.set_defaults(function=multabench)
    command = commands.add_parser("download", help="Kaggle copies of the MultiModalPFN data")
    command.add_argument("--dataset", choices=("pad", "cbis", "petfinder"), required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=download)
    command = commands.add_parser("mmpfn", help="the four MultiModalPFN tasks")
    command.add_argument("--task", choices=MMPFN_IMAGE, required=True)
    command.add_argument("--root", type=Path, required=True, help="downloaded dataset")
    command.add_argument("--data", type=Path, required=True)
    # CBIS-DDSM views: from the DICOM metadata, or the first directory entry as in MultiModalPFN
    command.add_argument("--cbis-policy", choices=("dicom_metadata", "author_first"),
                         default="dicom_metadata")
    command.add_argument("--mmpfn-source", type=Path, help="MultiModalPFN clone to verify")
    command.set_defaults(function=mmpfn)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
