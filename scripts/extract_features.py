import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rofe.features import EXTRACTORS, get_extractor  # noqa: E402

IMAGE_ENCODERS = ("dinov3", "siglip")  # read image files; text encoders read texts.jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen embeddings of the prepared tasks")
    parser.add_argument("--encoder", choices=sorted(EXTRACTORS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--image-root", type=Path,
                        help="dataset root of the images (default: the one in audit.json)")
    parser.add_argument("--dinov3-source", type=Path, help="DINOv3 clone")
    parser.add_argument("--checkpoint", type=Path, help="DINOv3 ViT-B/16 weights")
    parser.add_argument("--revision", help="SigLIP commit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.encoder == "dinov3" and (args.dinov3_source is None or args.checkpoint is None):
        parser.error("dinov3 needs --dinov3-source and --checkpoint")
    options = {"device": args.device}
    if args.encoder in IMAGE_ENCODERS:
        options["root"] = args.image_root
    if args.encoder == "dinov3":
        options |= {"source": args.dinov3_source, "checkpoint": args.checkpoint}
    if args.encoder == "siglip" and args.revision:
        options["revision"] = args.revision
    extract = get_extractor(args.encoder)
    for task in args.tasks:
        output = args.data / "features" / args.encoder / task
        if (output / "manifest.json").exists():
            continue
        bank = extract(args.data / "tasks" / task, output, **options)
        print(json.dumps({"task": task, "encoder": args.encoder,
                          "rows": len(bank.row_ids)}), flush=True)


if __name__ == "__main__":
    main()
