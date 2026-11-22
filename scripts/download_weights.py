import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rofe.pfn.tabicl import download_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the TabICLv2 classifier checkpoint")
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    args = parser.parse_args()
    args.weights.mkdir(parents=True, exist_ok=True)
    print(download_checkpoint(args.weights))


if __name__ == "__main__":
    main()
