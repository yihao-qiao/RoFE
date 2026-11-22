import subprocess
import sys
from pathlib import Path

# pinned commits of the cloned sources
MULTABENCH_REVISION = "54e2466006da11504e4ea7583ebb056f5b3d7343"
MMPFN_REVISION = "e7a8f93205a9901f199dead7d39b9afc78d1471d"
DINOV3_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"


def verify_clone(path: Path, revision: str) -> Path:
    path = Path(path).resolve()
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                      text=True).strip()
    if head != revision or changes:
        raise ValueError(f"{path} is not a clean clone at {revision}")
    return path


def import_from(path: Path) -> None:
    sys.dont_write_bytecode = True  # no bytecode written into the clone
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
