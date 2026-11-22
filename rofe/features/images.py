import hashlib
import io
import json
from pathlib import Path

from .bank import sha256_file


def image_record(row_id: str, path: Path, root: Path) -> dict:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    return {"row_id": row_id, "image_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path)}


# one JSON record per line, in extraction order
def write_images(task_dir: Path, records: list[dict]) -> None:
    (task_dir / "images.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def read_images(task_dir: Path) -> list[dict]:
    with (task_dir / "images.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or len({r["row_id"] for r in records}) != len(records):
        raise ValueError(f"Invalid image inventory in {task_dir}")
    return records


def image_root(task_dir: Path, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    # else the root used at preparation
    return Path(json.loads((task_dir / "audit.json").read_text())["image_root"])


def checked_rgb(record: dict, root: Path):
    from PIL import Image

    payload = (root / record["image_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"Image bytes changed: {record['row_id']}")
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")
