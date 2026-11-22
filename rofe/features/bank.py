import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FILES = ("features.npy", "row_ids.json")  # checksums in manifest.json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureBank:
    """Frozen embeddings, one row per image or text, with their identities."""

    features: np.ndarray  # (N, p) float32
    row_ids: tuple[str, ...]
    provenance: dict

    @classmethod
    def load(cls, directory: Path) -> "FeatureBank":
        directory = Path(directory)
        meta = json.loads((directory / "manifest.json").read_text())
        for name in FILES:
            if sha256_file(directory / name) != meta["sha256"][name]:
                raise ValueError(f"Checksum mismatch: {directory / name}")
        features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        if features.dtype != np.float32 or list(features.shape) != meta["shape"]:
            raise ValueError(f"Unexpected features in {directory}")
        ids = tuple(json.loads((directory / "row_ids.json").read_text()))
        if len(ids) != len(features) or len(set(ids)) != len(ids):
            raise ValueError(f"Row identities do not match the features in {directory}")
        return cls(features, ids, meta.get("provenance", {}))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "features.npy", np.asarray(self.features, np.float32),
                allow_pickle=False)
        (directory / "row_ids.json").write_text(json.dumps(list(self.row_ids)))
        # manifest last: only a complete bank has one
        meta = {"shape": list(self.features.shape), "dtype": "float32",
                "provenance": self.provenance,
                "sha256": {name: sha256_file(directory / name) for name in FILES}}
        (directory / "manifest.json").write_text(json.dumps(meta, indent=2))

    def rows(self, keys: list[list[str]]) -> np.ndarray:
        lookup = {key: i for i, key in enumerate(self.row_ids)}
        out = np.stack([np.concatenate([self.features[lookup[k]] for k in view]) for view in keys])
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite embeddings")
        return out  # (N, views * p), views concatenated in order
