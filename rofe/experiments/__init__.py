from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

EXPERIMENTS: dict[str, "Experiment"] = {}


@dataclass(frozen=True)
class Experiment:
    name: str
    run: Callable  # (fold, cfg, ctx) -> records
    export: Callable | None = None  # blocks for a PFN run in its own environment
    score: Callable | None = None  # records from that PFN's saved predictions


@dataclass(frozen=True)
class Context:
    data: Path
    weights: Path
    pfn: object = None  # in-process PFN; None in the export and score stages
    threads: int = 1
    options: dict = field(default_factory=dict)


def register_experiment(name: str, export: Callable | None = None,
                        score: Callable | None = None) -> Callable:
    def decorator(run: Callable) -> Callable:
        EXPERIMENTS[name] = Experiment(name, run, export, score)
        return run
    return decorator


def get_experiment(name: str) -> Experiment:
    from ..baselines import experiments  # noqa: F401  (registration)
    from . import (  # noqa: F401  (registration)
        basis,
        heads,
        interface,
        scaling,
    )

    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name}; known: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]
