__version__ = "0.1.0"

# public name -> module, imported on first access
_EXPORTS = {
    "RoFE": "method",
    "labeled_scores": "method",
    "support_pca": "pca",
    "haar_rotation": "rotation",
    "rotate": "rotation",
    "rotation_seeds": "rotation",
    "make_pfn": "pfn",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
