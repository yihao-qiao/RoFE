from collections.abc import Callable

PFNS: dict[str, Callable] = {}  # name -> wrapper class


def register_pfn(name: str) -> Callable:
    def decorator(cls):
        PFNS[name] = cls
        return cls
    return decorator


def make_pfn(name: str, **kwargs):
    """PFN wrapper by name: tabicl, tabpfn_v2, tabpfn_v2.5 or tabpfn_v3.5."""
    from . import tabicl, tabpfn  # noqa: F401  (registration)

    if name not in PFNS:
        raise ValueError(f"Unknown PFN {name}; known: {sorted(PFNS)}")
    return PFNS[name](**kwargs)


__all__ = ["PFNS", "make_pfn", "register_pfn"]
