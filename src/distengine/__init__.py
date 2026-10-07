"""DistEngine: a minimal disaggregated inference engine."""

__version__ = "0.1.0"


def main() -> None:
    """Report the package setup without loading models or allocating GPUs."""
    print(f"DistEngine {__version__}: project setup ready (stage 0).")
