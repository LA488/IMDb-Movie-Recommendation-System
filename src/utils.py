"""
Utility helpers: model persistence and fuzzy title matching.
"""
import logging
import pickle
from difflib import get_close_matches
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def save_model(obj: Any, path: Path) -> None:
    """Pickle-save *obj* to *path*, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved model -> %s", path)


def load_model(path: Path) -> Any:
    """Load and return a pickled object from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run `python src/train.py` first to train and save models."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded model <- %s", path)
    return obj


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def fuzzy_match(query: str, titles: list[str], n: int = 1, cutoff: float = 0.6) -> list[str]:
    """
    Return up to *n* titles from *titles* that are close to *query*.
    Returns an empty list when no close match is found.
    """
    return get_close_matches(query, titles, n=n, cutoff=cutoff)
