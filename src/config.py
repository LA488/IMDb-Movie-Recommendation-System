"""
Project-wide configuration.
All paths are resolved relative to the project root using pathlib,
so the app works regardless of the working directory it's launched from.
"""
from pathlib import Path

# ── Project root (two levels up from this file: src/config.py → src/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "imdb_top_1000.csv"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "imdb_clean.csv"

# ── Models
MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
SIM_MATRIX_PATH = MODELS_DIR / "similarity_matrix.pkl"

# ── Recommender defaults
TOP_N_DEFAULT: int = 10
MAX_TOP_N: int = 50
MIN_SIMILARITY_SCORE: float = 0.0

# ── TF-IDF settings
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple = (1, 2)
