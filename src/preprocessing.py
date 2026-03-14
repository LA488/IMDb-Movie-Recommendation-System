"""
Data loading, cleaning and feature engineering for the IMDb recommender.
"""
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(path: str | "Path") -> pd.DataFrame:
    """Load the raw IMDb CSV file."""
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows, %d cols", *df.shape)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw IMDb DataFrame.

    Fixes applied:
    - Released_Year  : extract 4-digit year, keep as int (NaN → 0 for safety)
    - Runtime        : strip ' min', cast to int
    - Gross          : strip commas, cast to float (NaN safe)
    - Meta_score     : fill NaN with column median
    - Certificate    : fill NaN with 'Unknown'
    - drop_duplicates
    """
    df = df.copy()

    # Released_Year — some rows have 'PG' instead of a year (e.g. poster links)
    df["Released_Year"] = (
        df["Released_Year"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
        .astype(float)          # float to accommodate NaN before fillna
        .fillna(0)
        .astype(int)
    )

    # Runtime — "142 min" → 142
    df["Runtime"] = (
        df["Runtime"]
        .astype(str)
        .str.replace(r"\s*min", "", regex=True)
        .str.strip()
        .replace("", "0")
        .astype(int)
    )

    # Gross — "12,345,678" → 12345678.0  (NaN rows stay NaN)
    df["Gross"] = (
        df["Gross"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", None)
        .astype(float)
    )

    # Meta_score — fill with median
    df["Meta_score"] = df["Meta_score"].fillna(df["Meta_score"].median())

    # Certificate — fill with 'Unknown'
    df["Certificate"] = df["Certificate"].fillna("Unknown")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Series_Title"])
    logger.info("Dropped %d duplicate rows", before - len(df))

    df = df.reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'tags' column used for TF-IDF vectorisation.

    tags = Genre + Director + Stars + Overview (lowercased, symbols stripped)
    Multi-word names are joined with underscores so TF-IDF treats them as tokens.
    """
    df = df.copy()

    def _clean_token(text: str) -> str:
        """Lowercase, strip punctuation, join multi-word with underscore."""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        return "_".join(text.split())

    def _build_row_tags(row: pd.Series) -> str:
        parts = []
        # Genre: "Crime, Drama" → "crime drama"
        for genre in str(row["Genre"]).split(","):
            parts.append(_clean_token(genre))
        # Director
        parts.append(_clean_token(row["Director"]))
        # Stars
        for star in ["Star1", "Star2", "Star3", "Star4"]:
            parts.append(_clean_token(row[star]))
        # Overview (free text — keep as-is after clean)
        overview_words = re.sub(r"[^a-z0-9 ]", " ", str(row["Overview"]).lower()).split()
        parts.extend(overview_words)
        return " ".join(parts)

    df["tags"] = df.apply(_build_row_tags, axis=1)
    return df
