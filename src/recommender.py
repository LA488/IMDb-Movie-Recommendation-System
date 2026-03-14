"""
Content-based MovieRecommender using TF-IDF + cosine similarity.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.utils import fuzzy_match

logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    Content-based recommender built on TF-IDF of combined movie metadata
    (genre, director, cast, overview).

    Usage
    -----
    >>> rec = MovieRecommender()
    >>> rec.build(df)            # fit on preprocessed DataFrame
    >>> results = rec.recommend("Inception")
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self._titles_lower: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> None:
        """
        Fit TF-IDF vectorizer on the 'tags' column and compute cosine
        similarity matrix.  *df* must already contain a 'tags' column
        (created by preprocessing.build_features).
        """
        if "tags" not in df.columns:
            raise ValueError("DataFrame must contain a 'tags' column. Run build_features() first.")

        self.df = df.reset_index(drop=True)
        self._titles_lower = self.df["Series_Title"].str.lower().tolist()

        self.vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        logger.info(
            "Model built: %d movies, similarity matrix %s",
            len(self.df),
            self.similarity_matrix.shape,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        movie_title: str,
        top_n: int = config.TOP_N_DEFAULT,
        use_fuzzy: bool = True,
    ) -> list[dict]:
        """
        Return *top_n* movies most similar to *movie_title*.

        Parameters
        ----------
        movie_title : str
            Exact or approximate movie title.
        top_n : int
            Number of recommendations to return.
        use_fuzzy : bool
            Fall back to fuzzy matching when exact match fails.

        Returns
        -------
        list[dict]
            Each dict contains: Series_Title, Genre, IMDB_Rating,
            Director, Overview, Released_Year, similarity_score.

        Raises
        ------
        RuntimeError
            If the model has not been built yet (call build() first).
        """
        if self.df is None:
            raise RuntimeError("Model not built. Call build() or load model files first.")

        # 1. Exact match (case-insensitive)
        query_lower = movie_title.strip().lower()
        matched_indices = [i for i, t in enumerate(self._titles_lower) if t == query_lower]

        # 2. Fuzzy fallback
        if not matched_indices and use_fuzzy:
            close = fuzzy_match(query_lower, self._titles_lower, n=1, cutoff=0.6)
            if close:
                matched_title = close[0]
                matched_indices = [i for i, t in enumerate(self._titles_lower) if t == matched_title]
                logger.info("Fuzzy match: '%s' → '%s'", movie_title, matched_title)

        if not matched_indices:
            return []

        idx = matched_indices[0]
        top_n = min(top_n, config.MAX_TOP_N)

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip the first result (the movie itself)
        sim_scores = [s for s in sim_scores if s[0] != idx][: top_n]

        results = []
        for movie_idx, score in sim_scores:
            row = self.df.iloc[movie_idx]
            results.append(
                {
                    "Series_Title": row["Series_Title"],
                    "Released_Year": int(row["Released_Year"]),
                    "Genre": row["Genre"],
                    "IMDB_Rating": float(row["IMDB_Rating"]),
                    "Director": row["Director"],
                    "Overview": row["Overview"],
                    "Poster_Link": row.get("Poster_Link", ""),
                    "similarity_score": round(float(score), 4),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_all_titles(self) -> list[str]:
        """Return list of all movie titles in the dataset."""
        if self.df is None:
            return []
        return self.df["Series_Title"].tolist()

    def get_movie_info(self, movie_title: str) -> Optional[dict]:
        """Return full metadata for a single movie, or None if not found."""
        if self.df is None:
            return None
        query_lower = movie_title.strip().lower()
        mask = self.df["Series_Title"].str.lower() == query_lower
        if not mask.any():
            return None
        row = self.df[mask].iloc[0]
        return row.drop("tags", errors="ignore").to_dict()
