"""
Tests for src/recommender.py
"""
import pandas as pd
import pytest
from src.recommender import MovieRecommender
from src.preprocessing import clean_dataset, build_features


@pytest.fixture(scope="module")
def trained_recommender():
    """Build a recommender from the real processed dataset."""
    from src.config import DATA_RAW_PATH
    from src.preprocessing import load_raw
    df = build_features(clean_dataset(load_raw(DATA_RAW_PATH)))
    rec = MovieRecommender()
    rec.build(df)
    return rec


class TestMovieRecommender:
    def test_build_creates_matrix(self, trained_recommender):
        rec = trained_recommender
        assert rec.similarity_matrix is not None
        assert rec.similarity_matrix.shape[0] == rec.similarity_matrix.shape[1]

    def test_recommend_known_title(self, trained_recommender):
        results = trained_recommender.recommend("The Dark Knight", top_n=5)
        assert isinstance(results, list)
        assert len(results) == 5
        # All results must have required keys
        for r in results:
            for key in ("Series_Title", "Genre", "IMDB_Rating", "similarity_score"):
                assert key in r

    def test_recommend_case_insensitive(self, trained_recommender):
        results_lower = trained_recommender.recommend("the dark knight", top_n=3)
        results_upper = trained_recommender.recommend("THE DARK KNIGHT", top_n=3)
        assert [r["Series_Title"] for r in results_lower] == [
            r["Series_Title"] for r in results_upper
        ]

    def test_recommend_unknown_title_returns_empty(self, trained_recommender):
        results = trained_recommender.recommend("ZZZZNOTAMOVIEXXXX12345", use_fuzzy=False)
        assert results == []

    def test_recommend_fuzzy_match(self, trained_recommender):
        # Slight typo "Inceptoin" should still return results via fuzzy
        results = trained_recommender.recommend("Inceptoin", top_n=3)
        assert len(results) > 0

    def test_get_all_titles(self, trained_recommender):
        titles = trained_recommender.get_all_titles()
        assert isinstance(titles, list)
        assert len(titles) > 900  # IMDb Top 1000 dataset

    def test_get_movie_info(self, trained_recommender):
        info = trained_recommender.get_movie_info("Inception")
        assert info is not None
        assert info["Series_Title"] == "Inception"
        assert "IMDB_Rating" in info

    def test_get_movie_info_not_found(self, trained_recommender):
        info = trained_recommender.get_movie_info("ZZZZNOTAMOVIE")
        assert info is None

    def test_similarity_score_range(self, trained_recommender):
        results = trained_recommender.recommend("Inception", top_n=10)
        for r in results:
            assert 0.0 <= r["similarity_score"] <= 1.0
