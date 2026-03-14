"""
Tests for src/preprocessing.py
"""
import pandas as pd
import pytest
from src.preprocessing import clean_dataset, build_features


@pytest.fixture
def raw_df():
    """Minimal raw DataFrame that mimics IMDb CSV structure."""
    return pd.DataFrame(
        {
            "Poster_Link": ["https://img.png"],
            "Series_Title": ["Test Movie"],
            "Released_Year": ["2003"],
            "Certificate": [None],       # NaN — should become "Unknown"
            "Runtime": ["142 min"],
            "Genre": ["Drama, Crime"],
            "IMDB_Rating": [9.3],
            "Overview": ["Two men bond in prison."],
            "Meta_score": [None],        # NaN — should become median
            "Director": ["Frank Darabont"],
            "Star1": ["Tim Robbins"],
            "Star2": ["Morgan Freeman"],
            "Star3": ["Bob Gunton"],
            "Star4": ["William Sadler"],
            "No_of_Votes": [2_600_000],
            "Gross": ["28,341,469"],     # should become float
        }
    )


class TestCleanDataset:
    def test_runtime_is_int(self, raw_df):
        df = clean_dataset(raw_df)
        assert df["Runtime"].dtype == int
        assert df["Runtime"].iloc[0] == 142

    def test_gross_is_float(self, raw_df):
        df = clean_dataset(raw_df)
        assert df["Gross"].dtype == float
        assert df["Gross"].iloc[0] == 28_341_469.0

    def test_released_year_is_int(self, raw_df):
        df = clean_dataset(raw_df)
        assert df["Released_Year"].dtype == int
        assert df["Released_Year"].iloc[0] == 2003

    def test_certificate_nan_filled(self, raw_df):
        df = clean_dataset(raw_df)
        assert df["Certificate"].iloc[0] == "Unknown"

    def test_meta_score_nan_filled(self, raw_df):
        # With only one row the median is also NaN, but fillna handles it
        raw_df2 = pd.concat([raw_df, raw_df.assign(
            Series_Title="Second Movie", Meta_score=70.0
        )], ignore_index=True)
        df = clean_dataset(raw_df2)
        assert df["Meta_score"].isna().sum() == 0

    def test_no_duplicates(self, raw_df):
        doubled = pd.concat([raw_df, raw_df], ignore_index=True)
        df = clean_dataset(doubled)
        assert df.duplicated(subset=["Series_Title"]).sum() == 0


class TestBuildFeatures:
    def test_tags_column_created(self, raw_df):
        df = build_features(clean_dataset(raw_df))
        assert "tags" in df.columns

    def test_tags_contains_genre(self, raw_df):
        df = build_features(clean_dataset(raw_df))
        assert "drama" in df["tags"].iloc[0]

    def test_tags_contains_director(self, raw_df):
        df = build_features(clean_dataset(raw_df))
        assert "frank_darabont" in df["tags"].iloc[0]
