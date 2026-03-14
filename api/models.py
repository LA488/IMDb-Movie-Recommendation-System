"""
Pydantic request/response schemas for the recommender API.
"""
from pydantic import BaseModel, Field


class MovieOut(BaseModel):
    """Single movie in a recommendation response."""

    Series_Title: str
    Released_Year: int
    Genre: str
    IMDB_Rating: float
    Director: str
    Overview: str
    Poster_Link: str
    similarity_score: float = Field(ge=0.0, le=1.0)

    model_config = {"from_attributes": True}


class RecommendResponse(BaseModel):
    """Response envelope for /recommend."""

    query: str
    matched_title: str
    total_results: int
    results: list[MovieOut]


class MovieDetailResponse(BaseModel):
    """Single movie full metadata response."""

    Series_Title: str
    Released_Year: int
    Certificate: str
    Runtime: int
    Genre: str
    IMDB_Rating: float
    Meta_score: float
    Director: str
    Star1: str
    Star2: str
    Star3: str
    Star4: str
    No_of_Votes: int
    Gross: float | None
    Overview: str
    Poster_Link: str

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_movies: int
