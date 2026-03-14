"""
Router: /recommend, /movies, /movie/{title}, /health
"""
from fastapi import APIRouter, HTTPException, Query, Request

from api.models import HealthResponse, MovieDetailResponse, RecommendResponse
from src import config

router = APIRouter()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["Utility"])
def health(request: Request):
    """Service health check."""
    rec = request.app.state.recommender
    return HealthResponse(
        status="ok",
        model_loaded=rec.df is not None,
        total_movies=len(rec.df) if rec.df is not None else 0,
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@router.get("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(
    request: Request,
    title: str = Query(..., description="Movie title to base recommendations on"),
    top_n: int = Query(
        config.TOP_N_DEFAULT,
        ge=1,
        le=config.MAX_TOP_N,
        description="Number of movies to return",
    ),
):
    """
    Return *top_n* content-based recommendations for the given movie title.

    Supports fuzzy matching — slight typos or partial titles are tolerated.
    """
    rec = request.app.state.recommender
    results = rec.recommend(title, top_n=top_n)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Movie '{title}' not found in the dataset. "
                "Check /movies for a full list of available titles."
            ),
        )

    # The first result reveals what title was actually matched
    matched_title = title
    if rec.df is not None:
        titles_lower = rec.df["Series_Title"].str.lower().tolist()
        query_lower = title.strip().lower()
        if query_lower not in titles_lower:
            from src.utils import fuzzy_match
            close = fuzzy_match(query_lower, titles_lower)
            if close:
                matched_title = rec.df["Series_Title"][
                    titles_lower.index(close[0])
                ]

    return RecommendResponse(
        query=title,
        matched_title=matched_title,
        total_results=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Movie catalogue
# ---------------------------------------------------------------------------

@router.get("/movies", tags=["Catalogue"])
def list_movies(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    genre: str | None = Query(None, description="Filter by genre (case-insensitive)"),
    min_rating: float = Query(0.0, ge=0.0, le=10.0, description="Minimum IMDb rating"),
):
    """
    Paginated list of all movies with optional genre and rating filters.
    Useful for building autocomplete or browsing the catalogue.
    """
    rec = request.app.state.recommender
    if rec.df is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    df = rec.df.copy()

    if genre:
        df = df[df["Genre"].str.contains(genre, case=False, na=False)]

    df = df[df["IMDB_Rating"] >= min_rating]
    df = df.sort_values("IMDB_Rating", ascending=False)

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": page_df[
            ["Series_Title", "Genre", "IMDB_Rating", "Released_Year", "Director", "Poster_Link"]
        ].to_dict(orient="records"),
    }


@router.get("/movie/{title}", response_model=MovieDetailResponse, tags=["Catalogue"])
def movie_detail(title: str, request: Request):
    """
    Return full metadata for a single movie.
    """
    rec = request.app.state.recommender
    info = rec.get_movie_info(title)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")

    # Gross can be NaN — convert to None for JSON
    if info.get("Gross") != info.get("Gross"):  # NaN check
        info["Gross"] = None

    return info
