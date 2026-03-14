"""
FastAPI application entry point.

Start the server:
    python -m uvicorn api.main:app --reload

Interactive docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup, release at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained recommender model at startup."""
    from src import config
    from src.recommender import MovieRecommender
    from src.utils import load_model

    logger.info("Loading models …")
    try:
        df = load_model(config.MODELS_DIR / "dataframe.pkl")
        vectorizer = load_model(config.VECTORIZER_PATH)
        sim_matrix = load_model(config.SIM_MATRIX_PATH)

        rec = MovieRecommender()
        rec.df = df
        rec.vectorizer = vectorizer
        rec.similarity_matrix = sim_matrix
        rec._titles_lower = df["Series_Title"].str.lower().tolist()

        app.state.recommender = rec
        logger.info("Models loaded [OK]  (%d movies)", len(df))
    except FileNotFoundError as exc:
        logger.error("Model files missing: %s", exc)
        logger.error("Run `python src/train.py` first, then restart the server.")
        # Still start the server so /health returns a meaningful error
        from src.recommender import MovieRecommender
        app.state.recommender = MovieRecommender()

    yield  # ← application runs here

    logger.info("Shutting down …")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IMDb Movie Recommendation API",
    description=(
        "Content-based movie recommendation system built on IMDb Top 1000 dataset.\n\n"
        "Uses TF-IDF vectorisation of movie metadata (genre, director, cast, overview) "
        "and cosine similarity to find the most similar films.\n\n"
        "**Before first use:** run `python src/train.py` to generate model files."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local dev (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from api.routers.recommend import router as recommend_router  # noqa: E402

app.include_router(recommend_router, prefix="")
