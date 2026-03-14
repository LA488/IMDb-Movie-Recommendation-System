"""
train.py — one-shot training script.

Run from the project root:
    python src/train.py

What it does:
    1. Load data/raw/imdb_top_1000.csv
    2. Clean + feature-engineer → data/processed/imdb_clean.csv
    3. Fit TF-IDF + cosine similarity
    4. Save models/vectorizer.pkl + models/similarity_matrix.pkl
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when script is run directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def main() -> None:
    from src import config
    from src.preprocessing import load_raw, clean_dataset, build_features
    from src.recommender import MovieRecommender
    from src.utils import save_model

    # 1. Load
    logger.info("Loading raw data from %s", config.DATA_RAW_PATH)
    df_raw = load_raw(config.DATA_RAW_PATH)

    # 2. Clean
    logger.info("Cleaning dataset …")
    df_clean = clean_dataset(df_raw)

    # 3. Feature engineering
    logger.info("Building feature tags …")
    df_feat = build_features(df_clean)

    # 4. Save processed CSV (used by API at startup)
    config.DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(config.DATA_PROCESSED_PATH, index=False)
    logger.info("Saved processed data -> %s", config.DATA_PROCESSED_PATH)

    # 5. Train recommender
    logger.info("Fitting TF-IDF + cosine similarity …")
    rec = MovieRecommender()
    rec.build(df_feat)

    # 6. Save models
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(rec.vectorizer, config.VECTORIZER_PATH)
    save_model(rec.similarity_matrix, config.SIM_MATRIX_PATH)
    # Also save the cleaned/featured DataFrame so the API can reload without re-training
    save_model(df_feat, config.MODELS_DIR / "dataframe.pkl")

    logger.info("Training complete [OK]")
    logger.info("  vectorizer   -> %s", config.VECTORIZER_PATH)
    logger.info("  sim_matrix   -> %s", config.SIM_MATRIX_PATH)


if __name__ == "__main__":
    main()
