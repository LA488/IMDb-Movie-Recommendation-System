"""
Integration tests for the FastAPI application.
Uses httpx + TestClient (no running server needed).
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """
    TestClient that exercises the full lifespan (model loading).
    Models must already exist — run `python src/train.py` first.
    """
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["total_movies"], int)

    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client):
        resp = client.get("/recommend", params={"title": "Inception", "top_n": 5})
        assert resp.status_code == 200

    def test_recommend_response_structure(self, client):
        resp = client.get("/recommend", params={"title": "Inception", "top_n": 3})
        data = resp.json()
        assert "query" in data
        assert "results" in data
        assert len(data["results"]) == 3

    def test_recommend_result_fields(self, client):
        resp = client.get("/recommend", params={"title": "Inception", "top_n": 1})
        movie = resp.json()["results"][0]
        for field in ("Series_Title", "Genre", "IMDB_Rating", "similarity_score"):
            assert field in movie

    def test_recommend_unknown_title_404(self, client):
        resp = client.get("/recommend", params={"title": "ZZZNOMOVIEHERE999"})
        assert resp.status_code == 404

    def test_recommend_top_n_respected(self, client):
        for n in (1, 5, 10):
            resp = client.get("/recommend", params={"title": "The Dark Knight", "top_n": n})
            assert len(resp.json()["results"]) == n

    def test_recommend_top_n_max_limit(self, client):
        """top_n > MAX_TOP_N should return 422 Unprocessable Entity."""
        resp = client.get("/recommend", params={"title": "Inception", "top_n": 9999})
        assert resp.status_code == 422


class TestMoviesEndpoint:
    def test_movies_returns_200(self, client):
        resp = client.get("/movies")
        assert resp.status_code == 200

    def test_movies_pagination(self, client):
        resp = client.get("/movies", params={"page": 1, "page_size": 10})
        data = resp.json()
        assert len(data["results"]) == 10
        assert data["page"] == 1

    def test_movies_genre_filter(self, client):
        resp = client.get("/movies", params={"genre": "Drama"})
        results = resp.json()["results"]
        assert all("Drama" in r["Genre"] for r in results)

    def test_movies_rating_filter(self, client):
        resp = client.get("/movies", params={"min_rating": 9.0})
        results = resp.json()["results"]
        assert all(r["IMDB_Rating"] >= 9.0 for r in results)


class TestMovieDetailEndpoint:
    def test_movie_detail_ok(self, client):
        resp = client.get("/movie/Inception")
        assert resp.status_code == 200

    def test_movie_detail_fields(self, client):
        resp = client.get("/movie/Inception")
        data = resp.json()
        assert data["Series_Title"] == "Inception"
        for field in ("Genre", "Director", "IMDB_Rating", "Overview"):
            assert field in data

    def test_movie_detail_not_found(self, client):
        resp = client.get("/movie/ZZZNOTAMOVIE")
        assert resp.status_code == 404
