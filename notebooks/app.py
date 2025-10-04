from fastapi import FastAPI, Query

import pandas as pd
import pickle
from typing import List

# Загружаем артефакты
with open("tfidf_model.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

df = pd.read_pickle("movies_df.pkl")
indices = pd.Series(df.index, index=df['Series_Title']).drop_duplicates()

# Инициализация приложения
app = FastAPI(title="Movie Recommendation API")

def get_recommendations(title: str, n: int = 5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # убираем сам фильм
    movie_indices = [i[0] for i in sim_scores]
    return df[['Series_Title', 'IMDB_Rating', 'Genre']].iloc[movie_indices].to_dict(orient="records")

@app.get("/recommend")
def recommend(title: str = Query(..., description="Название фильма"),
              n: int = Query(5, description="Количество рекомендаций")):
    """
    Получить список рекомендованных фильмов по названию.
    """
    recs = get_recommendations(title, n)
    if not recs:
        return {"error": f"Фильм '{title}' не найден в базе."}
    return {"title": title, "recommendations": recs}
