import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SHL Smart Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/catalog_clean.csv")
FAISS_PATH = os.path.join(BASE_DIR, "../embeddings/vector_store.faiss")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RequestData(BaseModel):
    query: str


def load_all():
    try:
        print(f"🔹 Trying to load {DATA_PATH} and {FAISS_PATH}")
        df = pd.read_csv(DATA_PATH)
        model = SentenceTransformer(MODEL_NAME)
        index = faiss.read_index(FAISS_PATH)
        print("✅ Loaded all resources successfully!")
        return df, model, index
    except Exception as e:
        print(f"❌ Error loading FAISS or CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend/recommend")
def recommend(request: RequestData):
    try:
        df, model, index = load_all()
        query_vector = model.encode([request.query])
        distances, indices = index.search(query_vector, k=5)
        results = []
        for i, score in zip(indices[0], distances[0]):
            if 0 <= i < len(df):
                item = df.iloc[i].to_dict()
                item["similarity_score"] = float(score)
                results.append(item)
        return {"query": request.query, "recommendations": results}
    except Exception as e:
        print(f"🔥 Recommend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
