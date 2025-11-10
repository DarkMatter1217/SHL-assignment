import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------
# ✅ Config
# ---------------------------------------------------------------------
app = FastAPI(title="SHL Smart Assessment Recommender")

# Allow CORS (important for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/catalog_clean.csv")
FAISS_PATH = os.path.join(BASE_DIR, "../embeddings/vector_store.faiss")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------
# ✅ Input Schema
# ---------------------------------------------------------------------
class RequestData(BaseModel):
    query: str


# ---------------------------------------------------------------------
# ✅ Safe loader for FAISS + CSV + Model
# ---------------------------------------------------------------------
def load_all():
    try:
        print("🔹 Loading dataset, model and FAISS index...")
        df = pd.read_csv(DATA_PATH)
        model = SentenceTransformer(MODEL_NAME)
        index = faiss.read_index(FAISS_PATH)
        print("✅ Successfully loaded resources")
        return df, model, index
    except Exception as e:
        print(f"❌ Resource loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resource load error: {str(e)}")


# ---------------------------------------------------------------------
# ✅ Health check
# ---------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# ✅ Recommendation Endpoint
# ---------------------------------------------------------------------
@app.post("/recommend/recommend")
def recommend(request: RequestData):
    try:
        df, model, index = load_all()

        # Generate query embedding
        query_vector = model.encode([request.query])

        # Search top 5 nearest vectors
        distances, indices = index.search(query_vector, k=5)

        # Prepare results
        results = []
        for i, score in zip(indices[0], distances[0]):
            if 0 <= i < len(df):
                item = df.iloc[i].to_dict()
                item["similarity_score"] = float(score)
                results.append(item)

        return {"query": request.query, "recommendations": results}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"🔥 Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# ✅ Main entry (for local dev and Render)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
