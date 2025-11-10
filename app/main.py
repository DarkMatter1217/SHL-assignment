from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pandas as pd
import gc
import threading
import os

# Paths for FAISS and CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "../embeddings/vector_store.faiss")
DATA_PATH = os.path.join(BASE_DIR, "../data/catalog_clean.csv")

# FastAPI setup
app = FastAPI(title="SHL GenAI Backend", version="1.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Streamlit or frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
index = None
dataset = None

# Input model
class QueryModel(BaseModel):
    query: str

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Backend is live 🚀"}

# Lazy load FAISS + dataset
def load_all():
    global index, dataset
    try:
        print(f"🔹 Trying to load {DATA_PATH} and {FAISS_PATH}")
        dataset = pd.read_csv(DATA_PATH)

        # Use mmap mode to reduce memory usage
        index = faiss.read_index(FAISS_PATH, faiss.IO_FLAG_MMAP)

        print("✅ FAISS index + dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
    finally:
        gc.collect()

# Background load on startup
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_all).start()
    gc.collect()

# Recommend endpoint
@app.post("/recommend/recommend")
def recommend(query: QueryModel):
    global index, dataset

    # Check if data loaded
    if index is None or dataset is None:
        return {"error": "Model not loaded yet. Please wait a few seconds and try again."}

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Convert query into embedding
        query_vector = model.encode([query.query])
        D, I = index.search(query_vector, 5)

        # Get top recommendations
        recommendations = []
        for idx, score in zip(I[0], D[0]):
            rec = dataset.iloc[idx].to_dict()
            rec["similarity_score"] = float(score)
            recommendations.append(rec)

        return {"query": query.query, "recommendations": recommendations}

    except Exception as e:
        return {"error": str(e)}

# Gunicorn port setup (Render requirement)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
