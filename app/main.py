from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pandas as pd
import gc
import threading
import os
import time

# === Absolute safe paths (Render environment independent) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

FAISS_PATH = os.path.join(ROOT_DIR, "embeddings", "vector_store.faiss")
DATA_PATH = os.path.join(ROOT_DIR, "data", "catalog_clean.csv")

print("🔍 FAISS_PATH =", FAISS_PATH)
print("🔍 DATA_PATH =", DATA_PATH)

# === FastAPI setup ===
app = FastAPI(title="SHL GenAI Backend", version="1.0")

# === CORS setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Globals ===
index = None
dataset = None


class QueryModel(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Backend is live 🚀"}


# === Load FAISS + dataset safely ===
def load_all():
    global index, dataset
    try:
        print(f"🔹 Loading dataset from: {DATA_PATH}")
        dataset = pd.read_csv(DATA_PATH)
        print(f"✅ Loaded dataset with {len(dataset)} rows")

        print(f"🔹 Loading FAISS index from: {FAISS_PATH}")
        index = faiss.read_index(FAISS_PATH, faiss.IO_FLAG_MMAP)
        print("✅ FAISS index loaded successfully!")

    except Exception as e:
        print(f"❌ Error while loading FAISS/data: {e}")
    finally:
        gc.collect()


@app.on_event("startup")
async def startup_event():
    print("🚀 Starting backend...")
    threading.Thread(target=load_all).start()
    gc.collect()


@app.post("/recommend/recommend")
def recommend(query: QueryModel):
    global index, dataset

    # Retry wait loop
    retries = 0
    while (index is None or dataset is None) and retries < 10:
        print("⏳ Waiting for FAISS + dataset to load...")
        time.sleep(3)
        retries += 1

    if index is None or dataset is None:
        return {"error": "Model not loaded yet. Please retry after 30 seconds."}

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        query_vector = model.encode([query.query])
        D, I = index.search(query_vector, 5)

        recommendations = []
        for idx, score in zip(I[0], D[0]):
            rec = dataset.iloc[idx].to_dict()
            rec["similarity_score"] = float(score)
            recommendations.append(rec)

        return {"query": query.query, "recommendations": recommendations}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
