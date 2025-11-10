import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="SHL Backend", version="final")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df, index, embedder, model = None, None, None, None
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "..", "data", "catalog_clean.csv")
FAISS_PATH = os.path.join(ROOT, "..", "embeddings", "vector_store.faiss")

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend running fine 🚀"}

def load_resources():
    global df, index, embedder, model
    if df is None:
        df = pd.read_csv(DATA_PATH)
    if index is None:
        index = faiss.read_index(FAISS_PATH)
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if model is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables")
        model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=key)
    print("✅ Resources loaded successfully")

@app.post("/recommend/recommend")
def recommend(req: Query):
    load_resources()
    query_emb = embedder.encode([req.query], convert_to_numpy=True)
    D, I = index.search(query_emb.astype("float32"), k=5)
    results = df.iloc[I[0]][["assessment_name", "category", "test_type", "url"]].to_dict(orient="records")
    return {"query": req.query, "recommendations": results}

@app.get("/")
def root():
    return {"message": "Use /recommend/recommend to query recommendations ✅"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting app on 0.0.0.0:{port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
