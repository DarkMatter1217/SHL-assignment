from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="SHL Backend", version="1.0")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
df, index, embedder, model = None, None, None, None
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "..", "data", "catalog_clean.csv")
FAISS_PATH = os.path.join(ROOT, "..", "embeddings", "vector_store.faiss")
EMB_PATH = os.path.join(ROOT, "..", "embeddings", "embeddings.npy")

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend live 🚀"}

def load_all():
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
            raise ValueError("GEMINI_API_KEY not set")
        model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=key)
    print("✅ Resources loaded")

@app.post("/recommend/recommend")
def recommend(req: Query):
    load_all()
    emb = embedder.encode([req.query], convert_to_numpy=True)
    D, I = index.search(emb.astype("float32"), k=5)
    results = df.iloc[I[0]][["assessment_name", "category", "test_type", "url"]].to_dict(orient="records")
    return {"query": req.query, "recommendations": results}

@app.get("/")
def root():
    return {"message": "Use /recommend/recommend to query ✅"}

# ⚠️ NO uvicorn.run() here! Gunicorn handles it.
