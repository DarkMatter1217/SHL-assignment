import os
import pandas as pd
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# ✅ Lazy load placeholders
df = None
index = None
embeddings = None
embedder = None
model = None

# ✅ FastAPI app setup
app = FastAPI(title="SHL GenAI Backend", version="1.0")

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Model paths
DATA_PATH = "data/data.csv"
FAISS_PATH = "faiss_store.index"
EMB_PATH = "embeddings.npy"

# ✅ Health check route
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "LangChain Gemini backend live 🚀"}

# ✅ Pydantic input model
class QueryRequest(BaseModel):
    query: str

# ✅ Function to load models and data only once
def load_resources():
    global df, index, embeddings, embedder, model

    if df is None:
        print("📂 Loading dataset...")
        df = pd.read_csv(DATA_PATH)

    if index is None:
        print("🧠 Loading FAISS index...")
        index = faiss.read_index(FAISS_PATH)

    if embeddings is None:
        print("📦 Loading embeddings...")
        embeddings = np.load(EMB_PATH)

    if embedder is None:
        print("🔍 Loading SentenceTransformer...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if model is None:
        print("🤖 Initializing Gemini model...")
        GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
        if not GEMINI_KEY:
            raise ValueError("❌ Missing GEMINI_API_KEY in Render environment variables")
        model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_KEY)

    print("✅ All resources loaded successfully.")

# ✅ Recommendation endpoint
@app.post("/recommend/recommend")
def recommend(req: QueryRequest):
    load_resources()

    query_vector = embedder.encode([req.query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k=5)

    results = df.iloc[indices[0]][["name", "description", "link"]].to_dict(orient="records")

    return {"query": req.query, "recommendations": results}

# ✅ Root route
@app.get("/")
def root():
    return {"message": "Backend live ✅ Use /recommend/recommend to query."}

# ⚠️ Note: Removed uvicorn.run() to prevent double boot under Gunicorn
