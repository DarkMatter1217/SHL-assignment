from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import faiss
import numpy as np
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from sentence_transformers import SentenceTransformer

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables.")
print("🔑 Gemini API key loaded successfully.")

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "catalog_clean.csv")
EMB_PATH = os.path.join(ROOT, "embeddings", "embeddings.npy")
FAISS_PATH = os.path.join(ROOT, "embeddings", "vector_store.faiss")

app = FastAPI(
    title="SHL Recommender with LangChain Gemini",
    description="FAISS + LangChain Gemini (2.5-flash) backend for AI-powered assessment recommendations",
    version="v5.0"
)

print("🧠 Loading FAISS, data, and embedding model...")
df = pd.read_csv(DATA_PATH)
embeddings = np.load(EMB_PATH)
index = faiss.read_index(FAISS_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ All resources loaded successfully.")

def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return 0.0 if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return 0.0

def rerank_with_gemini(query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Uses LangChain's ChatGoogleGenerativeAI (Gemini 2.5 Flash)
    to rerank SHL assessments based on JD relevance.
    Enforces valid JSON output.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=GEMINI_KEY,
        )

        prompt = f"""
You are an AI recruitment assistant.
Given a job description and a list of SHL assessments,
select the {top_k} most relevant ones in order of importance.

Job Description:
{query}

Assessments (JSON list):
{json.dumps([c['assessment_name'] for c in candidates])}

Return your answer **strictly as a valid JSON array only** —
no explanations, no text before or after.
Example:
["Assessment A", "Assessment B", "Assessment C"]
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        if "[" in text and "]" in text:
            text = text[text.find("["): text.rfind("]") + 1]

        try:
            ranked_names = json.loads(text)
        except Exception:
            print("⚠️ Gemini output not valid JSON, using fallback FAISS results.")
            return candidates[:top_k]

        ranked = [c for c in candidates if c["assessment_name"] in ranked_names]
        for c in candidates:
            if c not in ranked:
                ranked.append(c)

        return ranked[:top_k]

    except Exception as e:
        print(f"⚠️ Gemini rerank failed: {e}")
        return candidates[:top_k]

class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    rerank: Optional[bool] = True

@app.get("/health")
def health():
    return {"status": "ok", "message": "LangChain Gemini backend live 🚀"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Main API endpoint:
    - Searches FAISS for closest embeddings
    - Optionally reranks with Gemini (via LangChain)
    - Returns SHL formatted recommendations
    """
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Empty job description.")

        q_emb = embedder.encode([req.query], convert_to_numpy=True)
        D, I = index.search(q_emb.astype("float32"), max(20, req.top_k * 2))

        results = []
        for i, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(df):
                continue
            row = df.iloc[idx]
            results.append({
                "assessment_name": str(row.get("assessment_name", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "category": str(row.get("category", "")).strip(),
                "test_type": str(row.get("test_type", "")).strip(),
                "url": str(row.get("url", "")).strip(),
                "score": safe_float(D[0][i])
            })

        if req.rerank:
            results = rerank_with_gemini(req.query, results, req.top_k)

        final_output = [
            {"assessment_name": r["assessment_name"], "url": r["url"]}
            for r in results[:req.top_k]
        ]

        return {"query": req.query, "recommendations": final_output}

    except Exception as e:
        print("❌ Internal server error:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
