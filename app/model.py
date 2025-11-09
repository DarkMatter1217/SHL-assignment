import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import google.generativeai as genai
import json
from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN_CSV = os.path.join(ROOT, "data", "catalog_clean.csv")
EMB_NPY = os.path.join(ROOT, "embeddings", "embeddings.npy")
FAISS_IDX = os.path.join(ROOT, "embeddings", "vector_store.faiss")

class Recommender:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.df = pd.read_csv(CLEAN_CSV)
        self.embedder = SentenceTransformer(embed_model)
        self.index = faiss.read_index(FAISS_IDX)
        self.use_gpu = faiss.get_num_gpus() > 0
        if self.use_gpu:
            print("⚡ Using GPU FAISS index")

    def retrieve(self, query: str, top_k=20):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb.astype("float32"), top_k)
        results = []
        for i, idx in enumerate(I[0]):
            if idx >= 0 and idx < len(self.df):
                row = self.df.iloc[idx]
                results.append({
                    "assessment_name": row.get("assessment_name", ""),
                    "description": row.get("description", ""),
                    "category": row.get("category", ""),
                    "test_type": row.get("test_type", ""),
                    "url": row.get("url", ""),
                    "score": float(D[0][i])
                })
        return sorted(results, key=lambda x: x["score"])

    def rerank_with_gemini(self, query: str, candidates: List[Dict[str, Any]], top_k=10):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("⚠️ GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=api_key)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("⚠️ GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        cand_text = "\n".join(
            [f"- {c['assessment_name']} ({c.get('test_type','')}) : {c['description'][:250]} [URL: {c['url']}]" for c in candidates]
        )
        prompt = f"""
You are an AI assistant helping recruiters select the most relevant SHL assessments.

Job Query:
{query}

Assessments:
{cand_text}

Task:
Return a JSON list of top {top_k} most relevant assessment URLs, ranked by suitability for the job.
Only output valid JSON array.
"""
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()    
            urls = json.loads(text)
        except Exception as e:
            print("Gemini parsing error:", e)
            urls = [c["url"] for c in candidates[:top_k]]

        url_map = {c["url"]: c for c in candidates}
        ranked = [url_map[u] for u in urls if u in url_map]
        for c in candidates:
            if c not in ranked:
                ranked.append(c)
        return ranked[:top_k]

_recommender = None
def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender
