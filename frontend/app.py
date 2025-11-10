import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

<<<<<<< HEAD
=======
# ---------------- CONFIG ----------------
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
DATA_PATH = "./data/catalog_clean.csv"
EMB_PATH = "./embeddings/embeddings.npy"
FAISS_PATH = "./embeddings/vector_store.faiss"

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY" or st.secrets["GEMINI_API_KEY"] 

<<<<<<< HEAD
=======
# ---------------- MODEL INIT ----------------
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.3
)

<<<<<<< HEAD
=======
# ---------------- LOAD DATA ----------------
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
@st.cache_resource
def load_resources():
    df = pd.read_csv(DATA_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(FAISS_PATH)
    return df, embedder, index

df, embedder, index = load_resources()

<<<<<<< HEAD
=======
# ---------------- FAISS SEARCH ----------------
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
def search_faiss(query, k=10):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), k)
    return df.iloc[I[0]]

<<<<<<< HEAD
def rerank_with_gemini(query, candidates_df, top_k=5):
    """Use Gemini to rerank FAISS results with safe text cleaning."""
=======
# ---------------- GEMINI RERANK ----------------
def rerank_with_gemini(query, candidates_df, top_k=5):
    """Use Gemini to rerank FAISS results with safe text cleaning."""

    # Clean all text safely
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
    def safe_text(x):
        if isinstance(x, str):
            return x.strip()
        return ""

    candidates_df["assessment_name"] = candidates_df["assessment_name"].apply(safe_text)
    candidates_df["test_type"] = candidates_df["test_type"].apply(safe_text)
    candidates_df["description"] = candidates_df["description"].apply(safe_text)

    assessments_text = "\n".join([
        f"{i+1}. {row['assessment_name']} ({row['test_type']}) - {row['description'][:120]}"
        for i, (_, row) in enumerate(candidates_df.iterrows())
    ])

    prompt = PromptTemplate.from_template("""
    You are an SHL Assessment Recommendation AI.

    User Query: "{query}"

    Below are possible SHL assessments:
    {assessments}

    Your task:
    - Rank the **top {top_k} most relevant assessments** for this query.
    - Return ONLY a valid JSON array like:
    [
<<<<<<< HEAD
      {"name": "Data Analysis Test", "category": "Cognitive"},
      {"name": "Leadership Potential Test", "category": "Personality"}
=======
      {{"name": "Data Analysis Test", "category": "Cognitive"}},
      {{"name": "Leadership Potential Test", "category": "Personality"}}
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
    ]
    """)

    chain = prompt | llm

    try:
        response = chain.invoke({
            "query": query,
            "assessments": assessments_text,
            "top_k": top_k
        })

        text = response.content.strip().replace("```json", "").replace("```", "")
        start, end = text.find("["), text.rfind("]") + 1
        json_str = text[start:end]
        parsed = json.loads(json_str)

        ranked_names = [p["name"] for p in parsed if "name" in p]
        name_to_cat = {p["name"]: p.get("category", "N/A") for p in parsed}

        ranked = candidates_df[candidates_df["assessment_name"].isin(ranked_names)].copy()
        ranked["Category"] = ranked["assessment_name"].map(name_to_cat)
        return ranked.head(top_k)

    except Exception as e:
        st.error(f"⚠️ Gemini parsing error: {str(e)}")
        candidates_df["Category"] = candidates_df["test_type"]
        return candidates_df.head(top_k)

<<<<<<< HEAD
=======
# ---------------- STREAMLIT UI ----------------
>>>>>>> c93658ad633f2ee548ddef180e6de8f12487d230
st.set_page_config(page_title="SHL GenAI Recommender", layout="centered")
st.title("SHL GenAI — Smart Assessment Recommender")
st.write("Enter a **job role** or **skill**, and get SHL’s most relevant assessments.")

query = st.text_area("💬 Enter job description or skill", placeholder="e.g. Data Analyst with leadership and reasoning skills")

if st.button("🔍 Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Analyzing ..."):
            faiss_results = search_faiss(query, k=10)
            final_results = rerank_with_gemini(query, faiss_results, top_k=5)

        st.success("Top Recommended Assessments:")
        for _, row in final_results.iterrows():
            with st.expander(f"✅ {row['assessment_name']} ({row['Category']})"):
                st.write(row["description"])
                if "url" in row and isinstance(row["url"], str) and row["url"].startswith("http"):
                    st.markdown(f"[🔗 View Assessment]({row['url']})")
