# frontend/app.py
import os
import streamlit as st
import requests
import pandas as pd
import random

st.set_page_config(page_title=" SHL Smart Assessment Recommender", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #FFB74D;'>🌍 SHL Smart Assessment Recommender</h1>
    <p style='text-align: center; font-size:17px; color:#CCCCCC;'>
        AI-powered recommendation system for SHL assessments.<br>
        Uses FAISS + Gemini intelligence to rank the most relevant assessments automatically 🚀
    </p>
    <hr style='border: 1px solid #555555;'>
    """,
    unsafe_allow_html=True
)

query = st.text_area(
    "Enter Job Description or Hiring Requirement:",
    placeholder="e.g. Hiring a software engineer with strong Python and teamwork skills...",
    height=200
)

if st.button("✨ Getting Recommendations"):
    if not query.strip():
        st.warning("Please enter a job description first.")
    else:
        top_k = random.randint(5, 10)
        api_url = "https://shl-genai-backend.onrender.com"

        # Read key from Streamlit secrets (must exist)
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
        except Exception as e:
            st.error("Gemini API key not found in Streamlit secrets. Add it to .streamlit/secrets.toml")
            gemini_key = None

        payload = {
            "query": query,
            "top_k": top_k,
            "rerank": True,
            "gemini_api_key": gemini_key
        }

        with st.spinner("Generating Gemini-powered recommendations..."):
            try:
                res = requests.post(f"{api_url}/recommend", json=payload, timeout=60)
            except Exception as e:
                st.error(f"Request failed: {e}")
                res = None

            if res is None:
                pass
            else:
                if res.status_code != 200:
                    st.error(f"❌ API Error {res.status_code}: {res.text}")
                else:
                    data = res.json().get("recommendations", [])
                    if not data:
                        st.info("No recommendations found.")
                    else:
                        df = pd.DataFrame(data)
                        df["Rank"] = range(1, len(df) + 1)
                        df["Link"] = df["url"].apply(lambda u: f'<a href="{u}" target="_blank">🔗 Open</a>' if u else "")
                        display_df = df[["Rank", "assessment_name", "Link"]].rename(columns={"assessment_name": "Assessment Name"})
                        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                        st.success(f" Gemini reranked and returned {len(df)} assessments automatically.")
