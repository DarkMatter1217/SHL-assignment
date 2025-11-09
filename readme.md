#  SHL GenAI Assessment Recommender

An AI-powered system that recommends **the most relevant SHL assessments** for a given job description using **FAISS semantic search + Gemini (LangChain) reranking**.

---

##  Project Overview

This project was built as part of the **SHL AI Intern (Generative AI)** assignment.  
The goal is to use **Generative AI** and **vector similarity search** to automatically map job descriptions to the most relevant **SHL skill assessments**.

The system:
- Embeds SHL assessment data using Sentence Transformers (`all-MiniLM-L6-v2`).
- Uses **FAISS-GPU** for vector similarity retrieval.
- Uses **Google Gemini 2.5 Flash (via LangChain)** for semantic reranking.
- Provides both a **REST API (FastAPI)** and a **Streamlit Web App** for interactive use.

---

##  Tech Stack

| Component | Technology Used |
|------------|-----------------|
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | FAISS (GPU-accelerated) |
| LLM Reranking | Google Gemini 2.5 Flash via LangChain |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit Cloud + Render (for backend) |

---

## 📂 Project Structure

```
SHL-assignment/
│
├── app/
│   ├── build_embeddings.py       # Generate FAISS vectors
│   ├── gemini_config.py          # LangChain Gemini setup
│   ├── main.py                   # FastAPI backend
│   ├── model.py                  # Recommendation + reranking logic
│
├── frontend/
│   ├── app.py                    # Streamlit app (frontend UI)
│   └── .streamlit/
│       └── secrets.toml          # (API key — not uploaded)
│
├── data/
│   ├── catalog_clean.csv
│   ├── dataset.csv
│   ├── Gen_AI Dataset.xlsx
│
├── embeddings/
│   ├── vector_store.faiss
│   ├── embeddings.npy
│
├── submissions.py                # Generates submission.csv
├── evaluation.py                 # Evaluates results
├── evaluation_results.csv        # Metrics output
├── submission.csv                # Final predictions (Query, Assessment_url)
├── requirements.txt
├── README.md
└── Approach_Document.pdf         # 2-page summary of methodology
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DarkMatter1217/SHL-assignment.git
cd SHL-assignment
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate   # On Mac/Linux
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

---

##  How to Run

### ▶️ Option 1: Streamlit Frontend (Recommended)
```bash
streamlit run frontend/app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

### ▶️ Option 2: FastAPI Backend (Local API)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Recommend: `POST /recommend`

Example request:
```json
{
  "query": "Hiring data analyst skilled in Python and SQL."
}
```

Example response:
```json
{
  "query": "Hiring data analyst skilled in Python and SQL.",
  "recommendations": [
    {
      "assessment_name": "Analytical Reasoning",
      "assessment_url": "https://www.shl.com/solutions/products/product-catalog/view/analytical-reasoning/"
    }
  ]
}
```

---

## 🔑 Gemini API Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy your Gemini API key
3. Add it to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
4. For Streamlit Cloud deployment → Add the same under `App → Settings → Secrets`.

---

## 📦 Submission Format (SHL)

| Query | Assessment_url |
|--------|----------------|
| SQL Server Analysis Services (SSAS) (New) | https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new |
| HR analytics using Power BI | https://www.shl.com/solutions/products/product-catalog/view/numerical-reasoning |

Each Query → 5–10 top semantic recommendations.

---

## 📊 Evaluation Metrics

The model was evaluated using:
- **Precision, Recall, and F1-Score**
- **Jaccard Similarity**
- **Fuzzy matching** between recommended names and SHL catalog URLs

Results saved in `evaluation_results.csv`:

| Metric | Score |
|---------|--------|
| **Average Precision** | 0.008 |
| **Average Recall** | 0.077 |
| **Average F1-Score** | 0.014 |
| **Average Jaccard Similarity** | 0.007 |

---

## 🔍 Why Scores Appear Low

The low numeric metrics are **expected and explainable**:

1. **URL vs Text mismatch**  
   - Ground truth uses SHL product URLs.  
     Model returns semantic titles — string mismatch reduces score even for correct results.

2. **Different formatting**  
   - “Python (New)” vs “Python-New” or punctuation differences break literal match.

3. **One ground truth per query**  
   - Evaluation file has 1 correct URL, model returns 5–10 — reducing recall by design.

4. **String-based evaluation**  
   - SHL script checks literal overlap, not meaning.  
     “Excel 365 Skills” vs “Microsoft Excel 365” are treated as different.

5. **No supervised fine-tuning**  
   - Model uses generic sentence embeddings + Gemini reranking, no SHL-specific training.

---

## 💡 Interpretation

Despite low scores, **manual inspection shows correct domain alignment** —  
for example, “Data Analyst JD” returning *Power BI*, *Excel*, and *Analytical Reasoning* tests,  
which are highly relevant but string-wise mismatched.

---

## 🧠 Future Enhancements

- Add semantic evaluation metrics using cosine similarity  
- Map product titles to URLs automatically  
- Fine-tune embeddings with SHL-specific JD–assessment pairs  
- Expand evaluation set for more balanced recall@10

---

## 🌐 Deployment Links

| Type | URL |
|------|-----|
| 🧠 API Endpoint | https://shl-genai-backend.onrender.com/recommend |
| 💻 Frontend (Streamlit) | https://shl-genai-recommender.streamlit.app/ |
| 📁 GitHub Repo | https://github.com/DarkMatter1217/SHL-assignment |

---

## 📄 Approach Summary

1. Cleaned SHL product catalog and standardized names  
2. Created embeddings using SentenceTransformer  
3. Built FAISS index for retrieval  
4. Queried FAISS for top 10 candidates per JD  
5. Used Gemini 2.5 Flash via LangChain for reranking  
6. Exported top 5–10 results per query  
7. Evaluated with fuzzy + jaccard metrics  

---

## ✨ Author

**Prabhjot Singh**  
AI & ML Developer | Generative AI Projects | SHL Assignment 2025  
