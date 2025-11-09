import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_CSV = os.path.join(ROOT, "data", "dataset.csv")
CLEAN_CSV = os.path.join(ROOT, "data", "catalog_clean.csv")
EMB_DIR = os.path.join(ROOT, "embeddings")
EMB_PATH = os.path.join(EMB_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(EMB_DIR, "vector_store.faiss")

os.makedirs(EMB_DIR, exist_ok=True)

def load_and_prepare():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Missing dataset at {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

    rename_map = {
        "Individual Test Solution": "assessment_name",
        "Remote Testing": "description",
        "Adaptive/IRT": "category",
        "Test Type": "test_type",
        "URL": "url"
    }
    df.rename(columns=rename_map, inplace=True)
    df.fillna("", inplace=True)

    def clean_url(u):
        u = str(u).strip()
        if not u:
            return ""
        if u.startswith("http"):
            return u
        if u.startswith("/"):
            return "https://www.shl.com" + u
        if "shl.com" not in u:
            return "https://www.shl.com/solutions/products/" + u.replace(" ", "-").lower()
        return u

    df["url"] = df["url"].apply(clean_url)

    df["text_for_embed"] = (
        df["assessment_name"].astype(str)
        + " | " + df["description"].astype(str)
        + " | " + df["category"].astype(str)
        + " | " + df["test_type"].astype(str)
    ).str.strip()

    df.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Cleaned dataset saved to: {CLEAN_CSV}")
    return df

def build_embeddings():
    df = load_and_prepare()
    texts = df["text_for_embed"].tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, embeddings.astype("float32"))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_PATH)
    print(f"✅ Embeddings and FAISS index saved to {EMB_PATH}, {FAISS_PATH}")

if __name__ == "__main__":
    build_embeddings()
