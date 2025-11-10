import requests
import pandas as pd
from tqdm import tqdm
import os

API_URL = "http://localhost:8000/recommend"
INPUT_FILE = r"test.xlsx"
OUTPUT_FILE = "Prabhjot_Singh.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

if INPUT_FILE.endswith(".xlsx"):
    df = pd.read_excel(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)

print("\n📂 Columns found in dataset:", df.columns.tolist())

JD_COLUMN = "Query"

if JD_COLUMN not in df.columns:
    raise ValueError(f"❌ '{JD_COLUMN}' column not found in dataset. Found columns: {df.columns.tolist()}")

print(f"✅ Using column for job description: '{JD_COLUMN}'")

results = []
print("\n🚀 Generating AI-powered recommendations...\n")

for i, row in tqdm(df.iterrows(), total=len(df)):
    jd_text = str(row[JD_COLUMN]).strip()
    jd_id = row.get("ID", i + 1)

    if not jd_text or jd_text.lower() in ["nan", "none"]:
        results.append({
            "JD_ID": jd_id,
            "Job Description": "",
            "Recommended Assessments": "No JD provided"
        })
        continue

    payload = {
        "query": jd_text,
        "top_k": 10,
        "rerank": True
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            recs = response.json().get("recommendations", [])
            rec_names = [r["assessment_name"] for r in recs]
            results.append({
                "JD_ID": jd_id,
                "Job Description": jd_text,
                "Recommended Assessments": ", ".join(rec_names)
            })
        else:
            print(f"⚠️ API error {response.status_code} for JD {jd_id}")
            results.append({
                "JD_ID": jd_id,
                "Job Description": jd_text,
                "Recommended Assessments": f"API Error {response.status_code}"
            })
    except Exception as e:
        print(f"❌ Request failed for JD {jd_id}: {e}")
        results.append({
            "JD_ID": jd_id,
            "Job Description": jd_text,
            "Recommended Assessments": "Request Error"
        })

submission_df = pd.DataFrame(results)
submission_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✅ Submission file generated successfully → {OUTPUT_FILE}")
print(f"📈 Total JDs processed: {len(results)}")
