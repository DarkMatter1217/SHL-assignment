# ==========================================================
#  File: evaluation.py
#  Purpose: Evaluate submission.csv against SHL ground truth
#  Metrics: Precision, Recall, F1, Jaccard (with fuzzy + slug normalization)
# ==========================================================

import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
import re

# ==========================================================
# 🔧 CONFIGURATION
# ==========================================================
GROUND_TRUTH_FILE = r"data/Gen_AI Dataset.xlsx"   # Ground truth dataset
PRED_FILE = r"submission.csv"                     # Your generated submission
FUZZY_THRESHOLD = 0.6                             # Match ratio threshold

# ==========================================================
# 🧠 LOAD DATASETS
# ==========================================================
df_truth = pd.read_excel(GROUND_TRUTH_FILE)
df_pred = pd.read_csv(PRED_FILE)

print("\n✅ Loaded files successfully!")
print(f"Ground Truth Columns: {df_truth.columns.tolist()}")
print(f"Predicted Columns: {df_pred.columns.tolist()}\n")

# Adjust column names
truth_col_jd = "Query"              # Ground truth JD
truth_col_assess = "Assessment_url" # Ground truth assessments (URLs)
pred_col_jd = "Job Description"     # Predicted JD
pred_col_recs = "Recommended Assessments"  # Predicted assessments

# ==========================================================
# 🧩 HELPER FUNCTIONS
# ==========================================================

def normalize_text_list(x):
    """Normalizes URLs or text names into slug format for fair comparison."""
    if not isinstance(x, str):
        return []
    # Split by commas or semicolons
    items = [i.strip().lower() for i in re.split(r"[,;]", x) if i.strip()]
    clean = []
    for i in items:
        # remove url prefix
        i = re.sub(r"https?://(www\.)?shl\.com/", "", i)
        # remove html / trailing slash
        i = i.replace(".html", "").strip().strip("/")
        # replace separators with spaces
        i = i.replace("-", " ").replace("_", " ")
        # remove extra words like 'test', 'assessment' for more flexibility
        i = re.sub(r"\b(test|assessment|assessments)\b", "", i).strip()
        clean.append(i)
    return clean

def fuzzy_match(a, b, threshold=FUZZY_THRESHOLD):
    """Returns True if two strings are similar enough."""
    return SequenceMatcher(None, a, b).ratio() >= threshold

# ==========================================================
# 🚀 EVALUATION
# ==========================================================
results = []

for i, row in tqdm(df_pred.iterrows(), total=len(df_pred)):
    jd_text = str(row[pred_col_jd]).strip().lower()
    pred_assess = normalize_text_list(row[pred_col_recs])

    # Find the same JD in truth file (case-insensitive)
    truth_row = df_truth[df_truth[truth_col_jd].str.lower().str.strip() == jd_text]
    if truth_row.empty:
        results.append({"JD": jd_text, "Precision": 0, "Recall": 0, "F1": 0, "Jaccard": 0})
        continue

    truth_assess = normalize_text_list(truth_row.iloc[0][truth_col_assess])

    if not truth_assess or not pred_assess:
        results.append({"JD": jd_text, "Precision": 0, "Recall": 0, "F1": 0, "Jaccard": 0})
        continue

    set_true = set(truth_assess)
    set_pred = set(pred_assess)

    # For debugging, show first few comparisons
    if i < 3:
        print(f"\n🔍 JD: {jd_text}")
        print("Truth:", set_true)
        print("Pred :", set_pred)

    # Fuzzy intersection count
    matches = 0
    matched_true = set()
    for p in set_pred:
        for t in set_true:
            if t in matched_true:
                continue
            if fuzzy_match(p, t):
                matches += 1
                matched_true.add(t)
                break

    intersection = matches
    union = len(set_true | set_pred)

    precision = intersection / len(set_pred) if len(set_pred) else 0
    recall = intersection / len(set_true) if len(set_true) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    jaccard = intersection / union if union else 0

    results.append({
        "JD": jd_text,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "Jaccard": round(jaccard, 3),
    })

# ==========================================================
# 📊 SUMMARY
# ==========================================================
eval_df = pd.DataFrame(results)
mean_prec = eval_df["Precision"].mean()
mean_rec = eval_df["Recall"].mean()
mean_f1 = eval_df["F1"].mean()
mean_jacc = eval_df["Jaccard"].mean()

print("\n📈 Overall Evaluation Results (with fuzzy + slug normalization):")
print(f"  🔹 Average Precision: {mean_prec:.3f}")
print(f"  🔹 Average Recall:    {mean_rec:.3f}")
print(f"  🔹 Average F1 Score:  {mean_f1:.3f}")
print(f"  🔹 Avg Jaccard Sim:   {mean_jacc:.3f}")

eval_df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
print("\n✅ Detailed results saved to 'evaluation_results.csv'")
