# scripts/predict.py
"""
Standalone prediction module for Drug Side Effect Predictor.
Can be run directly from the command line for testing.
"""

import os
import re
import joblib
import pandas as pd
import difflib

# --- Setup paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")

# Load models and vectorizers
tfv = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "side_effect_model.pkl"))
label_binarizer = joblib.load(os.path.join(MODEL_DIR, "mlb.pkl"))

# Load dataset
df = pd.read_csv(DATA_PATH)
drug_list = df["Medicine Name"].dropna().unique().tolist()


# ---------- Severity Rules ----------
SEVERITY_RULES = {
    "severe": ["death", "anaphyl", "coma", "hospital", "failure", "insufficiency", "liver", "seizure", "cardiac"],
    "moderate": ["nausea", "vomit", "vomiting", "diarr", "headache", "dizzy", "rash", "pain", "fever", "swelling"],
}


def estimate_severity(text: str) -> str:
    """Classify a side-effect string as Severe/Moderate/Mild."""
    t = text.lower()
    for w in SEVERITY_RULES["severe"]:
        if w in t:
            return "Severe 🔴"
    for w in SEVERITY_RULES["moderate"]:
        if w in t:
            return "Moderate 🟠"
    return "Mild 🟢"


def clean_text(s):
    """Lowercase, strip URLs and special characters."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_side_effects(side_effects):
    """Clean and deduplicate side effect strings."""
    formatted = []
    for effect in side_effects:
        effect = re.sub(r"[^\w\s,]", "", effect)
        parts = re.split(r",| and | with ", effect)
        for p in parts:
            p = p.strip().capitalize()
            if len(p) > 1:
                formatted.append(p)
    return list(dict.fromkeys(formatted))


def predict_side_effects(drug_name: str) -> list | None:
    """Predict side effects for a given drug name. Returns list or None."""
    matches = df[df["Medicine Name"].str.lower() == drug_name.lower()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    description = clean_text(row["Composition"]) + " " + clean_text(row["Uses"])
    description_vec = tfv.transform([description])
    prediction = xgb_model.predict(description_vec)
    return [label_binarizer.classes_[i] for i in range(len(prediction[0])) if prediction[0][i] == 1]


def predict_review_label(drug_name: str) -> str | None:
    """Predict review sentiment for a given drug name."""
    matches = df[df["Medicine Name"].str.lower() == drug_name.lower()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    excellent = row.get("Excellent Review %", 0)
    average = row.get("Average Review %", 0)
    if excellent > 50:
        return "Excellent ✅"
    elif average > 50:
        return "Average ⚠"
    else:
        return "Poor ⚠️"


def suggest_drug_names(user_input: str, n: int = 5, cutoff: float = 0.4) -> list:
    """Return fuzzy-matched drug name suggestions."""
    return difflib.get_close_matches(user_input, drug_list, n=n, cutoff=cutoff)


# ----------- Main Program -----------
if __name__ == "__main__":
    print("🔹 Drug Side Effect & Review Predictor 🔹")

    while True:
        user_input = input("\n🔍 Enter a drug name (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("\n👋 Exiting. Stay healthy and safe!")
            break

        exact_match = next((d for d in drug_list if d.lower() == user_input.lower()), None)

        if not exact_match:
            suggestions = suggest_drug_names(user_input)
            if not suggestions:
                print("❌ Drug not found. Try a different name.")
                continue

            print("\n🤔 Did you mean:")
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s}")

            choice = input("👉 Choose a number (or press Enter to try again): ").strip()
            if not choice.isdigit() or int(choice) not in range(1, len(suggestions) + 1):
                print("🔁 Let's try again...")
                continue
            selected_drug = suggestions[int(choice) - 1]
        else:
            selected_drug = exact_match

        side_effects = predict_side_effects(selected_drug)
        review_label = predict_review_label(selected_drug)

        if side_effects is None:
            print("❌ Drug not found in dataset. Try again.")
        else:
            matches = df[df["Medicine Name"].str.lower() == selected_drug.lower()]
            row = matches.iloc[0]

            print("\n✅ Prediction Result")
            print("─" * 40)
            print(f"📌 Drug Name: {selected_drug}\n")

            print("💊 Medicine Details:")
            print("─" * 40)
            if pd.notna(row.get("Manufacturer")) and str(row.get("Manufacturer")).strip():
                print(f"🏭 Manufacturer: {row.get('Manufacturer')}")
            if pd.notna(row.get("Composition")) and str(row.get("Composition")).strip():
                print(f"🧪 Composition: {row.get('Composition')}")
            if pd.notna(row.get("Uses")) and str(row.get("Uses")).strip():
                print(f"💡 Uses: {row.get('Uses')}")
            print()

            clean_effects = format_side_effects(side_effects)
            mild, moderate, severe = [], [], []

            for effect in clean_effects:
                severity = estimate_severity(effect)
                if "Severe" in severity:
                    severe.append(f"  • {effect} ({severity})")
                elif "Moderate" in severity:
                    moderate.append(f"  • {effect} ({severity})")
                else:
                    mild.append(f"  • {effect} ({severity})")

            print("💊 Predicted Side Effects:\n")
            if severe:
                print("🔴 Severe Side Effects:")
                print("\n".join(severe), "\n")
            if moderate:
                print("🟠 Moderate Side Effects:")
                print("\n".join(moderate), "\n")
            if mild:
                print("🟢 Mild Side Effects:")
                print("\n".join(mild), "\n")

            print(f"🗣️ Review Sentiment: {review_label}")
