# app.py
import os
import re
import joblib
import pandas as pd
import difflib
import streamlit as st

# --- Paths: adjust if your project structure is different ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")

st.set_page_config(page_title="Drug Side Effect Predictor", layout="centered")

# --- Helpers & Caching ---
SEVERITY_RULES = {
    'severe': ['death', 'anaphyl', 'coma', 'hospital', 'failure'],
    'moderate': ['nausea', 'vomit', 'vomiting', 'diarr', 'headache', 'dizzy', 'rash', 'pain'],
}

def estimate_severity(text: str) -> str:
    t = text.lower()
    for w in SEVERITY_RULES['severe']:
        if w in t:
            return "Severe 🔴"
    for w in SEVERITY_RULES['moderate']:
        if w in t:
            return "Moderate 🟠"
    return "Mild 🟢"

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def format_side_effects(side_effects):
    formatted = []
    for effect in side_effects:
        effect = re.sub(r'[^\w\s,]', '', effect)
        parts = re.split(r',| and | with ', effect)
        for p in parts:
            p = p.strip().capitalize()
            if len(p) > 1:
                formatted.append(p)
    # preserve order, remove duplicates
    return list(dict.fromkeys(formatted))

@st.cache_resource
def load_models():
    tfv = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'side_effect_model.pkl'))
    label_binarizer = joblib.load(os.path.join(MODEL_DIR, 'mlb.pkl'))
    return tfv, xgb_model, label_binarizer

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # ensure medicine name column exists and is str
    df['Medicine Name'] = df['Medicine Name'].astype(str)
    drug_list = df['Medicine Name'].dropna().unique().tolist()
    return df, drug_list

tfv, xgb_model, label_binarizer = load_models()
df, drug_list = load_data()

# --- UI ---
st.title("🔹 Drug Side Effect & Review Predictor")
st.write("Enter a medicine name and get predicted side effects (grouped by severity) and review sentiment.")

col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_input("🔍 Drug name", value="", placeholder="Type drug name like 'botox'...")
with col2:
    if st.button("Predict"):
        query = user_input.strip()
        if not query:
            st.warning("Please enter a drug name.")
        else:
            # exact match
            exact_match = next((d for d in drug_list if d.lower() == query.lower()), None)
            suggestions = []
            if not exact_match:
                suggestions = difflib.get_close_matches(query, drug_list, n=6, cutoff=0.35)

            if not exact_match and not suggestions:
                st.error("❌ Drug not found. Try a different name.")
            elif not exact_match and suggestions:
                st.info("Did you mean:")
                choice = st.selectbox("Choose from suggestions", options=["-- select --"] + suggestions)
                if choice and choice != "-- select --":
                    selected_drug = choice
                else:
                    st.stop()
            else:
                selected_drug = exact_match

            # Fetch row
            matches = df[df['Medicine Name'].str.lower() == selected_drug.lower()]
            if matches.empty:
                st.error("❌ Drug not found in dataset.")
                st.stop()

            row = matches.iloc[0]
            # predict using model (reuse your logic)
            description = clean_text(row.get('Composition', '')) + " " + clean_text(row.get('Uses', ''))
            description_vec = tfv.transform([description])
            prediction = xgb_model.predict(description_vec)
            side_effects = [label_binarizer.classes_[i] for i in range(len(prediction[0])) if prediction[0][i] == 1]
            clean_effects = format_side_effects(side_effects)

            # group by severity
            mild, moderate, severe = [], [], []
            for eff in clean_effects:
                sev = estimate_severity(eff)
                if "Severe" in sev:
                    severe.append(f"{eff} — {sev}")
                elif "Moderate" in sev:
                    moderate.append(f"{eff} — {sev}")
                else:
                    mild.append(f"{eff} — {sev}")

            # review sentiment
            excellent = row.get('Excellent Review %', 0)
            average = row.get('Average Review %', 0)
            if excellent > 50:
                review = "Excellent ✅"
            elif average > 50:
                review = "Average ⚠"
            else:
                review = "Poor ⚠️"

            # Display results
            st.subheader(f"📌 {selected_drug}")
            st.markdown(f"**🗣️ Review Sentiment:** {review}")

            if mild:
                st.markdown("### ✅ Mild Side Effects")
                for m in mild:
                    st.write(f"- {m}")
            if moderate:
                st.markdown("### 🟠 Moderate Side Effects")
                for m in moderate:
                    st.write(f"- {m}")
            if severe:
                st.markdown("### 🔴 Severe Side Effects")
                for m in severe:
                    st.write(f"- {m}")

# Footer tips
st.markdown("---")
st.caption("Tip: If suggestions repeat or are odd, try increasing `cutoff` in the code's `difflib.get_close_matches` call.")

