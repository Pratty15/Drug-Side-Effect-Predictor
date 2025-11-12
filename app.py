# app.py - Classic, compact & readable Streamlit UI
import os
import re
import joblib
import pandas as pd
import difflib
import streamlit as st
from io import StringIO
from datetime import datetime

# -------- Paths (update if needed) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")

# -------- Page config ----------
st.set_page_config(page_title="Drug Side Effect Predictor", layout="centered")

# -------- Clean/classic CSS ----------
st.markdown(
    """
    <style>
    /* Page */
    .stApp { background: #0f1720; color: #e8eef6; }
    .block-container { padding: 28px 36px; max-width: 980px; }

    /* Titles */
    .title { font-size: 28px; font-weight:700; margin-bottom:4px; color: #ffffff; }
    .subtitle { color:#aebfd3; margin-bottom:18px; }

    /* Card */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: 10px;
        box-shadow: none;
    }

    /* Input area */
    .input-row { margin-bottom: 10px; }

    /* Result area: fixed width to avoid overflow */
    .result-card {
        background: #0b1220;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 18px;
        border-radius: 10px;
        max-width: 420px;
    }

    /* Headings inside result */
    .result-title { font-size:20px; font-weight:800; margin-bottom:4px; color:#fff; word-break:break-word; }
    .result-meta { color:#aebfd3; margin-bottom:12px; font-size:13px; }

    /* badges */
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; font-size:13px; margin-bottom:10px; }
    .green { background: rgba(45,212,191,0.10); color:#2DD4BF; border:1px solid rgba(45,212,191,0.14); }
    .orange { background: rgba(255,159,67,0.08); color:#FF9F43; border:1px solid rgba(255,159,67,0.12); }
    .red { background: rgba(255,99,132,0.06); color:#FF6384; border:1px solid rgba(255,99,132,0.10); }

    /* lists */
    ul.clean { padding-left: 18px; margin-top: 8px; }
    ul.clean li { margin-bottom: 6px; line-height:1.35; color:#dfeaf6; }

    /* small text */
    .muted { color:#9fb0c9; font-size:13px; }

    /* wrap long selectbox contents */
    .css-15tx938 { max-width: 620px; } /* Streamlit selectbox container tweak (may vary across versions) */

    /* ensure long inputs wrap */
    .stTextInput>div>div>input { color: #e6eef6; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------- Helpers ----------
SEVERITY_RULES = {
    'severe': ['death', 'anaphyl', 'coma', 'hospital', 'failure', 'insufficiency', 'liver'],
    'moderate': ['nausea', 'vomit', 'vomiting', 'diarr', 'headache', 'dizzy', 'rash', 'pain', 'nausea'],
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
        parts = re.split(r',| and | with |;|\||/', effect)
        for p in parts:
            p = p.strip().capitalize()
            if len(p) > 1:
                formatted.append(p)
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
    df['Medicine Name'] = df['Medicine Name'].astype(str)
    drug_list = df['Medicine Name'].dropna().unique().tolist()
    return df, drug_list

tfv, xgb_model, label_binarizer = load_models()
df, drug_list = load_data()

def get_predictions_for_drug(selected_drug):
    matches = df[df['Medicine Name'].str.lower() == selected_drug.lower()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    desc = clean_text(row.get('Composition', '')) + " " + clean_text(row.get('Uses', ''))
    vec = tfv.transform([desc])
    pred = xgb_model.predict(vec)
    raw = [label_binarizer.classes_[i] for i in range(len(pred[0])) if pred[0][i] == 1]
    formatted = format_side_effects(raw)
    groups = {"mild": [], "moderate": [], "severe": []}
    for eff in formatted:
        sev = estimate_severity(eff)
        if "Severe" in sev:
            groups["severe"].append((eff, sev))
        elif "Moderate" in sev:
            groups["moderate"].append((eff, sev))
        else:
            groups["mild"].append((eff, sev))
    excellent = row.get('Excellent Review %', 0)
    average = row.get('Average Review %', 0)
    if excellent > 50:
        review = "Excellent ✅"
    elif average > 50:
        review = "Average ⚠"
    else:
        review = "Poor ⚠️"
    return {"name": row['Medicine Name'], "raw": raw, "groups": groups, "review": review, "row": row.to_dict()}

# -------- Header ----------
st.markdown('<div class="title">🔷 Drug Side Effect & Review Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type a medicine name and view grouped side effects — classic, readable and demo-friendly.</div>', unsafe_allow_html=True)

# -------- Main layout: two columns, result on right with fixed width ----------
col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### 🔍 Search Medicine', unsafe_allow_html=True)

    query = st.text_input("Type medicine name", value="", placeholder="e.g. Paracetamol, Botox, StayHappi ...")
    suggestions = difflib.get_close_matches(query, drug_list, n=6, cutoff=0.30) if query else []
    chosen = None
    if suggestions:
        st.caption("Suggestions (choose one):")
        chosen = st.selectbox("", options=["-- choose --"] + suggestions, index=0)
    # Predict
    btn_predict = st.button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Dataset preview & tips (small sample)"):
        st.write("If suggestions are off, type more characters or check dataset for spelling.")
        st.dataframe(df[['Medicine Name']].drop_duplicates().head(8).reset_index(drop=True))

with col_right:
    # static container for result with fixed width to avoid overflow
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    result_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# -------- Handle predict ----------
selected_drug = None
if btn_predict:
    if chosen and chosen != "-- choose --":
        selected_drug = chosen
    elif query:
        exact = next((d for d in drug_list if d.lower() == query.lower()), None)
        if exact:
            selected_drug = exact
        elif suggestions:
            # if user typed and suggestions available, pick first suggestion but show that choice
            selected_drug = suggestions[0]
        else:
            selected_drug = None

    if not selected_drug:
        result_placeholder.error("❌ No matching drug found. Try different name or choose from suggestions.")
    else:
        pred = get_predictions_for_drug(selected_drug)
        if not pred:
            result_placeholder.error("❌ Drug found but prediction unavailable.")
        else:
            name = pred['name']
            review = pred['review']
            groups = pred['groups']
            now = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Build result HTML with safe wrapping
            header = f"<div class='result-title'>📍 {name}</div><div class='result-meta'>🗓 {now} &nbsp;&nbsp; • &nbsp;&nbsp; 🗣️ Review: <strong style='color:#e6eef6'>{review}</strong></div>"
            parts = header
            # mild
            if groups['mild']:
                parts += "<div style='margin-top:6px'><span class='pill green'>✅ Mild Side Effects</span><ul class='clean'>"
                for eff,_ in groups['mild']:
                    parts += f"<li>{eff} <span class='muted'>• Mild</span></li>"
                parts += "</ul></div>"
            if groups['moderate']:
                parts += "<div style='margin-top:6px'><span class='pill orange'>🟠 Moderate Side Effects</span><ul class='clean'>"
                for eff,_ in groups['moderate']:
                    parts += f"<li>{eff} <span class='muted'>• Moderate</span></li>"
                parts += "</ul></div>"
            if groups['severe']:
                parts += "<div style='margin-top:6px'><span class='pill red'>🔴 Severe Side Effects</span><ul class='clean'>"
                for eff,_ in groups['severe']:
                    parts += f"<li>{eff} <span class='muted'>• Severe</span></li>"
                parts += "</ul></div>"

            result_placeholder.markdown(f"<div class='card'>{parts}</div>", unsafe_allow_html=True)

            # prepare downloadable text
            txt = StringIO()
            txt.write(f"Drug: {name}\nReview: {review}\nGenerated: {now}\n\n")
            for grp_name, grp in (("Mild", groups['mild']), ("Moderate", groups['moderate']), ("Severe", groups['severe'])):
                if grp:
                    txt.write(f"{grp_name}:\n")
                    for eff,_ in grp:
                        txt.write(f"- {eff}\n")
                    txt.write("\n")
            download_text = txt.getvalue()
            # Show actions below card (copy/download)
            st.download_button("Download .txt", download_text, file_name=f"{name.replace(' ','_')}_prediction.txt", mime="text/plain")

# Footer small note
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Tip: For production, serve a React frontend with the FastAPI backend to get pixel-perfect layout and better control over responsive behavior.</div>", unsafe_allow_html=True)
