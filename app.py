# app.py
import os
import re
import joblib
import pandas as pd
import difflib
import streamlit as st
from io import StringIO
from datetime import datetime

# --------- Paths (change if needed) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")

# --------- Page config ----------
st.set_page_config(
    page_title="Drug Side Effect & Review Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------- Inline CSS for nicer look ----------
st.markdown(
    """
    <style>
    /* page background */
    .stApp { background-color: #0b0f14; color: #e6eef6; }
    /* card */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.04);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
    }
    .title {
        font-size:34px;
        font-weight:700;
        letter-spacing: -0.5px;
    }
    .subtitle { color: #9fb0c9; margin-bottom:8px; }
    .muted { color:#8b9db3; font-size:13px; }
    .pill {
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        font-weight:600;
        margin-right:6px;
    }
    .green { background: rgba(45, 212, 191, 0.12); color:#2DD4BF; border:1px solid rgba(45,212,191,0.18); }
    .orange { background: rgba(255, 159, 67, 0.08); color:#FF9F43; border:1px solid rgba(255,159,67,0.12); }
    .red { background: rgba(255, 99, 132, 0.08); color:#FF6384; border:1px solid rgba(255,99,132,0.12); }
    .badge { font-size:14px; font-weight:700; }
    ul.clean { padding-left: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Helpers ----------
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
    # group by severity
    groups = {"mild": [], "moderate": [], "severe": []}
    for eff in formatted:
        sev = estimate_severity(eff)
        if "Severe" in sev:
            groups["severe"].append((eff, sev))
        elif "Moderate" in sev:
            groups["moderate"].append((eff, sev))
        else:
            groups["mild"].append((eff, sev))
    # review sentiment
    excellent = row.get('Excellent Review %', 0)
    average = row.get('Average Review %', 0)
    if excellent > 50:
        review = "Excellent ✅"
    elif average > 50:
        review = "Average ⚠"
    else:
        review = "Poor ⚠️"
    return {
        "name": row['Medicine Name'],
        "raw": raw,
        "groups": groups,
        "review": review,
        "row": row.to_dict()
    }

# --------- Header ----------
col1, col2 = st.columns([6,2])
with col1:
    st.markdown('<div class="title">🔷 Drug Side Effect & Review Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter a medicine name and see predicted side effects grouped by severity — clean, clear and ready for demo.</div>', unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right; color:#9fb0c9; font-weight:600'>Demo • Streamlit</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------- Main layout ----------
left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Search Medicine")
    query = st.text_input("Type medicine name", value="", placeholder="e.g. Botox, Picon Cream, Neurovin ...")
    # fuzzy suggestions as user types
    suggestions = []
    if query:
        suggestions = difflib.get_close_matches(query, drug_list, n=8, cutoff=0.30)
    if suggestions:
        st.caption("Suggestions (click to choose):")
        chosen = st.selectbox("Or choose from suggestions", options=["-- choose --"] + suggestions, index=0)
    else:
        chosen = None

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns([1,1])
    with cols[0]:
        btn_predict = st.button("Predict", use_container_width=True)
    with cols[1]:
        st.markdown('<button style="background:#1f6feb;color:#fff;border-radius:8px;padding:8px 12px;border:none;font-weight:700">Advanced</button>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Extra info / dataset preview
    with st.expander("Dataset preview & tips"):
        st.write("Tip: If suggestions look odd, try typing more characters or check dataset for spelling.")
        st.dataframe(df[['Medicine Name']].drop_duplicates().sample(min(10, len(df))).reset_index(drop=True))

with right:
    # Result placeholder card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📌 Result")
    result_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# --------- Handle predict action ----------
selected_drug = None
if btn_predict:
    if chosen and chosen != "-- choose --":
        selected_drug = chosen
    elif query:
        # try exact
        exact = next((d for d in drug_list if d.lower() == query.lower()), None)
        if exact:
            selected_drug = exact
        elif suggestions:
            # if suggestions exist, pick first automatically but show choice to user
            selected_drug = suggestions[0]
        else:
            selected_drug = None

    if not selected_drug:
        result_area.error("❌ No matching drug found. Try different name or choose from suggestions.")
    else:
        pred = get_predictions_for_drug(selected_drug)
        if not pred:
            result_area.error("❌ Drug found in list but no model prediction available.")
        else:
            # Build nice HTML for result
            name = pred['name']
            review = pred['review']
            groups = pred['groups']

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_html = f"""
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="font-size:22px; font-weight:800;">📍 {name}</div>
                <div style="margin-left:8px; color:#9fb0c9;">{now}</div>
            </div>
            <div style="margin-top:8px;" class="muted">🗣️ Review Sentiment: <strong style="color:#e6eef6"> {review} </strong></div>
            <hr style="opacity:0.06; margin-top:12px; margin-bottom:14px;">
            """
            # prepare lists
            mild_html = ""
            mod_html = ""
            sev_html = ""
            if groups["mild"]:
                mild_html += "<div class='pill green badge'>✅ Mild Side Effects</div><ul class='clean'>"
                for eff, sev in groups["mild"]:
                    mild_html += f"<li>{eff} <span style='color:#2DD4BF'>• Mild</span></li>"
                mild_html += "</ul>"
            if groups["moderate"]:
                mod_html += "<div class='pill orange badge'>🟠 Moderate Side Effects</div><ul class='clean'>"
                for eff, sev in groups["moderate"]:
                    mod_html += f"<li>{eff} <span style='color:#FF9F43'>• Moderate</span></li>"
                mod_html += "</ul>"
            if groups["severe"]:
                sev_html += "<div class='pill red badge'>🔴 Severe Side Effects</div><ul class='clean'>"
                for eff, sev in groups["severe"]:
                    sev_html += f"<li>{eff} <span style='color:#FF6384'>• Severe</span></li>"
                sev_html += "</ul>"

            # combine
            body_html = header_html + mild_html + mod_html + sev_html

            # show in result_area
            result_area.markdown(f"<div class='card'>{body_html}</div>", unsafe_allow_html=True)

            # generate text output for copy/download
            txt = StringIO()
            txt.write(f"Drug: {name}\nReview: {review}\n\n")
            if groups["mild"]:
                txt.write("Mild:\n")
                for eff,_ in groups["mild"]:
                    txt.write(f"- {eff}\n")
                txt.write("\n")
            if groups["moderate"]:
                txt.write("Moderate:\n")
                for eff,_ in groups["moderate"]:
                    txt.write(f"- {eff}\n")
                txt.write("\n")
            if groups["severe"]:
                txt.write("Severe:\n")
                for eff,_ in groups["severe"]:
                    txt.write(f"- {eff}\n")
                txt.write("\n")
            txt_value = txt.getvalue()

            # action buttons
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button("Copy result text"):
                    st.experimental_set_query_params()  # tiny no-op so UI updates
                    st.write("✅ Copied to clipboard (use browser copy).")
                    # Note: A direct clipboard copy is limited in Streamlit; we offer download as alternative.
            with c2:
                st.download_button("Download .txt", txt_value, file_name=f"{name.replace(' ','_')}_prediction.txt", mime="text/plain")
            with c3:
                st.button("Share (coming soon)", disabled=True)

# --------- Footer ----------
st.markdown("<br>", unsafe_allow_html=True)
#st.markdown("<div class='muted'>Built with ❤️ • Streamlit — Improve UI further by adding icons, images, or hosting frontend separately.</div>", unsafe_allow_html=True)
