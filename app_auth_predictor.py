# app_auth_predictor.py
import os
import re
import sqlite3
import joblib
import pandas as pd
import difflib
import streamlit as st
import bcrypt
from io import StringIO
from datetime import datetime

# ----------------- CONFIG: update paths if needed -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")
DB_PATH = os.path.join(BASE_DIR, "users.db")  # sqlite db for credentials

# ----------------- Streamlit page config -----------------
st.set_page_config(page_title="Drug Predictor (Auth demo)", layout="centered")

# ----------------- Simple CSS -----------------
st.markdown(
    """
    <style>
      .center {text-align:center}
      .small {font-size:12px; color: #9fb0c9}
      .card { padding: 12px; border-radius:10px; background: #0b1220; border:1px solid rgba(255,255,255,0.04) }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- AUTH DB helpers -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn

def user_exists(conn, username):
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    return c.fetchone() is not None

def create_user(conn, username, password):
    pw_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    created_at = datetime.utcnow().isoformat()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                  (username, pw_hash, created_at))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(conn, username, password):
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if not row:
        return False
    stored_hash = row[0]
    try:
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    except Exception:
        return False

# init DB connection once
conn = init_db()

# ----------------- Model & predictor helpers (same logic as earlier) -----------------
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
def load_models_and_data():
    # load your vectorizer, model, mlb and data — adjust filenames if different
    tfv = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'side_effect_model.pkl'))
    label_binarizer = joblib.load(os.path.join(MODELS_DIR, 'mlb.pkl'))
    df = pd.read_csv(DATA_PATH)
    df['Medicine Name'] = df['Medicine Name'].astype(str)
    drug_list = df['Medicine Name'].dropna().unique().tolist()
    return tfv, xgb_model, label_binarizer, df, drug_list

tfv, xgb_model, label_binarizer, df, drug_list = load_models_and_data()

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

# ----------------- Session helpers -----------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'login_error' not in st.session_state:
    st.session_state['login_error'] = ""

def do_logout():
    st.session_state['authenticated'] = False
    st.session_state['username'] = None

# ----------------- UI: login/signup panel -----------------
st.title("🔷 Drug Side Effect Predictor — Auth Demo")
st.write("Signup / Login to access the predictor. (Local demo using SQLite + bcrypt)")

col1, col2 = st.columns(2)

with col1:
    st.header("🔑 Login")
    login_user = st.text_input("Username (login)", key="login_user")
    login_pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Login"):
        if not login_user or not login_pw:
            st.warning("Enter username & password")
        else:
            ok = verify_user(conn, login_user, login_pw)
            if ok:
                st.session_state['authenticated'] = True
                st.session_state['username'] = login_user
                st.success(f"Logged in as {login_user}")
            else:
                st.error("Invalid username or password")

    if st.session_state['authenticated']:
        if st.button("Logout"):
            do_logout()
            st.info("Logged out")

with col2:
    st.header("📝 Signup")
    new_user = st.text_input("Choose a username", key="signup_user")
    new_pw = st.text_input("Choose a password", type="password", key="signup_pw")
    new_pw2 = st.text_input("Confirm password", type="password", key="signup_pw2")
    if st.button("Create account"):
        if not new_user or not new_pw or not new_pw2:
            st.warning("Fill all signup fields")
        elif new_pw != new_pw2:
            st.error("Passwords do not match")
        elif user_exists(conn, new_user):
            st.error("Username already exists")
        else:
            created = create_user(conn, new_user, new_pw)
            if created:
                st.success("Account created — you can now login")
            else:
                st.error("Could not create account (username may already exist)")

st.markdown("---")

# ----------------- Protected predictor UI -----------------
if not st.session_state['authenticated']:
    st.info("Please login or signup to use the predictor. (This demo stores credentials locally in users.db)")
else:
    st.markdown(f"### Welcome, **{st.session_state['username']}**")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Search & Predict")
    query = st.text_input("Enter medicine name (search or type)", key="main_query")
    suggestions = difflib.get_close_matches(query, drug_list, n=8, cutoff=0.30) if query else []
    chosen = None
    if suggestions:
        chosen = st.selectbox("Choose from suggestions (optional)", options=["-- none --"] + suggestions, index=0)
    if st.button("Predict for selected name"):
        sel = None
        if chosen and chosen != "-- none --":
            sel = chosen
        elif query:
            exact = next((d for d in drug_list if d.lower() == query.lower()), None)
            if exact:
                sel = exact
            elif suggestions:
                sel = suggestions[0]
        if not sel:
            st.error("No matching drug found. Try different name or choose from suggestions.")
        else:
            pred = get_predictions_for_drug(sel)
            if not pred:
                st.error("Found drug but model returned no prediction.")
            else:
                name = pred['name']
                review = pred['review']
                groups = pred['groups']
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"**📍 {name}**")
                st.markdown(f"_{now}_")
                st.write("**Review:**", review)
                if groups['mild']:
                    st.markdown("**✅ Mild Side Effects**")
                    for eff,_ in groups['mild']:
                        st.write("- " + eff)
                if groups['moderate']:
                    st.markdown("**🟠 Moderate Side Effects**")
                    for eff,_ in groups['moderate']:
                        st.write("- " + eff)
                if groups['severe']:
                    st.markdown("**🔴 Severe Side Effects**")
                    for eff,_ in groups['severe']:
                        st.write("- " + eff)
                # download text
                txt = StringIO()
                txt.write(f"Drug: {name}\nReview: {review}\nGenerated: {now}\n\n")
                for grp_name, grp in (("Mild", groups['mild']), ("Moderate", groups['moderate']), ("Severe", groups['severe'])):
                    if grp:
                        txt.write(f"{grp_name}:\n")
                        for eff,_ in grp:
                            txt.write(f"- {eff}\n")
                        txt.write("\n")
                st.download_button("Download result as .txt", txt.getvalue(), file_name=f"{name.replace(' ','_')}_prediction.txt")
    st.markdown("</div>", unsafe_allow_html=True)
