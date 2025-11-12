# app_pages.py
"""
Streamlit multi-page Drug Predictor
- Login page → /?page=login
- Signup page → /?page=signup
- Predictor page → /?page=app (protected)
"""

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

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "users.db"))

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Drug Side Effect Predictor", layout="centered")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { color: #EAEAEA; background-color: #0f1720; }
h2, h3 { color: #FFFFFF; }
.card {
    background: #101922;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
}
.muted { color:#9fb0c9; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE FUNCTIONS ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
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
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
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
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash)

conn = init_db()

# ---------------- MODEL & PREDICTOR ----------------
SEVERITY_RULES = {
    "severe": ["death", "anaphyl", "coma", "hospital", "failure", "insufficiency", "liver"],
    "moderate": ["nausea", "vomit", "vomiting", "diarr", "headache", "dizzy", "rash", "pain"],
}

def estimate_severity(text: str) -> str:
    t = text.lower()
    for w in SEVERITY_RULES["severe"]:
        if w in t:
            return "Severe 🔴"
    for w in SEVERITY_RULES["moderate"]:
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
        effect = re.sub(r"[^\w\s,]", "", effect)
        parts = re.split(r",| and | with |;|\||/", effect)
        for p in parts:
            p = p.strip().capitalize()
            if len(p) > 1:
                formatted.append(p)
    return list(dict.fromkeys(formatted))

@st.cache_resource
def load_models_and_data():
    tfv = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "side_effect_model.pkl"))
    label_binarizer = joblib.load(os.path.join(MODELS_DIR, "mlb.pkl"))
    df = pd.read_csv(DATA_PATH)
    df["Medicine Name"] = df["Medicine Name"].astype(str)
    drug_list = df["Medicine Name"].dropna().unique().tolist()
    return tfv, xgb_model, label_binarizer, df, drug_list

tfv, xgb_model, label_binarizer, df, drug_list = load_models_and_data()

def get_predictions_for_drug(selected_drug):
    matches = df[df["Medicine Name"].str.lower() == selected_drug.lower()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    desc = clean_text(row.get("Composition", "")) + " " + clean_text(row.get("Uses", ""))
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
    excellent = row.get("Excellent Review %", 0)
    average = row.get("Average Review %", 0)
    if excellent > 50:
        review = "Excellent ✅"
    elif average > 50:
        review = "Average ⚠"
    else:
        review = "Poor ⚠️"
    return {"name": row["Medicine Name"], "groups": groups, "review": review}

# ---------------- SESSION ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# ---------------- QUERY PARAM ROUTER ----------------
def get_current_page():
    return st.query_params.get("page", "login")

def go_to_page(page_name):
    st.query_params["page"] = page_name
    st.rerun()

# ---------------- PAGE COMPONENTS ----------------
def header_nav():
    st.markdown("### 🔷 Drug Side Effect & Review Predictor")
    if st.session_state["authenticated"]:
        st.markdown(f"<div class='muted'>Logged in as {st.session_state['username']}</div>", unsafe_allow_html=True)
    st.markdown("---")

def login_page():
    st.markdown("## 🔐 Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pw")
    if st.button("Login"):
        if verify_user(conn, username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"Welcome {username}!")
            go_to_page("app")
        else:
            st.error("Invalid username or password")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Go to Signup"):
        go_to_page("signup")

def signup_page():
    st.markdown("## 📝 Signup")
    username = st.text_input("Choose username", key="signup_user")
    pw1 = st.text_input("Password", type="password", key="signup_pw1")
    pw2 = st.text_input("Confirm password", type="password", key="signup_pw2")
    if st.button("Create Account"):
        if not username or not pw1 or not pw2:
            st.warning("Fill all fields")
        elif pw1 != pw2:
            st.error("Passwords do not match")
        elif user_exists(conn, username):
            st.error("Username already exists")
        else:
            create_user(conn, username, pw1)
            st.success("Account created successfully!")
            go_to_page("login")
    if st.button("Back to Login"):
        go_to_page("login")

def predictor_page():
    if not st.session_state["authenticated"]:
        st.warning("Please login first.")
        if st.button("Go to Login"):
            go_to_page("login")
        return

    st.markdown("## 💊 Drug Predictor")
    query = st.text_input("Enter medicine name")
    suggestions = difflib.get_close_matches(query, drug_list, n=8, cutoff=0.30) if query else []
    chosen = st.selectbox("Choose from suggestions", ["-- None --"] + suggestions) if suggestions else None

    if st.button("Predict"):
        sel = chosen if chosen and chosen != "-- None --" else query
        if not sel:
            st.error("No drug selected")
        else:
            result = get_predictions_for_drug(sel)
            if not result:
                st.error("Drug not found in dataset")
            else:
                st.markdown(f"### 📌 {result['name']}")
                st.write(f"**Review:** {result['review']}")
                for cat, items in result["groups"].items():
                    if items:
                        emoji = "🟢" if cat == "mild" else "🟠" if cat == "moderate" else "🔴"
                        st.markdown(f"**{emoji} {cat.capitalize()} Side Effects**")
                        for eff, sev in items:
                            st.write(f"- {eff} ({sev})")

    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        go_to_page("login")

# ---------------- ROUTER ----------------
header_nav()
page = get_current_page()

if page == "login":
    login_page()
elif page == "signup":
    signup_page()
elif page == "app":
    predictor_page()
else:
    st.error("Page not found. Use ?page=login or ?page=signup or ?page=app")
