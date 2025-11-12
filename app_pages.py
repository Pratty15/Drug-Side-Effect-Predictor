# app_pages.py
"""
Elegant multi-page Streamlit app:
- /?page=login   -> Login page (elegant)
- /?page=signup  -> Signup page (elegant)
- /?page=app     -> Protected predictor
Run: streamlit run app_pages.py
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
st.set_page_config(page_title="Drug Predictor • Auth", layout="centered", initial_sidebar_state="collapsed")

# ---------------- STYLES: Elegant & Classic ----------------
st.markdown(
    """
    <style>
    :root {
      --bg:#f6f8fa;
      --card:#ffffff;
      --muted:#6b7280;
      --accent:#0f62fe;
      --accent-2:#06b6d4;
    }
    /* page background */
    .stApp { background: linear-gradient(180deg,#e6eefb 0%, #f7fbff 100%); color: #071126; }
    /* center container tweaks */
    .block-container { padding-top: 28px; padding-bottom: 28px; max-width: 960px; }

    /* header */
    .brand { font-weight:800; font-size:20px; color: #071126; margin-bottom:6px; }
    .tag { color:var(--muted); font-size:13px; margin-bottom:18px; }

    /* card */
    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 22px;
      box-shadow: 0 6px 20px rgba(15, 38, 76, 0.06);
      border: 1px solid rgba(15, 38, 76, 0.04);
    }

    /* form layout */
    .form-row { margin-bottom: 12px; }
    input[type="text"], input[type="password"] {
      border: 1px solid #d1d5db !important;
      padding: 10px 12px !important;
      border-radius: 8px !important;
      width: 100%;
      box-sizing: border-box;
    }

    /* big CTA */
    .btn {
      background: linear-gradient(90deg,var(--accent), var(--accent-2));
      color: white;
      padding: 10px 14px;
      border-radius: 10px;
      font-weight:700;
      border: none;
      cursor: pointer;
      box-shadow: 0 6px 18px rgba(15,98,254,0.12);
    }
    .btn:disabled { opacity:0.6; cursor:not-allowed; box-shadow:none; }

    .muted { color:var(--muted); font-size:13px; }
    .link { color:var(--accent); font-weight:700; cursor:pointer; text-decoration:underline; }

    .pill {
      display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; font-size:12px;
    }

    /* result card tweaks */
    .result-card { background: #ffffff; border-radius:10px; padding:14px; border:1px solid rgba(15,38,76,0.04); }
    .result-title { font-weight:800; font-size:18px; margin-bottom:6px; }
    ul.clean { padding-left:18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TEXT NOT NULL
        );"""
    )
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
        c.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, pw_hash, created_at),
        )
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

# ---------------- MODEL & PREDICTOR HELPERS ----------------
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

# ---------------- ROUTING ----------------
def get_current_page():
    return st.query_params.get("page", "login")

def go_to_page(page_name):
    st.query_params["page"] = page_name
    st.rerun()

# ---------------- UI: Elegant Pages ----------------
def top_header():
    st.markdown("<div class='brand'>Drug Side Effect Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='tag'>Secure demo • Login to access the predictor</div>", unsafe_allow_html=True)

def login_page():
    top_header()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Login", unsafe_allow_html=True)
    st.write("", unsafe_allow_html=True)

    # Inputs inside columns to center nicely
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uname = st.text_input("Username", key="login_user")
        pw = st.text_input("Password", type="password", key="login_pw")
        show_pw = st.checkbox("Show password", key="login_show_pw")
        if show_pw and pw:
            st.write(f"**Password:** `{pw}`")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        # Buttons
        if st.button("Login", key="login_btn"):
            if not uname or not pw:
                st.warning("Please enter username and password.")
            else:
                if verify_user(conn, uname, pw):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = uname
                    st.success(f"Welcome, {uname}!")
                    go_to_page("app")
                else:
                    st.error("Invalid username or password.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="muted">No account? <span class="link" id="toSignup">Create one</span></div>', unsafe_allow_html=True)
        # JS-less "link": use button to navigate
        if st.button("Create account", key="goto_signup_btn"):
            go_to_page("signup")
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    top_header()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Create an account", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        new_user = st.text_input("Choose username", key="signup_user")
        new_pw = st.text_input("Password", type="password", key="signup_pw")
        new_pw2 = st.text_input("Confirm password", type="password", key="signup_pw2")
        show_pw = st.checkbox("Show passwords", key="signup_show_pw")
        if show_pw:
            if new_pw:
                st.write(f"**Password:** `{new_pw}`")
            if new_pw2:
                st.write(f"**Confirm:** `{new_pw2}`")

        # Simple password strength (visual hint)
        strength_msg = "Too short"
        if new_pw and len(new_pw) >= 8:
            score = 1
            if re.search(r"\d", new_pw):
                score += 1
            if re.search(r"[A-Z]", new_pw):
                score += 1
            if score == 1:
                strength_msg = "Weak"
            elif score == 2:
                strength_msg = "Okay"
            else:
                strength_msg = "Strong"
        if new_pw:
            st.markdown(f"<div class='muted'>Password strength: <strong>{strength_msg}</strong></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Create account", key="signup_btn"):
            if not new_user or not new_pw or not new_pw2:
                st.warning("Please fill all fields.")
            elif new_pw != new_pw2:
                st.error("Passwords do not match.")
            elif user_exists(conn, new_user):
                st.error("Username already exists.")
            elif len(new_pw) < 6:
                st.error("Password too short (min 6 chars).")
            else:
                created = create_user(conn, new_user, new_pw)
                if created:
                    st.success("Account created — you can now login.")
                    go_to_page("login")
                else:
                    st.error("Could not create account. Try another username.")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Back to Login", key="signup_back"):
            go_to_page("login")
    st.markdown("</div>", unsafe_allow_html=True)

def predictor_page():
    top_header()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💊 Drug Predictor", unsafe_allow_html=True)
    if not st.session_state["authenticated"]:
        st.warning("Please login to access the predictor.")
        if st.button("Go to Login"):
            go_to_page("login")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(f"<div class='muted'>Signed in as <strong>{st.session_state['username']}</strong></div>", unsafe_allow_html=True)
    query = st.text_input("Enter medicine name", key="predict_query")
    suggestions = difflib.get_close_matches(query, drug_list, n=8, cutoff=0.30) if query else []
    chosen = st.selectbox("Choose from suggestions", ["-- None --"] + suggestions) if suggestions else None

    if st.button("Predict", key="predict_btn"):
        sel = chosen if chosen and chosen != "-- None --" else query
        if not sel:
            st.error("No drug selected.")
        else:
            result = get_predictions_for_drug(sel)
            if not result:
                st.error("Drug not found in dataset.")
            else:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-title'>📌 {result['name']}</div>", unsafe_allow_html=True)
                st.write("**Review:**", result["review"])
                for cat, items in result["groups"].items():
                    if items:
                        emoji = "🟢" if cat == "mild" else "🟠" if cat == "moderate" else "🔴"
                        st.markdown(f"**{emoji} {cat.capitalize()} Side Effects**")
                        st.markdown("<ul class='clean'>", unsafe_allow_html=True)
                        for eff, sev in items:
                            st.markdown(f"<li>{eff} <span class='muted'>• {sev}</span></li>", unsafe_allow_html=True)
                        st.markdown("</ul>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        go_to_page("login")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ROUTER ----------------
# Load models/data once (cached earlier)
@st.cache_resource
def ensure_models_loaded():
    # Already loaded above via load_models_and_data() call; ensure keys exist
    return True

# Ensure models and data exist before mapping pages (this caches inside load_models_and_data)
tfv, xgb_model, label_binarizer, df, drug_list = load_models_and_data()

# Header & routing
page = get_current_page()

if page == "login":
    login_page()
elif page == "signup":
    signup_page()
elif page == "app":
    predictor_page()
else:
    # fallback
    st.error("Page not found. Use ?page=login or ?page=signup or ?page=app")
