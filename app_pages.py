import os
import re
import sqlite3
import joblib
import pandas as pd
import difflib
import streamlit as st
import bcrypt
from datetime import datetime
import plotly.graph_objects as go

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
h2, h3, h4 { color: #FFFFFF; }
.card {
    background: #101922;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
}
.muted { color:#9fb0c9; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
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

# ---------------- MODEL & DATA ----------------
SEVERITY_RULES = {
    "severe": ["death", "anaphyl", "coma", "hospital", "failure", "insufficiency", "liver", "seizure"],
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
    return tfv, xgb_model, label_binarizer, df, df["Medicine Name"].dropna().unique().tolist()

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
            groups["severe"].append(eff)
        elif "Moderate" in sev:
            groups["moderate"].append(eff)
        else:
            groups["mild"].append(eff)

    excellent = row.get("Excellent Review %", 0)
    average = row.get("Average Review %", 0)
    poor = 100 - (excellent + average)
    review_label = "Excellent ✅" if excellent > 50 else "Average ⚠" if average > 50 else "Poor ⚠️"

    return {
        "name": row["Medicine Name"],
        "groups": groups,
        "review": review_label,
        "image_url": row.get("Image URL", ""),
        "excellent": excellent,
        "average": average,
        "poor": poor
    }

# ---------------- SESSION ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

# ---------------- ROUTER ----------------
def get_current_page():
    return st.query_params.get("page", "login")

def go_to_page(page_name):
    st.query_params["page"] = page_name
    st.rerun()

# ---------------- PAGES ----------------
def header_nav():
    st.markdown("### 🔷 Drug Side Effect & Review Predictor")
    if st.session_state["authenticated"]:
        st.markdown(f"<div class='muted'>Logged in as {st.session_state['username']}</div>", unsafe_allow_html=True)
    st.markdown("---")

def login_page():
    st.markdown("## 🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(conn, username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"Welcome {username}!")
            go_to_page("app")
        else:
            st.error("Invalid username or password")
    if st.button("Go to Signup"):
        go_to_page("signup")

def signup_page():
    st.markdown("## 📝 Signup")
    username = st.text_input("Choose username")
    pw1 = st.text_input("Password", type="password")
    pw2 = st.text_input("Confirm password", type="password")
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
    suggestions = difflib.get_close_matches(query, drug_list, n=8, cutoff=0.3) if query else []
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
                col1, col2 = st.columns([1, 2])

                with col1:
                    if result["image_url"] and str(result["image_url"]).startswith("http"):
                        st.image(result["image_url"], caption=result["name"], use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/300x200?text=No+Image", caption="No image available", use_container_width=True)
                    st.write(f"**Review:** {result['review']}")
                    st.markdown(f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"### 📌 {result['name']}")
                    # Severity Chart
                    counts = {
                        "Mild": len(result["groups"]["mild"]),
                        "Moderate": len(result["groups"]["moderate"]),
                        "Severe": len(result["groups"]["severe"])
                    }
                    fig_bar = go.Figure(go.Bar(
                        x=list(counts.keys()), y=list(counts.values()),
                        marker_color=["#2ecc71", "#f1c40f", "#e74c3c"]
                    ))
                    fig_bar.update_layout(title="Side-effects by severity", xaxis_title="Severity", yaxis_title="Count", height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Review Pie Chart
                    fig_pie = go.Figure(go.Pie(
                        labels=["Excellent", "Average", "Poor"],
                        values=[result["excellent"], result["average"], result["poor"]],
                        marker_colors=["#f1c40f", "#3498db", "#e74c3c"],
                        hole=0.3
                    ))
                    fig_pie.update_layout(title="Review distribution", height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Side effects below
                for cat, items in result["groups"].items():
                    if items:
                        emoji = "🟢" if cat == "mild" else "🟠" if cat == "moderate" else "🔴"
                        st.markdown(f"**{emoji} {cat.capitalize()} Side Effects**")
                        for eff in items:
                            st.write(f"- {eff}")

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
