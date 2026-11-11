# app.py — Drug Side Effect Predictor (Overhauled)
# ═══════════════════════════════════════════════════
# Premium Streamlit app with dark glassmorphic UI,
# drug comparison, search history, export, and more.
# ═══════════════════════════════════════════════════

import os
import re
import io
import sqlite3
import joblib
import pandas as pd
import difflib
import streamlit as st
import bcrypt
from datetime import datetime, timezone
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64

# Optional faster fuzzy matching
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False


# ════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "Medicine_Details.csv")
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "users.db"))

st.set_page_config(
    page_title="Drug Side Effect Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════
# PREMIUM CSS DESIGN SYSTEM
# ════════════════════════════════════════════════════
def inject_css():
    """Inject the full premium CSS design system."""

    # Encode background image
    bg_b64 = ""
    bg_path = os.path.join(BASE_DIR, "dataset-cover.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()

    bg_rule = ""
    if bg_b64:
        bg_rule = f"""
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(135deg, rgba(10,12,20,0.92) 0%, rgba(18,24,38,0.88) 50%, rgba(10,12,20,0.94) 100%),
                url("data:image/jpg;base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_rule = """
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0a0c14 0%, #121826 50%, #0a0c14 100%);
        }
        """

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Base ────────────────────────────────── */
    *, *::before, *::after {{ font-family: 'Inter', sans-serif !important; }}

    {bg_rule}

    [data-testid="stHeader"] {{
        background: rgba(10,12,20,0.7) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(108,99,255,0.15) !important;
    }}

    /* ── Sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] > div:first-child {{
        background: rgba(14,18,30,0.85) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(108,99,255,0.12) !important;
    }}
    [data-testid="stSidebar"] .stRadio > label {{
        color: #8892b0 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
    }}
    [data-testid="stSidebar"] .stRadio > div {{
        gap: 2px !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label {{
        padding: 10px 16px !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        color: #c4d2e3 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        cursor: pointer !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: rgba(108,99,255,0.12) !important;
        color: #fff !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
        background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(108,99,255,0.1)) !important;
        color: #fff !important;
        border-left: 3px solid #6C63FF !important;
        font-weight: 600 !important;
    }}

    /* ── Main content container ───────────────── */
    .block-container {{
        background: rgba(14,18,30,0.6) !important;
        backdrop-filter: blur(16px) !important;
        padding: 28px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(108,99,255,0.08) !important;
    }}

    /* ── Glass Card ───────────────────────────── */
    .glass-card {{
        background: rgba(20,26,44,0.65);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(108,99,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 18px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108,99,255,0.15);
        border-color: rgba(108,99,255,0.3);
    }}

    /* ── Stat Card ────────────────────────────── */
    .stat-card {{
        background: linear-gradient(135deg, rgba(20,26,44,0.7), rgba(30,38,60,0.5));
        backdrop-filter: blur(16px);
        border: 1px solid rgba(108,99,255,0.15);
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108,99,255,0.2);
        border-color: rgba(108,99,255,0.4);
    }}
    .stat-number {{
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }}
    .stat-label {{
        font-size: 13px;
        color: #8892b0;
        font-weight: 500;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    /* ── Animated Header ─────────────────────── */
    .app-header {{
        text-align: center;
        padding: 32px 20px 18px;
        margin-bottom: 10px;
    }}
    .app-header h1 {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #a78bfa 40%, #00D68F 80%, #6C63FF 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }}
    .app-header p {{
        color: #8892b0;
        font-size: 15px;
        font-weight: 400;
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* ── User badge ──────────────────────────── */
    .user-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0,214,143,0.1);
        border: 1px solid rgba(0,214,143,0.25);
        border-radius: 20px;
        padding: 6px 16px;
        float: right;
    }}
    .user-badge .dot {{
        width: 8px; height: 8px;
        background: #00D68F;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(0,214,143,0.6);
        animation: pulse 2s ease infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    .user-badge span {{
        color: #c4f0dd;
        font-size: 13px;
        font-weight: 500;
    }}

    /* ── Severity Badges ─────────────────────── */
    .sev-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 3px 4px;
        transition: transform 0.2s ease;
    }}
    .sev-badge:hover {{ transform: scale(1.05); }}
    .sev-mild {{
        background: rgba(0,214,143,0.12);
        color: #00D68F;
        border: 1px solid rgba(0,214,143,0.25);
    }}
    .sev-moderate {{
        background: rgba(255,176,32,0.12);
        color: #FFB020;
        border: 1px solid rgba(255,176,32,0.25);
    }}
    .sev-severe {{
        background: rgba(255,107,107,0.12);
        color: #FF6B6B;
        border: 1px solid rgba(255,107,107,0.25);
    }}

    /* ── Section Titles ──────────────────────── */
    .section-title {{
        font-size: 18px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 20px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(108,99,255,0.2);
    }}

    /* ── Medicine detail label ────────────────── */
    .detail-label {{
        font-size: 12px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 2px;
    }}
    .detail-value {{
        font-size: 15px;
        color: #e2e8f0;
        font-weight: 500;
        margin-bottom: 14px;
        line-height: 1.6;
    }}

    /* ── Search history chips ─────────────────── */
    .chip {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(108,99,255,0.1);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 20px;
        padding: 5px 14px;
        margin: 3px 4px;
        color: #a78bfa;
        font-size: 13px;
        font-weight: 500;
        cursor: default;
        transition: all 0.2s ease;
    }}
    .chip:hover {{
        background: rgba(108,99,255,0.2);
        border-color: rgba(108,99,255,0.4);
        color: #c4b5fd;
    }}

    /* ── Interaction warning ──────────────────── */
    .interaction-warning {{
        background: rgba(255,176,32,0.08);
        border: 1px solid rgba(255,176,32,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #FFB020;
        font-size: 14px;
    }}
    .interaction-warning strong {{ color: #ffd166; }}

    /* ── Buttons ──────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, #6C63FF, #8b5cf6) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(108,99,255,0.25) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108,99,255,0.4) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* ── Inputs ───────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stTextInput input,
    .stTextArea textarea,
    input[type="text"],
    input[type="password"],
    textarea,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    [data-baseweb="textarea"] textarea {{
        background: rgba(20,26,44,0.8) !important;
        background-color: rgba(20,26,44,0.8) !important;
        border: 1px solid rgba(108,99,255,0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        caret-color: #6C63FF !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }}
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    input[type="text"]:focus,
    input[type="password"]:focus,
    textarea:focus,
    [data-baseweb="input"] input:focus,
    [data-baseweb="base-input"] input:focus {{
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
        background: rgba(20,26,44,0.9) !important;
    }}
    /* Input container / wrapper */
    [data-baseweb="input"],
    [data-baseweb="base-input"] {{
        background-color: rgba(20,26,44,0.8) !important;
        border-color: rgba(108,99,255,0.2) !important;
        border-radius: 10px !important;
    }}
    /* Input placeholder */
    ::placeholder {{
        color: rgba(136,146,176,0.6) !important;
    }}
    /* Input labels */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label {{
        color: #8892b0 !important;
        font-weight: 500 !important;
    }}

    /* ── Select box ──────────────────────────── */
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"],
    [data-baseweb="select"] > div {{
        background: rgba(20,26,44,0.8) !important;
        background-color: rgba(20,26,44,0.8) !important;
        border: 1px solid rgba(108,99,255,0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }}
    /* Select dropdown menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {{
        background: rgba(14,18,30,0.95) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(108,99,255,0.2) !important;
        border-radius: 10px !important;
    }}
    [data-baseweb="menu"] li,
    ul[role="listbox"] li {{
        color: #c4d2e3 !important;
    }}
    [data-baseweb="menu"] li:hover,
    ul[role="listbox"] li:hover {{
        background: rgba(108,99,255,0.15) !important;
    }}

    /* ── Dataframe styling ────────────────────── */
    .stDataFrame {{ border-radius: 12px !important; overflow: hidden !important; }}

    /* ── Dividers ─────────────────────────────── */
    hr {{ border-color: rgba(108,99,255,0.12) !important; }}

    /* ── Comparison columns ───────────────────── */
    .compare-col {{
        background: rgba(20,26,44,0.5);
        border: 1px solid rgba(108,99,255,0.1);
        border-radius: 14px;
        padding: 20px;
    }}

    /* ── Toast / success ─────────────────────── */
    .stSuccess {{ border-radius: 10px !important; }}
    .stError {{ border-radius: 10px !important; }}
    .stWarning {{ border-radius: 10px !important; }}

    /* ── Smooth scrollbar ─────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: rgba(10,12,20,0.5); }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(108,99,255,0.3);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(108,99,255,0.5); }}

    /* ── Hide Streamlit branding ──────────────── */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    </style>
    """, unsafe_allow_html=True)


inject_css()


# ════════════════════════════════════════════════════
# DATABASE HELPERS
# ════════════════════════════════════════════════════
@st.cache_resource
def get_db_connection():
    """Initialize SQLite database and return a connection."""
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
    """Check if a username exists in the database."""
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    return c.fetchone() is not None


def create_user(conn, username, password):
    """Create a new user with bcrypt-hashed password. Returns True on success."""
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    created_at = datetime.now(timezone.utc).isoformat()
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
    """Verify a username/password pair against the database."""
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if not row:
        return False
    stored_hash = row[0]
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash)


conn = get_db_connection()


# ════════════════════════════════════════════════════
# MODEL & PREDICTION HELPERS
# ════════════════════════════════════════════════════
SEVERITY_RULES = {
    "severe": ["death", "anaphyl", "coma", "hospital", "failure", "insufficiency", "liver", "seizure", "cardiac"],
    "moderate": ["nausea", "vomit", "vomiting", "diarr", "headache", "dizzy", "rash", "pain", "fever", "swelling"],
}


def estimate_severity(text: str) -> str:
    """Classify a side-effect string as Severe/Moderate/Mild based on keyword rules."""
    t = text.lower()
    for w in SEVERITY_RULES["severe"]:
        if w in t:
            return "Severe"
    for w in SEVERITY_RULES["moderate"]:
        if w in t:
            return "Moderate"
    return "Mild"


def clean_text(s):
    """Lowercase, strip URLs and special characters for ML vectorization."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_side_effects(side_effects):
    """Clean and deduplicate a list of raw side-effect strings."""
    formatted = []
    for effect in side_effects:
        effect = re.sub(r"[^\w\s,]", "", effect)
        parts = re.split(r",| and | with |;|\||/", effect)
        for p in parts:
            p = p.strip().capitalize()
            if len(p) > 1:
                formatted.append(p)
    return list(dict.fromkeys(formatted))


def detect_multi_ingredient(composition: str) -> list:
    """Detect multiple active ingredients from a composition string.
    Returns a list of ingredient names (empty if single/none).
    """
    if not composition or composition == "Not available":
        return []
    # Split on common separators: +, &, and
    parts = re.split(r"\s*[+&]\s*|\s+and\s+", composition, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    if len(parts) > 1:
        return parts
    return []


# ════════════════════════════════════════════════════
# LOAD MODELS & DATA (cached)
# ════════════════════════════════════════════════════
@st.cache_resource
def load_models_and_data():
    """Load ML models, vectorizer, and dataset. Cached across sessions."""
    tfv = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "side_effect_model.pkl"))
    label_binarizer = joblib.load(os.path.join(MODELS_DIR, "mlb.pkl"))
    df = pd.read_csv(DATA_PATH)
    df["Medicine Name"] = df["Medicine Name"].astype(str)

    drug_map = {}
    drug_display_list = []
    for _, r in df.iterrows():
        name = str(r["Medicine Name"]).strip()
        if name:
            key = name.lower()
            if key not in drug_map:
                drug_map[key] = r
                drug_display_list.append(name)
    return tfv, xgb_model, label_binarizer, df, drug_map, drug_display_list


tfv, xgb_model, label_binarizer, df, drug_map, drug_display_list = load_models_and_data()


def fuzzy_search(query: str, limit: int = 8) -> list:
    """Return fuzzy-matched drug names for a query string."""
    if not query:
        return []
    if HAVE_RAPIDFUZZ:
        matches = rf_process.extract(query, drug_display_list, limit=limit, score_cutoff=45)
        return [m[0] for m in matches]
    return difflib.get_close_matches(query, drug_display_list, n=limit, cutoff=0.3)


def fuzzy_best_match(query: str) -> str | None:
    """Return the single best fuzzy match for a query, or None."""
    if not query:
        return None
    if HAVE_RAPIDFUZZ:
        result = rf_process.extractOne(query, drug_display_list)
        if result and result[1] >= 50:
            return result[0]
    else:
        close = difflib.get_close_matches(query, drug_display_list, n=1, cutoff=0.25)
        if close:
            return close[0]
    return None


def get_predictions_for_drug_key(key: str) -> dict | None:
    """Run the ML pipeline for a drug key and return a result dict."""
    if key not in drug_map:
        return None
    row = drug_map[key]
    desc = clean_text(row.get("Composition", "")) + " " + clean_text(row.get("Uses", ""))
    vec = tfv.transform([desc])
    pred = xgb_model.predict(vec)
    raw = [label_binarizer.classes_[i] for i in range(len(pred[0])) if pred[0][i] == 1]
    formatted = format_side_effects(raw)

    groups = {"mild": [], "moderate": [], "severe": []}
    for eff in formatted:
        sev = estimate_severity(eff)
        groups[sev.lower()].append(eff)

    # Read review percentages directly from the CSV (Bug Fix #4)
    excellent = row.get("Excellent Review %", 0) if "Excellent Review %" in row.index else 0
    average = row.get("Average Review %", 0) if "Average Review %" in row.index else 0
    poor = row.get("Poor Review %", 0) if "Poor Review %" in row.index else max(0, 100 - (excellent + average))

    # Image URL heuristic
    img_val = ""
    for c in ["Image URL", "Image_URL", "Image", "image"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            img_val = str(row[c]).strip()
            break

    # Medicine details
    composition = str(row.get("Composition", "")) if pd.notna(row.get("Composition")) and str(row.get("Composition")).strip() else "Not available"
    uses = str(row.get("Uses", "")) if pd.notna(row.get("Uses")) and str(row.get("Uses")).strip() else "Not available"
    manufacturer = str(row.get("Manufacturer", "")) if pd.notna(row.get("Manufacturer")) and str(row.get("Manufacturer")).strip() else "Not available"

    return {
        "name": row["Medicine Name"],
        "groups": groups,
        "image_url": img_val,
        "excellent": excellent,
        "average": average,
        "poor": poor,
        "composition": composition,
        "uses": uses,
        "manufacturer": manufacturer,
    }


def generate_report(result: dict) -> str:
    """Generate a downloadable text report for a drug prediction."""
    lines = [
        "═" * 50,
        "  DRUG SIDE EFFECT PREDICTION REPORT",
        "═" * 50,
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"  Drug Name: {result['name']}",
        f"  Manufacturer: {result['manufacturer']}",
        f"  Composition: {result['composition']}",
        f"  Uses: {result['uses']}",
        "",
        "─" * 50,
        "  PREDICTED SIDE EFFECTS",
        "─" * 50,
    ]
    for severity in ["severe", "moderate", "mild"]:
        items = result["groups"][severity]
        if items:
            icon = {"severe": "🔴", "moderate": "🟠", "mild": "🟢"}[severity]
            lines.append(f"\n  {icon} {severity.upper()} ({len(items)}):")
            for eff in items:
                lines.append(f"    • {eff}")
    total = sum(len(v) for v in result["groups"].values())
    if total == 0:
        lines.append("\n  ✔ No significant side effects predicted.")

    lines.extend([
        "",
        "─" * 50,
        "  REVIEW DISTRIBUTION",
        "─" * 50,
        f"  Excellent: {result['excellent']}%",
        f"  Average:   {result['average']}%",
        f"  Poor:      {result['poor']}%",
        "",
        "═" * 50,
        "  ⚠ This report is AI-generated and for informational",
        "  purposes only. Consult a healthcare professional.",
        "═" * 50,
    ])
    return "\n".join(lines)


# ════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════
_DEFAULTS = {
    "authenticated": False,
    "username": None,
    "page": "login",
    "search_history": [],
    "voice_text": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def add_to_history(drug_name: str):
    """Add a drug name to search history (max 10, no duplicates)."""
    history = st.session_state.get("search_history", [])
    if drug_name in history:
        history.remove(drug_name)
    history.insert(0, drug_name)
    st.session_state["search_history"] = history[:10]


# ════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════
def render_header():
    """Render the animated app header with user badge."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="app-header" style="text-align:left; padding:16px 0 8px;">
            <h1 style="font-size:1.8rem; margin-bottom:2px;">💊 Drug Side Effect Predictor</h1>
            <p>AI-powered medicine analysis with real-time side effect predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.session_state.get("authenticated"):
            user = st.session_state.get("username", "")
            st.markdown(f"""
            <div style="padding-top:28px;">
                <div class="user-badge">
                    <div class="dot"></div>
                    <span>Logged in as <strong>{user}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")


# ════════════════════════════════════════════════════
# SHARED: PREDICTION UI
# ════════════════════════════════════════════════════
def show_prediction(result: dict, compact: bool = False):
    """Render a full prediction result with charts, details, and export."""

    add_to_history(result["name"])

    if not compact:
        st.markdown(f'<div class="section-title">📊 Results for {result["name"]}</div>', unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 1.6])

    with col_img:
        # Drug image
        if result["image_url"] and str(result["image_url"]).startswith("http"):
            try:
                st.image(result["image_url"], caption=result["name"], use_container_width=True)
            except Exception:
                st.info("🖼 Image not available")
        else:
            st.info("🖼 Image not available")

        # Medicine Details
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:0;">💊 Medicine Details</div>', unsafe_allow_html=True)

        for label, value in [("Manufacturer", result["manufacturer"]), ("Composition", result["composition"]), ("Uses", result["uses"])]:
            if value and value != "Not available":
                st.markdown(f'<div class="detail-label">{label}</div><div class="detail-value">{value}</div>', unsafe_allow_html=True)

        # Drug interaction warning
        ingredients = detect_multi_ingredient(result["composition"])
        if len(ingredients) >= 2:
            st.markdown(f"""
            <div class="interaction-warning">
                <strong>⚠ Multi-Ingredient Medicine</strong><br>
                This medicine contains <strong>{len(ingredients)} active ingredients</strong>:
                {', '.join(ingredients[:5])}{'...' if len(ingredients) > 5 else ''}.
                <br><em>Consult your doctor about potential drug interactions.</em>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        # Severity counts
        counts = {
            "Mild": len(result["groups"]["mild"]),
            "Moderate": len(result["groups"]["moderate"]),
            "Severe": len(result["groups"]["severe"]),
        }
        total = sum(counts.values())

        if total == 0:
            st.success("✔ No significant side effects predicted for this drug.")
        else:
            # Side effects bar chart
            fig_bar = go.Figure(go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=["#00D68F", "#FFB020", "#FF6B6B"],
                marker_line_width=0,
                text=list(counts.values()),
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=14, family="Inter"),
            ))
            fig_bar.update_layout(
                title=dict(text="Side Effects by Severity", font=dict(color="#e2e8f0", size=16, family="Inter")),
                xaxis=dict(color="#8892b0", gridcolor="rgba(108,99,255,0.08)"),
                yaxis=dict(color="#8892b0", gridcolor="rgba(108,99,255,0.08)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{result['name']}")
 
        # Review pie chart
        fig_pie = go.Figure(go.Pie(
            labels=["Excellent", "Average", "Poor"],
            values=[result["excellent"], result["average"], result["poor"]],
            hole=0.45,
            marker=dict(colors=["#00D68F", "#FFB020", "#FF6B6B"], line=dict(color="rgba(10,12,20,0.8)", width=2)),
            textfont=dict(color="#e2e8f0", size=13, family="Inter"),
        ))
        fig_pie.update_layout(
            title=dict(text="Review Distribution", font=dict(color="#e2e8f0", size=16, family="Inter")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#c4d2e3", size=12)),
            height=280,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{result['name']}")

    # Side effects detail list
    if total > 0:
        st.markdown('<div class="section-title">🔍 Detailed Side Effects</div>', unsafe_allow_html=True)
        for cat, class_name, icon in [("severe", "sev-severe", "🔴"), ("moderate", "sev-moderate", "🟠"), ("mild", "sev-mild", "🟢")]:
            items = result["groups"][cat]
            if items:
                st.markdown(f"**{icon} {cat.capitalize()} ({len(items)})**")
                badges_html = " ".join(f'<span class="sev-badge {class_name}">{eff}</span>' for eff in items)
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown("")

    # Export report button
    if not compact:
        report = generate_report(result)
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"{result['name'].replace(' ', '_')}_report.txt",
            mime="text/plain",
            key=f"dl_{result['name']}",
        )


# ════════════════════════════════════════════════════
# SHARED: SEARCH + PREDICT LOGIC
# ════════════════════════════════════════════════════
def search_and_predict(query: str, suggestions_key: str):
    """Shared search + predict logic used by both text and voice pages."""
    suggestions = fuzzy_search(query)

    chosen = None
    if suggestions:
        chosen = st.selectbox(
            "💡 Suggestions",
            options=["— Select or type your own —"] + suggestions,
            key=suggestions_key,
        )

    if st.button("🔬 Predict Side Effects", key=f"btn_{suggestions_key}"):
        sel = chosen if (chosen and chosen != "— Select or type your own —") else query
        if not sel or not sel.strip():
            st.error("Please enter or select a drug name.")
            return

        sel_key = sel.strip().lower()
        if sel_key not in drug_map:
            candidate = fuzzy_best_match(sel)
            if candidate:
                sel_key = candidate.lower()
            else:
                st.error("❌ No matching drug found in the dataset. Try a different name.")
                return

        with st.spinner("Analyzing..."):
            result = get_predictions_for_drug_key(sel_key)

        if not result:
            st.error("❌ Drug not found in dataset.")
        else:
            show_prediction(result)


def render_search_history():
    """Render search history as clickable chips."""
    history = st.session_state.get("search_history", [])
    if history:
        chips_html = "".join(f'<span class="chip">🕑 {name}</span>' for name in history)
        st.markdown(f'<div style="margin-bottom:12px;">{chips_html}</div>', unsafe_allow_html=True)
        # Also provide a selectbox for functional re-search
        rerun_choice = st.selectbox(
            "Quick re-search from history",
            options=["— Select —"] + history,
            key="history_rerun",
            label_visibility="collapsed",
        )
        if rerun_choice and rerun_choice != "— Select —":
            sel_key = rerun_choice.lower()
            result = get_predictions_for_drug_key(sel_key)
            if result:
                show_prediction(result)


# ════════════════════════════════════════════════════
# PAGE: LOGIN
# ════════════════════════════════════════════════════
def login_page():
    """Render the login page."""
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="app-header">
            <h1>💊 Drug Side Effect Predictor</h1>
            <p>Sign in to access AI-powered medicine analysis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Login")
        username = st.text_input("Username", key="login_user", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_pw", placeholder="Enter your password")

        if st.button("Login", use_container_width=True):
            if not username or not password:
                st.warning("Please fill in all fields.")
            elif verify_user(conn, username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["page"] = "dashboard"
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("---")
        st.markdown("<p style='text-align:center; color:#8892b0;'>Don't have an account?</p>", unsafe_allow_html=True)
        if st.button("Create Account", use_container_width=True, key="goto_signup"):
            st.session_state["page"] = "signup"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE: SIGNUP
# ════════════════════════════════════════════════════
def signup_page():
    """Render the signup page."""
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="app-header">
            <h1>💊 Drug Side Effect Predictor</h1>
            <p>Create your account to get started</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Create Account")
        username = st.text_input("Choose username", key="signup_user", placeholder="Pick a unique username")
        pw1 = st.text_input("Password", type="password", key="signup_pw1", placeholder="Create a strong password")
        pw2 = st.text_input("Confirm password", type="password", key="signup_pw2", placeholder="Repeat your password")

        if st.button("Create Account", use_container_width=True, key="do_signup"):
            if not username or not pw1 or not pw2:
                st.warning("Please fill in all fields.")
            elif len(pw1) < 4:
                st.warning("Password must be at least 4 characters.")
            elif pw1 != pw2:
                st.error("Passwords do not match.")
            elif user_exists(conn, username):
                st.error("Username already exists. Choose another.")
            else:
                if create_user(conn, username, pw1):
                    st.success("✅ Account created! You can now login.")
                    st.session_state["page"] = "login"
                else:
                    st.error("Could not create account. Try a different username.")

        st.markdown("---")
        if st.button("← Back to Login", use_container_width=True, key="back_login"):
            st.session_state["page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════
def dashboard_page():
    """Render the dashboard home page with stats and quick search."""
    st.markdown("""
    <div class="app-header" style="text-align:left; padding-bottom:0;">
        <h1 style="font-size:1.6rem;">📊 Dashboard</h1>
        <p>Overview of your drug analysis workspace</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    total_drugs = len(drug_display_list)
    total_manufacturers = df["Manufacturer"].nunique() if "Manufacturer" in df.columns else 0
    avg_excellent = df["Excellent Review %"].mean() if "Excellent Review %" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total_drugs:,}</div>
            <div class="stat-label">💊 Drugs in Database</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{total_manufacturers:,}</div>
            <div class="stat-label">🏭 Manufacturers</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_excellent:.0f}%</div>
            <div class="stat-label">⭐ Avg Excellent Review</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        history_len = len(st.session_state.get("search_history", []))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{history_len}</div>
            <div class="stat-label">🕑 Recent Searches</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Quick search
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-top:0;">🔍 Quick Search</div>', unsafe_allow_html=True)
    q = st.text_input("Enter a medicine name", key="dashboard_search", placeholder="e.g. Paracetamol, Aspirin, Augmentin...")
    if q:
        search_and_predict(q, "dash_suggest")
    st.markdown('</div>', unsafe_allow_html=True)

    # Recent searches
    history = st.session_state.get("search_history", [])
    if history:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:0;">🕑 Recent Searches</div>', unsafe_allow_html=True)
        render_search_history()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE: TEXT PREDICTOR
# ════════════════════════════════════════════════════
def text_predictor_page():
    """Render the text-based drug predictor page."""
    st.markdown("""
    <div class="app-header" style="text-align:left; padding-bottom:0;">
        <h1 style="font-size:1.6rem;">⌨️ Text Predictor</h1>
        <p>Type a medicine name to predict its side effects</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    q = st.text_input("Enter medicine name", key="typed_query_text", placeholder="Start typing a drug name...")
    if q:
        search_and_predict(q, "suggest_text")
    st.markdown('</div>', unsafe_allow_html=True)

    # Search history
    history = st.session_state.get("search_history", [])
    if history:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:0;">🕑 Recent Searches</div>', unsafe_allow_html=True)
        render_search_history()
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE: VOICE PREDICTOR
# ════════════════════════════════════════════════════
def voice_predictor_page():
    """Render the voice-based drug predictor page.
    Uses st_voice_component to listen to speech, update state, and display predictions.
    """
    st.markdown("""
    <div class="app-header" style="text-align:left; padding-bottom:0;">
        <h1 style="font-size:1.6rem;">🎤 Voice Predictor</h1>
        <p>Speak a medicine name to automatically predict its side effects</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Import the voice component function
    from st_voice_component import voice_component
    
    # Render the custom voice component iframe
    voice_data = voice_component(key="voice_rec_widget")
    
    transcript = voice_data.get("transcript", "").strip()
    is_final = voice_data.get("is_final", False)
    supported = voice_data.get("supported", True)

    if not supported:
        st.error("⚠️ Speech Recognition is not supported by your browser. Please use Chrome or Edge.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if transcript:
        st.markdown(f"🗣️ **Detected Speech:** *{transcript}*")
        
        # Suggestions matching the transcript
        suggestions = fuzzy_search(transcript)
        chosen = None
        if suggestions:
            chosen = st.selectbox(
                "💡 Suggestions matching your voice",
                options=["— Select or speak again —"] + suggestions,
                key="suggest_voice",
            )

        # Trigger prediction on button click or when final speech ends
        if st.button("🔬 Predict Side Effects", key="btn_suggest_voice") or is_final:
            sel = chosen if (chosen and chosen != "— Select or speak again —") else transcript
            if not sel or not sel.strip():
                st.error("No drug text captured.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            sel_key = sel.strip().lower()
            if sel_key not in drug_map:
                candidate = fuzzy_best_match(sel)
                if candidate:
                    sel_key = candidate.lower()
                else:
                    st.error("❌ No matching drug found in the dataset. Try speaking more clearly.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

            with st.spinner("Analyzing..."):
                result = get_predictions_for_drug_key(sel_key)
            if not result:
                st.error("❌ Drug not found in dataset.")
            else:
                show_prediction(result)
                
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE: DRUG COMPARISON
# ════════════════════════════════════════════════════
def drug_comparison_page():
    """Render a side-by-side drug comparison page."""
    st.markdown("""
    <div class="app-header" style="text-align:left; padding-bottom:0;">
        <h1 style="font-size:1.6rem;">⚖️ Drug Comparison</h1>
        <p>Compare side effects and reviews of two medicines side by side</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        q1 = st.text_input("First drug name", key="cmp_drug1", placeholder="Type first drug (e.g. Augmentin)...")
        suggestions1 = fuzzy_search(q1) if q1 else []
        chosen1 = None
        if suggestions1:
            chosen1 = st.selectbox(
                "💡 Related Drugs (First)",
                options=["— Select or use typed text —"] + suggestions1,
                key="cmp_suggest1"
            )
            
    with col_b:
        q2 = st.text_input("Second drug name", key="cmp_drug2", placeholder="Type second drug (e.g. Avastin)...")
        suggestions2 = fuzzy_search(q2) if q2 else []
        chosen2 = None
        if suggestions2:
            chosen2 = st.selectbox(
                "💡 Related Drugs (Second)",
                options=["— Select or use typed text —"] + suggestions2,
                key="cmp_suggest2"
            )

    if st.button("⚖️ Compare", use_container_width=True, key="cmp_btn"):
        sel1 = chosen1 if (chosen1 and chosen1 != "— Select or use typed text —") else q1
        sel2 = chosen2 if (chosen2 and chosen2 != "— Select or use typed text —") else q2

        if not sel1 or not sel2 or not sel1.strip() or not sel2.strip():
            st.error("Please enter or select both drug names.")
        else:
            key1 = sel1.strip().lower()
            key2 = sel2.strip().lower()

            # Fuzzy resolve
            if key1 not in drug_map:
                m = fuzzy_best_match(sel1)
                if m:
                    key1 = m.lower()
            if key2 not in drug_map:
                m = fuzzy_best_match(sel2)
                if m:
                    key2 = m.lower()

            r1 = get_predictions_for_drug_key(key1)
            r2 = get_predictions_for_drug_key(key2)

            if not r1:
                st.error(f"Drug '{sel1}' not found in the dataset.")
            if not r2:
                st.error(f"Drug '{sel2}' not found in the dataset.")

            if r1 and r2:
                st.markdown("---")

                # Side-by-side severity comparison chart
                categories = ["Severe", "Moderate", "Mild"]
                vals1 = [len(r1["groups"]["severe"]), len(r1["groups"]["moderate"]), len(r1["groups"]["mild"])]
                vals2 = [len(r2["groups"]["severe"]), len(r2["groups"]["moderate"]), len(r2["groups"]["mild"])]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name=r1["name"][:30], x=categories, y=vals1,
                    marker_color=["#FF6B6B", "#FFB020", "#00D68F"],
                    marker_line_width=0, opacity=0.85,
                    text=vals1, textposition="outside",
                    textfont=dict(color="#e2e8f0", size=13),
                ))
                fig.add_trace(go.Bar(
                    name=r2["name"][:30], x=categories, y=vals2,
                    marker_color=["#ff9999", "#ffd480", "#66e8b8"],
                    marker_line_width=0, opacity=0.85,
                    text=vals2, textposition="outside",
                    textfont=dict(color="#e2e8f0", size=13),
                ))
                fig.update_layout(
                    barmode="group",
                    title=dict(text="Side Effects Comparison", font=dict(color="#e2e8f0", size=16, family="Inter")),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#c4d2e3", size=12)),
                    xaxis=dict(color="#8892b0"),
                    yaxis=dict(color="#8892b0", gridcolor="rgba(108,99,255,0.08)"),
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Review comparison
                fig_rev = go.Figure()
                for name_label, r in [(r1["name"][:25], r1), (r2["name"][:25], r2)]:
                    fig_rev.add_trace(go.Bar(
                        name=name_label,
                        x=["Excellent", "Average", "Poor"],
                        y=[r["excellent"], r["average"], r["poor"]],
                        text=[f'{r["excellent"]}%', f'{r["average"]}%', f'{r["poor"]}%'],
                        textposition="outside",
                        textfont=dict(color="#e2e8f0", size=12),
                    ))
                fig_rev.update_layout(
                    barmode="group",
                    title=dict(text="Review Distribution Comparison", font=dict(color="#e2e8f0", size=16, family="Inter")),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#c4d2e3", size=12)),
                    xaxis=dict(color="#8892b0"),
                    yaxis=dict(color="#8892b0", gridcolor="rgba(108,99,255,0.08)", title="Percentage"),
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_rev, use_container_width=True)

                # Detailed side-by-side
                st.markdown("---")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown(f'<div class="compare-col">', unsafe_allow_html=True)
                    show_prediction(r1, compact=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with d2:
                    st.markdown(f'<div class="compare-col">', unsafe_allow_html=True)
                    show_prediction(r2, compact=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# PAGE: DRUG CATALOG
# ════════════════════════════════════════════════════
def drug_catalog_page():
    """Render the drug catalog with search, pagination, and sorting."""
    st.markdown("""
    <div class="app-header" style="text-align:left; padding-bottom:0;">
        <h1 style="font-size:1.6rem;">📋 Drug Catalog</h1>
        <p>Browse and search the complete medicine database</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Search and filter controls
    col_search, col_rows = st.columns([3, 1])
    with col_search:
        search = st.text_input("🔍 Search by name, manufacturer, composition, or use", key="catalog_search", placeholder="Type to filter...")
    with col_rows:
        rows_per_page = st.selectbox("Rows per page", [25, 50, 100, 200], key="catalog_rows")

    # Filter
    base_cols = ["Medicine Name", "Manufacturer", "Composition", "Uses", "Excellent Review %", "Average Review %", "Poor Review %"]
    cols_to_show = [c for c in base_cols if c in df.columns]
    if not cols_to_show:
        st.error("No displayable columns found in the dataset.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    filtered_df = df.copy()
    search_term = search.strip().lower() if search else ""
    if search_term:
        mask = pd.Series(False, index=filtered_df.index)
        for col in ["Medicine Name", "Manufacturer", "Composition", "Uses"]:
            if col in filtered_df.columns:
                mask |= filtered_df[col].astype(str).str.lower().str.contains(search_term, na=False)
        filtered_df = filtered_df[mask]

    total_results = len(filtered_df)
    if total_results == 0:
        st.warning("No drugs match your search. Try different keywords.")
    else:
        # Pagination
        total_pages = max(1, (total_results + rows_per_page - 1) // rows_per_page)

        if "catalog_page" not in st.session_state:
            st.session_state["catalog_page"] = 0
        current_page = st.session_state["catalog_page"]
        current_page = min(current_page, total_pages - 1)

        start_idx = current_page * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_results)

        st.caption(f"Showing {start_idx + 1}–{end_idx} of {total_results:,} results")
        st.dataframe(
            filtered_df[cols_to_show].iloc[start_idx:end_idx].reset_index(drop=True),
            use_container_width=True,
            height=min(rows_per_page * 35 + 40, 600),
        )

        # Pagination controls
        if total_pages > 1:
            p1, p2, p3, p4, p5 = st.columns([1, 1, 2, 1, 1])
            with p1:
                if st.button("⏮ First", key="pg_first", disabled=current_page == 0):
                    st.session_state["catalog_page"] = 0
                    st.rerun()
            with p2:
                if st.button("◀ Prev", key="pg_prev", disabled=current_page == 0):
                    st.session_state["catalog_page"] = current_page - 1
                    st.rerun()
            with p3:
                st.markdown(f"<div style='text-align:center;color:#8892b0;padding-top:8px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with p4:
                if st.button("Next ▶", key="pg_next", disabled=current_page >= total_pages - 1):
                    st.session_state["catalog_page"] = current_page + 1
                    st.rerun()
            with p5:
                if st.button("Last ⏭", key="pg_last", disabled=current_page >= total_pages - 1):
                    st.session_state["catalog_page"] = total_pages - 1
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="text-align:center; padding:14px 0 8px;">
    <div style="font-size:28px;">💊</div>
    <div style="font-size:14px; font-weight:700; color:#e2e8f0; letter-spacing:0.5px;">Drug Predictor</div>
    <div style="font-size:11px; color:#8892b0;">AI-Powered Analysis</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.session_state.get("authenticated"):
    nav_options = ["📊 Dashboard", "⌨️ Text Predictor", "🎤 Voice Predictor", "⚖️ Compare Drugs", "📋 Drug Catalog"]
    nav_map = {
        "📊 Dashboard": "dashboard",
        "⌨️ Text Predictor": "text",
        "🎤 Voice Predictor": "voice",
        "⚖️ Compare Drugs": "compare",
        "📋 Drug Catalog": "drugs",
    }
    # Reverse map to find current selection
    reverse_map = {v: k for k, v in nav_map.items()}
    current_page = st.session_state.get("page", "dashboard")
    default_idx = 0
    if current_page in reverse_map:
        try:
            default_idx = nav_options.index(reverse_map[current_page])
        except ValueError:
            default_idx = 0

    selected_nav = st.sidebar.radio(
        "NAVIGATION",
        nav_options,
        index=default_idx,
        key="nav_radio",
        label_visibility="visible",
    )
    new_page = nav_map.get(selected_nav, "dashboard")
    if new_page != st.session_state.get("page"):
        st.session_state["page"] = new_page
        st.rerun()

    st.sidebar.markdown("---")

    # User info in sidebar
    user = st.session_state.get("username", "")
    st.sidebar.markdown(f"""
    <div style="background:rgba(0,214,143,0.06);border:1px solid rgba(0,214,143,0.15);border-radius:10px;padding:12px;margin-bottom:12px;">
        <div style="color:#8892b0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">Signed in as</div>
        <div style="color:#e2e8f0;font-size:15px;font-weight:600;">{user}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["page"] = "login"
        st.session_state["search_history"] = []
        st.rerun()
else:
    auth_options = ["🔐 Login", "📝 Signup"]
    auth_map = {"🔐 Login": "login", "📝 Signup": "signup"}
    reverse_auth = {v: k for k, v in auth_map.items()}
    current_auth = st.session_state.get("page", "login")
    auth_idx = 0
    if current_auth in reverse_auth:
        try:
            auth_idx = auth_options.index(reverse_auth[current_auth])
        except ValueError:
            auth_idx = 0

    selected_auth = st.sidebar.radio(
        "ACCOUNT",
        auth_options,
        index=auth_idx,
        key="auth_radio",
        label_visibility="visible",
    )
    new_page = auth_map.get(selected_auth, "login")
    if new_page != st.session_state.get("page"):
        st.session_state["page"] = new_page
        st.rerun()


# ════════════════════════════════════════════════════
# PAGE ROUTING
# ════════════════════════════════════════════════════
page = st.session_state.get("page", "login")

if page in ("login", "signup"):
    if page == "login":
        login_page()
    else:
        signup_page()
else:
    # Protected pages
    if not st.session_state.get("authenticated", False):
        st.warning("⚠ Please login to access this page.")
        st.stop()

    render_header()

    if page == "dashboard":
        dashboard_page()
    elif page == "text":
        text_predictor_page()
    elif page == "voice":
        voice_predictor_page()
    elif page == "compare":
        drug_comparison_page()
    elif page == "drugs":
        drug_catalog_page()
    else:
        st.info("Unknown page. Use the sidebar to navigate.")
