# 💊 Drug Side Effect & Safety Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, full-stack healthcare web application that predicts medicine side-effect severities, review distributions, and drug interactions based on active composition. Built with **Machine Learning**, **Streamlit**, and **Web Speech API** in a modern glassmorphic interface.

---

## 🌟 Key Features

- **🔍 Smart Search & Fuzzy Matching:**
  - Auto-suggests and searches across hundreds of medicines with typo-tolerant fuzzy matching (powered by `rapidfuzz`).
- **🎙️ Voice-Activated Predictor:**
  - Integrated custom browser speech component using the **Web Speech API** for hands-free voice input and auto-prediction.
- **📊 Multi-Label Side Effect Severity Prediction:**
  - Classifies side effects into **Mild**, **Moderate**, and **Severe** categories using a trained **Random Forest** model.
- **⚖️ Side-by-Side Drug Comparison:**
  - Compare two medicines simultaneously with interactive **Plotly** severity charts, review breakdown, and active ingredients.
- **⚠️ Multi-Ingredient Interaction Warnings:**
  - Identifies combination drugs with multiple active components and alerts users to consult medical professionals.
- **📈 Interactive Data Visualizations:**
  - Dynamic severity bar charts and user review satisfaction donut charts (Excellent / Average / Poor).
- **📥 Downloadable Clinical Reports:**
  - Generates comprehensive, downloadable text summary reports of any medicine for patient or doctor reference.
- **🔐 User Authentication & Search History:**
  - Secure SQLite backend with hashed passwords (`bcrypt`) and user session history tracking.

---

## 🛠️ Tech Stack & Architecture

- **Frontend / UI:** [Streamlit](https://streamlit.io/), HTML5/CSS3 (Glassmorphism), Vanilla JS (Web Speech API Component)
- **Machine Learning & NLP:** [Scikit-Learn](https://scikit-learn.org/) (Random Forest Classifier, MultiOutput), TF-IDF Vectorization, XGBoost
- **Data Processing:** Pandas, NumPy, RapidFuzz
- **Data Visualization:** [Plotly](https://plotly.com/python/)
- **Database & Security:** SQLite3, bcrypt

---

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .gitattributes             # Git LFS configuration for models
├── data/
│   └── Medicine_Details.csv   # Dataset of medicines, compositions, and reviews
├── models/
│   ├── side_effect_model.pkl  # Trained ML model for side effects
│   ├── review_model.pkl       # Trained ML model for review scores
│   ├── vectorizer.pkl         # TF-IDF composition vectorizer
│   └── mlb.pkl                # MultiLabelBinarizer for side effects
├── scripts/
│   ├── train_model.py         # Model training pipeline
│   └── predict.py             # Standalone prediction module
└── st_voice_component/        # Custom Streamlit Web Speech API component
    └── frontend/
        └── index.html
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git & Git LFS installed (for downloading large model `.pkl` files)

### 2. Clone the Repository
```bash
git clone https://github.com/Pratty15/Drug-Side-Effect-Predictor.git
cd Drug-Side-Effect-Predictor
```

### 3. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔬 How the Model Works

1. **Preprocessing & Feature Extraction:**
   - Active compositions are parsed, cleaned, and vectorized using **TF-IDF Vectorization** to capture pharmaceutical component significance.
2. **Multi-Output Classification:**
   - Multi-label classification models predict binary presence across individual side effect indicators.
   - Predictions are mapped into clinical severity bins (**Mild**, **Moderate**, **Severe**) based on standard pharmacovigilance annotations.
3. **Sentiment & Review Prediction:**
   - Predicts patient review sentiment distribution (percentage of positive, neutral, and adverse feedback).

---

## 📄 Disclaimer

> **Medical Disclaimer:** This application is developed for educational, informational, and academic research purposes. It is **not** a replacement for professional medical advice, diagnosis, or clinical prescription. Always consult a qualified healthcare provider regarding medications and health conditions.

---

## 👤 Author

- **Pratyush** - [GitHub Profile](https://github.com/Pratty15)
