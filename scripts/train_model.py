# scripts/train_model.py
"""
Optimized training with XGBoost (Option B).
Saves: models/vectorizer.pkl, models/mlb.pkl, models/le_review.pkl,
       models/side_effect_model.pkl, models/review_model.xgb, models/meta.json
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb  # added for low-level API

# ---------- Config ----------
DATA_PATH = os.path.join('..', 'data', 'Medicine_Details.csv')  # adjust if needed
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

TFV_MAX_FEATURES = 4000
TFV_NGRAM = (1, 2)

# XGBoost (side-effect) params (lightweight but good)
XGB_SIDE_PARAMS = {
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_jobs': -1,
    'random_state': 42
}

# XGBoost (review) params and early stopping
XGB_REVIEW_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_label_encoder': False,  # ignored in xgb.train()
    'eval_metric': 'mlogloss',
    'verbosity': 0,
    'n_jobs': -1,  # ignored in xgb.train()
    'random_state': 42
}
EARLY_STOPPING_ROUNDS = 20
VALIDATION_RATIO = 0.1  # of the training set used as validation for early stopping

# Minimum positive count for a side-effect label to be trained
MIN_POS_COUNT = 5

# ---------- Helpers ----------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9, ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_side_effects(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    parts = re.split(r"[,;]\s*", str(s))
    parts = [p.strip().lower() for p in parts if p.strip()]
    return parts

def map_review(row):
    if row.get('Excellent Review %', 0) > 50:
        return 'Excellent'
    elif row.get('Average Review %', 0) > 50:
        return 'Average'
    else:
        return 'Poor'

def map_severity(effect):
    e = effect.lower()
    if any(x in e for x in ['death','anaphyl','coma','hospital']):
        return 'severe'
    if any(x in e for x in ['nausea','vomit','vomiting','diarr','diarrhoea','diarrhea','headache','dizzy','dizziness','rash']):
        return 'moderate'
    return 'mild'

# ---------- Load dataset ----------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df), "Columns:", list(df.columns))

# ---------- Preprocess ----------
print("Cleaning text columns...")
if 'Composition' not in df.columns or 'Uses' not in df.columns or 'Side_effects' not in df.columns:
    raise ValueError("Expected columns missing. Make sure CSV has 'Composition', 'Uses', and 'Side_effects' columns.")

df['Composition'] = df['Composition'].apply(clean_text)
df['Uses'] = df['Uses'].apply(clean_text)
df['Side_effects'] = df['Side_effects'].apply(parse_side_effects)
df['Review_Label'] = df.apply(map_review, axis=1)
df['Text_Feature'] = df['Composition'] + " " + df['Uses']

# ---------- TF-IDF ----------
print(f"Vectorizing text with TF-IDF (max_features={TFV_MAX_FEATURES}, ngram={TFV_NGRAM})...")
tfv = TfidfVectorizer(max_features=TFV_MAX_FEATURES, ngram_range=TFV_NGRAM)
X_text = tfv.fit_transform(df['Text_Feature'])

# ---------- Side-effect multi-label setup ----------
print("Preparing multi-label targets for side effects...")
mlb_all = MultiLabelBinarizer()
Y_side_all = mlb_all.fit_transform(df['Side_effects'])
all_labels = np.array(mlb_all.classes_)
label_counts = Y_side_all.sum(axis=0)

print("Total unique side-effect labels:", len(all_labels))

keep_mask = label_counts >= MIN_POS_COUNT
kept_labels = all_labels[keep_mask]
print(f"Keeping {len(kept_labels)} labels with >= {MIN_POS_COUNT} positive examples (out of {len(all_labels)})")

if len(kept_labels) < len(all_labels):
    mlb = MultiLabelBinarizer(classes=list(kept_labels))
    mlb.fit([[l] for l in kept_labels])
else:
    mlb = mlb_all

Y_side = mlb.transform(df['Side_effects'])

# ---------- Train/test split ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X_text, Y_side, test_size=0.2, random_state=42
)

# ---------- Side-effect model: OneVsRest XGBoost ----------
print("\nTraining One-vs-Rest XGBoost for multi-label side-effect prediction...")
start = time()
base_xgb = XGBClassifier(**XGB_SIDE_PARAMS)
side_clf = OneVsRestClassifier(base_xgb, n_jobs=-1)

side_clf.fit(X_train, Y_train)
elapsed = time() - start
print(f"Trained OneVsRest side-effect model in {elapsed:.1f}s")

print("\nEvaluating side-effect model (multi-label)...")
Y_pred = side_clf.predict(X_test)
from sklearn.metrics import classification_report as cr
print("Classification report (first 30 labels show):")
n_show = min(30, Y_pred.shape[1])
for i in range(n_show):
    lbl = mlb.classes_[i]
    print(f"Label {i+1}/{Y_pred.shape[1]}: {lbl}")
    print(cr(Y_test[:, i], Y_pred[:, i], zero_division=0))

from sklearn.metrics import f1_score
print("micro f1:", f1_score(Y_test, Y_pred, average='micro'))
print("macro f1:", f1_score(Y_test, Y_pred, average='macro'))

# ---------- Review prediction (xgb.train with early stopping) ----------
print("\nPreparing review prediction...")
le_review = LabelEncoder()
y_review = le_review.fit_transform(df['Review_Label'])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_text, y_review, test_size=0.2, random_state=42, stratify=y_review)
Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(Xr_train, yr_train, test_size=VALIDATION_RATIO, random_state=42, stratify=yr_train)

dtrain = xgb.DMatrix(Xr_tr, label=yr_tr)
dval = xgb.DMatrix(Xr_val, label=yr_val)
dtest = xgb.DMatrix(Xr_test)

params = {
    'max_depth': XGB_REVIEW_PARAMS['max_depth'],
    'eta': XGB_REVIEW_PARAMS['learning_rate'],
    'objective': 'multi:softprob',
    'num_class': len(le_review.classes_),
    'subsample': XGB_REVIEW_PARAMS['subsample'],
    'colsample_bytree': XGB_REVIEW_PARAMS['colsample_bytree'],
    'eval_metric': 'mlogloss',
    'seed': XGB_REVIEW_PARAMS['random_state'],
    'verbosity': 0,
}

evals = [(dtrain, 'train'), (dval, 'eval')]

print("Training XGBoost review model with early stopping (xgb.train)...")
start = time()
review_model = xgb.train(
    params,
    dtrain,
    num_boost_round=XGB_REVIEW_PARAMS['n_estimators'],
    evals=evals,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=False
)
elapsed = time() - start
print(f"Trained review XGBoost in {elapsed:.1f}s")

y_pred_prob = review_model.predict(dtest)
y_pred = y_pred_prob.argmax(axis=1)

acc = accuracy_score(yr_test, y_pred)
print(f"\nReview model accuracy on test set: {acc:.4f}")
print(classification_report(yr_test, y_pred, target_names=le_review.classes_))

# ---------- Save artifacts ----------
print("\nSaving artifacts to", MODELS_DIR)
joblib.dump(tfv, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
joblib.dump(mlb, os.path.join(MODELS_DIR, 'mlb.pkl'))
joblib.dump(le_review, os.path.join(MODELS_DIR, 'le_review.pkl'))
joblib.dump(side_clf, os.path.join(MODELS_DIR, 'side_effect_model.pkl'))
review_model.save_model(os.path.join(MODELS_DIR, 'review_model.xgb'))

meta = {
    'side_effect_classes': mlb.classes_.tolist(),
    'review_classes': le_review.classes_.tolist(),
    'severity_rules': {
        'severe_keywords': ['death', 'anaphyl', 'coma', 'hospital'],
        'moderate_keywords': ['nausea', 'vomit', 'vomiting', 'diarr', 'diarrhoea', 'diarrhea', 'headache', 'dizzy', 'dizziness', 'rash'],
        'default': 'mild'
    }
}
with open(os.path.join(MODELS_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("All models and metadata saved. ✅")
