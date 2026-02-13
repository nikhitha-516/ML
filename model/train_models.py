import json
from pathlib import Path

#import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
DATASET_URL = "https://raw.githubusercontent.com/caravanuden/cardio/master/cardio_train.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset"


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw_df = pd.read_csv(DATASET_URL, sep=";")

    # Basic data cleaning: remove physiologically implausible records.
    clean_df = raw_df[
        (raw_df["ap_hi"] > 0)
        & (raw_df["ap_lo"] > 0)
        & (raw_df["ap_hi"] > raw_df["ap_lo"])
        & (raw_df["ap_hi"] < 260)
        & (raw_df["ap_lo"] < 200)
        & (raw_df["height"] >= 120)
        & (raw_df["weight"] >= 35)
    ].copy()

    # Feature engineering to strengthen signal and keep feature count > 12.
    clean_df["age_years"] = clean_df["age"] / 365.25
    clean_df["bmi"] = clean_df["weight"] / ((clean_df["height"] / 100.0) ** 2)
    clean_df["pulse_pressure"] = clean_df["ap_hi"] - clean_df["ap_lo"]

    y = clean_df["cardio"].astype(int)
    X = clean_df.drop(columns=["cardio", "id"])

    # Raw upload-compatible dataframe (without engineered columns) for sample CSV.
    upload_df = clean_df.drop(columns=["age_years", "bmi", "pulse_pressure"])
    return X, y, upload_df


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "kNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": safe_auc(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    X, y, upload_df = load_dataset()

    idx_train, idx_test = train_test_split(
        X.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    # Save default evaluation set for Streamlit demo.
    default_test = X_test.copy()
    default_test["cardio"] = y_test.values
    default_test.to_csv(DATA_DIR / "default_test.csv", index=False)

    # Save upload sample in raw-cardio format.
    sample_upload = upload_df.loc[idx_test].head(500).copy()
    sample_upload.to_csv(DATA_DIR / "sample_upload_test.csv", index=False)

    metrics_rows = []
    models = build_models()

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        metrics = evaluate(y_test, y_pred, y_prob)
        metrics_rows.append({"ML Model Name": model_name, **metrics})

        model_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = MODEL_DIR / f"{model_filename}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)

    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    with open(MODEL_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": "Cardiovascular Disease Dataset",
                "source": "Kaggle / Open-source mirror",
                "dataset_url": DATASET_URL,
                "kaggle_url": KAGGLE_URL,
                "instances": int(X.shape[0]),
                "features": int(X.shape[1]),
                "target_name": "Cardio (0 = no disease, 1 = disease)",
                "notes": "11 base features + 3 engineered features (age_years, bmi, pulse_pressure)",
            },
            f,
            indent=2,
        )

    print("Training complete. Artifacts saved in model/ and data/.")
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
