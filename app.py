import json
from io import StringIO
from pathlib import Path

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"


@st.cache_resource
def load_models() -> dict:
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
        "XGBoost (Ensemble)": "xgboost_ensemble.pkl",
    }

    models = {}
    for name, file_name in model_files.items():
        model_path = MODEL_DIR / file_name
        with open(model_path, "rb") as f:
            models[name] = pickle.load(f)

    return models


@st.cache_data
def load_feature_columns() -> list:
    with open(MODEL_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_baseline_metrics() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "metrics.csv")


@st.cache_data
def load_dataset_info() -> dict:
    with open(MODEL_DIR / "dataset_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dataset_from_url(dataset_url: str) -> pd.DataFrame:
    return pd.read_csv(dataset_url, sep=";")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict:
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if len(pd.Series(y_true).unique()) > 1:
        result["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        result["AUC"] = float("nan")

    return result


def detect_target_column(df: pd.DataFrame) -> str | None:
    candidates = ["cardio", "target", "label", "class", "diagnosis"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "age_years" not in out.columns and "age" in out.columns:
        out["age_years"] = out["age"] / 365.25

    if "bmi" not in out.columns and {"weight", "height"}.issubset(out.columns):
        out["bmi"] = out["weight"] / ((out["height"] / 100.0) ** 2)

    if "pulse_pressure" not in out.columns and {"ap_hi", "ap_lo"}.issubset(out.columns):
        out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]

    return out


def prepare_input_dataframe(
    uploaded_df: pd.DataFrame, feature_cols: list
) -> tuple[pd.DataFrame, pd.Series | None]:

    enriched_df = add_engineered_features(uploaded_df)

    target_col = detect_target_column(enriched_df)
    y_true = enriched_df[target_col] if target_col else None

    missing_cols = [col for col in feature_cols if col not in enriched_df.columns]
    if missing_cols:
        raise ValueError(
            "Uploaded CSV is missing required feature columns: "
            + ", ".join(missing_cols[:8])
            + (" ..." if len(missing_cols) > 8 else "")
        )

    X = enriched_df[feature_cols].copy()
    return X, y_true


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue().decode("utf-8")
    return pd.read_csv(StringIO(content), sep=None, engine="python")


def main() -> None:
    st.set_page_config(page_title="ML Assignment 2 - Classification", layout="wide")

    st.title("Machine Learning Assignment 2")
    st.subheader("Classification Model Comparison and Inference App")

    dataset_info = load_dataset_info()
    models = load_models()
    feature_cols = load_feature_columns()
    baseline_metrics = load_baseline_metrics()

    with st.expander("Dataset Information", expanded=True):
        st.write(f"Dataset: **{dataset_info['dataset_name']}**")
        st.write(f"Source: **{dataset_info['source']}**")

        if "dataset_url" in dataset_info:
            st.write(f"Dataset URL: {dataset_info['dataset_url']}")

        if "kaggle_url" in dataset_info:
            st.write(f"Kaggle URL: {dataset_info['kaggle_url']}")

        st.write(f"Instances: **{dataset_info['instances']}**")
        st.write(f"Features: **{dataset_info['features']}**")
        st.write(f"Target: **{dataset_info['target_name']}**")

        if "notes" in dataset_info:
            st.write(f"Notes: {dataset_info['notes']}")

    st.sidebar.header("Controls")
    selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload CSV test dataset", type=["csv"])

    sample_file_path = DATA_DIR / "sample_upload_test.csv"
    if sample_file_path.exists():
        with open(sample_file_path, "rb") as f:
            st.sidebar.download_button(
                "Download Sample Test CSV",
                data=f,
                file_name="sample_upload_test.csv",
                mime="text/csv",
            )

    model = models[selected_model_name]

    if uploaded_file is not None:
        test_df = read_uploaded_csv(uploaded_file)
        source_label = "Uploaded CSV"
    else:
        test_df = load_dataset_from_url(dataset_info["dataset_url"])
        source_label = "Kaggle cardiovascular dataset URL"

    st.write(f"Using dataset source: **{source_label}**")
    st.write(f"Rows: **{len(test_df)}**")

    try:
        X_test, y_true = prepare_input_dataframe(test_df, feature_cols)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    result_df = X_test.copy()
    result_df["prediction"] = y_pred

    st.write("Preview of predictions:")
    st.dataframe(result_df.head(20), use_container_width=True)

    if y_true is not None:
        metrics = compute_metrics(y_true, y_pred, y_prob)

        st.markdown("### Evaluation Metrics")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        c2.metric("AUC", f"{metrics['AUC']:.4f}" if pd.notna(metrics["AUC"]) else "N/A")
        c3.metric("Precision", f"{metrics['Precision']:.4f}")
        c4.metric("Recall", f"{metrics['Recall']:.4f}")
        c5.metric("F1", f"{metrics['F1']:.4f}")
        c6.metric("MCC", f"{metrics['MCC']:.4f}")

        st.markdown("### Confusion Matrix")
        plot_confusion_matrix(y_true, y_pred)

        st.markdown("### Classification Report")
        cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cls_report_df = pd.DataFrame(cls_report).transpose().round(4)
        st.dataframe(cls_report_df, use_container_width=True)

    else:
        st.warning(
            "Uploaded CSV does not include a target column (`cardio`, `target`, `label`, `class`, or `diagnosis`). "
            "Predictions are shown, but metrics and confusion matrix need true labels."
        )

    st.markdown("### Baseline Comparison on Hold-Out Set")
    st.dataframe(baseline_metrics.round(4), use_container_width=True)


if __name__ == "__main__":
    main()
