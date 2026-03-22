from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

RESULTS_DIR = Path("./results")

st.set_page_config(page_title="Stock CNN Dashboard", layout="wide")

st.title("Stock Price Trend CNN Dashboard")
st.caption("Model evaluation dashboard for image-based stock movement prediction")

# -----------------------------
# Helpers
# -----------------------------
def read_metrics_txt(path: Path):
    metrics = {}
    if not path.exists():
        return metrics

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            metrics[key.strip()] = value.strip()

    return metrics


# -----------------------------
# File paths
# -----------------------------
metrics_file = RESULTS_DIR / "test_metrics.txt"
summary_file = RESULTS_DIR / "test_summary.csv"
history_file = RESULTS_DIR / "training_history.csv"
preds_file = RESULTS_DIR / "test_predictions.csv"
loss_img_file = RESULTS_DIR / "loss_curves.png"
cm_img_file = RESULTS_DIR / "confusion_matrix.png"

# -----------------------------
# Top metrics
# -----------------------------
st.subheader("Final Test Metrics")

metrics = read_metrics_txt(metrics_file)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", metrics.get("Accuracy", "N/A"))
col2.metric("Precision", metrics.get("Precision", "N/A"))
col3.metric("Recall", metrics.get("Recall", "N/A"))
col4.metric("F1 Score", metrics.get("F1 Score", "N/A"))
col5.metric("Threshold", metrics.get("Threshold", "N/A"))

# -----------------------------
# Charts row
# -----------------------------
st.subheader("Model Visuals")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Training vs Validation Loss**")
    if loss_img_file.exists():
        st.image(Image.open(loss_img_file), use_container_width=True)
    else:
        st.warning("loss_curves.png not found in results/")

with c2:
    st.markdown("**Confusion Matrix**")
    if cm_img_file.exists():
        st.image(Image.open(cm_img_file), use_container_width=True)
    else:
        st.warning("confusion_matrix.png not found in results/")

# -----------------------------
# Training history
# -----------------------------
st.subheader("Training History")

if history_file.exists():
    history_df = pd.read_csv(history_file)
    st.dataframe(history_df, use_container_width=True)

    numeric_cols = [c for c in history_df.columns if c != "epoch"]
    selected_col = st.selectbox("Select metric to visualize", numeric_cols)

    if "epoch" in history_df.columns:
        chart_df = history_df.set_index("epoch")[[selected_col]]
        st.line_chart(chart_df)
else:
    st.info("training_history.csv not found in results/")

# -----------------------------
# Test summary
# -----------------------------
st.subheader("Test Summary")

if summary_file.exists():
    summary_df = pd.read_csv(summary_file)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("test_summary.csv not found in results/")

# -----------------------------
# Predictions explorer
# -----------------------------
st.subheader("Predictions Explorer")

if preds_file.exists():
    preds_df = pd.read_csv(preds_file)

    st.write(f"Total prediction rows: {len(preds_df)}")

    show_cols = [c for c in preds_df.columns if c in ["Date", "StockID", "Ret_5d", "y_true", "y_pred", "y_prob", "year"]]
    if len(show_cols) > 0:
        st.dataframe(preds_df[show_cols].head(100), use_container_width=True)
    else:
        st.dataframe(preds_df.head(100), use_container_width=True)

    if {"y_true", "y_pred"}.issubset(preds_df.columns):
        st.markdown("**Misclassified Samples**")
        mis_df = preds_df[preds_df["y_true"] != preds_df["y_pred"]]
        st.write(f"Misclassified rows: {len(mis_df)}")

        if len(mis_df) > 0:
            mis_cols = [c for c in mis_df.columns if c in ["Date", "StockID", "Ret_5d", "y_true", "y_pred", "y_prob", "year"]]
            if len(mis_cols) > 0:
                st.dataframe(mis_df[mis_cols].head(100), use_container_width=True)
            else:
                st.dataframe(mis_df.head(100), use_container_width=True)
else:
    st.info("test_predictions.csv not found in results/")