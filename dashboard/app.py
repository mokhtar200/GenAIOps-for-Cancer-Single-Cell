import sys
import pathlib
import streamlit as st
import pandas as pd

# Ensure project root is on sys.path so `src` can be imported when Streamlit runs
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_breast_cancer_dataset
from src.model import load_model, train_baseline
from src.explainer import explain


st.title("Cancer Detection — Demo Dashboard")

if st.button("Train baseline model"):
    clf, acc = train_baseline()
    st.success(f"Trained baseline RandomForest — test accuracy={acc:.3f}")

X, y = load_breast_cancer_dataset()
st.sidebar.header("Sample inputs")
idx = st.sidebar.slider("Sample index", 0, len(X) - 1, 0)
sample = X.iloc[[idx]]
st.write("### Sample features")
st.dataframe(sample)

if st.button("Predict sample"):
    model = load_model()
    pred = model.predict(sample)[0]
    st.write("**Prediction (1=malignant, 0=benign):**", int(pred))
    ex = explain(model, sample)
    st.write("### Explanation (feature -> importance)")
    ex_df = pd.DataFrame(list(ex.items()), columns=["feature", "importance"]).sort_values("importance", ascending=False)
    st.table(ex_df.head(10))
