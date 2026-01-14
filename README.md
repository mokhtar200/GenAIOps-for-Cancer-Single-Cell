# QC → ML → Cancer Detection → GenAI Explanation → MLOps → Dashboard

Minimal scaffold demonstrating a pipeline combining:
- QC stub (quantum feature mapping placeholder)
- Classical ML baseline for cancer detection (sklearn)
- Simple explanation module (SHAP fallback to feature importances)
- FastAPI model serving
- Streamlit dashboard for predictions and explanations

Quick start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a baseline model (Python REPL):

```bash
python -c "from src.model import train_baseline; train_baseline()"
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Serve the model (FastAPI):

```bash
uvicorn src.serve:app --reload
```
