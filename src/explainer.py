from typing import Any, Dict
import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def explain(model: Any, X: pd.DataFrame) -> Dict[str, float]:
    """Return a simple explanation mapping feature -> importance.

    Uses SHAP when available; otherwise falls back to model.feature_importances_.
    """
    if SHAP_AVAILABLE:
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        return dict(zip(X.columns.tolist(), mean_abs.tolist()))

    # Fallback
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        return dict(zip(X.columns.tolist(), imps.tolist()))

    # Last-resort: zeroed importance
    return {c: 0.0 for c in X.columns}
