from typing import Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load sklearn breast cancer dataset and return (X, y)."""
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=[data.target.name])
    y = data.frame[data.target.name]
    return X, y
