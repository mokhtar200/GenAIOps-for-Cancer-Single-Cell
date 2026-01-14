from typing import Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .data import load_breast_cancer_dataset


def train_baseline(save_path: str = "models/baseline.joblib") -> Tuple[RandomForestClassifier, float]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    X, y = load_breast_cancer_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    joblib.dump(clf, save_path)
    return clf, acc


def load_model(path: str = "models/baseline.joblib") -> RandomForestClassifier:
    return joblib.load(path)


def predict(features: pd.DataFrame, model_path: str = "models/baseline.joblib") -> np.ndarray:
    clf = load_model(model_path)
    return clf.predict(features)
