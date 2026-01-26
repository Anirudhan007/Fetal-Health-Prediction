import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score


# -------------------------------------------------
# Load model class
# -------------------------------------------------

def load_model_class():
    source_path = Path("/workspace/results/utils.py")
    assert source_path.exists(), "utils.py not found at /workspace/results/utils.py"

    spec = importlib.util.spec_from_file_location("utils_module", source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils_module"] = module
    spec.loader.exec_module(module)

    assert hasattr(module, "FetalHealthPredictor"), "FetalHealthPredictor class missing"
    return module.FetalHealthPredictor


# -------------------------------------------------
# Main test
# -------------------------------------------------

def test_fetal_health_model_end_to_end():
    # Load data
    train_path = Path("/workspace/data/train.csv")
    test_path = Path("/tests/test.csv")

    assert train_path.exists(), "train.csv not found"
    assert test_path.exists(), "test.csv not found"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    assert "target" in df_train.columns, "'target' missing in train.csv"
    assert "target" in df_test.columns, "'target' missing in test.csv"

    # Load model
    FetalHealthPredictor = load_model_class()
    model = FetalHealthPredictor()

    # Train
    model.fit(df_train)

    # Predict
    X_test = df_test.drop("target", axis=1)
    y_true = df_test["target"].astype(float)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # -----------------------------
    # Basic output checks
    # -----------------------------
    assert len(y_pred) == len(X_test), "predict() length mismatch"
    assert isinstance(y_proba, np.ndarray), "predict_proba() must return numpy array"
    assert y_proba.shape == (len(X_test), 3), "predict_proba() must return shape (n, 3)"
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Invalid probability values"
    assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-3), "Probabilities must sum to 1"

    # -----------------------------
    # Metrics (HONEST evaluation)
    # -----------------------------
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recall_c3 = recall_score(
        y_true, y_pred, labels=[1.0, 2.0, 3.0], average=None
    )[2]

    print("\nFINAL TEST METRICS")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Recall C3: {recall_c3:.4f}")

    # -----------------------------
    # Thresholds
    # -----------------------------
    assert accuracy >= 0.92, f"Accuracy too low: {accuracy:.4f}"
    assert macro_f1 >= 0.86, f"Macro F1 too low: {macro_f1:.4f}"
    assert recall_c3 >= 0.84, f"Recall(class 3) too low: {recall_c3:.4f}"
