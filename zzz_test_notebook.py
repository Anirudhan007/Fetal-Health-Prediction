import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score


# -------------------------------------------------------------
# Fixture: Handle Dynamic Import of the Model
# -------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor_class():
    """
    Locates /workspace/results/utils.py, dynamically imports it as a module,
    and returns the FetalHealthPredictor class.
    """
    source_path = Path("/workspace/results/utils.py")

    if not source_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Source file '{source_path}' does not exist.")

    try:
        spec = importlib.util.spec_from_file_location("utils_module", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "FetalHealthPredictor"):
            pytest.fail("CRITICAL FAILURE: Class 'FetalHealthPredictor' not found.")

        return module.FetalHealthPredictor

    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Failed to import model. Error: {e}")


# -------------------------------------------------------------
# Fixture: Handle Data Loading
# -------------------------------------------------------------

@pytest.fixture(scope="module")
def fetal_health_data():
    """
    Loads Fetal Health train/test datasets as full dataframes (including `target`).
    """
    train_path = Path("/workspace/data/train.csv")
    test_path = Path("/tests/test.csv")

    if not train_path.exists() or not test_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Data files missing.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if "target" not in df_train.columns:
        pytest.fail("CRITICAL FAILURE: 'target' column missing from train.csv")

    if "target" not in df_test.columns:
        pytest.fail("CRITICAL FAILURE: 'target' column missing from test.csv")

    return df_train, df_test


# -------------------------------------------------------------
# Fixture: Trained Model Instance
# -------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model(predictor_class, fetal_health_data):
    """
    Creates and trains a model instance.
    """
    FetalHealthPredictor = predictor_class
    df_train, df_test = fetal_health_data

    model = FetalHealthPredictor()
    model.fit(df_train)

    return model, df_train, df_test


# -------------------------------------------------------------
# Test: File Existence
# -------------------------------------------------------------

def test_file_exists():
    source_path = Path("/workspace/results/utils.py")
    assert source_path.exists(), "utils.py does not exist"


# -------------------------------------------------------------
# Test: Class Structure
# -------------------------------------------------------------

def test_class_structure(predictor_class):
    model = predictor_class()

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    assert callable(model.fit)
    assert callable(model.predict)
    assert callable(model.predict_proba)


# -------------------------------------------------------------
# Test: Model Training
# -------------------------------------------------------------

def test_model_training(predictor_class, fetal_health_data):
    df_train, df_test = fetal_health_data
    model = predictor_class()

    try:
        model.fit(df_train)
    except Exception as e:
        pytest.fail(f"Model training failed: {e}")


# -------------------------------------------------------------
# Test: Predict Method
# -------------------------------------------------------------

def test_predict_method(trained_model):
    model, df_train, df_test = trained_model

    X = df_test.drop("target", axis=1)
    y_pred = model.predict(X)

    assert len(y_pred) == len(X)
    assert isinstance(y_pred, (list, np.ndarray))

    allowed_classes = {1.0, 2.0, 3.0}
    unique_vals = set(np.array(y_pred).astype(float))
    assert unique_vals.issubset(allowed_classes)


# -------------------------------------------------------------
# Test: Predict Proba Method
# -------------------------------------------------------------

def test_predict_proba_method(trained_model):
    model, df_train, df_test = trained_model

    X = df_test.drop("target", axis=1)
    y_pred_proba = model.predict_proba(X)

    assert isinstance(y_pred_proba, np.ndarray)
    assert y_pred_proba.ndim == 2
    assert y_pred_proba.shape == (len(X), 3)

    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0, atol=1e-3)


# -------------------------------------------------------------
# Test: Final Model Performance (HONEST EVALUATION)
# -------------------------------------------------------------

def test_model_performance(trained_model):
    model, df_train, df_test = trained_model

    X_test = df_test.drop("target", axis=1)
    y_true = df_test["target"].astype(float)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recall_c3 = recall_score(y_true, y_pred, labels=[1.0, 2.0, 3.0], average=None)[2]

    print("\nFINAL TEST METRICS")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Recall C3: {recall_c3:.4f}")

    assert accuracy >= 0.92, f"Accuracy too low: {accuracy:.4f}"
    assert macro_f1 >= 0.86, f"Macro F1 too low: {macro_f1:.4f}"
    assert recall_c3 >= 0.84, f"Recall(class 3) too low: {recall_c3:.4f}"
