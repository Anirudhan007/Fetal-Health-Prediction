import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path


# -------------------------------------------------------------
# Fixture: Handle Dynamic Import of the Model
# -------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor_class():
    """
    Locates /workspace/results/utils.py, dynamically imports it as a module,
    and returns the FetalHealthPredictor class.

    Fails the test suite immediately if the file or class is missing.
    """
    source_path = Path("/workspace/results/utils.py")

    # 1. Check file existence
    if not source_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Source file '{source_path}' does not exist.")

    # 2. Dynamic Import
    try:
        spec = importlib.util.spec_from_file_location("utils_module", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)

        # 3. Check for Class Existence
        if not hasattr(module, "FetalHealthPredictor"):
            pytest.fail(f"CRITICAL FAILURE: Class 'FetalHealthPredictor' not found in {source_path}")

        return module.FetalHealthPredictor

    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Failed to import model from {source_path}. Error: {e}")


# -------------------------------------------------------------
# Fixture: Handle Data Loading
# -------------------------------------------------------------

@pytest.fixture(scope="module")
def fetal_health_data():
    """
    Loads Fetal Health train/test datasets as full dataframes (including `fetal_health`).
    """
    train_path = Path("/workspace/data/train.csv")
    test_path = Path("/tests/test.csv")

    if not train_path.exists() or not test_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Data files missing. Train: {train_path}, Test: {test_path}")

    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        if "fetal_health" not in df_train.columns:
            pytest.fail("CRITICAL FAILURE: 'fetal_health' column missing from train.csv")

        return df_train, df_test

    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Error processing data files: {e}")


# -------------------------------------------------------------
# Fixture: Trained Model Instance
# -------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model(predictor_class, fetal_health_data):
    """
    Creates and trains a model instance for use in multiple tests.
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
    """
    Verify that the utils.py file exists at the expected location.
    """
    source_path = Path("/workspace/results/utils.py")
    assert source_path.exists(), f"Source file '{source_path}' does not exist."


# -------------------------------------------------------------
# Test: Class Structure
# -------------------------------------------------------------

def test_class_structure(predictor_class):
    """
    Verify that the FetalHealthPredictor class has the required methods.
    """
    FetalHealthPredictor = predictor_class

    # Test instantiation
    model = FetalHealthPredictor()

    # Check required methods exist
    assert hasattr(model, "fit"), "Model must have 'fit' method"
    assert hasattr(model, "predict"), "Model must have 'predict' method"
    assert hasattr(model, "predict_proba"), "Model must have 'predict_proba' method"

    # Check methods are callable
    assert callable(model.fit), "'fit' must be callable"
    assert callable(model.predict), "'predict' must be callable"
    assert callable(model.predict_proba), "'predict_proba' must be callable"


# -------------------------------------------------------------
# Test: Model Training
# -------------------------------------------------------------

def test_model_training(predictor_class, fetal_health_data):
    """
    Verify that the model can be trained without errors.
    """
    FetalHealthPredictor = predictor_class
    df_train, df_test = fetal_health_data

    model = FetalHealthPredictor()

    # Training should complete without exception
    try:
        model.fit(df_train)
    except Exception as e:
        pytest.fail(f"Model training failed with error: {e}")


# -------------------------------------------------------------
# Test: Predict Method
# -------------------------------------------------------------

def test_predict_method(trained_model):
    """
    Verify that the predict() method returns valid multi-class predictions.
    """
    model, df_train, df_test = trained_model

    # Get predictions
    y_pred = model.predict(df_test.drop("fetal_health", axis=1))

    # Check output shape
    assert len(y_pred) == len(df_test), (
        f"Prediction length {len(y_pred)} doesn't match test data length {len(df_test)}"
    )

    # Check output type
    assert isinstance(y_pred, (np.ndarray, list)), "Predictions must be array-like"

    # Validate class labels (as floats: 1.0, 2.0, 3.0)
    allowed_classes = {1.0, 2.0, 3.0}
    unique_values = set(np.array(y_pred).astype(float).tolist())
    assert unique_values.issubset(allowed_classes), (
        f"Predictions must be in {allowed_classes}, got {unique_values}"
    )


# -------------------------------------------------------------
# Test: Predict Proba Method
# -------------------------------------------------------------

def test_predict_proba_method(trained_model):
    """
    Verify that the predict_proba() method returns valid probability matrix.
    """
    model, df_train, df_test = trained_model

    # Get probability predictions
    y_pred_proba = model.predict_proba(df_test.drop("fetal_health", axis=1))

    # Must be numpy array
    assert isinstance(y_pred_proba, np.ndarray), "predict_proba must return a numpy array"

    # Must be 2D array: (n_samples, n_classes)
    assert y_pred_proba.ndim == 2, f"predict_proba must return a 2D array, got ndim={y_pred_proba.ndim}"

    n_samples = len(df_test)
    assert y_pred_proba.shape[0] == n_samples, (
        f"predict_proba must return shape (n_samples, n_classes). "
        f"Expected first dimension {n_samples}, got {y_pred_proba.shape[0]}"
    )

    # n_classes should be 3 for fetal_health (Normal, Suspect, Pathological)
    assert y_pred_proba.shape[1] == 3, (
        f"predict_proba must return probabilities for 3 classes, got shape {y_pred_proba.shape}"
    )

    # Probabilities must lie in [0, 1]
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "All probabilities must be between 0 and 1"

    # Row sums should be approx 1
    row_sums = y_pred_proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3), (
        "Each row of predict_proba output must sum to 1 (within tolerance)"
    )


# -------------------------------------------------------------
# Test: Model Performance
# -------------------------------------------------------------

def test_model_performance(trained_model):
    """
    Verify that the model achieves the required accuracy performance (>= 85%).
    """
    model, df_train, df_test = trained_model

    # Get true labels
    y_true = df_test["fetal_health"]

    # Get predictions
    y_pred = model.predict(df_test.drop("fetal_health", axis=1))

    # Calculate accuracy independently
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)

    target_accuracy = 0.85
    print(f"\nModel Accuracy Score: {accuracy:.4f}")  # Visible with pytest -s

    # Check accuracy is in valid range
    assert 0.0 <= accuracy <= 1.0, f"Accuracy must be in [0, 1], got {accuracy:.4f}"

    # Check accuracy meets minimum threshold
    assert accuracy >= target_accuracy, (
        f"Performance Failure: Model Accuracy {accuracy:.4f} is below the "
        f"required threshold of {target_accuracy} (85%)"
    )
