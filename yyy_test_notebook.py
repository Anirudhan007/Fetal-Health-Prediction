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
    Loads Fetal Health train/test datasets as full dataframes (including `target`).
    """
    train_path = Path("/workspace/data/train.csv")
    test_path = Path("/tests/test.csv")

    if not train_path.exists() or not test_path.exists():
        pytest.fail(f"CRITICAL FAILURE: Data files missing. Train: {train_path}, Test: {test_path}")

    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        if "target" not in df_train.columns:
            pytest.fail("CRITICAL FAILURE: 'target' column missing from train.csv")

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

    # Get predictions (pass full feature DF except target)
    y_pred = model.predict(df_test.drop("target", axis=1))

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
    y_pred_proba = model.predict_proba(df_test.drop("target", axis=1))

    # Must be numpy array
    assert isinstance(y_pred_proba, np.ndarray), "predict_proba must return a numpy array"

    # Must be 2D array: (n_samples, n_classes)
    assert y_pred_proba.ndim == 2, f"predict_proba must return a 2D array, got ndim={y_pred_proba.ndim}"

    n_samples = len(df_test)
    assert y_pred_proba.shape[0] == n_samples, (
        f"predict_proba must return shape (n_samples, n_classes). "
        f"Expected first dimension {n_samples}, got {y_pred_proba.shape[0]}"
    )

    # n_classes should be 3 for target (Normal, Suspect, Pathological)
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
# Additional Test 1: Robustness to Missing Values
# -------------------------------------------------------------

def test_predict_with_missing_values(trained_model):
    """
    Verify model can handle missing values in input features.
    """
    model, df_train, df_test = trained_model

    X = df_test.drop("target", axis=1).copy()
    if X.shape[1] > 0:
        X.iloc[0, 0] = np.nan

    y_pred = model.predict(X)

    assert len(y_pred) == len(X), "Model must still return predictions with missing values present"


# -------------------------------------------------------------
# Additional Test 2: Robustness to Shuffled Columns
# -------------------------------------------------------------

def test_predict_with_shuffled_columns(trained_model):
    """
    Verify model can handle shuffled feature column order.
    """
    model, df_train, df_test = trained_model

    X = df_test.drop("target", axis=1).copy()
    cols = list(X.columns)
    np.random.shuffle(cols)
    X_shuffled = X[cols]

    y_pred = model.predict(X_shuffled)

    assert len(y_pred) == len(X_shuffled), "Model must handle shuffled feature order"


# -------------------------------------------------------------
# Additional Test 3: Robustness to Missing Column
# -------------------------------------------------------------

def test_predict_with_missing_column(trained_model):
    """
    Verify model can handle a missing feature column gracefully.
    """
    model, df_train, df_test = trained_model

    X = df_test.drop("target", axis=1).copy()

    # Drop a random column if possible (do not guide feature choice)
    if X.shape[1] > 1:
        drop_col = np.random.choice(list(X.columns))
        X = X.drop(drop_col, axis=1)

    y_pred = model.predict(X)

    assert len(y_pred) == len(X), "Model must still return predictions even if a column is missing"


# -------------------------------------------------------------
# Additional Test 4: Probability Quality Sanity Check
# -------------------------------------------------------------

def test_probability_quality(trained_model):
    """
    Ensure probabilities are not degenerate (e.g., always uniform).
    """
    model, df_train, df_test = trained_model

    X = df_test.drop("target", axis=1)
    proba = model.predict_proba(X)

    mean_max = float(np.mean(np.max(proba, axis=1)))
    assert mean_max >= 0.45, f"Probabilities appear too uniform/uncertain. Mean max={mean_max:.4f}"


# -------------------------------------------------------------
# Test: Model Performance (Hard Mode)
# -------------------------------------------------------------

def test_model_performance(trained_model):
    """
    Harder evaluation:
    - Accuracy >= 0.92
    - Macro F1 >= 0.86
    - Recall(class 3) >= 0.84
    """
    model, df_train, df_test = trained_model

    # Get true labels
    y_true = df_test["target"].astype(float)

    # Get predictions
    y_pred = model.predict(df_test.drop("target", axis=1))
    y_pred = np.array(y_pred).astype(float)

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recall_per_class = recall_score(y_true, y_pred, labels=[1.0, 2.0, 3.0], average=None)
    recall_class3 = float(recall_per_class[2])

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Recall C3: {recall_class3:.4f}")

    # Thresholds
    assert accuracy >= 0.92, f"Accuracy too low: {accuracy:.4f} < 0.92"
    assert macro_f1 >= 0.86, f"Macro F1 too low: {macro_f1:.4f} < 0.86"
    assert recall_class3 >= 0.84, f"Class-3 recall too low: {recall_class3:.4f} < 0.84"
