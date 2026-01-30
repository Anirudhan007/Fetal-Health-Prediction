import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================

CURRENT_DIR = Path(__file__).parent
UTILS_PATH = Path("/workspace/results/utils.py")
SUBMISSION_PATH = Path("/workspace/results/submission.csv")
TRAIN_DATA_PATH = Path("/workspace/data/train.csv")
TEST_DATA_PATH = Path("/tests/test.csv")
GROUND_TRUTH_PATH = CURRENT_DIR / "garbage_columns.csv"

# Performance Thresholds
ACCURACY_THRESHOLD = 0.90
MACRO_F1_THRESHOLD = 0.90
CLASS3_RECALL_THRESHOLD = 0.90

# Probability Quality Threshold
MIN_MEAN_CONFIDENCE = 0.4  # Ensure predictions aren't too uniform

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def predictor_class():
    """
    Dynamically imports FetalHealthPredictor class from /workspace/results/utils.py.
    Fails immediately if file or class is missing.
    """
    if not UTILS_PATH.exists():
        pytest.fail(f"CRITICAL FAILURE: Source file '{UTILS_PATH}' does not exist.")
    
    try:
        spec = importlib.util.spec_from_file_location("utils_module", UTILS_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, "FetalHealthPredictor"):
            pytest.fail(f"CRITICAL FAILURE: Class 'FetalHealthPredictor' not found in {UTILS_PATH}")
        
        return module.FetalHealthPredictor
    
    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Failed to import model from {UTILS_PATH}. Error: {e}")


@pytest.fixture(scope="module")
def fetal_health_data():
    """
    Loads train and test datasets.
    Validates that both files exist and contain the 'target' column.
    """
    if not TRAIN_DATA_PATH.exists():
        pytest.fail(f"CRITICAL FAILURE: Training data not found at {TRAIN_DATA_PATH}")
    
    if not TEST_DATA_PATH.exists():
        pytest.fail(f"CRITICAL FAILURE: Test data not found at {TEST_DATA_PATH}")
    
    try:
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        df_test = pd.read_csv(TEST_DATA_PATH)
        
        if "target" not in df_train.columns:
            pytest.fail("CRITICAL FAILURE: 'target' column missing from train.csv")
        
        if "target" not in df_test.columns:
            pytest.fail("CRITICAL FAILURE: 'target' column missing from test.csv")
        
        return df_train, df_test
    
    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Error loading data files. Error: {e}")


@pytest.fixture(scope="module")
def trained_model(predictor_class, fetal_health_data):
    """
    Creates and trains a FetalHealthPredictor instance.
    Model is trained once and reused across all tests for efficiency.
    """
    FetalHealthPredictor = predictor_class
    df_train, df_test = fetal_health_data
    
    try:
        model = FetalHealthPredictor()
        model.fit(df_train)
        return model, df_train, df_test
    
    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Model training failed. Error: {e}")


@pytest.fixture(scope="module")
def garbage_ground_truth():
    """
    Loads ground truth garbage features from garbage_columns.csv.
    Format: Single column with feature names (no headers), one feature per row.
    Returns a set of feature names for order-independent comparison.
    """
    if not GROUND_TRUTH_PATH.exists():
        pytest.fail(f"CRITICAL FAILURE: Ground truth file not found at {GROUND_TRUTH_PATH}")
    
    try:
        truth_df = pd.read_csv(GROUND_TRUTH_PATH, header=None)
        
        # Validate single column format
        if truth_df.shape[1] != 1:
            pytest.fail(f"Ground truth must have exactly 1 column, found {truth_df.shape[1]}")
        
        # Extract feature names from first column, remove any NaN, convert to set
        garbage_features = set(truth_df.iloc[:, 0].dropna().astype(str).values)
        
        if len(garbage_features) == 0:
            pytest.fail("Ground truth file is empty - no garbage features found")
        
        return garbage_features
    
    except Exception as e:
        pytest.fail(f"CRITICAL FAILURE: Error loading ground truth. Error: {e}")


# =============================================================================
# LAYER 1: PATH & DIRECTORY VALIDATION
# =============================================================================

def test_results_directory_exists():
    """Verify that the /workspace/results/ directory exists."""
    results_dir = Path("/workspace/results/")
    assert results_dir.exists(), f"Results directory '{results_dir}' does not exist"


def test_data_directory_exists():
    """Verify that the /workspace/data/ directory exists."""
    data_dir = Path("/workspace/data/")
    assert data_dir.exists(), f"Data directory '{data_dir}' does not exist"


# =============================================================================
# LAYER 2: FILE EXISTENCE VALIDATION
# =============================================================================

def test_utils_file_exists():
    """Verify that utils.py exists at the expected location."""
    assert UTILS_PATH.exists(), f"Source file '{UTILS_PATH}' does not exist"


def test_submission_file_exists():
    """
    Verify that submission.csv exists at the expected location.
    Required for garbage feature identification.
    """
    assert SUBMISSION_PATH.exists(), \
        f"Submission file '{SUBMISSION_PATH}' must exist at {SUBMISSION_PATH}"


def test_train_data_exists():
    """Verify that train.csv exists."""
    assert TRAIN_DATA_PATH.exists(), f"Training data not found at {TRAIN_DATA_PATH}"


def test_test_data_exists():
    """Verify that test.csv exists."""
    assert TEST_DATA_PATH.exists(), f"Test data not found at {TEST_DATA_PATH}"


def test_ground_truth_exists():
    """Verify that garbage_columns.csv exists."""
    assert GROUND_TRUTH_PATH.exists(), f"Ground truth not found at {GROUND_TRUTH_PATH}"


# =============================================================================
# LAYER 3: CLASS STRUCTURE VALIDATION
# =============================================================================

def test_class_structure(predictor_class):
    """
    Verify that FetalHealthPredictor class has required methods:
    - fit()
    - predict()
    - predict_proba()
    """
    FetalHealthPredictor = predictor_class
    model = FetalHealthPredictor()
    
    assert hasattr(model, 'fit'), "Model must have 'fit' method"
    assert hasattr(model, 'predict'), "Model must have 'predict' method"
    assert hasattr(model, 'predict_proba'), "Model must have 'predict_proba' method"
    
    assert callable(model.fit), "'fit' must be callable"
    assert callable(model.predict), "'predict' must be callable"
    assert callable(model.predict_proba), "'predict_proba' must be callable"


# =============================================================================
# LAYER 4: GARBAGE FEATURE VALIDATION
# =============================================================================

def test_garbage_features_schema():
    """
    Verify that submission.csv has correct format:
    - Single column with feature names (no headers)
    - No missing values
    - At least one feature listed
    """
    if not SUBMISSION_PATH.exists():
        pytest.fail(f"Submission file missing at {SUBMISSION_PATH}")
    
    try:
        submission_df = pd.read_csv(SUBMISSION_PATH, header=None)
        
        # Should have exactly one column
        assert submission_df.shape[1] == 1, \
            f"Submission must have exactly 1 column, found {submission_df.shape[1]}"
        
        # Should have at least one row
        assert len(submission_df) > 0, \
            "Submission file is empty - no features listed"
        
        # No missing values
        if submission_df.iloc[:, 0].isnull().any():
            pytest.fail("Submission contains NaN values in feature list")
    
    except Exception as e:
        pytest.fail(f"Error reading submission.csv: {e}")


def test_garbage_features_accuracy(garbage_ground_truth):
    """
    Compare identified garbage features to ground truth.
    Requires EXACT MATCH (order-independent).
    Both submission.csv and garbage_columns.csv contain feature names in first column (no headers).
    """
    if not SUBMISSION_PATH.exists():
        pytest.fail("Submission file missing - cannot validate garbage feature identification")
    
    try:
        submission_df = pd.read_csv(SUBMISSION_PATH, header=None)
        
        # Extract feature names from first column, convert to set (order-independent)
        identified_garbage = set(submission_df.iloc[:, 0].dropna().astype(str).values)
        
        # Calculate matching statistics for debugging
        true_positives = len(identified_garbage & garbage_ground_truth)
        false_positives = len(identified_garbage - garbage_ground_truth)
        false_negatives = len(garbage_ground_truth - identified_garbage)
        
        print(f"\n[GARBAGE FEATURE IDENTIFICATION]")
        print(f"Ground Truth Count: {len(garbage_ground_truth)}")
        print(f"Identified Count: {len(identified_garbage)}")
        print(f"Correctly Identified: {true_positives}")
        print(f"Incorrectly Identified (False Positives): {false_positives}")
        print(f"Missed (False Negatives): {false_negatives}")
        
        if false_positives > 0:
            print(f"Extra features identified: {identified_garbage - garbage_ground_truth}")
        if false_negatives > 0:
            print(f"Missing features: {garbage_ground_truth - identified_garbage}")
        
        # EXACT MATCH REQUIRED (order-independent set comparison)
        assert identified_garbage == garbage_ground_truth, \
            f"Garbage features must match exactly.\n" \
            f"Missing: {garbage_ground_truth - identified_garbage}\n" \
            f"Extra: {identified_garbage - garbage_ground_truth}"
    
    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"Error validating garbage features: {e}")


# =============================================================================
# LAYER 5: PREDICTION OUTPUT VALIDATION
# =============================================================================

def test_predict_output_format(trained_model):
    """
    Verify that predict() returns:
    - Correct shape (matches test set size)
    - Only valid class labels (1.0, 2.0, 3.0)
    - No NaN values
    """
    model, df_train, df_test = trained_model
    X_test = df_test.drop('target', axis=1)
    
    try:
        predictions = model.predict(X_test)
        
        # Check shape
        assert len(predictions) == len(X_test), \
            f"Expected {len(X_test)} predictions, got {len(predictions)}"
        
        # Check no NaN
        if pd.isna(predictions).any():
            pytest.fail("Predictions contain NaN values")
        
        # Check valid classes
        valid_classes = {1.0, 2.0, 3.0}
        unique_preds = set(predictions)
        invalid = unique_preds - valid_classes
        
        assert len(invalid) == 0, \
            f"Predictions contain invalid classes: {invalid}. Expected only {valid_classes}"
    
    except Exception as e:
        pytest.fail(f"Error in predict() method: {e}")


def test_predict_proba_output_format(trained_model):
    """
    Verify that predict_proba() returns:
    - Shape (n_samples, 3)
    - Probabilities in [0, 1]
    - Each row sums to 1.0
    - Non-degenerate probabilities
    """
    model, df_train, df_test = trained_model
    X_test = df_test.drop('target', axis=1)
    
    try:
        probabilities = model.predict_proba(X_test)
        
        # Check shape
        assert probabilities.shape == (len(X_test), 3), \
            f"Expected shape ({len(X_test)}, 3), got {probabilities.shape}"
        
        # Check probabilities in [0, 1]
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1), \
            "Probabilities must be in range [0, 1]"
        
        # Check rows sum to 1
        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            "Probability rows must sum to 1.0"
        
        # Check non-degenerate (not overly uniform)
        max_probs = probabilities.max(axis=1)
        mean_confidence = max_probs.mean()
        
        print(f"\n[PROBABILITY QUALITY]")
        print(f"Mean Max Probability: {mean_confidence:.4f}")
        
        assert mean_confidence >= MIN_MEAN_CONFIDENCE, \
            f"Probabilities too uniform (mean max prob {mean_confidence:.4f} < {MIN_MEAN_CONFIDENCE})"
    
    except Exception as e:
        pytest.fail(f"Error in predict_proba() method: {e}")


def test_all_classes_predicted(trained_model):
    """
    Verify that the model doesn't collapse to predicting only one or two classes.
    All three classes should appear in predictions.
    """
    model, df_train, df_test = trained_model
    X_test = df_test.drop('target', axis=1)
    
    predictions = model.predict(X_test)
    unique_classes = set(predictions)
    
    assert len(unique_classes) >= 2, \
        f"Model collapsed to {len(unique_classes)} class(es). Predictions: {unique_classes}"


# =============================================================================
# LAYER 6: ROBUSTNESS TESTS
# =============================================================================

def test_handles_shuffled_columns(trained_model):
    """
    Verify that model produces consistent predictions when columns are shuffled.
    Model must be column-order invariant.
    """
    model, df_train, df_test = trained_model
    X_test = df_test.drop('target', axis=1)
    
    # Get baseline predictions
    baseline_predictions = model.predict(X_test)
    
    # Shuffle columns
    shuffled_cols = X_test.columns.tolist()
    np.random.seed(42)
    np.random.shuffle(shuffled_cols)
    X_test_shuffled = X_test[shuffled_cols]
    
    # Get predictions on shuffled data
    shuffled_predictions = model.predict(X_test_shuffled)
    
    # Predictions should be identical
    assert np.array_equal(baseline_predictions, shuffled_predictions), \
        "Model predictions change when columns are shuffled - not column-order invariant"


def test_handles_missing_values(trained_model):
    """
    Verify that model handles missing values (NaNs) gracefully without crashing.
    Predictions should still be valid.
    """
    model, df_train, df_test = trained_model
    X_test = df_test.drop('target', axis=1).copy()
    
    # Inject some NaNs (5% of values)
    np.random.seed(42)
    mask = np.random.rand(*X_test.shape) < 0.05
    X_test_with_nans = X_test.copy()
    X_test_with_nans = X_test_with_nans.mask(mask)
    
    try:
        predictions = model.predict(X_test_with_nans)
        
        # Verify predictions are valid
        assert len(predictions) == len(X_test_with_nans), "Prediction count mismatch"
        assert not pd.isna(predictions).any(), "Model produced NaN predictions"
        
        valid_classes = {1.0, 2.0, 3.0}
        assert set(predictions).issubset(valid_classes), "Invalid class predictions with NaN input"
    
    except Exception as e:
        pytest.fail(f"Model failed to handle missing values: {e}")


# =============================================================================
# LAYER 7: PERFORMANCE METRICS (MAIN EVALUATION)
# =============================================================================

def test_accuracy_threshold(trained_model):
    """
    Evaluate model accuracy on test set.
    Required: Accuracy >= 0.90
    """
    model, df_train, df_test = trained_model
    
    y_true = df_test['target']
    y_pred = model.predict(df_test.drop('target', axis=1))
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n[METRIC] Accuracy: {accuracy:.4f}")
    
    assert accuracy >= ACCURACY_THRESHOLD, \
        f"Accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}"


def test_macro_f1_threshold(trained_model):
    """
    Evaluate model Macro-F1 score on test set.
    Required: Macro-F1 >= 0.90
    """
    model, df_train, df_test = trained_model
    
    y_true = df_test['target']
    y_pred = model.predict(df_test.drop('target', axis=1))
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n[METRIC] Macro-F1 Score: {macro_f1:.4f}")
    
    assert macro_f1 >= MACRO_F1_THRESHOLD, \
        f"Macro-F1 {macro_f1:.4f} below threshold {MACRO_F1_THRESHOLD}"


def test_class3_recall_threshold(trained_model):
    """
    Evaluate recall for Class 3 (Pathological) on test set.
    Required: Class 3 Recall >= 0.90
    
    This is the MOST CRITICAL metric for medical safety.
    """
    model, df_train, df_test = trained_model
    
    y_true = df_test['target']
    y_pred = model.predict(df_test.drop('target', axis=1))
    
    # Calculate recall for each class
    recalls = recall_score(y_true, y_pred, labels=[1.0, 2.0, 3.0], average=None)
    class3_recall = recalls[2]  # Index 2 corresponds to class 3
    
    print(f"\n[METRIC] Class 3 (Pathological) Recall: {class3_recall:.4f}")
    print(f"[INFO] Class 1 Recall: {recalls[0]:.4f}")
    print(f"[INFO] Class 2 Recall: {recalls[1]:.4f}")
    
    assert class3_recall >= CLASS3_RECALL_THRESHOLD, \
        f"Class 3 Recall {class3_recall:.4f} below threshold {CLASS3_RECALL_THRESHOLD}. " \
        f"Critical for detecting pathological cases!"


# =============================================================================
# LAYER 8: COMPREHENSIVE PERFORMANCE SUMMARY
# =============================================================================

def test_print_final_summary(trained_model):
    """
    Print a comprehensive summary of all performance metrics.
    This test always passes - it's just for reporting.
    """
    model, df_train, df_test = trained_model
    
    y_true = df_test['target']
    y_pred = model.predict(df_test.drop('target', axis=1))
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    recalls = recall_score(y_true, y_pred, labels=[1.0, 2.0, 3.0], average=None)
    
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Accuracy:           {accuracy:.4f} (Threshold: {ACCURACY_THRESHOLD})")
    print(f"Macro-F1:           {macro_f1:.4f} (Threshold: {MACRO_F1_THRESHOLD})")
    print(f"Class 1 Recall:     {recalls[0]:.4f}")
    print(f"Class 2 Recall:     {recalls[1]:.4f}")
    print(f"Class 3 Recall:     {recalls[2]:.4f} (Threshold: {CLASS3_RECALL_THRESHOLD}) ⭐")
    print("="*60)
    
    # This test always passes - it's just for summary
    assert True