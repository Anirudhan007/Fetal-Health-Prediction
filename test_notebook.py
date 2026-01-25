"""
Use this file to define pytest tests that verify the outputs of the Fetal Health Prediction task.
This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# ==============================================================================
# CONFIGURATION - File paths inside the container
# ==============================================================================

NOTEBOOK_DIR = Path("/workspace")
TEST_DIR = Path("/tests")
VERIFIER_DIR = Path("/logs/verifier")

# Where the harness saves captured notebook variables (if needed)
NOTEBOOK_VARS_PATH = VERIFIER_DIR / "notebook_variables.json"

# ==============================================================================
# EXPECTED VALUES FROM solution.ipynb output
# ==============================================================================

EXPECTED_MODEL_QUALITY = {
    "auc": 0.98783,
    "f1":  0.8941,
}

EXPECTED_FEATURE_IMPORTANCE = {
    'baseline value': 0.03411, 
    'accelerations': 0.03331, 
    'fetal_movement': 0.0155, 
    'uterine_contractions': 0.0247, 
    'light_decelerations': 0.00546, 
    'severe_decelerations': 0.00029, 
    'prolongued_decelerations': 0.0385, 
    'abnormal_short_term_variability': 0.1031, 
    'mean_value_of_short_term_variability': 0.0988, 
    'percentage_of_time_with_abnormal_long_term_variability': 0.09199, 
    'mean_value_of_long_term_variability': 0.04015, 
    'MajorDecelBurden': 0.03871, 
    'VariabilityAbnormalityIndex': 0.12912, 
    'TotalDecelerations': 0.00874, 
    'ReassuranceRatio': 0.03089, 
    'histogram_width': 0.02905, 
    'histogram_min': 0.02733, 
    'histogram_max': 0.02505, 
    'histogram_number_of_peaks': 0.0184, 
    'histogram_number_of_zeroes': 0.00562, 
    'histogram_mode': 0.04659, 
    'histogram_mean': 0.07398, 
    'histogram_median': 0.04725, 
    'histogram_variance': 0.02622, 
    'histogram_tendency': 0.00713, 
    'health_insurance': 0.0
}

REQUIRED_MODEL_QUALITY_KEYS = ["f1", "auc"]

# NOTE: We expect ALL keys in EXPECTED_FEATURE_IMPORTANCE
REQUIRED_FEATURE_IMPORTANCE_KEYS = list(EXPECTED_FEATURE_IMPORTANCE.keys())


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def notebook_variables() -> dict:
    """Load and return the variables assigned in the notebook."""
    if not NOTEBOOK_VARS_PATH.exists():
        pytest.skip("notebook_variables.json not found")
    with open(NOTEBOOK_VARS_PATH, "r") as f:
        return json.load(f)


# ==============================================================================
# BASIC EXISTENCE TESTS
# ==============================================================================

def test_notebook_exists() -> None:
    """Test that the notebook exists."""
    notebook_path = NOTEBOOK_DIR / "notebook.ipynb"
    assert notebook_path.exists(), (
        f"Notebook 'notebook.ipynb' not found in {NOTEBOOK_DIR}"
    )


# ==============================================================================
# VARIABLE EXISTENCE TESTS
# ==============================================================================

def test_feature_importance_dict_exists(notebook_variables: dict) -> None:
    """Test that feature_importance_dict exists in the environment."""
    assert "feature_importance_dict" in notebook_variables, (
        "feature_importance_dict must exist in the environment"
    )
    assert isinstance(notebook_variables["feature_importance_dict"], dict), (
        "feature_importance_dict must be a dictionary"
    )


def test_model_quality_exists(notebook_variables: dict) -> None:
    """Test that model_quality exists in the environment."""
    assert "model_quality" in notebook_variables, (
        "model_quality must exist in the environment"
    )
    assert isinstance(notebook_variables["model_quality"], dict), (
        "model_quality must be a dictionary"
    )


def test_fetal_status_exists(notebook_variables: dict) -> None:
    """Test that fetal_status exists in the environment."""
    assert "fetal_status" in notebook_variables, (
        "fetal_status must exist in the environment"
    )
    # fetal_status should be serialized using orient='split', therefore a dict
    assert isinstance(notebook_variables["fetal_status"], dict), (
        "fetal_status must be serialized as a dictionary using to_dict(orient='split')"
    )


# ==============================================================================
# KEYS TESTS
# ==============================================================================

def test_model_quality_keys_correct(notebook_variables: dict) -> None:
    """Test that model_quality has f1 and auc keys."""
    model_quality = notebook_variables.get("model_quality", {})
    actual_keys = set(model_quality.keys())
    expected_keys = set(REQUIRED_MODEL_QUALITY_KEYS)

    assert actual_keys == expected_keys, (
        f"model_quality must contain exactly these keys: {REQUIRED_MODEL_QUALITY_KEYS}. "
        f"Got: {sorted(actual_keys)}"
    )


def test_feature_importance_dict_keys_correct(notebook_variables: dict) -> None:
    """Test that feature_importance_dict has all required feature keys."""
    feature_importance_dict = notebook_variables.get("feature_importance_dict", {})
    actual_keys = set(feature_importance_dict.keys())
    expected_keys = set(REQUIRED_FEATURE_IMPORTANCE_KEYS)

    assert actual_keys == expected_keys, (
        "feature_importance_dict must contain exactly the expected feature keys. "
        f"Missing: {sorted(expected_keys - actual_keys)} | "
        f"Extra: {sorted(actual_keys - expected_keys)}"
    )


# ==============================================================================
# VALUE ACCURACY TESTS (MODEL QUALITY)
# ==============================================================================

def test_auc_within_tolerance(notebook_variables: dict) -> None:
    """Test that AUC matches expected value within tolerance."""
    model_quality = notebook_variables.get("model_quality", {})
    auc = model_quality.get("auc")

    assert auc is not None, "auc key not found in model_quality"

    expected = EXPECTED_MODEL_QUALITY["auc"]
    tolerance = 0.005  # absolute tolerance

    assert abs(auc - expected) <= tolerance, (
        f"AUC should be {expected} ± {tolerance}, got {auc}"
    )


def test_f1_within_tolerance(notebook_variables: dict) -> None:
    """Test that F1 matches expected value within tolerance."""
    model_quality = notebook_variables.get("model_quality", {})
    f1 = model_quality.get("f1")

    assert f1 is not None, "f1 key not found in model_quality"

    expected = EXPECTED_MODEL_QUALITY["f1"]
    tolerance = 0.01  # absolute tolerance

    assert abs(f1 - expected) <= tolerance, (
        f"F1 should be {expected} ± {tolerance}, got {f1}"
    )


def test_model_quality_values_reasonable_bounds(notebook_variables: dict) -> None:
    """Test that model_quality values are valid probabilities in [0, 1]."""
    model_quality = notebook_variables.get("model_quality", {})

    auc = model_quality.get("auc")
    f1 = model_quality.get("f1")

    assert auc is not None, "auc key not found in model_quality"
    assert f1 is not None, "f1 key not found in model_quality"

    assert 0.0 <= auc <= 1.0, f"AUC must be between 0 and 1, got {auc}"
    assert 0.0 <= f1 <= 1.0, f"F1 must be between 0 and 1, got {f1}"


# ==============================================================================
# VALUE ACCURACY TESTS (FEATURE IMPORTANCE)
# ==============================================================================

def test_feature_importance_values_within_tolerance(notebook_variables: dict) -> None:
    """
    Test that all feature importances are within tolerance of expected values.
    Since RF importances can vary, we allow small absolute tolerance.
    """
    feature_importance_dict = notebook_variables.get("feature_importance_dict", {})

    abs_tolerance = 0.01  # absolute tolerance for each feature importance

    for feat in REQUIRED_FEATURE_IMPORTANCE_KEYS:
        actual = feature_importance_dict.get(feat)
        expected = EXPECTED_FEATURE_IMPORTANCE[feat]

        assert actual is not None, f"{feat} key not found in feature_importance_dict"
        assert abs(actual - expected) <= abs_tolerance, (
            f"Feature importance for '{feat}' should be {expected} ± {abs_tolerance}, got {actual}"
        )


def test_feature_importance_sum_close_to_one(notebook_variables: dict) -> None:
    """
    Feature importances in sklearn RF generally sum to 1.0.
    We check they are close to 1 within a tolerance.
    """
    feature_importance_dict = notebook_variables.get("feature_importance_dict", {})
    total = sum(feature_importance_dict.values())

    assert abs(total - 1.0) <= 0.05, (
        f"Sum of feature importances should be close to 1.0 (±0.05), got {total}"
    )

def test_final_df_exists_and_patient_id_dropped(notebook_variables: dict) -> None:
    """
    Test that final_df exists and patient_id has been dropped after merging,
    as required by instruction.md (ID leakage prevention).
    """
    assert "final_df" in notebook_variables, (
        "final_df must exist in the notebook environment after dataset integration."
    )

    final_df_obj = notebook_variables["final_df"]

    # final_df might be saved as a dict (split) or as a normal dict-like structure
    # reconstruct it safely into a DataFrame
    final_df = reconstruct_dataframe_from_dict(final_df_obj)

    assert isinstance(final_df, pd.DataFrame), "final_df must be reconstructable as a pandas DataFrame"
    assert "patient_id" not in final_df.columns, (
        "patient_id must be dropped from final_df before training (identifier leakage)."
    )

def test_health_insurance_all_true_in_final_df(notebook_variables: dict) -> None:
    assert "final_df" in notebook_variables, "final_df must exist"

    final_df = reconstruct_dataframe_from_dict(notebook_variables["final_df"])
    assert "health_insurance" in final_df.columns, "health_insurance must exist in final_df"

    vals = set(final_df["health_insurance"].astype(str).str.lower().unique())
    assert vals.issubset({"1", "true"}), f"Expected only insured rows, got values: {vals}"

# ==============================================================================
# fetal_status VALIDATION
# ==============================================================================

def reconstruct_dataframe_from_dict(data: dict) -> pd.DataFrame:
    """
    Reconstruct a pandas DataFrame from a dictionary.
    Handles dictionaries with 'split' orientation (from to_dict(orient='split')).
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        if "columns" in data and "data" in data:
            return pd.DataFrame(data["data"], columns=data["columns"])
        return pd.DataFrame(data)
    raise ValueError(f"Cannot reconstruct DataFrame from type: {type(data)}")


def test_fetal_status_structure(notebook_variables: dict) -> None:
    """Test fetal_status reconstructs to a DataFrame and contains expected info."""
    fetal_status = reconstruct_dataframe_from_dict(notebook_variables["fetal_status"])

    assert fetal_status.shape[0] > 0, "fetal_status must not be empty"

    # We don't enforce strict column names because user may name columns differently,
    # but we do ensure the table has 2 columns minimum: category and count.
    assert fetal_status.shape[1] >= 2, (
        f"fetal_status should have at least 2 columns (category + count), got {fetal_status.shape[1]}"
    )


def test_fetal_status_contains_all_classes(notebook_variables: dict) -> None:
    """Test fetal_status contains fetal_health categories 1, 2, 3."""
    fetal_status = reconstruct_dataframe_from_dict(notebook_variables["fetal_status"])

    # Attempt to locate class/category column
    # Common patterns: 'fetal_health', 'class', 'label', 'category'
    possible_class_cols = ["fetal_health", "class", "label", "category"]
    class_col = None
    for c in possible_class_cols:
        if c in fetal_status.columns:
            class_col = c
            break

    # If not found, just use first column heuristically
    if class_col is None:
        class_col = fetal_status.columns[0]

    present_classes = set(pd.Series(fetal_status[class_col]).astype(int).tolist())

    assert {1, 2, 3}.issubset(present_classes), (
        f"fetal_status must contain fetal health classes 1, 2, 3. Got: {sorted(present_classes)}"
    )


def test_fetal_status_counts_are_non_negative(notebook_variables: dict) -> None:
    """Test fetal_status counts are valid (non-negative integers)."""
    fetal_status = reconstruct_dataframe_from_dict(notebook_variables["fetal_status"])

    # Attempt to locate count column
    possible_count_cols = ["count", "counts", "n", "frequency"]
    count_col = None
    for c in possible_count_cols:
        if c in fetal_status.columns:
            count_col = c
            break

    # If not found, assume second column is count
    if count_col is None and fetal_status.shape[1] >= 2:
        count_col = fetal_status.columns[1]

    counts = pd.Series(fetal_status[count_col]).astype(int)

    assert (counts >= 0).all(), (
        "All fetal_status counts must be non-negative"
    )
