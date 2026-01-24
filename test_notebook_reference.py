import json
from pathlib import Path

import pandas as pd
import pytest

NOTEBOOK_DIR = Path("/workspace")
TEST_DIR = Path("/tests")
VERIFIER_DIR = Path("/logs/verifier")

# Expected values from solution.ipynb output
EXPECTED_COEFFICIENTS = {
    "intercept": -864401.0248,
    "cost_per_kg_local": 8554.6041,
    "kg_consumed": 0.6149,
    "rate_to_USD": 36320300.4742,
    "daily_sales_kg": -0.1208,
    "lag_kg_consumed_1d": -0.0142,
    "lag_daily_sales_kg_1d": 0.1909,
}

EXPECTED_MODEL_QUALITY = {
    "rmse": 1054.041,
    "r2": 0.9964,
}

REQUIRED_COEFFICIENT_KEYS = [
    "intercept",
    "cost_per_kg_local",
    "kg_consumed",
    "rate_to_USD",
    "daily_sales_kg",
    "lag_kg_consumed_1d",
    "lag_daily_sales_kg_1d",
]

REQUIRED_MODEL_QUALITY_KEYS = ["rmse", "r2"]

@pytest.fixture
def notebook_variables() -> dict:
    """Load and return the variables assigned in the notebook."""
    variables_file = VERIFIER_DIR / "notebook_variables.json"
    if not variables_file.exists():
        pytest.skip("notebook_variables.json not found")
    with open(variables_file, "r") as f:
        return json.load(f)

def test_notebook_exists() -> None:
    """Test that the notebook exists."""
    notebook_path = NOTEBOOK_DIR / "notebook.ipynb"
    assert notebook_path.exists(), (
        f"Notebook 'notebook.ipynb' not found in {NOTEBOOK_DIR}"
    )

# ==================== Variable Existence Tests ====================

def test_coefficients_dict_exists(notebook_variables: dict) -> None:
    """Test that coefficients_dict exists in the environment."""
    assert "coefficients_dict" in notebook_variables, (
        "coefficients_dict must exist in the environment"
    )
    assert isinstance(notebook_variables["coefficients_dict"], dict), (
        "coefficients_dict must be a dictionary"
    )

def test_model_quality_exists(notebook_variables: dict) -> None:
    """Test that model_quality exists in the environment."""
    assert "model_quality" in notebook_variables, (
        "model_quality must exist in the environment"
    )
    assert isinstance(notebook_variables["model_quality"], dict), (
        "model_quality must be a dictionary"
    )

# ==================== Keys Tests ====================

def test_coefficients_dict_keys_correct(notebook_variables: dict) -> None:
    """Test that coefficients_dict has all 7 required keys."""
    coefficients_dict = notebook_variables.get("coefficients_dict", {})
    actual_keys = set(coefficients_dict.keys())
    expected_keys = set(REQUIRED_COEFFICIENT_KEYS)

    assert actual_keys == expected_keys, (
        f"coefficients_dict must contain exactly these keys: {REQUIRED_COEFFICIENT_KEYS}. "
        f"Got: {sorted(actual_keys)}"
    )

def test_model_quality_keys_correct(notebook_variables: dict) -> None:
    """Test that model_quality has rmse and r2 keys."""
    model_quality = notebook_variables.get("model_quality", {})
    actual_keys = set(model_quality.keys())
    expected_keys = set(REQUIRED_MODEL_QUALITY_KEYS)

    assert actual_keys == expected_keys, (
        f"model_quality must contain exactly these keys: {REQUIRED_MODEL_QUALITY_KEYS}. "
        f"Got: {sorted(actual_keys)}"
    )

# ==================== Value Accuracy Tests ====================

def test_intercept_value_correct(notebook_variables: dict) -> None:
    """Test that intercept is within 0.5% of expected value."""
    coefficients_dict = notebook_variables.get("coefficients_dict", {})
    intercept = coefficients_dict.get("intercept")

    assert intercept is not None, "intercept key not found in coefficients_dict"

    expected = EXPECTED_COEFFICIENTS["intercept"]
    tolerance = abs(expected * 0.005) # 0.5% tolerance

    assert abs(intercept - expected) <= tolerance, (
        f"intercept should be {expected} ± {tolerance}, got {intercept}"
    ) 

def test_kg_consumed_coef_correct(notebook_variables: dict) -> None:
    """Test that kg_consumed coefficient is within 0.5% of expected value."""
    coefficients_dict = notebook_variables.get("coefficients_dict", {})
    kg_consumed_coef = coefficients_dict.get("kg_consumed")

    assert kg_consumed_coef is not None, "kg_consumed key not found in coefficients_dict"

    expected = EXPECTED_COEFFICIENTS["kg_consumed"]
    tolerance = abs(expected * 0.005) # 0.5% tolerance

    assert abs(kg_consumed_coef - expected) <= tolerance, (
        f"kg_consumed coefficient should be {expected} ± {tolerance}, got {kg_consumed_coef}"
    )

def test_rmse_within_tolerance(notebook_variables: dict) -> None:
    """Test that RMSE is within 5 units of expected value."""
    model_quality = notebook_variables.get("model_quality", {})
    rmse = model_quality.get("rmse")

    assert rmse is not None, "rmse key not found in model_quality"

    expected = EXPECTED_MODEL_QUALITY["rmse"]
    tolerance = 5.0

    assert abs(rmse - expected) <= tolerance, (
        f"RMSE should be {expected} ± {tolerance}, got {rmse}"
    )

def test_r2_above_threshold(notebook_variables: dict) -> None:
    """Test that R2 is >= 0.99."""
    model_quality = notebook_variables.get("model_quality", {})
    r2 = model_quality.get("r2")

    assert r2 is not None, "r2 key not found in model_quality"
    assert r2 >= 0.99, (
        f"R2 should be >= 0.99, got {r2}"
    )

def test_all_coefficients_within_tolerance(notebook_variables: dict) -> None:
    """Test that all coefficients are within 0.5% of expected values."""
    coefficients_dict = notebook_variables.get("coefficients_dict", {})

    for key in REQUIRED_COEFFICIENT_KEYS:
        actual = coefficients_dict.get(key)
        expected = EXPECTED_COEFFICIENTS[key]

        assert actual is not None, f"{key} key not found in coefficients_dict"

        # For very large values (like rate_to_USD), use absolute tolerance
        if abs(expected) > 1000:
            tolerance = abs(expected * 0.005) # 0.5% tolerance
        else:
            tolerance = max(abs(expected * 0.005), 0.0001) # At least 0.0001
        
        assert abs(actual - expected) <= tolerance, (
            f"{key} should be {expected} ± {tolerance}, got {actual}"
        )

def test_model_quality_values_within_tolerance(notebook_variables: dict) -> None:
    """Test that model_quality values match expected values within tolerance."""
    model_quality = notebook_variables.get("model_quality", {})

    # Test RMSE (within 5 units)
    rmse = model_quality.get("rmse")
    assert rmse is not None, "rmse key not found in model_quality"
    assert abs(rmse - EXPECTED_MODEL_QUALITY["rmse"]) <= 5.0, (
        f"RMSE should be {EXPECTED_MODEL_QUALITY['rmse']} ± 5.0, got {rmse}"
    )

    # Test R2 (>= 0.99)
    r2 = model_quality.get("r2")
    assert r2 is not None, "r2 key not found in model_quality"
    assert r2 >= 0.99, (
        f"R2 should be >= 0.99, got {r2}"
    )

def reconstruct_dataframe_from_dict(data: dict) -> pd.DataFrame:
    """
    Reconstruct a pandas DataFrame from a dictionary.
    Handles dictionaries with 'split' orientation (from to_dict(orient='split')).
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        # Check if it's a dict with 'split' orientation (from to_dict(orient='split'))
        if "columns" in data and "data" in data:
            return pd.DataFrame(data["data"], columns=data["columns"])
        # Otherwise, try to create DataFrame directly
        return pd.DataFrame(data)
    raise ValueError(f"Cannot reconstruct DataFrame from type: {type(data)}")

def test_daily_sales_parsed(notebook_variables: dict) -> None:
    """Test that daily_sales_kg includes 'dozen' conversions (sum > 2500/day for some days)."""
    if "daily_prod" not in notebook_variables:
        pytest.skip("daily_prod not found in notebook_variables")

    daily_prod = reconstruct_dataframe_from_dict(notebook_variables["daily_prod"])

    # Check that daily_prod has the expected structure
    assert "date" in daily_prod.columns, "daily_prod must have 'date' column"
    assert "daily_sales_kg" in daily_prod.columns, "daily_prod must have 'daily_sales_kg' column"

    # Check that some days have daily_sales_kg > 2500, indicating "dozen" conversions were applied
    # If the agent only parsed numeric values without handling "dozen", values would be much lower
    max_daily_sales = daily_prod["daily_sales_kg"].max()
    assert max_daily_sales > 2500, (
        f"Expected some days to have daily_sales_kg > 2500 (indicating 'dozen' conversions), "
        f"but maximum value is {max_daily_sales}. This suggests 'dozen' conversions were not properly applied."
    )

    # Additional check: verify that there are multiple days with substantial sales
    days_above_2500 = (daily_prod["daily_sales_kg"] > 2500).sum()
    assert days_above_2500 > 0, (
        f"Expected at least one day with daily_sales_kg > 2500, but found {days_above_2500} days. "
        f"This suggests 'dozen' conversions were not properly applied."
    )