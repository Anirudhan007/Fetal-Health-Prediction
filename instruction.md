# Task Instructions

You are the **Perinatologist / Maternal–Fetal Medicine (MFM) Specialist** working with the hospital data science team. Your task is to build a **Fetal Health Prediction** model using **Cardiotocography (CTG)** data to classify fetal status into three categories using the following two datasets:

- `medical_data_file` - Core CTG clinical features
- `histogram_data` - Histogram-based CTG features + `health_insurance` status

## Data Preparation

1. In the `medical_data_file`, create a new feature called `MajorDecelBurden = severe_decelerations + prolongued_decelerations`.
2. In the `medical_data_file`, create another new feature called `VariabilityAbnormalityIndex = (abnormal_short_term_variability + percentage_of_time_with_abnormal_long_term_variability)`.
3. In the `medical_data_file`, create reassurance-related features:
   - Compute `TotalDecelerations = (light_decelerations + severe_decelerations + prolongued_decelerations)`.
   - Compute `ReassuranceRatio = accelerations / (TotalDecelerations + EPS)` where `EPS` is a small constant (e.g., `1e-6`) to avoid divide-by-zero.
4. The target variable is `fetal_health` with classes `1, 2, 3`.

## Dataset Integration

1. Combine all data in both input files on the basis of `patient_id`.
2. Remove all rows where `health_insurance == 0` or `health_insurance == False`.
3. Drop `patient_id` after merging, as it is an identifier and should not be used as a model feature.
4. Save the combined and processed dataset as **final_df**.

## Feature Selection and NA Handling

1. Use all the features in the final integrated table (`final_df`).
2. Drop any row containing NA in features or target (`fetal_health`) in `final_df`.

## Model Setup

1. Split the dataset into training and testing partitions using a `train_test_split` (70/30) with a random seed of 42 to ensure reproducibility.
2. Fit a Random Forest classification model.
3. Evaluate the model using **F1 (macro)** and **AUC** on the testing set.

## Deliverables

Return the following:

1. **`feature_importance_dict`** (Python dict) with feature → importance (rounded to 5 decimals).
2. **`model_quality`** (Python dict) with keys: `f1` and `auc` (both rounded to 5 decimals).
3. **`fetal_status`** (pandas DataFrame) containing fetal health category counts (classes 1, 2, 3).

## Variable Serialization

Convert `fetal_status` DataFrame to a dictionary using `to_dict(orient='split')` before the end of your notebook.
