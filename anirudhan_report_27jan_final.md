============================================================
FETAL HEALTH PREDICTION — BENCHMARK REPORT
============================================================

Task Definition
---------------

You are given a modified fetal health dataset for predicting fetal health
status as a multi-class classification task. The dataset (data/train.csv)
contains a large number of heterogeneous features, including both clinically
relevant signals and intentionally injected irrelevant or high-noise attributes.

The target variable `target` represents fetal health categories:
- 1: Normal
- 2: Suspect
- 3: Pathological

Participants are required to implement a `FetalHealthPredictor` class that
trains on the full dataset and produces both class predictions and calibrated
class probabilities. The solution must be robust to shuffled feature columns,
missing feature columns at inference time, and missing values within features.

Objective
---------
1. Train a high-performing multi-class classification model under strict
   runtime constraints.
2. Perform disciplined feature selection to exclude irrelevant or noisy
   columns (e.g., date, blood_group) that negatively impact generalization.
3. Ensure robustness and correctness under all specified evaluation conditions.
4. Meet all required metric thresholds on a hidden test dataset.

Constraints
-----------
- Accuracy ≥ 0.945
- Macro-F1 ≥ 0.914
- Recall (Class 3 – Pathological) ≥ 0.9
- End-to-end execution (training + inference + evaluation) must complete
  within 10 minutes.
- The final implementation must be written to:
  /workspace/results/utils.py

------------------------------------------------------------

Why This Is a Good Task
----------------------

This benchmark effectively distinguishes between surface-level model
construction and disciplined machine learning practice:

- Naive agents that consume all available features fail due to noise injection
  from irrelevant columns.
- Overfitted validation strategies (single train–test split, aggressive feature
  selection without cross-validation) produce inflated internal metrics but fail
  on the hidden test set.
- The task requires explicit handling of column order variance, missing
  features, and missing values.
- Success depends on pipeline robustness, feature hygiene, calibration quality,
  and strict instruction compliance.
- Grading is deterministic and metric-driven, ensuring unambiguous evaluation.

------------------------------------------------------------

Outcome Metrics (Pass@5)
-----------------------

Model              Pass@5
------------------------
Gemini-3 Pro        3 / 5
Claude Opus 4.5     3 / 5
GPT-5.2             2 / 5

No evaluated agent achieved consistent success across all runs, indicating that
the task meaningfully stresses robustness, generalization, and execution
discipline rather than raw modeling capability.

------------------------------------------------------------

Failure Analysis (Gemini-3 Pro)
-------------------------------

Analysis of the two failed Gemini-3 Pro runs reveals consistent failure modes:

- The agent failed to meet the required accuracy threshold
  (0.9379 vs. 0.945) due to an inadequate validation strategy.
- Recursive Feature Elimination (RFE) and hyperparameter tuning were performed
  on a single train–test split, resulting in overfitting to a narrow subset
  of features.
- This produced an inflated internal validation estimate (~0.955) that did not
  generalize to the hidden test dataset.
- Robustness issues were compounded by task execution failures:
  * The agent ignored the explicit instruction to save the final
    FetalHealthPredictor class to /workspace/results/utils.py.
  * A NotFittedError occurred due to accessing feature importance before
    calling the fit() method.
- As a result, the required deliverable was never created, leading to immediate
  critical failures across all evaluation checks.

------------------------------------------------------------

Conclusion
----------

This benchmark exposes failure modes common in real-world machine learning
systems: overfitting caused by weak validation design, insufficient feature
discipline, brittle pipelines, and incomplete adherence to operational
requirements. Partial success across strong models confirms the task’s
difficulty and its effectiveness in evaluating end-to-end ML competence rather
than isolated model performance.

============================================================
