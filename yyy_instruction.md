You are given a modified Fetal Health dataset for predicting fetal health status (Multi-class classification task). The training data `train.csv` is in the directory `data/` and contains multiple features along with the target column `target`.

The target column `target` represents fetal health category:
- 1 = Normal
- 2 = Suspect
- 3 = Pathological

Complete the class `FetalHealthPredictor` in the provided initial notebook. You have `scikit-learn`, `numpy`, and `pandas` available. Note that both `fit` and `predict` functions should take in dataframes as input.

**Development Approach (Feature Flexibility Allowed)**:
- Train your model on the full `train.csv` dataset
- You may choose which feature columns to include (you are not required to use all columns)
- You may tune hyperparameters and apply feature engineering to improve generalization

**Time Constraint**:
- Your entire solution (training + inference + evaluation checks) must complete within **10 minutes**

**Evaluation / Benchmarks (Hidden Test Dataset)**:
Your final model will be evaluated on a separate hidden test dataset using:
- Accuracy **>= 0.92**
- Macro-F1 score **>= 0.86**
- Recall for class 3 (Pathological) **>= 0.84**

**Required Methods**:
1. `fit(train_df)`  
   - Takes a full dataframe (including the `target` column) and trains the model  
2. `predict(df)`  
   - Returns predictions (1.0, 2.0, or 3.0) for the given dataframe (without the `target` column)  
3. `predict_proba(df)`  
   - Returns predicted probabilities for the given dataframe (without the `target` column)  
   - Must return a 2D numpy array of shape `(n_samples, 3)` with valid probabilities that sum to 1 across classes  

**Robustness Requirements**:
- Your model must work even if the input feature columns are provided in a different order (shuffled columns).
- Your model must work even if one feature column is missing at inference time (handle missing columns gracefully).

**Important Note (Metrics Printing Rule)**:
- Do NOT print training metrics as final performance.
- If you report metrics, they must be computed from an evaluation run that completes within the **10 minute** limit.

Once the model is complete, write the entire class `FetalHealthPredictor` along with necessary imports and helper functions into the file:

`/workspace/results/utils.py`

**Testing**: The model will be loaded and tested as follows:

```python
predictor = FetalHealthPredictor()
predictor.fit(df_train)

y_true = df_test['target']
y_pred = predictor.predict(df_test.drop('target', axis=1))

accuracy = metrics.accuracy_score(y_true, y_pred)
macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
recall_class3 = metrics.recall_score(y_true, y_pred, labels=[1.0,2.0,3.0], average=None)[2]
