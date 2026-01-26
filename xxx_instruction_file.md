You are given the Fetal Health dataset for predicting fetal health status (Multi-class classification task). The training data `train.csv` is in the directory `data/` and contains CTG-based features along with the target column `fetal_health`.

The target column `fetal_health` represents fetal health category:
- 1 = Normal
- 2 = Suspect
- 3 = Pathological

Complete the class `FetalHealthPredictor` in the provided initial notebook. You have `scikit-learn`, `numpy`, and `pandas` available. Note that both `fit` and `predict` functions should take in dataframes as input.

**Development Approach**:
- Train your model on the full `train.csv` dataset
- Tune hyperparameters and model architecture to maximize performance

**Evaluation**:
- Your final model will be evaluated on a **separate hidden test dataset**
- The test dataset has the same format as `train.csv` (including the `fetal_health` column for evaluation purposes)

**Goal**:
- Achieve **at least 92% accuracy**, **macro-F1 score >= 0.86**, and **Recall for class 3 (Pathological) >= 0.84** on the hidden test dataset

**Required Methods**:
1. `fit(train_df)`  
   - Takes a full dataframe (including the `fetal_health` column) and trains the model  
2. `predict(df)`  
   - Returns predictions (1.0, 2.0, or 3.0) for the given dataframe (without the `fetal_health` column)  
3. `predict_proba(df)`  
   - Returns predicted probabilities for the given dataframe (without the `fetal_health` column)  
   - Must return a 2D numpy array of shape `(n_samples, 3)` with valid probabilities that sum to 1 across classes  

**Robustness Requirements**:
- Your model must work even if the input feature columns are provided in a different order (shuffled columns).
- Your model must work even if one feature column is missing at inference time (handle missing columns gracefully).

**Important Note (Do NOT print training metrics)**:
- Do NOT print training accuracy / training macro-F1 / training recall during `fit()`
- The evaluator will compute metrics independently on the hidden test dataset
- Printing training metrics is not considered valid evaluation and may cause your solution to fail automated checks

**Note**: The model is evaluated using accuracy, macro-F1, and class-3 recall calculated independently by the test suite using `predict()` output and true labels from the hidden test dataset.

Once the model is complete, write the entire class `FetalHealthPredictor` along with necessary imports and helper functions into the file:

`/workspace/results/utils.py`

**Testing**: The model will be loaded and tested as follows:

```python
predictor = FetalHealthPredictor()
predictor.fit(df_train)

y_true = df_test['fetal_health']
y_pred = predictor.predict(df_test.drop('fetal_health', axis=1))

accuracy = metrics.accuracy_score(y_true, y_pred)
macro_f1 = metrics.f1_score(y_true, y_pred, average="macro")
recall_class3 = metrics.recall_score(y_true, y_pred, labels=[1.0,2.0,3.0], average=None)[2]
