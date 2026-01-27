 You are given the Fetal Health dataset for predicting fetal_health outcomes (Multi-class classification task). The training data `train.csv` is in the directory `data/` with columns ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency', 'fetal_health']. 'fetal_health' is our target column (1.0 = Normal, 2.0 = Suspect, 3.0 = Pathological) and the rest are available features.

Complete the class 'FetalHealthPredictor' in the provided initial notebook. You have `scikit-learn`, `numpy`, and `pandas` available. Note that both `fit` and `predict` functions should take in dataframes as input.

**Development Approach**:
- Train your model on the full `train.csv` dataset
- Tune hyperparameters and model architecture to maximize performance
- Ensure robustness to real-world clinical data issues such as missing values and schema mismatches

**Evaluation**:
- Your final model will be evaluated on a **separate hidden test dataset**
- The test dataset has the same format as `train.csv` (including the 'fetal_health' column for evaluation purposes)
- **Goal**: Achieve strong performance and robustness on the hidden test dataset

**Hard Mode Requirements**:
Your model must satisfy ALL of the following:
- Accuracy >= 90%
- Macro F1 Score >= 0.88
- Recall for Class 3 (Pathological) >= 0.75

**Required Methods**:
1. `fit(train_df)` - Takes a full dataframe (including the 'fetal_health' column) and trains the model
2. `predict(df)` - Returns multi-class predictions (1.0, 2.0, or 3.0) for the given dataframe (without the 'fetal_health' column)
3. `predict_proba(df)` - Returns predicted class probabilities for the given dataframe (without the 'fetal_health' column). Should return a 2D array of probabilities for all classes with shape (n_samples, 3)

**Note**: The model is evaluated using accuracy, macro F1 score, and class-wise recall calculated independently by the test suite using `predict()` output and true labels from the test dataset.

Once the model is complete, write the entire class 'FetalHealthPredictor' along with necessary imports and helper functions into 

**Testing**: The model will be loaded and tested as follows:

```python
predictor = FetalHealthPredictor()
predictor.fit(df_train)

y_true = df_test['fetal_health']
y_pred = predictor.predict(df_test.drop('fetal_health', axis=1))
accuracy = metrics.accuracy_score(y_true, y_pred)
