You are given a modified Fetal Health dataset for predicting fetal health outcomes (multi-class classification task).  
The training data `train.csv` is located in the directory `data/` and contains multiple feature columns along with a target column named **`target`**.

The target column `target` represents fetal health status:
- **1** = Normal  
- **2** = Suspect  
- **3** = Pathological  

All remaining columns are available input features. Some columns may be noisy or irrelevant and do not need to be used.

---

Complete the class **`FetalHealthPredictor`** in the provided initial notebook.  
You have **`scikit-learn`**, **`numpy`**, and **`pandas`** available.  
Both `fit` and `predict` functions must take **pandas DataFrames** as input.

---

### **Development Approach**
- Train your model on the full `train.csv` dataset
- You may select or drop feature columns internally
- Tune hyperparameters and model configuration to improve generalization
- Ensure robustness to real-world data issues such as missing values and schema mismatches
- All preprocessing and feature handling must be implemented **inside the model**

---

### **Evaluation**
- Your final model will be evaluated on a **separate hidden test dataset**
- The hidden test dataset has the same structure as `train.csv` (including the `target` column for evaluation)
- Performance metrics are computed **externally** by the test suite

---

### **Performance Expectations**
Your model is expected to achieve strong performance on the hidden test dataset, measured using:
- Accuracy
- Macro F1 score
- Recall for class 3 (Pathological)

---

### **Required Methods**
Your implementation must define the following methods:

1. **`fit(train_df)`**  
   - Takes a full dataframe including the `target` column  
   - Trains the model and performs all preprocessing internally  

2. **`predict(df)`**  
   - Takes a dataframe **without** the `target` column  
   - Returns multi-class predictions `{1, 2, 3}`  
   - Output must be array-like with length equal to the number of input rows  

---

### **Robustness Requirements**
Your model must:
- Work correctly if feature columns are provided in a different order
- Handle missing feature columns gracefully at inference time
- Handle missing values in the input data
- Not assume a fixed input schema beyond what is learned during training

---

### **Output Requirement**
Once the model is complete, write the entire class **`FetalHealthPredictor`**, along with all required imports and helper methods, to the following file: `/workspace/results/utils.py`.

### **Testing**
The model will be loaded and evaluated as follows:

```python
predictor = FetalHealthPredictor()
predictor.fit(df_train)

y_true = df_test["target"]
y_pred = predictor.predict(df_test.drop("target", axis=1))

