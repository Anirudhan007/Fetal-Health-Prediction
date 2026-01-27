import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from results.utils import FetalHealthPredictor


def test_multiclass_metrics():
    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Train model
    model = FetalHealthPredictor()
    model.fit(train_df)

    # Split test data
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"].astype(int)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # -------------------
    # Accuracy
    # -------------------
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.85, f"Accuracy too low: {acc:.4f}"

    # -------------------
    # Macro F1
    # -------------------
    f1 = f1_score(y_test, y_pred, average="macro")
    assert f1 >= 0.80, f"Macro F1 too low: {f1:.4f}"

    # -------------------
    # Macro ROC-AUC (OvR)
    # -------------------
    classes = [1, 2, 3]
    y_test_bin = label_binarize(y_test, classes=classes)

    auc = roc_auc_score(
        y_test_bin,
        y_proba,
        average="macro",
        multi_class="ovr"
    )

    assert auc >= 0.85, f"Macro AUC too low: {auc:.4f}"
