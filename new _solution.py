import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FetalHealthPredictor:
    def __init__(self):
        # Target column
        self.target_col = "target"

        # Columns that should NOT be used even if present
        self.excluded_columns = {
            "date",
            "heart_rate_status",
            "fetal_age_in_days",
            "blood_group",
            "placenta_grade",
            "amniotic_fluid",
            "fetal_size",
            "maternal_stress_index",
        }

        self.feature_columns = None

        # Preprocessing
        self.preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Gradient Boosting model
        self.model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=600,
            max_depth=8,
            min_samples_leaf=15,
            random_state=42,
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _select_features(self, df):
        """Select usable feature columns from dataframe."""
        return [
            c for c in df.columns
            if c != self.target_col and c not in self.excluded_columns
        ]

    def _align_features(self, df):
        """Ensure dataframe matches training feature schema."""
        X = df.copy()

        # Add missing columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0

        # Keep only known columns in correct order
        X = X[self.feature_columns]

        # Force numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        return X

    # --------------------------------------------------
    # Required API
    # --------------------------------------------------

    def fit(self, train_df):
        """Train the model. train_df must include 'target'."""
        if self.target_col not in train_df.columns:
            raise ValueError("Training dataframe must include 'target' column")

        # Decide features at training time
        self.feature_columns = self._select_features(train_df)

        X = train_df.drop(self.target_col, axis=1)
        y = train_df[self.target_col].astype(int).values

        X = self._align_features(X)
        Xp = self.preprocessor.fit_transform(X)

        self.model.fit(Xp, y)

    def predict_proba(self, df):
        """Return class probabilities (n_samples, 3)."""
        X = self._align_features(df)
        Xp = self.preprocessor.transform(X)

        proba = self.model.predict_proba(Xp)

        # Safety: always return (n, 3)
        if proba.shape[1] != 3:
            full = np.zeros((proba.shape[0], 3))
            for i, cls in enumerate(self.model.classes_):
                if cls in (1, 2, 3):
                    full[:, int(cls) - 1] = proba[:, i]
            full = full / full.sum(axis=1, keepdims=True)
            return full

        return proba.astype(float)

    def predict(self, df):
        """Return class labels (1.0, 2.0, 3.0)."""
        proba = self.predict_proba(df)
        preds = np.argmax(proba, axis=1) + 1
        return preds.astype(float)
