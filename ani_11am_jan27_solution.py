
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# Load training data (Harbor provides full train.csv)
# ------------------------------------------------------------
data_dir = Path("data/")
assert data_dir.exists(), "Expected /data directory to exist"

df_train = pd.read_csv(data_dir / "train.csv")
print(f"Training rows: {len(df_train)}")
print(f"Columns: {list(df_train.columns)}")


# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
class FetalHealthPredictor:
    def __init__(self):
        self.target_col = "target"

        # Explicit garbage columns (domain-driven, deterministic)
        self.garbage_columns = [
            "date",
            "heart_rate_status",
            "fetal_age_in_days",
            "blood_group",
            "placenta_grade",
            "amniotic_fluid",
            "fetal_size",
            "maternal_stress_index",
        ]

        self.feature_columns = None
        self.preprocessor = None
        self.model = None
        self.classes_ = np.array([1, 2, 3])

    # ------------------------
    # Internal feature cleaning
    # ------------------------
    def _clean_features(self, df):
        X = df.copy()

        # Drop target if present
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        # Drop garbage columns (only if present)
        drop_cols = [c for c in self.garbage_columns if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X

    def _build_preprocessor(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

    # ------------------------
    # Training
    # ------------------------
    def fit(self, train_df):
        if self.target_col not in train_df.columns:
            raise ValueError("Training data must include 'target' column")

        X = self._clean_features(train_df)
        y = train_df[self.target_col].astype(int)

        # Freeze feature schema
        self.feature_columns = list(X.columns)
        assert len(self.feature_columns) > 0, "No usable features after cleaning"

        self.preprocessor = self._build_preprocessor()
        X_proc = self.preprocessor.fit_transform(X)

        base_model = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        self.model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv=5,
        )

        self.model.fit(X_proc, y)

    # ------------------------
    # Prediction
    # ------------------------
    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model has not been trained")

        X = self._clean_features(df)

        # Align to frozen schema
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_columns]
        X_proc = self.preprocessor.transform(X)

        return self.model.predict(X_proc).astype(int)


# ------------------------------------------------------------
# Train model (NO evaluation here)
# ------------------------------------------------------------
predictor = FetalHealthPredictor()
predictor.fit(df_train)

print("Training complete. Model ready for evaluation.")


# ------------------------------------------------------------
# Export utils.py (this is what test_notebook.py imports)
# ------------------------------------------------------------
Path("/workspace/results").mkdir(parents=True, exist_ok=True)

utils_code = '''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class FetalHealthPredictor:
    def __init__(self):
        self.target_col = "target"

        self.garbage_columns = [
            "date",
            "heart_rate_status",
            "fetal_age_in_days",
            "blood_group",
            "placenta_grade",
            "amniotic_fluid",
            "fetal_size",
            "maternal_stress_index",
        ]

        self.feature_columns = None
        self.preprocessor = None
        self.model = None
        self.classes_ = np.array([1, 2, 3])

    def _clean_features(self, df):
        X = df.copy()

        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        drop_cols = [c for c in self.garbage_columns if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X

    def _build_preprocessor(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

    def fit(self, train_df):
        if self.target_col not in train_df.columns:
            raise ValueError("Training data must include 'target'")

        X = self._clean_features(train_df)
        y = train_df[self.target_col].astype(int)

        self.feature_columns = list(X.columns)
        assert len(self.feature_columns) > 0

        self.preprocessor = self._build_preprocessor()
        X_proc = self.preprocessor.fit_transform(X)

        base_model = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        self.model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv=5,
        )

        self.model.fit(X_proc, y)

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model not trained")

        X = self._clean_features(df)

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_columns]
        X_proc = self.preprocessor.transform(X)

        return self.model.predict(X_proc).astype(int)
'''

with open("/workspace/results/utils.py", "w") as f:
    f.write(utils_code)

print("utils.py written successfully")
