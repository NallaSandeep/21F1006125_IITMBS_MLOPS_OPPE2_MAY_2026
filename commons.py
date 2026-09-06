"""Shared configuration and schema constants for the heart-disease service."""

DATA_PATH = "data/data.csv"
MLFLOW_TRACKING_URI = "http://35.202.51.100:8100"
MODEL_NAME = "HeartDiseaseLogisticRegression"
MODEL_URI = f"models:/{MODEL_NAME}/latest"
MODEL_PATH = "/app/model"

# ``sno`` is a record identifier, not a clinical model input.
IDENTIFIER_COLUMN = "sno"
FEATURE_COLUMNS = (
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
)
TARGET_COLUMN = "target"
EXPECTED_COLUMNS = (IDENTIFIER_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
TARGET_CLASSES = frozenset({"yes", "no"})
GENDER_VALUES = frozenset({0, 1})  # notebook factorization: male=0, female=1

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
FEATURE_RANGES = {
    "age": (29, 77), "gender": (0, 1), "cp": (0, 3),
    "trestbps": (0, 200), "chol": (0, 564), "fbs": (0, 1), "restecg": (0, 2),
    "thalach": (0, 202), "exang": (0, 1), "oldpeak": (0, 6.2),
    "slope": (0, 2), "ca": (0, 4), "thal": (0, 3),
}
