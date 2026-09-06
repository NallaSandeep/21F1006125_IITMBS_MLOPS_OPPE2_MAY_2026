"""Train and register the notebook's heart-disease logistic-regression model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from commons import (
    DATA_PATH,
    FEATURE_COLUMNS,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    TARGET_COLUMN,
)

def load_data(path=DATA_PATH):
    """Load and apply the same cleaning/encoding used in the notebook."""
    data = pd.read_csv(path)
    data["gender"] = pd.factorize(data["gender"])[0]
    return data.dropna().copy()


def split_data(data):
    """Reproduce the notebook's 80/20 split with its NumPy seed."""
    np.random.seed(42)
    return train_test_split(data[list(FEATURE_COLUMNS)], data[TARGET_COLUMN], test_size=0.2)


def fit_model(X_train, y_train):
    grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
    return RandomizedSearchCV(LogisticRegression(), grid, cv=5, n_iter=20, verbose=1).fit(X_train, y_train)


def train_model_log_mlflow(X_train, y_train, X_test, y_test):
    # Keep local training/tests independent of the optional MLflow client.
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mlflow.set_experiment("heart_disease_prediction")
    
    with mlflow.start_run():
        model = fit_model(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label="yes", zero_division=0)
        recall = recall_score(y_test, predictions, pos_label="yes", zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label="yes", zero_division=0)

        mlflow.log_params(model.best_params_)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        mlflow.sklearn.log_model(
            sk_model=model.best_estimator_,
            name="heart_disease_model",
            registered_model_name=MODEL_NAME,
            input_example=X_test[:5],
            signature=infer_signature(X_test, predictions),
        )

        print("Best parameters:", model.best_params_)
        print(f"Accuracy={accuracy:.4f}")
        return model.best_estimator_, predictions, accuracy


def main():
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    train_model_log_mlflow(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
