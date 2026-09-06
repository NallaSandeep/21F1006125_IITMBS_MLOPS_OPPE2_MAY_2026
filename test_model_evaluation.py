from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from build_model import (
    fit_model,
    load_data,
    split_data,
)
from commons import DATA_PATH


def get_trained_model():
    """Train the notebook model locally so tests do not need an MLflow server."""
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    return fit_model(X_train, y_train), X_test, y_test


def test_model_accuracy():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.70


def test_model_precision():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="yes",
        zero_division=0,
    )

    assert precision >= 0.70


def test_model_recall():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    recall = recall_score(
        y_test,
        predictions,
        pos_label="yes",
        zero_division=0,
    )

    assert recall >= 0.70


def test_model_f1_score():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="yes",
        zero_division=0,
    )

    assert f1 >= 0.70
