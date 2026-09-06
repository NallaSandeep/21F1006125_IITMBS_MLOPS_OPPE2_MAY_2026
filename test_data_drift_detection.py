import pandas as pd

from data_drift_detection import detect_input_drift, prepare_features


def test_identical_data_has_no_detected_input_drift():
    training = pd.read_csv("data/data.csv")
    report = detect_input_drift(training, training)

    assert not report["drift_detected"].any()
    assert set(report["feature"]) == set(prepare_features(training).columns)
