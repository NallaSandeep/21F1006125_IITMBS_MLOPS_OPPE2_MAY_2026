from pathlib import Path

from commons import FEATURE_COLUMNS
from per_sample_predictions import generate_random_dataset


def test_random_prediction_sample_has_100_model_input_rows(tmp_path):
    sample = generate_random_dataset(
        Path("data/data.csv"), tmp_path / "random_100_inputs.csv", random_state=42
    )

    assert len(sample) == 100
    assert list(sample.columns) == list(FEATURE_COLUMNS)
    assert "sno" not in sample.columns
