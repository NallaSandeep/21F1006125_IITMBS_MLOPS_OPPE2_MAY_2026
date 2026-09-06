import pandas as pd

from build_model import load_data, split_data

from commons import DATA_PATH, EXPECTED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


def test_expected_schema():
    """Verify the dataset contains the expected columns."""
    data = load_data(DATA_PATH)

    assert list(data.columns) == list(EXPECTED_COLUMNS)


def test_no_missing_values():
    """Verify there are no missing values."""
    data = load_data(DATA_PATH)

    assert data.isnull().sum().sum() == 0


def test_feature_data_types():
    """Verify feature columns are numeric."""
    data = load_data(DATA_PATH)

    for column in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(data[column])

    assert pd.api.types.is_string_dtype(data[TARGET_COLUMN])


def test_target_classes():
    """Verify expected target classes are present."""
    data = load_data(DATA_PATH)

    expected_classes = {"yes", "no"}

    assert set(data[TARGET_COLUMN].unique()) == expected_classes


def test_feature_ranges():
    """Verify feature values are within reasonable ranges."""
    data = load_data(DATA_PATH)

    from commons import FEATURE_RANGES
    for feature, (low, high) in FEATURE_RANGES.items():
        assert data[feature].between(low, high).all()
