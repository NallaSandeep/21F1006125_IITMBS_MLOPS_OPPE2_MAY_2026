import pandas as pd

from fairness_audit import AGE_LABELS, create_age_groups, save_fairness_report


def test_create_age_groups_uses_documented_boundaries():
    ages = pd.Series([29, 44, 45, 54, 55, 64, 65, 77])

    assert list(create_age_groups(ages).astype(str)) == [
        "under_45",
        "under_45",
        "45_to_54",
        "45_to_54",
        "55_to_64",
        "55_to_64",
        "65_and_over",
        "65_and_over",
    ]
    assert len(AGE_LABELS) == 4
