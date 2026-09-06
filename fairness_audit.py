"""Audit heart-disease model performance by age group with Fairlearn."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from commons import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


AGE_BINS = [0, 45, 55, 65, float("inf")]
AGE_LABELS = ["under_45", "45_to_54", "55_to_64", "65_and_over"]


def create_age_groups(age: pd.Series) -> pd.Series:
    """Convert continuous age values into sufficiently sized audit groups."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def audit_fairness(
    data_path: Path, random_state: int = DEFAULT_RANDOM_STATE
) -> MetricFrame:
    """Train the model and return Fairlearn metrics grouped by age band."""
    data = pd.read_csv(data_path)
    data["gender"] = pd.factorize(data["gender"])[0]
    data = data.dropna()

    train, test = train_test_split(
        data,
        test_size=DEFAULT_TEST_SIZE,
        random_state=random_state,
    )
    model = LogisticRegression(solver="liblinear")
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    predictions = model.predict(test[list(FEATURE_COLUMNS)])

    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }
    return MetricFrame(
        metrics=metrics,
        y_true=test[TARGET_COLUMN],
        y_pred=predictions,
        sensitive_features=create_age_groups(test["age"]),
    )


def save_fairness_report(metric_frame: MetricFrame, output_dir: Path) -> None:
    """Persist Fairlearn results for review and CI/CD artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.overall.rename("overall").to_csv(output_dir / "overall_metrics.csv")
    metric_frame.by_group.to_csv(output_dir / "metrics_by_age_group.csv")
    (metric_frame.by_group.max() - metric_frame.by_group.min()).rename(
        "maximum_group_difference"
    ).to_csv(output_dir / "maximum_group_difference.csv")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fairness"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metric_frame = audit_fairness(args.input)
    save_fairness_report(metric_frame, args.output_dir)
    print("Overall metrics:")
    print(metric_frame.overall.round(3))
    print("\nMetrics by age group:")
    print(metric_frame.by_group.round(3))
    print("\nMaximum group difference:")
    print((metric_frame.by_group.max() - metric_frame.by_group.min()).round(3))
    print(f"\nArtifacts saved to {args.output_dir}")
