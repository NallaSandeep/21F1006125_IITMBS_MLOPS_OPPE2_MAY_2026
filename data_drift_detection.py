"""Detect input drift between training data and the Deliverable 5 API sample.

Continuous features use the two-sample Kolmogorov-Smirnov test. Categorical
features use a chi-square test over their observed category frequencies. A
p-value below ``alpha`` indicates statistically detectable input drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from commons import DATA_PATH, FEATURE_COLUMNS


CATEGORICAL_FEATURES = frozenset(
    {"gender", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's gender encoding and retain only model inputs."""
    missing = set(FEATURE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing prediction features: {sorted(missing)}")
    prepared = data.copy()
    if pd.api.types.is_string_dtype(prepared["gender"]):
        prepared["gender"] = prepared["gender"].map({"male": 0, "female": 1})
    return prepared[list(FEATURE_COLUMNS)].dropna()


def detect_input_drift(
    training_features: pd.DataFrame, incoming_features: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """Return a drift decision and test statistic for every model feature."""
    training = prepare_features(training_features)
    incoming = prepare_features(incoming_features)
    results = []

    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_FEATURES:
            categories = sorted(set(training[feature]).union(incoming[feature]))
            observed = pd.crosstab(
                pd.Series(["training"] * len(training) + ["incoming"] * len(incoming)),
                pd.Categorical(
                    pd.concat([training[feature], incoming[feature]], ignore_index=True),
                    categories=categories,
                ),
            )
            statistic, p_value, _, _ = chi2_contingency(observed)
            test = "chi_square"
        else:
            statistic, p_value = ks_2samp(training[feature], incoming[feature])
            test = "kolmogorov_smirnov"

        results.append(
            {
                "feature": feature,
                "test": test,
                "statistic": statistic,
                "p_value": p_value,
                "drift_detected": bool(p_value < alpha),
                "training_mean": training[feature].mean(),
                "incoming_mean": incoming[feature].mean(),
            }
        )

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, default=Path(DATA_PATH))
    parser.add_argument(
        "--incoming-data",
        type=Path,
        default=Path("artifacts/observability/random_100_inputs.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/drift"))
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = detect_input_drift(
        pd.read_csv(args.training_data), pd.read_csv(args.incoming_data), args.alpha
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "input_drift_report.csv", index=False)
    summary = {
        "training_data": str(args.training_data),
        "incoming_data": str(args.incoming_data),
        "alpha": args.alpha,
        "features_checked": len(report),
        "drifted_features": report.loc[report["drift_detected"], "feature"].tolist(),
        "drift_detected": bool(report["drift_detected"].any()),
    }
    (args.output_dir / "input_drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(report.to_string(index=False))
    print(f"\nInput drift detected: {summary['drift_detected']}")
    print(f"Artifacts saved to {args.output_dir}")
