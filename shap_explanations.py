"""Create a SHAP summary plot for the heart-disease logistic-regression model."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a graphical display.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from commons import (
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def generate_shap_summary_plots(data_path: Path, output_dir: Path) -> list[Path]:
    """Fit the identifier-free notebook model and save a SHAP summary plot."""
    data = pd.read_csv(data_path)
    data["gender"] = pd.factorize(data["gender"])[0]
    data = data.dropna()
    features = data[list(FEATURE_COLUMNS)]
    # Use the same estimator search as the training notebook.
    search = RandomizedSearchCV(
        LogisticRegression(),
        {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]},
        cv=5,
        n_iter=20,
    ).fit(features, data[TARGET_COLUMN])
    model = search.best_estimator_

    # Passing the complete feature frame makes every sample part of the explanation.
    masker = shap.maskers.Independent(features, max_samples=len(features))
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_absolute_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap")
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    print("Mean absolute SHAP impact (lowest to highest):")
    print(importance.to_string(index=False))
    plt.figure()
    shap.summary_plot(shap_values, features, show=False)
    plt.title("SHAP summary: heart-disease prediction")
    plt.tight_layout()
    output_path = output_dir / "shap_summary_heart_disease.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return [output_path]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shap"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for path in generate_shap_summary_plots(args.input, args.output_dir):
        print(f"Saved {path}")
