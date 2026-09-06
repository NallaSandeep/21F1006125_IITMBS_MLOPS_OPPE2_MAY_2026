"""Send 100 individually logged heart-disease predictions to the deployed API.

Set ``PREDICTION_API_URL`` to the GKE LoadBalancer URL, for example
``http://203.0.113.10/predict``. Each POST is intentionally sent one at a time
so GKE/Cloud Logging receives one structured prediction-request log per row.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from commons import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def generate_random_dataset(
    data_path: Path, output_path: Path, rows: int = 100, random_state: int = 42
) -> pd.DataFrame:
    """Create a reproducible random feature-only sample for API inference."""
    data = pd.read_csv(data_path)
    data["gender"] = pd.factorize(data["gender"])[0]
    data = data.dropna()
    if rows > len(data):
        raise ValueError(f"Requested {rows} rows, but only {len(data)} clean rows are available.")

    sample = data.sample(n=rows, random_state=random_state)[list(FEATURE_COLUMNS)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return sample


def post_prediction(api_url: str, payload: dict) -> dict:
    """Submit one JSON request and return the API response."""
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run_per_sample_predictions(api_url: str, sample: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """POST each row separately and save a local timestamped audit trail."""
    records = []
    for sample_number, (_, row) in enumerate(sample.iterrows(), start=1):
        payload = json.loads(row.to_json())
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            prediction = post_prediction(api_url, payload)
            records.append(
                {
                    "sample_number": sample_number,
                    "timestamp_utc": timestamp,
                    "input_features": json.dumps(payload),
                    "predicted_class": prediction.get("predicted_class"),
                    "has_heart_disease": prediction.get("has_heart_disease"),
                    "request_status": "success",
                }
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            records.append(
                {
                    "sample_number": sample_number,
                    "timestamp_utc": timestamp,
                    "input_features": json.dumps(payload),
                    "predicted_class": None,
                    "has_heart_disease": None,
                    "request_status": "failed",
                    "error": str(error),
                }
            )

    report = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("PREDICTION_API_URL"))
    parser.add_argument("--input", type=Path, default=Path(DATA_PATH))
    parser.add_argument("--sample-output", type=Path, default=Path("artifacts/observability/random_100_inputs.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("artifacts/observability/per_sample_predictions.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_url:
        raise SystemExit("Provide --api-url or set PREDICTION_API_URL to the deployed /predict URL.")
    random_sample = generate_random_dataset(args.input, args.sample_output, random_state=args.random_state)
    report = run_per_sample_predictions(args.api_url, random_sample, args.report_output)
    print(f"Submitted {len(report)} individual prediction requests.")
    print(f"Successful requests: {(report['request_status'] == 'success').sum()}")
    print(f"Local audit trail: {args.report_output}")
