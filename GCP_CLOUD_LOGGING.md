# GCP Cloud Logging: Per-Sample Prediction Observability

The FastAPI access-log middleware writes a JSON log record for every request to
standard output. For each `POST /predict`, the record includes:

- `event: "prediction_request"`
- `timestamp`
- `request_body` (the 13 model inputs)
- `response_body` (the predicted class and boolean decision)
- HTTP status, duration, client IP, and OpenTelemetry trace/span identifiers

GKE captures container standard output in Cloud Logging automatically. After the
service is deployed and exposed, submit 100 individual requests:

```bash
export PREDICTION_API_URL="http://136.111.70.98:80/predict"
python per_sample_predictions.py
```

This generates `artifacts/observability/random_100_inputs.csv` and a local,
timestamped request/result trail at
`artifacts/observability/per_sample_predictions.csv`.

In Google Cloud Logging, select the GKE workload logs or run:

```bash
gcloud logging read \
  'resource.type="k8s_container" AND jsonPayload.event="prediction_request"' \
  --project=project-eada5958-ab21-4f76-b53 \
  --limit=100 \
  --format=json
```

To limit results to the production cluster, add:

```text
resource.labels.cluster_name="oppe2-heart-disease-prediction"
```

The resulting 100 JSON records demonstrate per-sample prediction logging and
observability in GCP Cloud Logging.
