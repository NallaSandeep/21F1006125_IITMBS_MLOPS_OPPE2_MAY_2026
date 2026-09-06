# Heart Disease Prediction MLOps

The service productionizes `HeartDiseaseTrainingAndPrediction.ipynb`. It
encodes `gender` (male=0, female=1), drops missing rows, trains the notebook's
13-feature logistic-regression search, registers it in MLflow, and exposes a
FastAPI `/predict` endpoint accepting the same clinical columns. The `sno`
record identifier is excluded from training and inference.

## Features with the least prediction impact

The SHAP analysis identifies the following features as having the least impact
on predicting whether a patient has heart disease. Lower mean absolute SHAP
values indicate lower average influence on the model's predictions:

| Rank | Feature | Mean absolute SHAP |
| ---: | --- | ---: |
| 1 | `fbs` | 0.013946819152676454 |
| 2 | `age` | 0.04615069640002585 |
| 3 | `chol` | 0.08762638683187057 |

These features still contribute to individual predictions, but they have the
smallest average effect among the features included in this SHAP analysis.

## Age-group fairness results

The fairness audit evaluated model performance across age groups:

| Rank | Age group | Accuracy | Precision | Recall |
| ---: | --- | ---: | ---: | ---: |
| 1 | `45_to_54` | 1.0 | 1.0 | 1.0 |
| 2 | `55_to_64` | 0.65 | 0.8133333333333332 | 0.65 |
| 3 | `65_and_over` | 0.875 | 0.9 | 0.875 |
| 4 | `under_45` | 0.7333333333333333 | 0.7333333333333333 | 0.7333333333333333 |

These results show unequal performance across age groups. In particular, the
`55_to_64` group has substantially lower accuracy and recall than the
`45_to_54` and `65_and_over` groups. Therefore, the model does not demonstrate
fairness across age groups in this audit.

## Input drift results

Input drift was evaluated by comparing the training data with the incoming
100-row prediction sample using an alpha threshold of `0.05`:

| Field | Result |
| --- | --- |
| Training data | `data/data.csv` |
| Incoming data | `artifacts/observability/random_100_inputs.csv` |
| Alpha | `0.05` |
| Features checked | `13` |
| Drifted features | None (`[]`) |
| Drift detected | `false` |

No statistically detectable input drift was found across the 13 checked
features. The detailed results are saved in
`artifacts/drift/input_drift_report.csv`, with the summary in
`artifacts/drift/input_drift_summary.json`.

## setup python
* Run 'python3 -m venv .env'
* Run 'source .env/bin/activate'
* Run 'pip install -r requirements.txt'

## Setup Kubernetes
* Create a cluster with default settings -> Takes about 5 min
* Create a workload from the existing image
* Expose it via load balancer

## Steps to start MLFlow instance
* Open the workbench instance in SSH mode
* Install mlflow library (```pip install mlflow```)
* Create a new screen (screen -S mlflow_experiment)
* Start mlflow server
  ```
  mlflow server \
    --host 0.0.0.0 \
    --port 8100 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"
  ```
* Press keys Ctrl + A and Ctrl + D to detach from screen
* To list the existing screens, use 'screen -list'
* To reattach to previous screen, use 'screen -R mlflow_experiment)
* Create a firewall rule to allow mlflow instance (External IP address of VPC instance, port: 8100)
* Get the external IP address of the VM instance and access the IP (Say 35.202.51.100:8100) -> MLFlow UI page displays

## Stress testing commands
* Install wrk library
```sudo apt-get install -y wrk```
* Create a lua file (Say with file name - stress-test.lua)
```
wrk.method = "POST"

wrk.body = [[
{
  "age": 63,
  "gender": 0,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
]]

wrk.headers["Content-Type"] = "application/json"
```
* Run the following command
```wrk -t4 -c2500 -d30s  -s stress-test.lua http://136.111.70.98:80/predict```

### Stress test results

The `/predict` endpoint was tested for 30 seconds with 4 threads and 2,500
connections:

| Metric | Result |
| --- | ---: |
| Requests completed | 2,810 |
| Data read | 635.78 KB |
| Average latency | 1.40 s |
| Latency standard deviation | 456.13 ms |
| Maximum latency | 2.00 s |
| Requests per second | 93.42 |
| Transfer per second | 21.14 KB |
| Connect socket errors | 1,483 |
| Read socket errors | 12 |
| Write socket errors | 5 |
| Timeout errors | 1,762 |

The test achieved 93.42 requests per second, but the high number of connection
and timeout errors indicates that the deployment could not reliably serve the
full 2,500-connection load.

## File utility guide

### Project and model pipeline

| File | Utility |
| --- | --- |
| `commons.py` | Defines shared paths, feature names, target values, model configuration, and validation ranges. |
| `build_model.py` | Loads and validates data, trains the logistic-regression model, evaluates it, and logs it to MLflow. |
| `build_fastapi.py` | Runs the FastAPI service, health probes, structured logging, tracing, and `/predict` inference. |
| `download_best_model.py` | Downloads the latest registered MLflow model artifact for deployment. |
| `evaluate.py` | Evaluates the downloaded model and prints accuracy, precision, recall, and F1 metrics. |
| `per_sample_predictions.py` | Generates a 100-row sample, sends individual requests, and saves an observability audit trail. |
| `data_drift_detection.py` | Compares training and incoming feature distributions and writes drift reports. |
| `fairness_audit.py` | Measures model performance across age groups and writes fairness metrics. |
| `shap_explanations.py` | Computes SHAP feature-importance results and saves explainability evidence. |
| `Dockerfile` | Defines the container image used to package and run the FastAPI service. |
| `requirements.txt` | Lists dependencies for training, inference, testing, monitoring, and explainability. |

### Tests and source data

| File | Utility |
| --- | --- |
| `data/data.csv` | Heart-disease training and evaluation dataset. |
| `HeartDiseaseTrainingAndPrediction.ipynb` | Original notebook reference for preprocessing, training, and prediction. |
| `test_data_validation.py` | Tests input columns, missing-value handling, and validation behavior. |
| `test_model_evaluation.py` | Tests model accuracy, precision, recall, and F1 thresholds. |
| `test_per_sample_predictions.py` | Tests generation of the 100-row prediction sample. |
| `test_data_drift_detection.py` | Tests feature drift detection and no-drift behavior. |
| `test_fairness_audit.py` | Tests age-group assignment and fairness-audit behavior. |

### Deployment and documentation

| File | Utility |
| --- | --- |
| `k8s/deployment.yaml` | Defines the Kubernetes deployment, image, replicas, probes, and runtime settings. |
| `k8s/service.yaml` | Exposes the prediction service through Kubernetes. |
| `k8s/hpa.yaml` | Configures horizontal pod autoscaling based on resource usage. |
| `k8s/service-account.yaml` | Defines the workload identity and cloud permissions. |
| `GCP_CLOUD_LOGGING.md` | Documents Google Cloud Logging and observability setup. |
| `AGENTS.md` | Lists repository-specific examination scope and working instructions. |
| `AI_USAGE_DOC.md` | Records AI tools, prompts, and uses for examination work. |
| `README.md` | Provides project overview, setup instructions, results, deployment notes, and this guide. |

### Generated artifacts

| Path | Utility |
| --- | --- |
| `artifacts/drift/input_drift_report.csv` | Per-feature statistical drift results. |
| `artifacts/drift/input_drift_summary.json` | Overall drift inputs, threshold, feature count, and result. |
| `artifacts/fairness/metrics_by_age_group.csv` | Accuracy, precision, and recall for each age group. |
| `artifacts/fairness/overall_metrics.csv` | Overall model metrics for fairness comparison. |
| `artifacts/fairness/maximum_group_difference.csv` | Largest observed performance difference between age groups. |
| `artifacts/observability/random_100_inputs.csv` | Reproducible sample sent to the deployed API. |
| `artifacts/observability/per_sample_predictions.csv` | Per-request inputs, timestamps, predictions, and status. |
| `artifacts/shap/feature_importance.csv` | Mean absolute SHAP importance for each feature. |
| `artifacts/shap/shap_summary_heart_disease.png` | Visual SHAP summary of feature impact. |
