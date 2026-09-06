# AI Usage Documentation

## 2026-09-06

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "As an MLOps engineer at a healthcare firm, you need to transition a heart disease prediction model from a notebook to a production-ready environment. The model classifies whether a patient is likely to have heart disease based on 14 clinical attributes."
- **Use:** Initial repository inspection to identify the available OPPE-2 starter artifacts.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (Deliverable 7)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "Detect whether the distribution of incoming data has shifted from the training data. Compute input drift by comparing the training data distribution against the 100-row generated dataset used for prediction in Deliverable 5."
- **Use:** Implemented and ran per-feature statistical input-drift checks between the training and generated prediction data.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (Deliverable 5)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "Generate a 100-row random dataset and run per-sample predictions through your deployed API. Log each prediction request individually with its input features, predicted output, and timestamp. Demonstrate observability using GCP Cloud Logging."
- **Use:** Implemented individual-request generation, structured prediction logging, a local audit trail, and Cloud Logging verification instructions.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (Deliverable 3)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "Please implement deliverable 3"
- **Use:** Implemented a Fairlearn age-group fairness audit and a unit test for its age-band policy.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (Deployment identity)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "replace oppe2-mock use oppe2-heart-disease-prediction-service as deployment file name"
- **Use:** Renamed the Kubernetes deployment identity and its CI/CD image/deployment references.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (GKE configuration)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "gcp cluster name is oppe2-heart-disease-prediction"
- **Use:** Updated the CI/CD GKE cluster configuration.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (Identifier removal)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "sno ingeneral doesn't play a role in predictions. It can be removed from model building, inputs and predictions"
- **Use:** Excluded the serial-number identifier from training, inference, monitoring, explainability, and deployment payloads.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (Deliverable 2)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "Using explainability tools, describe in plain English the factors that have the least impact on predicting whether a patient has heart disease."
- **Use:** Ran SHAP on the notebook-aligned logistic-regression model and saved its feature-impact evidence.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (continued)

- **AI tool:** OpenAI Codex
- **Prompt supplied:** "Please change other files according to HeartDiseaseTrainingAndPrediction.ipynb"
- **Use:** Aligned the production artifacts with the notebook's heart-disease schema, preprocessing, logistic-regression model, and inference payload.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (MLflow inference schema fix)

- **AI tool:** GitHub Copilot
- **Prompt supplied:** "Fix the unhandled `/predict` exception caused by incompatible `int64` input values for MLflow signature columns declared as `double`."
- **Use:** Updated the FastAPI request model so `trestbps`, `chol`, and `thalach` are emitted as floating-point values for model inference.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (SHAP impact documentation)

- **AI tool:** GitHub Copilot
- **Prompt supplied:** "Update README with the features that have least impact on predicting whether a patient has heart disease, using the supplied mean absolute SHAP values."
- **Use:** Added the ranked `fbs`, `age`, and `chol` SHAP results and an interpretation to the README.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (fairness results documentation)

- **AI tool:** GitHub Copilot
- **Prompt supplied:** "Include the supplied age-group fairness results in the README and state that the model does not demonstrate fairness."
- **Use:** Added the age-group accuracy, precision, and recall table and its fairness conclusion to the README.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (input drift results documentation)

- **AI tool:** GitHub Copilot
- **Prompt supplied:** "Document that input drift was not found using the supplied training data, incoming data, alpha, feature count, and drift summary."
- **Use:** Added the input-drift evaluation sources, parameters, results, and artifact paths to the README.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (stress test results documentation)

- **AI tool:** GitHub Copilot
- **Prompt supplied:** "Include the supplied wrk stress-test results for the `/predict` endpoint in the README."
- **Use:** Added the load-test configuration, throughput, latency, transfer, and socket-error results to the README.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.

## 2026-09-06 (repository file utility guide)

- **AI tool:** GitHub Copilot
- **Prompt supplied:** "Include the utility of each file in the repository README."
- **Use:** Added grouped descriptions for project files, tests, deployment manifests, documentation, source data, and generated artifacts.
- **Shareable conversation:** Not available; retain a PDF copy of this conversation if submission requirements require one.
