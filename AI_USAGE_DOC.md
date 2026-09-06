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
