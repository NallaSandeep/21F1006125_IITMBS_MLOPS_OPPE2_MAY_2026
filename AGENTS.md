# OPPE-2 Workspace Instructions

This repository is for the IIT Madras MLOps OPPE-2 examination work. Keep all
changes focused on the examination task the user requests.

## In-scope syllabus

OPPE-2 covers Weeks 4-9 topics, including:

- Observability, explainability, containerization, continuous deployment,
  scaling, monitoring, and security for ML models.
- CI/CD with GitHub Actions.
- Model and experiment tracking with MLflow.
- Docker containerization and Artifact Registry.
- Kubernetes/GKE deployment orchestration and scaling.
- MLSecurityOps.
- Data and concept drift.
- Explainability with SHAP and LIME, logging, observability, and performance
  monitoring.
- Code versioning and experiment tracking.

## Out of scope

Do not introduce or spend effort on these OPPE-2-excluded topics unless the
user explicitly requests them:

- LLMOps
- Data versioning with DVC
- Feast feature management

## AI usage documentation

Maintain an `AI_USAGE_DOC.md` file in the repository whenever AI assistance is
used for examination work. It should record:

- The AI tools used.
- The prompts supplied.
- Links to shareable conversations.
- If a public conversation link cannot be shared, a reference to the PDF copy
  of that conversation.

Use the exact filename `AI_USAGE_DOC.md`.

## Working conventions

- Preserve existing user changes; do not discard or overwrite unrelated work.
- Prefer small, explainable changes and verify relevant commands before
  reporting work as complete.
- Do not add credentials, tokens, private keys, or other secrets to the
  repository. Use environment variables or documented placeholders instead.
- Keep README instructions and submitted artifacts aligned with the requested
  task.
