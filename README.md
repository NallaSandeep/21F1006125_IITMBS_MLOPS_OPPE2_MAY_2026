# Heart Disease Prediction MLOps

The service productionizes `HeartDiseaseTrainingAndPrediction.ipynb`. It
encodes `gender` (male=0, female=1), drops missing rows, trains the notebook's
13-feature logistic-regression search, registers it in MLflow, and exposes a
FastAPI `/predict` endpoint accepting the same clinical columns. The `sno`
record identifier is excluded from training and inference.

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
```wrk -t4 -c1000 -d30s  -s stress-test.lua http://136.111.70.98:80/predict```
