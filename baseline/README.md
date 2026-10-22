# FastAPI-Prometheus-Grafana-MLOPs
MLOPs with Prometheus and Grafana for monitoring ML models deployed in FastAPI

Link to article can be found [here](https://medium.com/stackademic/simplified-monitoring-for-ml-models-why-prometheus-and-grafana-are-the-tools-you-need-06a12fdc91f6)

## How to run

1. Build and start the containers (from this `baseline` folder; Docker Desktop must be running)

    ```bash
    docker compose up -d --build
    ```

2. Train the first model and make it the champion (from the repo root)

    ```bash
    python baseline/train.py
    ```

3. Tell the API to load it, then try a prediction

    ```bash
    curl -X POST http://localhost:8000/reload
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"data": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]}'
    ```

4. Send traffic for the dashboards: `python baseline/simulate_requests.py`

* MLflow UI: http://localhost:5000
* API: http://localhost:8000 (`/model` shows which version it serves, `/metrics` the Prometheus metrics)
* Prometheus: http://localhost:9090
* Grafana: http://localhost:3000 (admin / admin)

## Containerization

The MLflow tracking server is composed of 2 docker containers:

* MLflow server, which also stores and serves the model files (artifacts) from the `mlartifacts` volume
* MySQL database server (runs, metrics, the model registry)

MinIO was dropped: its images are no longer published on Docker Hub, so `docker compose up` failed.
Nobody needs S3 credentials now; clients upload and download models through the MLflow server.

## Example

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Install MLflow with extra dependencies, including scikit-learn

    ```bash
    pip install mlflow boto3
    ```
3. Set environmental variables

    ```bash
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    ```
4. Set MinIO credentials

    ```bash
    cat <<EOF > ~/.aws/credentials
    [default]
    aws_access_key_id=minio
    aws_secret_access_key=minio123
    EOF
    ```

5. Train a sample MLflow model

    ```bash
    mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.23
    ```

 6. Serve the model (replace ${MODEL_ID} with your model's ID)
    ```bash
    export MODEL_ID=0cgd240633e8417fbbcb2c441a7d2f07 # Replace this with your model's ID
    mlflow models serve -m runs:/${MODEL_ID}/model -p 1234 --env-manager conda
    ```

 7. You can check the input with this command
    ```bash
    curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [1.4, 0.7, 0, 1.9, 2.076, 1, 44, 0.9978, 3.51, 0.56, 9.4]}'
    ```
    
    Or you can simulate sending a 1000 requests using the `simulate_requests.sh` script and then view the metrics in the Grafana dashboard.