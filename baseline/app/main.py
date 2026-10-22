import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import psutil
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

import mlflow

try:
    from release import ReleaseManager            # dans l'image Docker
except ImportError:
    from baseline.app.release import ReleaseManager   # dans les tests

loaded_model = None
release = ReleaseManager()

# Create Prometheus metrics
cpu_usage_gauge = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_usage_gauge = Gauge("memory_usage_bytes", "Memory usage in bytes")

prediction_gauge = Gauge("model_prediction", "Model prediction value")
model_latency = Gauge("model_latency_seconds", "Time taken for model prediction")
# Counters
request_counter = Counter("total_requests", "Total number of prediction requests")
success_counter = Counter(
    "successful_predictions", "Total number of successful predictions"
)
failure_counter = Counter("failed_predictions", "Total number of failed predictions")

# Metrics per variant (champion / candidate), used by the Deployer
variant_requests = Counter(
    "variant_requests_total", "Prediction requests by variant", ["variant"]
)
variant_failures = Counter(
    "variant_failures_total", "Failed predictions by variant", ["variant"]
)
variant_latency = Histogram(
    "variant_request_latency_seconds",
    "Request latency by variant",
    ["variant"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

# Histograms for latency
latency_histogram = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10),  # Customize buckets as needed
)
# Histograms for input features
feature_histograms = {
    "fixed_acidity": Histogram(
        "input_fixed_acidity",
        "Fixed acidity of input data",
        buckets=[5, 6, 7, 8, 9, 10, 11, 12, 13],
    ),
    "volatile_acidity": Histogram(
        "input_volatile_acidity",
        "Volatile acidity of input data",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ),
    # Add histograms for other features as needed...
}
# Histogram for prediction values
prediction_histogram = Histogram(
    "prediction_values", "Distribution of prediction values"
)

mlflow.set_tracking_uri("http://mlflow_server:5000")


# Instantiate the FastAPI app
app = FastAPI(debug=True)
# Expose default metrics
instrumentator = Instrumentator()

instrumentator.instrument(app).expose(app)

# Expose custom metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model

    # Load champion (alias) and, if set, the candidate; then keep them in sync
    release.refresh()
    if release.champion is None:
        raise RuntimeError("champion model could not be loaded")
    loaded_model = release.champion
    release.start()
    print("Model loaded and cached at startup")

    yield

    release.stop()
    print("Application shutdown, resources cleaned")


# Assign the lifespan context manager to the app
app.router.lifespan_context = lifespan


@app.middleware("http")
async def add_process_metrics(request: Request, call_next):
    response = await call_next(request)
    cpu_usage_gauge.set(psutil.cpu_percent())
    memory_usage_gauge.set(psutil.virtual_memory().used)
    return response


@app.get("/")
async def home():
    return "Hello World"


class PredictionInput(BaseModel):
    data: list
    wine_id: Optional[int] = None


class ReleaseUpdate(BaseModel):
    stage: Optional[str] = None
    candidate_share: Optional[float] = None
    shadow_on: Optional[int] = None


class FaultUpdate(BaseModel):
    slow_ms: Optional[float] = None      # lenteur ajoutee au candidat (scenario G)
    err_pct: Optional[float] = None      # % d'erreurs injectees sur le candidat


def prepare_features_serve(values: list) -> pd.DataFrame:
    """Pure feature transformation function for prediction serving."""
    feature_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    return pd.DataFrame(columns=feature_columns, data=[values])


@app.get("/admin/release")
async def get_release():
    return {
        **release.state,
        "champion_loaded": release.champion_version,
        "candidate_loaded": release.candidate_version,
        "faults": release.faults,
    }


@app.post("/admin/release")
async def set_release(update: ReleaseUpdate):
    """Only the Deployer calls this: changes the traffic share immediately."""
    return release.apply(**update.dict(exclude_none=True))


@app.post("/admin/fault")
async def set_fault(update: FaultUpdate):
    """Fautes du scenario G a chaud. Un corps vide `{}` supprime les fautes."""
    return release.set_fault(update.slow_ms, update.err_pct)


@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    x_variant: Optional[str] = Header(None),
    x_dry_run: Optional[str] = Header(None),
):
    global loaded_model
    data = input_data.data
    # Dry run (traffic_replay) : variant force, rien dans les metriques ni dans le journal
    dry = x_dry_run == "1"
    forced = x_variant if x_variant in ("candidate", "champion") else None
    variant, shadow = release.route(forced)
    if dry:
        shadow = False
    if not dry:
        request_counter.inc()
        variant_requests.labels(variant).inc()
    start_time = time.time()
    version = (
        release.candidate_version if variant == "candidate" else release.champion_version
    )
    try:
        # Validate input data
        # Increment failure_counter if validation fails

        # Record feature statistics
        for i, feature_name in enumerate(feature_histograms.keys()):
            feature_value = input_data.data[i]
            if not dry:
                feature_histograms[feature_name].observe(feature_value)
        # Prepare the input data
        df = prepare_features_serve(data)

        if variant == "candidate":
            delay = release.candidate_fault_plan()
            if delay:
                await asyncio.sleep(delay)       # lenteur injectee, sans bloquer les autres requetes
            prediction = release.candidate.predict(df)
        else:
            prediction = release.champion.predict(df)

        prediction_value = prediction[0]
        if not dry:
            prediction_gauge.set(prediction_value)
            prediction_histogram.observe(prediction_value)

        # Shadow mode: the champion answers, the candidate only gets logged
        if shadow and release.candidate is not None:
            t0 = time.time()
            try:
                delay = release.candidate_fault_plan()
                if delay:
                    await asyncio.sleep(delay)
                shadow_value = release.candidate.predict(df)[0]
                release.log("shadow", release.candidate_version, input_data.wine_id,
                            data, shadow_value, (time.time() - t0) * 1000, 200)
            except Exception:
                release.log("shadow", release.candidate_version, input_data.wine_id,
                            data, None, (time.time() - t0) * 1000, 500)

        if not dry:
            release.log(variant, version, input_data.wine_id, data, prediction_value,
                        (time.time() - start_time) * 1000, 200)
            success_counter.inc()
    except Exception as e:
        if not dry:
            failure_counter.inc()
            variant_failures.labels(variant).inc()
            release.log(variant, version, input_data.wine_id, data, None,
                        (time.time() - start_time) * 1000, 500)
        raise e
    finally:
        elapsed_time = time.time() - start_time
        if not dry:
            latency_histogram.observe(elapsed_time)
            variant_latency.labels(variant).observe(elapsed_time)
        return JSONResponse({"prediction": prediction_value},
                            headers={"X-Served-Variant": variant})


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
