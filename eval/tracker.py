import mlflow

def log_run(prompt_version: str, model_name: str, metrics: dict):
    """
    Log one evaluation run to MLflow.
    metrics example: {"precision": 0.85, "recall": 0.78}
    """
    with mlflow.start_run():
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("model", model_name)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)