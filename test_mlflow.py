from eval.tracker import log_run

log_run(
    prompt_version="v1_test",
    model_name="gemini-1.5-flash",
    metrics={"precision": 0.85, "recall": 0.78}
)
print("Logged a test run — check the MLflow UI!")