import pandas as pd
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
import numpy as np


def store_results(id, y_pred, model_name, path="outputs"):
    """
    Store results in a CSV file
    args:
        results: list, list of dictionaries with results
    """

    preds = pd.DataFrame(
        {
            "test_case": "EXIST2025",
            "id": id,
            "value": np.where(y_pred == 0, "NO", "YES"),
        }
    )

    path = f"{path}/{model_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(preds.to_json(orient="records"))

    return path