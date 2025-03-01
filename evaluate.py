import pandas as pd
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
import numpy as np

def store_results(id, y_pred, model_name):
    """
    Store results in a CSV file
    args:
        results: list, list of dictionaries with results
    """

    preds = pd.DataFrame({
        "test_case":  'EXIST2025',
        "id": id,
        "value": np.where(y_pred == 0, 'NO', 'YES')
    })

    path = f'outputs/{model_name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(preds.to_json(orient='records'))
    
    return path

def get_scores(path):
    test = PyEvALLEvaluation()
    preds = path
    labels = "data/golds_task1_exist2025/training.json"
    metrics = [
    MetricFactory.Accuracy.value,
    MetricFactory.FMeasure.value,
    ]
    params= dict()
    report = test.evaluate(preds, labels, metrics, **params)
    report.print_report()
    accuracy = report.report["metrics"]["Accuracy"]["results"]["average_per_test_case"]
    f1 = report.report["metrics"]["FMeasure"]["results"]["average_per_test_case"]
    return f1, accuracy
    