import pandas as pd
from train import train_predict
from modelos_clasicos import get_model
from bert import train_predict_bert
from dataloader import load_data, load_test
from preprocess import preprocess_data
from evaluate import store_results, get_scores

train_data_path = "data/dataset_task1_exist2025/training.json"
test_data_path = "data/dataset_task1_exist2025/test.json"

# Evaluamos modelos
models = ["bert3"]  # "rf", "svm",
PREPROCCES = True
FOLDS = 3
results = []

train_data = load_data(train_data_path)
test_data = load_test(test_data_path)
if PREPROCCES:
    train_data["tweet"] = train_data["tweet"].apply(preprocess_data)
    test_data["tweet"] = test_data["tweet"].apply(preprocess_data)

for model_type in models:
    print(f"\n🔹 Generando Test Outputs modelo: {model_type.upper()}")

    if "bert" in model_type:
        y_pred = train_predict_bert(train_data, test_data["tweet"])
    else:
        model, vectorizer = get_model(model_type)
        y_pred = train_predict(train_data, model, vectorizer, test_data["tweet"])
    path = store_results(test_data["id"], y_pred, model_type, path="outputs/test")
