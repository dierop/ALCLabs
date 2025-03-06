import pandas as pd
from modelos_clasicos import get_model
from bert import train_bert_kfold
from dataloader import load_data
from preprocess import preprocess_data
from evaluate import store_results, get_scores

train_data_path = "Lab2/data/lab2_materials/dataset_task4_exist2025/training.json"

# Evaluamos modelos
models = [ "rf", "xgb", "logreg", "nb", "svm"] #falta agregar bert
PREPROCCES = True
FOLDS = 3
results = []

train_data = load_data(train_data_path)
if PREPROCCES:
    train_data["tweet"] = train_data["tweet"].apply(preprocess_data)

for model_type in models:
    print(f"\n🔹 Evaluando modelo: {model_type.upper()}")

    if "bert" in model_type:
        y_pred, _ = train_bert_kfold(train_data, FOLDS)
    else:
        model, vectorizer = get_model(model_type)
        y_pred = cross_validation(train_data, model, vectorizer, FOLDS)
    path = store_results(train_data["id"], y_pred, model_type)
    ac, f1 = get_scores(path)
    results.append(
        {"Modelo": model_type.upper(), "Accuracy": round(ac, 4), "F1": round(f1, 4)}
    )


# # Guardar resultados en CSV
df_results = pd.DataFrame(results)
df_results.to_csv("resultados.csv", index=False)

# Mostrar resultados en consola
print("\n📊 Comparación de Modelos:")
print(df_results)
