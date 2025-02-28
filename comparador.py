import pandas as pd
from cross_validation import cross_validation
from modelos_clasicos import train_model
from bert import train_bert  

train_data_path = "data/dataset_task1_exist2025/training.json"

# Evaluamos modelos 
models = ["svm", "rf", "xgb", "logreg", "nb", "bert"]
results = []

for model_type in models:
    print(f"\n🔹 Evaluando modelo: {model_type.upper()}")

    if model_type == "bert":
        model, accuracy = train_bert(train_data_path)
    else:
        accuracy, _ = cross_validation(train_data_path, lambda x: train_model(x, model_type=model_type))

    results.append({"Modelo": model_type.upper(), "Accuracy": round(accuracy, 4)})

# Guardar resultados en CSV
df_results = pd.DataFrame(results)
df_results.to_csv("resultados.csv", index=False)

# Mostrar resultados en consola
print("\n📊 Comparación de Modelos:")
print(df_results)