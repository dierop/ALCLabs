import pandas as pd
from sklearn.svm import SVC
from dataloader import load_data_json, load_test_json, load_data_csv
from preprocess import preprocess_data

# Datos
train_data_path = "Lab2/data/lab2_materials/dataset_task4_exist2025/training.json"
train_data_path_blip = 'Lab2/data/lab2_materials/dataset_task4_exist2025/blip_captions_training.csv'
train_data_path_mami = 'Lab2/data/mami_dataset/training_mami.csv'

test_data_path = "Lab2/data/lab2_materials/dataset_task4_exist2025/test.json"
test_data_path_blip = 'Lab2/data/lab2_materials/dataset_task4_exist2025/blip_captions_test.csv'
test_data_path_mami = 'Lab2/data/mami_dataset/test_mami.csv'


# Data Loader
train_data = load_data_json(train_data_path)
train_data_blip = load_data_csv(train_data_path_blip)
train_data_mami = load_data_csv(train_data_path_mami)

test_data = load_test_json(test_data_path)
test_data_blip = load_data_csv(test_data_path_blip)
test_data_mami = load_data_csv(test_data_path_mami)


# Preprocesar datos
train_data['text'] = train_data['text'].apply(preprocess_data)
train_data_blip['blip_caption'] = train_data_blip['blip_caption'].apply(preprocess_data)
train_data_mami['blip_caption'] = train_data_mami['blip_caption'].apply(preprocess_data)

test_data['text'] = test_data['text'].apply(preprocess_data)
test_data_blip['blip_caption'] = test_data_blip['blip_caption'].apply(preprocess_data)
test_data_mami['blip_caption'] = test_data_mami['blip_caption'].apply(preprocess_data)



"""
model = SVC(kernel="linear", probability=True, random_state=42)
results = []



# # Guardar resultados en CSV
df_results = pd.DataFrame(results)
df_results.to_csv("Lab2/resultados.csv", index=False)

# Mostrar resultados en consola
print("\n📊 Comparación de Modelos:")
print(df_results)
"""