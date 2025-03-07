from dataloader import load_data_json, load_test_json, load_data_csv
from preprocess import preprocess_data
from SVC import train_and_evaluate_svc
import pandas as pd

# Datos
train_data_path = "Lab2/data/lab2_materials/dataset_task4_exist2025/training.json"
train_data_path_blip = 'Lab2/data/lab2_materials/dataset_task4_exist2025/blip_captions_training.csv'
train_data_path_mami = 'Lab2/data/mami_dataset/training_mami.csv'

test_data_path = "Lab2/data/lab2_materials/dataset_task4_exist2025/test.json"
test_data_path_blip = 'Lab2/data/lab2_materials/dataset_task4_exist2025/blip_captions_test.csv'
test_data_path_mami = 'Lab2/data/mami_dataset/test_mami.csv'


# Data Loader
train_data_meme = load_data_json(train_data_path)
#train_data_blip = load_data_csv(train_data_path_blip)
train_data_mami = load_data_csv(train_data_path_mami)

#test_data_blip = load_data_csv(test_data_path_blip)
test_data_mami = load_data_csv(test_data_path_mami)

# Conjunto a evaluar
test_data = load_test_json(test_data_path)


# Preprocesar datos
train_data_meme['text'] = train_data_meme['text'].apply(preprocess_data)
#train_data_blip['blip_caption'] = train_data_blip['blip_caption'].apply(preprocess_data)
train_data_mami['text'] = train_data_mami['text'].apply(preprocess_data)

test_data['text'] = test_data['text'].apply(preprocess_data)
#test_data_blip['blip_caption'] = test_data_blip['blip_caption'].apply(preprocess_data)
test_data_mami['text'] = test_data_mami['text'].apply(preprocess_data)


# Consolidado
train_data = pd.concat([train_data_meme, train_data_mami], ignore_index=True)
test_data = test_data_mami

# SVC Data 1
X_train = train_data['text']
y_train = train_data['label']
X_test = test_data['text']
y_test = test_data['label']

accuracy, f1, _, _ = train_and_evaluate_svc(X_train, y_train, X_test, y_test)
print(accuracy, f1)

# SVC 2