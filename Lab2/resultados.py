from dataloader import load_data_json, load_test_json, load_data_csv
from preprocess import preprocess_data
from SVC import train_and_evaluate_svc
from sklearn.model_selection import train_test_split
import pandas as pd
from bert import train_and_evaluate_bert

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
train_data = pd.concat([train_data_meme, train_data_mami, test_data_mami], ignore_index=True)
X = train_data['text']
y = train_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resultados SVC
accuracy_svc, f1_svc, _, _ = train_and_evaluate_svc(X_train, y_train, X_test, y_test)
print(accuracy_svc, f1_svc)

# Resultados BERT
accuracy_bert, f1_bert, _, _ = train_and_evaluate_bert(X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist())
print(accuracy_bert, f1_bert)
