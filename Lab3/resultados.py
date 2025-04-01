from dataloader import load_data_json, load_test_json
from sklearn.model_selection import train_test_split
import pandas as pd
from bert import train_and_evaluate_bert
from evaluate import store_results



# Datos
path = "data/" # add Lab2/data if run from outisde Lab2
train_data_path = f"{path}dataset_task3_exist2025/training.json"
test_data_path = f"{path}/dataset_task3_exist2025/test.json"

def main(language=["EN"], gen_test=False):
    # Data Loader
    train_data_meme = load_data_json(train_data_path)

    # Conjunto a evaluar
    test_data = load_test_json(test_data_path)

    if len(language) == 2:
        model = "bert-base-uncased"
    elif "EN" in language:
        model = ""
        train_data_meme = train_data_meme[train_data_meme['split']=='EN']
    else:
        model = ""
        train_data_meme = train_data_meme[train_data_meme['split']=='ES']
        

    # Preprocesar datos
    # train_data_meme['text'] = train_data_meme['text'].apply(preprocess_data)

    # test_data['text'] = test_data['text'].apply(preprocess_data)

    # Consolidado
    X = train_data_meme['text']
    y = train_data_meme['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    if not gen_test:

        # Resultados BERT
        accuracy_bert, f1_bert, y_preds = train_and_evaluate_bert(X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist())
        print(accuracy_bert, f1_bert)
    else:
        X_train = pd.concat([X_train, X_test], ignore_index=True)
        y_train = pd.concat([y_train, y_test], ignore_index=True)
        if type == 'bert':
        # Resultados BERT
            _,_,y_preds = train_and_evaluate_bert(X_train.tolist(), y_train.tolist(), test_data['text'].tolist())
            store_results(test_data['id'], y_preds, "bert", path="outputs")

