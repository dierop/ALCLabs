from dataloader import load_data_json, load_test_json
from sklearn.model_selection import train_test_split
import pandas as pd
from bert import train_and_evaluate_bert
from evaluate import store_results



# Datos
path = "data/" # add Lab2/data if run from outisde Lab2
train_data_path = f"{path}dataset_task3_exist2025/training.json"
test_data_path = f"{path}dataset_task3_exist2025/test.json"

def main(language=["EN"], gen_test=False, model_name='hate-speech-CNERG/bert-base-uncased-hatexplain'):
    # Data Loader
    train_data_meme = load_data_json(train_data_path)

    # Conjunto a evaluar
    test_data = load_test_json(test_data_path)

    if len(language) == 2:
        # model = "bert-base-uncased"
        pass
    elif "EN" in language:
        # model = "bert-base-uncased"
        train_data_meme = train_data_meme[train_data_meme['split']=='TRAIN-VIDEO_EN']
        train_data_meme = train_data_meme.reindex(columns=["id", "text", "label"])
        test_data = test_data[test_data['split']=='DEV-VIDEO_EN']
        test_data = test_data.reindex(columns=["id", "text", "label"])
        print("Len EN", len(train_data_meme))
    else:
        # model = "dccuchile/bert-base-spanish-wwm-cased"
        train_data_meme = train_data_meme[train_data_meme['split']=='TRAIN-VIDEO_ES']
        train_data_meme = train_data_meme.reindex(columns=["id", "text", "label"])
        test_data = test_data[test_data['split']=='DEV-VIDEO_ES']
        test_data = test_data.reindex(columns=["id", "text", "label"])
        print("Len ES", len(train_data_meme))
        

    # Preprocesar datos
    # train_data_meme['text'] = train_data_meme['text'].apply(preprocess_data)

    # test_data['text'] = test_data['text'].apply(preprocess_data)

    # Consolidado
    X = train_data_meme['text']
    y = train_data_meme['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,)


    if not gen_test:

        # Resultados BERT
        accuracy_bert, f1_bert, y_preds = train_and_evaluate_bert(X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist(), epochs=5, model_name=model_name)
        print(accuracy_bert, f1_bert)
    else:
        print("Generando test")
        X_train = pd.concat([X_train, X_test], ignore_index=True)
        y_train = pd.concat([y_train, y_test], ignore_index=True)

        _,_,y_preds = train_and_evaluate_bert(X_train.tolist(), y_train.tolist(), test_data['text'].tolist(),  [0] * len(test_data['text']), epochs=5, pred=True, model_name=model_name)
        store_results(test_data['id'], y_preds, model_name, path="outputs")

