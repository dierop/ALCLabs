from dataloader import load_data_json, load_test_json, load_data_csv, load_blip_csv
from preprocess import preprocess_data
from classic import train_and_evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
from bert import train_and_evaluate_bert
from evaluate import store_results



# Datos
path = "data/" # add Lab2/data if run from outisde Lab2
train_data_path = f"{path}lab2_materials/dataset_task4_exist2025/training.json"
train_data_path_blip = f'{path}lab2_materials/dataset_task4_exist2025/blip_captions_training.csv'
train_data_path_mami = f'{path}mami_dataset/training_mami.csv'

test_data_path = f"{path}lab2_materials/dataset_task4_exist2025/test.json"
test_data_path_blip = f'{path}lab2_materials/dataset_task4_exist2025/blip_captions_test.csv'
test_data_path_mami = f'{path}mami_dataset/test_mami.csv'

def main(blip=0, mami=False, gen_test=False, type="SVC"):
    print("Blip: ", blip)
    print("Mami: ", mami)
    print("Gen Test: ", gen_test)
    # Data Loader
    train_data_meme = load_data_json(train_data_path)
    train_data_blip = load_blip_csv(train_data_path_blip)
    train_data_mami = load_data_csv(train_data_path_mami)

    test_data_blip = load_blip_csv(test_data_path_blip)
    test_data_mami = load_data_csv(test_data_path_mami)

    # Conjunto a evaluar
    test_data = load_test_json(test_data_path)

    if blip == 1:
        train_data_meme['text'] = train_data_blip['text']
        test_data['text'] = test_data_blip['text']
    elif blip == 2:
        train_data_meme['text'] = train_data_meme['text'].astype(str)
        train_data_blip['text'] = train_data_blip['text'].astype(str)
        train_data_meme['text'] = train_data_meme['text'].str.cat(train_data_blip['text'], sep=' ')
        test_data['text'] = test_data['text'].str.cat(test_data_blip['text'], sep=' ')



    # Preprocesar datos
    train_data_meme['text'] = train_data_meme['text'].apply(preprocess_data)
    #train_data_blip['blip_caption'] = train_data_blip['blip_caption'].apply(preprocess_data)
    train_data_mami['text'] = train_data_mami['text'].apply(preprocess_data)

    test_data['text'] = test_data['text'].apply(preprocess_data)
    #test_data_blip['blip_caption'] = test_data_blip['blip_caption'].apply(preprocess_data)
    test_data_mami['text'] = test_data_mami['text'].apply(preprocess_data)


    # Consolidado
    X = train_data_meme['text']
    y = train_data_meme['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add MAMI data to train set
    # It is used as data augmentation
    if mami:
        for data in [train_data_mami, test_data_mami]:
            X = data['text']
            y = data['label']
            X_train = pd.concat([X_train, X], ignore_index=True)
            y_train = pd.concat([y_train, y], ignore_index=True)

    if not gen_test:
    # Resultados SVC
        accuracy_svc, f1_svc, y_preds = train_and_evaluate(X_train, y_train, X_test, y_test, kernel="linear", tpye="SVC")
        print("SVC")
        print(accuracy_svc, f1_svc)

        # Resultados RF
        accuracy_rf, f1_rf, y_preds = train_and_evaluate(X_train, y_train, X_test, y_test, kernel="linear", tpye="RF")
        print("RF")
        print(accuracy_rf, f1_rf)

        # # Resultados BERT
        # accuracy_bert, f1_bert, y_preds = train_and_evaluate_bert(X_train.tolist()[:5000], y_train.tolist()[:5000], X_test.tolist(), y_test.tolist())
        # print(accuracy_bert, f1_bert)
    else:
        X_train = pd.concat([X_train, X_test], ignore_index=True)
        y_train = pd.concat([y_train, y_test], ignore_index=True)
        if type == 'bert':
        # Resultados BERT
            _,_,y_preds = train_and_evaluate_bert(X_train.tolist()[:5000], y_train.tolist()[:5000], test_data['text'].tolist())
            store_results(test_data['id'], y_preds, "bert", path="outputs")

        elif type == 'SVC':
        # Resultados SVC
            _,_,y_preds = train_and_evaluate(X_train, y_train, test_data['text'], [], kernel="linear", tpye="SVC", pred=True)
            store_results(test_data['id'], y_preds, "svc", path="outputs")

        # Resultados RF
        elif type == 'RF':
            _,_,y_preds = train_and_evaluate(X_train, y_train, test_data['text'], [], kernel="linear", tpye="RF", pred=True)
            store_results(test_data['id'], y_preds, "rf", path="outputs")


