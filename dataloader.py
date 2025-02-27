import json
import pandas as pd

def load_data(json_file):
    '''
    Load data from json file and return as pandas DataFrame
    args:
        json_file: str, path to json file
    return:
        df: pd.DataFrame, DataFrame with columns 'tweet' and 'labels_task1'
    '''
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame([
        {'tweet': v['tweet'], 'labels_task1': v['labels_task1']}
        for k, v in data.items()
    ])

    df['label'] = df['labels_task1'].apply(lambda x: max(x, key=x.count))
    df['label'] = df['label'].apply(lambda x: 0 if x == 'NO' else 1)
    df.drop('labels_task1', axis=1, inplace=True)
    return df