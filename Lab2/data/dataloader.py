import json
import pandas as pd


def load_data(json_file):
    """
    Load data from json file and return as pandas DataFrame
    args:
        json_file: str, path to json file
    return:
        df: pd.DataFrame, DataFrame with columns 'tweet', 'id' and 'labels_task1'
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(
        [
            {
                "text": v["text"],
                "label": v["labels_task4"],
                "id": v["id_EXIST"],
            }
            for _, v in data.items()
        ]
    )

    df["label"] = df["label"].apply(lambda x: max(x, key=x.count))
    df["label"] = df["label"].apply(lambda x: 0 if x == "NO" else 1)
    df = df.reindex(columns=["id", "text", "label"])
    return df

def load_test(json_file):
    """
    Load data from json file and return as pandas DataFrame
    args:
        json_file: str, path to json file
    return:
        df: pd.DataFrame, DataFrame with columns 'tweet' and 'id'
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(
        [
            {
                "text": v["text"],
                "id": v["id_EXIST"],
            }
            for _, v in data.items()
        ]
    )

    return df
