import json
import pandas as pd


def load_data_json(json_file):
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


def load_data_csv(csv_file):
    """
    Load data from csv file and return as pandas DataFrame
    args:
        csv_file: str, path to csv file
    return:
        df: pd.DataFrame, DataFrame with columns 'tweet', 'id' and 'labels_task1'
    """
    df2 = pd.read_csv(csv_file)
    df = df2[['meme_id','blip_caption', 'misogynous']]
    df = df.rename(columns={'meme_id': 'id', 'misogynous': 'label', 'blip_caption': 'text'})

    return df

def load_blip_csv(csv_file):
    """
    Load data from csv file and return as pandas DataFrame
    args:
        csv_file: str, path to csv file
    return:
        df: pd.DataFrame, DataFrame with columns 'tweet', 'id' and 'labels_task1'
    """
    df2 = pd.read_csv(csv_file)
    df = df2[['meme_id','blip_caption']]
    df = df.rename(columns={'meme_id': 'id', 'blip_caption': 'text'})

    return df


def load_test_json(json_file):
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
    df = df.reindex(columns=["id", "text"])
    return df
