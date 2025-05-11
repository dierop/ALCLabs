import json
import pandas as pd

from collections import Counter

# Función general para mayoría
def get_majority_label(label_list):
    """Devuelve la etiqueta mayoritaria si tiene mayoría; si no, devuelve None."""
    counts = Counter(label_list)
    most_common_label, count = counts.most_common(1)[0]
    if count > len(label_list) / 2:
        return most_common_label
    return None

# Función específica para label3 (lista de listas)
def get_majority_label3(label_list_of_lists):
    flat = [item for sublist in label_list_of_lists for item in sublist]
    if not flat:
        return None
    
    counts = Counter(flat)
    # Filtra las etiquetas con count > 1
    repeated = [label for label, cnt in counts.items() if cnt > 1]

    if not repeated:          # lista vacía  →  no hay ninguna repetida
        return None


    return repeated



def load_data_json(json_file, soft=True):
    """
    Load data from json file and return as pandas DataFrame
    args:
        json_file: str, path to json file
    return:
        df: pd.DataFrame, DataFrame with columns 'tweet', 'id' and 'labels_task1'
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)


    # Crear DataFrame desde diccionario
    df = pd.DataFrame(
        [
            {
                "text": v["text"],
                "label1": v["labels_task3_1"],
                "label2": v["labels_task3_2"],
                "label3": v["labels_task3_3"],
                "id": v["id_EXIST"],
                "split": v["split"],
            }
            for _, v in data.items()
        ]
    )

    # Si se quiere suavizar (soft) las etiquetas, convertir a proporciones
    if soft:
        # Para label1: proporción de "YES"
        df["label1"] = df["label1"].apply(
            lambda x: {k: x.count(k) / len(x) for k in ["YES", "NO"]}
        )

        # Para label2: proporción por categoría
        keys2 = ['-', 'DIRECT', 'JUDGEMENTAL']
        df["label2"] = df["label2"].apply(
            lambda x: {k: x.count(k) / len(x) for k in keys2}
        )

        # Para label3: dict con proporción por categoría extendida
        keys3 = [
            '-', 'IDEOLOGICAL-INEQUALITY', 'STEREOTYPING-DOMINANCE',
            'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'MISOGYNY-NON-SEXUAL-VIOLENCE'
        ]
        df["label3"] = df["label3"].apply(
            lambda x: (
                lambda flat: {k: flat.count(k) / len(flat) if len(flat) > 0 else 0.0 for k in keys3}
            )([item for sublist in x for item in sublist])
        )

    else:
        # Aplicar solo si soft == False

        df["label1"] = df["label1"].apply(get_majority_label)
        df["label2"] = df["label2"].apply(get_majority_label)
        df["label3"] = df["label3"].apply(get_majority_label3)
        

    

    df = df.reindex(columns=["id", "text", "label1","label2","label3","split"])
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
