from dataloader import load_data
from preprocess import preprocess_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def get_model(model_type="svm"):
    """
    Carga los datos, preprocesa y entrena el modelo especificado.

    Parámetros:
    - model_type: Tipo de modelo a entrenar ("rf", "xgb", "logreg", "nb").

    Retorna:
    - Modelo entrenado, vectorizador.
    """

    vectorizer = TfidfVectorizer(max_features=5000)

    # Seleccionar modelo
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_type == "svm":
        model = SVC(kernel="linear", probability=True, random_state=42)
    elif model_type == "xgb":
        model = XGBClassifier(eval_metric="logloss", random_state=42)
    elif model_type == "logreg":
        model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
    elif model_type == "nb":
        model = MultinomialNB()
    else:
        raise ValueError("Modelo no soportado. Usa 'rf', 'xgb', 'logreg', o 'nb'.")

    return model, vectorizer
