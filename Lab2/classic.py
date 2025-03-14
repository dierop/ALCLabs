from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_and_evaluate(X_train, y_train, X_test, y_test, kernel="linear", tpye="SVC"):
    """
    Entrena un modelo SVC y lo evalúa en el conjunto de prueba.
    
    Retorna:
    - accuracy: Precisión del modelo
    - f1: F1-score del modelo
    - svc_model: Modelo SVC entrenado
    - vectorizer: Vectorizador TF-IDF ajustado
    """

    # Convertir texto a representación numérica (TF-IDF)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Inicializar y entrenar el modelo SVC
    model = SVC(kernel=kernel) if tpye == "SVC" else RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test_tfidf)

    # Calcular métricas
    accuracy = np.round(accuracy_score(y_test, y_pred), 4)
    f1 = np.round(f1_score(y_test, y_pred, average="weighted"), 4)

    return accuracy, f1, y_pred
