from sklearn.model_selection import StratifiedKFold, cross_val_predict


def cross_validation(df, model, vectorizer, k_folds=5):
    """
    Realiza validación cruzada en cualquier modelo de clasificación.

    Parámetros:
    - train_data_path: Ruta del dataset de entrenamiento.
    - train_func: Función que entrena el modelo (debe devolver modelo, X, y).
    - k_folds: Número de folds para la validación cruzada (default=5).
    - scoring: Métrica de evaluación para cross validation (default="accuracy").

    Retorna:
    - Media de los scores de validación.
    - Desviación estándar de los scores.
    """
    X = df["tweet"]
    y = df["label"]

    X = vectorizer.fit_transform(X)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=skf)

    return y_pred


def train_predict(df, model, vectorizer, x_test):
    model.fit(vectorizer.fit_transform(df["tweet"]), df["label"])
    return model.predict(vectorizer.transform(x_test))
