from sklearn.model_selection import StratifiedKFold, cross_val_score

def cross_validation(train_data_path, train_func, k_folds=5, scoring="accuracy"):
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
    model, X, y, _ = train_func(train_data_path)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

    print(f"{scoring} media: {scores.mean():.4f}")
    print(f"desviación estandar: {scores.std():.4f}")

    return scores.mean(), scores.std()
