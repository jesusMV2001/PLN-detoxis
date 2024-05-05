from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_random_forest(X_train, y_train):
    """ Entrena un modelo de bosque aleatorio con ajustes para manejar desbalance de clases y parámetros ajustados. """
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                                   min_samples_leaf=5, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """ Evalúa el modelo con el conjunto de prueba y muestra métricas de rendimiento. """
    y_pred = model.predict(X_test)
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print("F1-score Macro:", f1_macro)
