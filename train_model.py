from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_logistic_regression(X_train, y_train):
    """ Entrena un modelo de regresión logística. """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """ Evalúa el modelo con el conjunto de prueba y muestra métricas de rendimiento. """
    y_pred = model.predict(X_test)
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print("F1-score Macro:", f1_macro)

