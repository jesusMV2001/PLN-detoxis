from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train_logistic_regression(X_train, y_train):
    """ Entrena un modelo de regresión logística con pesos de clase ajustados para manejar desbalance de clases. """
    # Calcular los pesos para las clases
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights)
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
