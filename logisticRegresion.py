from sklearn.linear_model import LogisticRegression
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
