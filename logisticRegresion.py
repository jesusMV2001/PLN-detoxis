from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train_logistic_regression(x_train, y_train):
    """ Entrena un modelo de regresión logística con ajustes óptimos para manejar desbalance de clases. """
    # Calcular los pesos para las clases
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Configuración para la búsqueda de hiperparámetros
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # liblinear trabaja bien con l1 y l2
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(x_train, y_train)

    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score F1-macro:", grid_search.best_score_)

    # Retorna el modelo con los mejores parámetros
    return grid_search.best_estimator_
