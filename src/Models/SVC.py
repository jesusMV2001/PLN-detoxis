from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_svc(x_train, y_train):
    print("Entrenando modelo de máquina de soporte vectorial...")
    # Busca los mejores parametros
    # best_params = search_best_params(x_train, y_train)
    # Entrenar el modelo SVC con los mejores parámetros
    # svc_best = SVC(**best_params, probability=True)
    svc_best = SVC(C=10, gamma='scale', kernel='linear', probability=True)

    svc_best.fit(x_train, y_train)

    return svc_best


def search_best_params(x_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear'],
    }

    # Crear el modelo SVC
    svc = SVC(probability=True)

    # Crear la búsqueda en cuadrícula
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='f1_macro')

    # Ajustar la búsqueda en cuadrícula a los datos
    grid_search.fit(x_train, y_train)

    # Obtener los mejores parámetros
    best_params = grid_search.best_params_

    print(f'Best parameters: {best_params}')

    return best_params
