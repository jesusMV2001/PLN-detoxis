from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_random_forest(x_train, y_train, settings):
    print("Entrenando modelo de bosque aleatorio...")
    # Busca los mejores parametros
    if settings['search_best_params']:
        print("Buscando mejores parametros...")
        best_params = search_best_params(x_train, y_train)
        # Entrena un modelo de bosque aleatorio con los mejores parámetros encontrados
        model = RandomForestClassifier(**best_params, random_state=42, class_weight="balanced")
    else:
        # Para no hacer la busqueda de los mejores parametros
        model = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_split=5,
                                       min_samples_leaf=2, random_state=42, class_weight="balanced")

    model.fit(x_train, y_train)

    return model


def search_best_params(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 250, 300, 400],
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Crear el modelo de bosque aleatorio
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    # Crear la búsqueda en cuadrícula
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro')

    # Ajustar la búsqueda en cuadrícula a los datos
    grid_search.fit(x_train, y_train)

    # Obtener los mejores parámetros
    best_params = grid_search.best_params_

    print(f'Best parameters: {best_params}')

    return best_params
