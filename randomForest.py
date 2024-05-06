from sklearn.ensemble import RandomForestClassifier


def train_random_forest(x_train, y_train):
    """ Entrena un modelo de bosque aleatorio con ajustes para manejar desbalance de clases y parámetros ajustados. """
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                                   min_samples_leaf=5, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    return model
