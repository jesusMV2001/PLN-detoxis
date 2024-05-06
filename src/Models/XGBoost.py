import numpy as np
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier


def train_xgboost(x_train, y_train, sample_weights):
    model = XGBClassifier(
        n_estimators=500,  # Incrementar si es necesario para mejorar el aprendizaje
        max_depth=5,  # Reducir para limitar el sobreajuste
        learning_rate=0.1,
        subsample=0.7,  # Ajustar para que el modelo no vea el 100% de las muestras cada vez
        colsample_bytree=0.7,  # Regularización por columnas
        gamma=0.3,  # Penalización por la complejidad del árbol
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        objective='multi:softmax'
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


def calcular_pesos_clases(y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights = {i: class_weights[i] for i in range(len(class_weights))}
    return [weights[y] for y in y_train]
