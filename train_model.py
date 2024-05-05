from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_xgboost(X_train, y_train):
    """ Entrena un modelo XGBoost con ajustes para manejar desbalance de clases. """
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                          scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss',
                          random_state=42)
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
