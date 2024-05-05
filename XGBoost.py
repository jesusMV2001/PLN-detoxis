from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """ Entrena un modelo XGBoost con ajustes para manejar desbalance de clases. """
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                          scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss',
                          random_state=42)
    model.fit(X_train, y_train)
    return model
