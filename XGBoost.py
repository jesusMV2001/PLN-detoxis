from xgboost import XGBClassifier


def train_xgboost(x_train, y_train, sample_weights):
    model = XGBClassifier(n_estimators=100, max_depth=9, learning_rate=0.05, subsample=0.8,
                          use_label_encoder=False, eval_metric='mlogloss', random_state=42,
                          objective='multi:softmax')
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model
