from sklearn.ensemble import VotingClassifier
from src.Models.SVC import train_svc
from src.Models.logisticRegresion import train_logistic_regression
from src.Models.randomForest import train_random_forest


def train_ensemble_model(x_train, y_train, settings):
    # Crear los modelos individuales
    clf1 = train_random_forest(x_train, y_train, settings)
    clf2 = train_svc(x_train, y_train)
    clf3 = train_logistic_regression(x_train, y_train)

    # Crear el modelo de votación
    eclf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('lr', clf3)], voting='soft')

    # Entrenar el modelo de votación
    eclf.fit(x_train, y_train)

    return eclf
