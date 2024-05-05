from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_model(model, x_test, y_test):
    """ Evalúa el modelo con el conjunto de prueba y muestra métricas de rendimiento. """
    y_pred = model.predict(x_test)
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print("F1-score Macro:", f1_macro)
