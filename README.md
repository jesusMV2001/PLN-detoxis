# Resumen de los archivos

## data_loader.py
Este archivo define una función load_data que carga los datos desde un archivo CSV utilizando la biblioteca pandas.

## logistic_regression.py
Este archivo define una función train_logistic_regression que entrena un modelo de regresión logística utilizando GridSearchCV para encontrar los mejores hiperparámetros. También maneja el desbalance de clases calculando los pesos de las clases.

## main.py
Este es el archivo principal del proyecto. Carga los datos, realiza el preprocesamiento de texto, divide los datos en conjuntos de entrenamiento y prueba, entrena el modelo de regresión logística, evalúa el modelo y genera predicciones. 

## predict.py
Este archivo define una función generate_predictions que carga y preprocesa los datos de prueba, genera predicciones utilizando el modelo proporcionado y guarda las predicciones en un archivo CSV.

## randomForest.py
Este archivo define una función train_random_forest que entrena un modelo de bosque aleatorio con ajustes para manejar el desbalance de clases.

## text_preprocessing.py
Este archivo define dos funciones, clean_text y prepare_features. clean_text limpia el texto eliminando caracteres especiales, convirtiendo a minúsculas y eliminando espacios extra. prepare_features prepara las características del texto utilizando TF-IDF. 

## train_model.py
Este archivo define una función evaluate_model que evalúa el modelo con el conjunto de prueba y muestra métricas de rendimiento como el informe de clasificación, la matriz de confusión y el F1-score macro. 

## XGBoost.py
Este archivo define una función train_xgboost que entrena un modelo XGBoost. Los hiperparámetros del modelo están predefinidos en la función.