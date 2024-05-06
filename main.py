import joblib
import numpy as np

from XGBoost import train_xgboost, calcular_pesos_clases
from logisticRegresion import train_logistic_regression
from data_loader import load_data
from predict import generate_predictions
from randomForest import train_random_forest
from text_preprocessing import clean_text, prepare_features
from sklearn.model_selection import train_test_split
from train_model import evaluate_model

# Cargar los datos
data = load_data('Data/Detoxis_train_kaggle.csv')

# Preprocesamiento
data['text_clean'] = data['text'].apply(clean_text)
features = prepare_features(data['text_clean'])
labels = data['label']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Entrenar y evaluar el modelo utilizando los datos y los pesos de las clases
model = train_random_forest(X_train, y_train)
joblib.dump(model, 'Model/model.pkl')

evaluate_model(joblib.load('Model/model.pkl'), X_test, y_test)

generate_predictions('Model/model.pkl', 'Model/tfidf_vectorizer.pkl', 'Data/Detoxis_test_kaggle.csv', 'submission.csv')
