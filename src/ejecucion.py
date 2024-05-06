import joblib

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.TextProcessing.data_loader import load_data
from src.TextProcessing.predict import generate_predictions
from src.Models.randomForest import train_random_forest
from src.TextProcessing.text_preprocessing import clean_text, prepare_features
from sklearn.model_selection import train_test_split
from src.Models.evaluate_model import evaluate_model


def main():
    # Cargar los datos
    data = load_data('Data/Detoxis_train_kaggle.csv')

    # Preprocesamiento
    data['text_clean'] = data['text'].apply(clean_text)
    features = prepare_features(data['text_clean'])
    labels = data['label']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Crear los samplers
    over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    # Aplicar el sobremuestreo
    X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
    # Aplicar el submuestreo
    X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

    # Entrenar y evaluar el modelo utilizando los datos y los pesos de las clases
    model = train_random_forest(X_train_over, y_train_over)
    joblib.dump(model, 'DataModels/model.pkl')

    evaluate_model(joblib.load('DataModels/model.pkl'), X_test, y_test)

    generate_predictions('DataModels/model.pkl', 'DataModels/tfidf_vectorizer.pkl', 'Data/Detoxis_test_kaggle.csv',
                         'submission.csv')
