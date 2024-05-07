import joblib

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.TextProcessing.data_loader import load_data
from src.TextProcessing.predict import generate_predictions
from src.Models.randomForest import train_random_forest
from src.TextProcessing.text_preprocessing import clean_text, prepare_features
from sklearn.model_selection import train_test_split
from src.Models.evaluate_model import evaluate_model


def main(settings):
    # Cargar los datos
    data = load_data('Data/Detoxis_train_kaggle.csv')

    # Preprocesamiento
    data['text_clean'] = data['text'].apply(clean_text)
    features = prepare_features(data['text_clean'])
    labels = data['label']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Aplicar muestreo si se especifica en los settings
    x_train, y_train = aplicar_muestreo(X_train, y_train, settings)

    # Entrenar y evaluar el modelo utilizando los datos y los pesos de las clases
    model = train_random_forest(x_train, y_train)
    # Guardar el modelo
    joblib.dump(model, 'DataModels/model.pkl')
    # Evaluar el modelo
    evaluate_model(joblib.load('DataModels/model.pkl'), X_test, y_test)
    # Generar predicciones
    generate_predictions('DataModels/model.pkl', 'DataModels/tfidf_vectorizer.pkl', 'Data/Detoxis_test_kaggle.csv',
                         'submission.csv')


def aplicar_muestreo(x_train, y_train, settings):
    if settings['sobremuestreo']:
        over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        x_train, y_train = over_sampler.fit_resample(x_train, y_train)
        print("Sobremuestreo aplicado")
    elif settings['submuestreo']:
        under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        x_train, y_train = under_sampler.fit_resample(x_train, y_train)
        print("Submuestreo aplicado")

    return x_train, y_train
