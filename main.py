from data_loader import load_data
from logisticRegresion import train_logistic_regression
from predict import generate_predictions
from text_preprocessing import clean_text, prepare_features
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from train_model import evaluate_model

# Cargar los datos
data = load_data('Data/Detoxis_train_kaggle.csv')

# Preprocesamiento
data['text_clean'] = data['text'].apply(clean_text)
features = prepare_features(data['text_clean'])
labels = data['label']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Aplicar sobremuestreo a las clases minoritarias
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Entrenar y evaluar el modelo utilizando los datos sobremuestreados
model = train_logistic_regression(X_resampled, y_resampled)
evaluate_model(model, X_test, y_test)

generate_predictions(model, 'Data/Detoxis_test_kaggle.csv', 'submission.csv')
