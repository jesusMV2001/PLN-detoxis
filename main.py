from XGBoost import train_xgboost
from data_loader import load_data
from predict import generate_predictions
from text_preprocessing import clean_text, prepare_features
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from train_model import evaluate_model

# Cargar los datos
data = load_data('Data/Detoxis_train_kaggle.csv')

# Preprocesamiento
data['text_clean'] = data['text'].apply(clean_text)
features = prepare_features(data['text_clean'])
labels = data['label']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Calcular los pesos para las clases
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = {i: class_weights[i] for i in range(len(class_weights))}
sample_weights = np.array([weights[i] for i in y_train])

# Entrenar y evaluar el modelo utilizando los datos y los pesos de las clases
model = train_xgboost(X_train, y_train, sample_weights)
evaluate_model(model, X_test, y_test)

generate_predictions(model, 'Data/Detoxis_test_kaggle.csv', 'submission.csv')
