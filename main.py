from data_loader import load_data
from text_preprocessing import clean_text, prepare_features
from train_model import train_logistic_regression, evaluate_model
from sklearn.model_selection import train_test_split

# Cargar los datos
data = load_data('Data/Detoxis_train_kaggle.csv')

# Preprocesamiento
data['text_clean'] = data['text'].apply(clean_text)
features = prepare_features(data['text_clean'])
labels = data['label']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Entrenar y evaluar el modelo
model = train_logistic_regression(X_train, y_train)
evaluate_model(model, X_test, y_test)

