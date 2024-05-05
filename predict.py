import pandas as pd
from text_preprocessing import prepare_features


def generate_predictions(model, test_data_path, submission_path):
    # Carga y preprocesa los datos de prueba
    test_data = pd.read_csv(test_data_path)
    test_features = prepare_features(test_data['comment'])  # Asumiendo que 'comment' es la columna de texto

    # Generar predicciones
    predictions = model.predict(test_features)

    # Preparar y guardar el archivo de envío
    submission = pd.DataFrame({
        'id': test_data['id'],
        'label': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Archivo de envío guardado en: {submission_path}")

# Ejemplo de cómo llamar a la función
#generate_predictions('path_to_trained_model.pkl', 'Data/Detoxis_test_kaggle.csv', 'submission.csv')
