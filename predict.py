import pandas as pd
import joblib

from text_preprocessing import clean_text


def generate_predictions(model_path, vectorizer_path, test_data_path, submission_path):
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    test_data = pd.read_csv(test_data_path)
    test_data['text_clean'] = test_data['comment'].apply(clean_text)
    test_features = tfidf.transform(test_data['text_clean'])

    predictions = model.predict(test_features)
    submission = pd.DataFrame({
        'id': test_data['id'],
        'label': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Archivo de envío guardado en: {submission_path}")
