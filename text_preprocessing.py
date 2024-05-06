import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import joblib

# Inicialización de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Descargar las stopwords de NLTK
nltk.download('stopwords')


def leer_stopwords(route):
    try:
        with open(route, 'r', encoding='utf-8') as archivo:
            newstopwords = archivo.read().splitlines()
        return newstopwords
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo: {e}")
        return []


def expand_stopwords():
    base_stopwords = stopwords.words('spanish')
    custom_stopwords = leer_stopwords('Data/spanishStopWords.txt')  # Asegúrate de que esto devuelve una lista o un set
    combined_stopwords = set(base_stopwords).union(custom_stopwords)  # Combina ambos conjuntos de stopwords
    return list(combined_stopwords)  # Convertir el conjunto resultante en una lista


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'\W', ' ', text)  # Eliminar caracteres especiales
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
    return text


def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def prepare_features(text_data, max_features=5000):
    text_data = [lemmatize_text(clean_text(text)) for text in text_data]
    stop_words = expand_stopwords()
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=(1, 3))
    features = tfidf.fit_transform(text_data).toarray()
    joblib.dump(tfidf, 'Model/tfidf_vectorizer.pkl')  # Guardar el vectorizador para usarlo en predicciones
    return features
