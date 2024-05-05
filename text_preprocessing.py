import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar las stopwords de NLTK
nltk.download('stopwords')


def clean_text(text):
    """ Limpia el texto reduciendo a minúsculas, eliminando caracteres especiales y espacios extra. """
    text = text.lower()  # convertir a minúsculas
    text = re.sub(r'\W', ' ', text)  # eliminar caracteres especiales
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # eliminar múltiples espacios
    text = re.sub(r'^b\s+', '', text)  # eliminar el prefijo b
    return text


def prepare_features(text_data, max_features=5000):
    """ Prepara las características del texto utilizando TF-IDF. """
    stop_words = stopwords.words('spanish')  # Obtener la lista de stopwords
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    features = tfidf.fit_transform(text_data).toarray()
    return features
