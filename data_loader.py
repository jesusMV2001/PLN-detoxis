import pandas as pd


def load_data(filepath):
    """ Carga los datos desde un archivo CSV. """
    return pd.read_csv(filepath)
