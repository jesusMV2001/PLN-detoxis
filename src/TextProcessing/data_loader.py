import json

import pandas as pd


def load_data(filepath):
    """ Carga los datos desde un archivo CSV. """
    return pd.read_csv(filepath)


def load_settings_from_file(filepath):
    with open(filepath, 'r') as f:
        settings = json.load(f)
    return settings
