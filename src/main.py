from src import ejecucion
from src.Interfaz import interfaz

from src.TextProcessing.data_loader import load_settings_from_file


# Cargar settings
settings = load_settings_from_file('settings.json')

if settings['interfaz']:
    interfaz.main(settings)
else:
    ejecucion.main(settings)

