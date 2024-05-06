from src.Interfaz import interfaz

from src.TextProcessing.data_loader import load_settings_from_file
from src.ejecucion import main


# Cargar settings
settings = load_settings_from_file('settings.json')

if settings['interfaz']:
    interfaz.main()
else:
    main()

