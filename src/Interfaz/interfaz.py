import threading
import tkinter as tk
from tkinter import ttk

from src import ejecucion


def main(settings):
    # Crear ventana
    window = tk.Tk()
    window.title("Configuracion")
    # Establecer tamaño de la ventana
    window.geometry('300x200')
    # Pantalla aparece en el centro
    window.eval('tk::PlaceWindow . center')

    # Crear menú desplegable para seleccionar el tipo de modelo
    model_types = ['Modelo 1', 'Modelo 2', 'Modelo 3']
    model_var = tk.StringVar()
    model_dropdown = ttk.Combobox(window, textvariable=model_var, values=model_types)
    model_dropdown.grid(column=0, row=0)
    model_dropdown.current(0)  # Seleccionar el primer elemento por defecto
    model_dropdown.state(['readonly'])  # no permitir editar el campo

    # Crear botones de radio para seleccionar el tipo de muestreo
    sampling_var = tk.StringVar(value='none')
    sampling_options = ['Sobremuestreo', 'Submuestreo', 'Ninguno']
    for i, option in enumerate(sampling_options):
        tk.Radiobutton(window, text=option, variable=sampling_var, value=option.lower()).grid(column=0, row=i + 1)

    # check box para seleccionar si se quiere hacer la busqueda de los mejores parametros
    check_var = tk.IntVar()
    check_box = tk.Checkbutton(window, text="Buscar mejores parametros", variable=check_var)
    check_box.grid(column=0, row=6)

    # Crear etiqueta para mostrar el estado de la ejecución
    status_label = tk.Label(window, text="")
    status_label.grid(column=0, row=5)

    def run_model():
        settings['search_best_params'] = check_var.get()
        settings['sobremuestreo'] = False
        settings['submuestreo'] = False
        settings[sampling_var.get()] = True
        status_label.config(text="Ejecutando...")
        # bloquear botones hasta que termine la ejecución
        run_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=ejecucion.main, args=(settings,))
        thread.start()
        window.after(1000, check_thread, thread)

    def check_thread(thread):
        if thread.is_alive():
            window.after(1000, check_thread, thread)
        else:
            status_label.config(text="Ejecución completada")
            # desbloquear botones
            run_button.config(state=tk.NORMAL)

    run_button = tk.Button(window, text="Ejecutar modelo", command=run_model)
    run_button.grid(column=0, row=4)

    # Inicia el bucle principal de la interfaz gráfica
    window.mainloop()
