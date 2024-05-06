import tkinter as tk
from tkinter import ttk


def main():
    # Crear ventana
    window = tk.Tk()
    window.title("Configuracion")

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

    def run_model():
        model_type = model_var.get()
        sampling_type = sampling_var.get()
        #run_model_func(model_type, sampling_type)

    run_button = tk.Button(window, text="Ejecutar modelo", command=run_model)
    run_button.grid(column=0, row=4)

    # Inicia el bucle principal de la interfaz gráfica
    window.mainloop()
