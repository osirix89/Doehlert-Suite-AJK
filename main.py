#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""
main.py — Interfaz gráfica (Tkinter + Matplotlib) de la Suite Doehlert.

Toda la lógica matemática/estadística vive en `doehlert_engine.py` (ambos archivos
deben estar en la misma carpeta). Este módulo solo contiene la vista y los
controladores. Dependencias: numpy, matplotlib (openpyxl opcional para .xlsx).
"""
import csv
import math
import os
import sys
import json
import hashlib
import platform
import traceback
import datetime
import importlib.metadata
import argparse
import subprocess

__version__ = "0.1.0"


def _construir_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description=("Doehlert Suite: Tkinter desktop application to design, fit, and diagnose "
                     "three-factor Doehlert designs. For command-line CSV analysis, run: "
                     "python3 doehlert_engine.py --help"))
    p.add_argument("archivo", nargs="?", metavar="ARCHIVO.csv",
                   help="CSV con respuestas para cargar y ajustar al abrir la ventana (opcional)")
    return p


# Se interpreta antes de importar Tkinter/Matplotlib: así `--help` funciona incluso sin entorno gráfico.
_ARGS = _construir_parser().parse_args() if __name__ == "__main__" else None

import tkinter as tk
try:
    import tkinter.font as tkfont
except Exception:          # entornos sin Tk completo (p. ej. auditoría headless)
    tkfont = None
from tkinter import filedialog, messagebox, ttk
import matplotlib
# Backend interactivo solo cuando hay servidor gráfico; permite importar/auditar en CI y headless.
try:
    matplotlib.use("TkAgg" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.name == "nt") else "Agg")
except ImportError:  # DISPLAY puede estar definido sin un servidor X accesible (CI/headless).
    matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np

from i18n import get_language, set_language, tr, install_tkinter_translations
install_tkinter_translations(tk, ttk, messagebox, filedialog)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doehlert_engine as eng
from doehlert_engine import (   # noqa: F401  (se re-exportan: auditores/pruebas externas las buscan en main)
    ALPHA, DOEHLERT_BASE, SQRT3, SQRT_2_3, ModelError, fit_quadratic_model,
    normal_ppf, _beta_continued_fraction, betainc_approx, f_sf, t_ppf_975, calcular_jacobiano_optimo,
)

# ==============================================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ==============================================================
class DoehlertSuiteApp(tk.Tk):
    def __init__(self):
        """Inicializa la ventana principal, el estado numérico de la aplicación y las tres pestañas funcionales. Define las transformaciones iniciales entre factores físicos y codificados, prepara los contenedores de resultados y construye los componentes gráficos. Los valores de centro y escala iniciales son valores predeterminados; se reemplazan al generar o importar un diseño."""
        super().__init__()
        self.title(tr(f"Doehlert Suite {__version__} — Diseño, Optimización y Diagnóstico"))
        self.geometry("1320x880")
        self.minsize(1100, 720)

        # Variables de estado
        self.matriz_generada = []
        self.headers_diseno = []
        self.factores_activos = []
        self.factor_centers = np.array([70.0, 20.0, 5.0], dtype=float)
        self.factor_scales = np.array([40.0 / SQRT3, 10.0, 2.0 / SQRT_2_3], dtype=float)
        self.x_opt_cod = None
        self.last_csv_path = None
        self.last_error = None
        self.debug_events = []
        # Fuente monoespaciada real del sistema (Consolas no existe en Debian)
        if tkfont is not None:
            self.fuente_mono = tkfont.nametofont("TkFixedFont").copy()
            self.fuente_mono.configure(size=9)
        else:
            self.fuente_mono = ("Courier", 9)
        self.fit = None
        self.cod_exp = None
        self.leverage = None
        self._hl_artists = []
        self._qq_pos = {}

        self.betas = None
        self.cov_beta = None
        self.X_exp = None
        self.y_exp = None
        self.y_pred = None
        self.residuos = None
        self.residuos_std = None
        self.x_opt = None
        self.ic_coords = []
        self.t_crit = 1.96
        self.s_res = 1.0
        self.nombres_factores = ["X1", "X2", "X3"]
        self.limites_factores = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        self.construir_menu_y_barra()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tab_diseno = ttk.Frame(self.notebook)
        self.tab_analisis = ttk.Frame(self.notebook)
        self.tab_modelo = ttk.Frame(self.notebook)
        self.tab_diagnostico = ttk.Frame(self.notebook)
        self.tab_auditoria = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_diseno, text=" 1. Diseño ")
        self.notebook.add(self.tab_analisis, text=" 2. Análisis y óptimo ")
        self.notebook.add(self.tab_modelo, text=" 3. ANOVA y coeficientes ")
        self.notebook.add(self.tab_diagnostico, text=" 4. Diagnóstico ")
        self.notebook.add(self.tab_auditoria, text=" 5. Auditoría y varias respuestas ")

        self.construir_tab_diseno()
        self.construir_tab_analisis()
        self.construir_tab_modelo()
        self.construir_tab_diagnostico()
        self.construir_tab_auditoria()
        self.multiresponse_fits = {}
        self.log_event("Aplicación iniciada")

    def log_event(self, message):
        """Añade un evento con marca temporal ISO 8601 y zona horaria local al registro interno de depuración. Conserva solamente los 300 eventos más recientes para limitar el crecimiento de memoria. No escribe archivos; la exportación se realiza desde exportar_log_debug."""
        timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        if not hasattr(self, "debug_events"):
            self.debug_events = []
        self.debug_events.append(f"[{timestamp}] {message}")
        self.debug_events = self.debug_events[-300:]

    # Organización de archivos: todo lo generado por el usuario queda agrupado por fecha.
    def carpeta_aplicacion(self):
        return os.path.dirname(os.path.abspath(__file__))

    def carpeta_exportacion(self, crear=True):
        fecha = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
        ruta = os.path.join(self.carpeta_aplicacion(), "Exports", fecha)
        if crear:
            os.makedirs(ruta, exist_ok=True)
        return ruta

    def nombre_exportacion(self, etiqueta, extension):
        fecha = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
        limpio = "_".join("".join(c if c.isalnum() else "_" for c in str(etiqueta)).strip("_").split())
        return f"{fecha}_Doehlert_{limpio or 'Resultado'}{extension}"

    def dialogo_guardar(self, etiqueta, extension, filetypes, title):
        return filedialog.asksaveasfilename(
            initialdir=self.carpeta_exportacion(),
            initialfile=self.nombre_exportacion(etiqueta, extension),
            defaultextension=extension, filetypes=filetypes, title=title)

    def importar_archivo_csv_desde_menu(self):
        ruta = filedialog.askopenfilename(
            initialdir=os.path.join(self.carpeta_aplicacion(), "Examples"),
            filetypes=[("Datos experimentales CSV", "*.csv"), ("Todos los archivos", "*.*")],
            title="Importar datos experimentales")
        if ruta:
            self.cargar_csv_respuestas(ruta)

    def mostrar_ayuda_rapida(self):
        messagebox.showinfo(
            "Cómo usar Doehlert Suite",
            "FLUJO RECOMENDADO\n\n"
            "1. Pestaña 1: define los tres factores y genera el diseño.\n"
            "2. Exporta la planilla CSV y completa la columna Respuesta_Y en Excel.\n"
            "3. Menú Archivo > Importar datos (CSV), o pestaña 2 > Cargar CSV.\n"
            "4. Revisa superficies, ANOVA y diagnóstico en las pestañas 2, 3 y 4.\n"
            "5. Pestaña 5: ejecuta la auditoría y consulta sus hallazgos.\n\n"
            "Los resultados se guardan por defecto en Exports/AÑO-MES-DÍA.\n"
            "Los CSV de práctica están en la carpeta Examples.")

    def report_callback_exception(self, exc, val, tb):
        """Captura excepciones no controladas originadas en callbacks de Tkinter. Conserva el traceback completo en `last_error`, registra el evento e intenta mostrar un diálogo sin permitir que un fallo del propio diálogo oculte la excepción original."""
        self.last_error = "".join(traceback.format_exception(exc, val, tb))
        self.log_event(f"Excepción Tkinter: {val}")
        try:
            messagebox.showerror("Error interno", f"Se produjo un error. Genera el LOG de depuración para revisarlo.\n\n{val}")
        except Exception:
            pass

    # ==============================================================
    # PESTAÑA 1: GENERADOR Y GEOMETRÍA 3D
    # ==============================================================
    def construir_tab_diseno(self):
        """Construye la pestaña de generación del diseño experimental. Crea entradas para nombres y límites de los tres factores, selector de réplicas centrales, tabla de corridas, controles de exportación/simulación y visualización 3D. Esta función configura widgets; no ejecuta cálculos estadísticos."""
        frame_top = ttk.LabelFrame(self.tab_diseno, text=" Configuración de Variables ", padding=10)
        frame_top.pack(fill="x", padx=12, pady=8)

        ttk.Label(frame_top, text="Factor / Nivel", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=6, pady=2, sticky="w")
        ttk.Label(frame_top, text="Nombre", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(frame_top, text="Mínimo", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=6, pady=2)
        ttk.Label(frame_top, text="Máximo", font=("Segoe UI", 9, "bold")).grid(row=0, column=3, padx=6, pady=2)

        self.entries = []
        labels = ["X1 (7 Niveles):", "X2 (5 Niveles):", "X3 (3 Niveles):"]
        defaults = [("Temperatura", "50", "90"), ("Tiempo", "10", "30"), ("pH", "3", "7")]

        for i in range(3):
            ttk.Label(frame_top, text=labels[i]).grid(row=i+1, column=0, padx=6, pady=3, sticky="w")
            ent_nom = ttk.Entry(frame_top, width=18)
            ent_nom.insert(0, defaults[i][0])
            ent_nom.grid(row=i+1, column=1, padx=6, pady=3)

            ent_min = ttk.Entry(frame_top, width=10)
            ent_min.insert(0, defaults[i][1])
            ent_min.grid(row=i+1, column=2, padx=6, pady=3)

            ent_max = ttk.Entry(frame_top, width=10)
            ent_max.insert(0, defaults[i][2])
            ent_max.grid(row=i+1, column=3, padx=6, pady=3)

            self.entries.append({"nombre": ent_nom, "min": ent_min, "max": ent_max})

        frame_acciones = ttk.Frame(frame_top)
        frame_acciones.grid(row=4, column=0, columnspan=4, pady=(10, 2), sticky="w")

        ttk.Label(frame_acciones, text="Réplicas del centro:").pack(side="left", padx=(0, 4))
        self.spin_replicas = ttk.Spinbox(frame_acciones, from_=1, to=10, width=4)
        self.spin_replicas.set(3)
        self.spin_replicas.pack(side="left", padx=(0, 10))

        ttk.Button(frame_acciones, text="1. Calcular Matriz y Ver 3D", command=self.generar_matriz_diseno).pack(side="left", padx=4)

        self.btn_simular = ttk.Button(frame_acciones, text="2. Simular Respuestas de Prueba", command=self.simular_datos_y_enviar, state="disabled")
        self.btn_simular.pack(side="left", padx=4)

        self.btn_enviar_directo = ttk.Button(frame_acciones, text="3. Enviar a Análisis (Pest. 2)", command=self.enviar_datos_a_analisis, state="disabled")
        self.btn_enviar_directo.pack(side="left", padx=4)

        self.btn_exportar = ttk.Button(frame_acciones, text="Exportar CSV...", command=self.exportar_csv_plantilla, state="disabled")
        self.btn_exportar.pack(side="left", padx=6)

        paned = ttk.PanedWindow(self.tab_diseno, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        frame_izq = ttk.LabelFrame(paned, text=" Matriz (Doble clic en Respuesta_Y para editar valor) ", padding=6)
        paned.add(frame_izq, weight=1)

        self.tree_diseno = ttk.Treeview(frame_izq, show="headings", selectmode="browse")
        sy = ttk.Scrollbar(frame_izq, orient="vertical", command=self.tree_diseno.yview)
        sx = ttk.Scrollbar(frame_izq, orient="horizontal", command=self.tree_diseno.xview)
        self.tree_diseno.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.tree_diseno.pack(fill="both", expand=True)

        self.tree_diseno.bind("<Double-1>", self.on_double_click_tree)

        frame_der = ttk.LabelFrame(paned, text=" Geometría Esférica de Doehlert (3D) ", padding=6)
        paned.add(frame_der, weight=2)

        self.fig_diseno = Figure(figsize=(5, 5), dpi=95)
        self.ax_diseno_3d = self.fig_diseno.add_subplot(111, projection="3d")
        self.canvas_diseno = FigureCanvasTkAgg(self.fig_diseno, master=frame_der)
        self.canvas_diseno.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(frame_der)
        toolbar_frame.pack(fill="x")
        self.toolbar_diseno = NavigationToolbar2Tk(self.canvas_diseno, toolbar_frame)
        self.toolbar_diseno.update()

    def on_double_click_tree(self, event):
        """Inicia la edición en línea de la respuesta de una corrida al hacer doble clic sobre una fila de la tabla. Identifica la fila y columna seleccionadas, coloca una entrada sobre la celda y valida el valor antes de actualizar la matriz interna y la vista. Solo permite editar la columna de respuesta."""
        region = self.tree_diseno.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.tree_diseno.identify_column(event.x)
        if column != "#5":
            return

        row_id = self.tree_diseno.identify_row(event.y)
        if not row_id:
            return

        x, y, w, h = self.tree_diseno.bbox(row_id, column)
        valores_actuales = self.tree_diseno.item(row_id, "values")
        val_prev = valores_actuales[4]

        entry_edit = ttk.Entry(self.tree_diseno, width=w)
        entry_edit.place(x=x, y=y, width=w, height=h)
        entry_edit.insert(0, val_prev)
        entry_edit.focus()

        def guardar_edicion(evt=None):
            """Rutina auxiliar interna de `guardar_edicion`. Procesa los datos y actualiza el estado de la interfaz según el contexto de la llamada."""
            nuevo_val = entry_edit.get().strip().replace(",", ".")
            entry_edit.destroy()
            if nuevo_val:
                try:
                    float(nuevo_val)
                    idx_fila = self.tree_diseno.index(row_id)
                    fila_act = list(self.matriz_generada[idx_fila])
                    fila_act[4] = nuevo_val
                    self.matriz_generada[idx_fila] = fila_act
                    self.tree_diseno.item(row_id, values=fila_act)
                    self.btn_enviar_directo.config(state="normal")
                except ValueError:
                    messagebox.showerror("Error", "Ingresa un número válido.")

        entry_edit.bind("<Return>", guardar_edicion)
        entry_edit.bind("<FocusOut>", lambda e: entry_edit.destroy())

    def generar_matriz_diseno(self):
        """Genera el diseño Doehlert canónico de tres factores a partir de los límites físicos introducidos. Calcula centro y escala física de cada factor usando los semiejes y ALPHA, añade las réplicas solicitadas en el centro y conserva ambas representaciones: factores reales y factores codificados. Rechaza límites inválidos y cantidades de réplicas fuera de 1–10."""
        factores = []
        for i, entry in enumerate(self.entries):
            nombre = entry["nombre"].get().strip() or f"X{i+1}"
            try:
                vmin = float(entry["min"].get().replace(",", "."))
                vmax = float(entry["max"].get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Error", f"Valores inválidos en {nombre}")
                return
            if vmin >= vmax:
                messagebox.showerror("Error", f"Mínimo >= Máximo en {nombre}")
                return
            factores.append({"nombre": nombre, "centro": (vmax+vmin)/2.0,
                             "delta": (vmax-vmin)/2.0, "alpha": ALPHA[i], "min": vmin, "max": vmax})
        try:
            n_cent = int(self.spin_replicas.get())
            if not 1 <= n_cent <= 10: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Las réplicas del centro deben ser un entero entre 1 y 10.")
            return
        self.factores_activos = factores
        self.factor_centers = np.asarray([f["centro"] for f in factores], dtype=float)
        self.factor_scales = np.asarray([f["delta"]/f["alpha"] for f in factores], dtype=float)
        corridas = list(DOEHLERT_BASE) + [(0.0, 0.0, 0.0)]*(n_cent-1)
        self.matriz_generada = []
        for num, coded in enumerate(corridas, start=1):
            real = self._real_from_coded(coded)
            self.matriz_generada.append([num, *[round(float(v), 6) for v in real], "", *[round(float(v), 6) for v in coded]])
        self.headers_diseno = ["Corrida", factores[0]["nombre"], factores[1]["nombre"], factores[2]["nombre"], "Respuesta_Y", "x1_cod", "x2_cod", "x3_cod"]
        self._refresh_design_table()
        self.log_event(f"Diseño Doehlert generado: {len(self.matriz_generada)} corridas; réplicas centro={n_cent}.")

    def simular_datos_y_enviar(self):
        """Genera valores de respuesta realistas con curvatura y ruido gaussiano."""
        if not self.matriz_generada:
            return

        rng = np.random.RandomState(42)
        nueva_matriz = []
        for fila in self.matriz_generada:
            c1 = float(fila[5])
            c2 = float(fila[6])
            c3 = float(fila[7])
            y_sim = 85.0 - 12.0*(c1 - 0.2)**2 - 15.0*(c2 + 0.1)**2 - 9.0*(c3 - 0.3)**2 + 4.0*c1*c2 + rng.normal(0, 0.8)
            fila_copia = list(fila)
            fila_copia[4] = f"{y_sim:.2f}"
            nueva_matriz.append(fila_copia)

        self.matriz_generada = nueva_matriz
        self._refresh_design_table()
        self.log_event("Se generaron respuestas sintéticas con semilla 42 (no son datos experimentales).")

        self.enviar_datos_a_analisis()
        messagebox.showinfo("Simulación Exitosa", "Se generaron respuestas simuladas con curvatura y ruido de laboratorio.\nYa están calculados los modelos en las Pestañas 2 y 3.")

    def enviar_datos_a_analisis(self):
        """Transfiere las respuestas ingresadas en la tabla al estado numérico del análisis. Comprueba que exista un diseño, convierte las columnas a arreglos, prepara nombres, límites y coordenadas codificadas, ejecuta el ajuste y actualiza los gráficos y reportes. Informa los errores de conversión o ajuste mediante la interfaz."""
        if not self.matriz_generada:
            messagebox.showwarning("Sin datos", "Primero genera o carga una matriz.")
            return
        try:
            X_vals, y_vals, cod_vals = [], [], []
            for idx, row in enumerate(self.matriz_generada, start=1):
                if len(row) < 8:
                    raise ValueError(f"La corrida {idx} no contiene las 8 columnas esperadas.")
                y_text = str(row[4]).strip().replace(",", ".")
                if not y_text:
                    raise ValueError(f"La corrida {idx} no tiene Respuesta_Y.")
                try:
                    values = [float(str(v).replace(",", ".")) for v in row[1:4]]
                    coded = [float(str(v).replace(",", ".")) for v in row[5:8]]
                    y_num = float(y_text)
                except ValueError:
                    raise ValueError(f"La corrida {idx} contiene un valor no numérico (factores, códigos o Respuesta_Y).")
                y_vals.append(y_num); X_vals.append(values); cod_vals.append(coded)
            self.nombres_factores = self.headers_diseno[1:4]
            self.X_exp = np.asarray(X_vals, dtype=float)
            self.y_exp = np.asarray(y_vals, dtype=float)
            self.cod_exp = np.asarray(cod_vals, dtype=float)
            self._configure_factor_transform(self.X_exp, self.cod_exp)
            self.limites_factores = [(float(np.min(self.X_exp[:, i])), float(np.max(self.X_exp[:, i]))) for i in range(3)]
            self.ejecutar_ajuste_completo()
            self.cbo_eje_x["values"] = self.nombres_factores
            self.cbo_eje_y["values"] = self.nombres_factores
            self.cbo_eje_x.current(0); self.cbo_eje_y.current(1)
            self.lbl_estado.config(text=f"Activo: {len(y_vals)} corridas procesadas.", foreground="#006600")
            self.actualizar_graficos_superficie(); self.actualizar_graficos_diagnostico()
            self.notebook.select(self.tab_analisis)
            self.log_event(f"Datos enviados al análisis: N={len(y_vals)}.")
        except ValueError as err:       # datos vacíos / no numéricos / diseño singular (ModelError)
            self.last_error = traceback.format_exc(); self.log_event(f"Datos no válidos al enviar: {err}")
            messagebox.showwarning("Datos no válidos", str(err))
        except Exception as err:
            self.last_error = traceback.format_exc(); self.log_event(f"Error enviando datos: {err}")
            messagebox.showerror("Error al ajustar", str(err))

    def exportar_csv_plantilla(self):
        """Permite guardar la matriz Doehlert actual como CSV para completar las respuestas experimentalmente. Incluye encabezados descriptivos, factores en unidades físicas, columna de respuesta vacía y coordenadas codificadas; usa UTF-8 y una estructura compatible con la rutina de importación."""
        if not self.matriz_generada:
            return
        ruta = self.dialogo_guardar("Planilla_Diseno", ".csv",
            [("Archivo CSV (decimal con punto)", "*.csv")], "Guardar planilla del diseño")
        if not ruta:
            return
        try:
            with open(ruta, mode="w", newline="", encoding="utf-8") as f:
                # Directiva compatible con Excel regional: separador de campos coma.
                f.write("sep=,\n")
                writer = csv.writer(f, delimiter=",")
                writer.writerow(self.headers_diseno)
                for fila in self.matriz_generada:
                    fila_csv = [
                        f"{float(valor):.8f}" if isinstance(valor, (float, np.floating)) else str(valor)
                        for valor in fila
                    ]
                    writer.writerow(fila_csv)
            self.log_event(f"CSV exportado con punto decimal y 8 decimales: {ruta}")
            messagebox.showinfo("Exportado", f"Archivo guardado con punto decimal:\n{ruta}")
        except Exception as e:
            self.last_error = traceback.format_exc()
            self.log_event(f"Error exportando CSV: {e}")
            messagebox.showerror("Error", f"No se pudo guardar: {e}")

    # ==============================================================
    # PESTAÑA 2: ANÁLISIS Y SUPERFICIES
    # ==============================================================
    def construir_tab_analisis(self):
        """Construye la pestaña de superficies de respuesta y optimización. Configura selectores de ejes, controles del factor mantenido fijo, paneles 3D/contorno, reporte de regresión y acciones de visualización. No ajusta un modelo por sí misma."""
        frame_controles = ttk.LabelFrame(self.tab_analisis, text=" Datos y Ajuste ", padding=8)
        frame_controles.pack(fill="x", padx=12, pady=6)

        ttk.Button(frame_controles, text="Cargar CSV con Respuestas...", command=self.cargar_csv_respuestas).pack(side="left", padx=6)
        self.lbl_estado = ttk.Label(frame_controles, text="Ningún dato procesado aún.", foreground="#555555")
        self.lbl_estado.pack(side="left", padx=10)

        frame_ejes = ttk.Frame(frame_controles)
        frame_ejes.pack(side="right", padx=10)

        ttk.Label(frame_ejes, text="Eje X:").pack(side="left", padx=2)
        self.cbo_eje_x = ttk.Combobox(frame_ejes, state="readonly", width=11)
        self.cbo_eje_x.pack(side="left", padx=4)

        ttk.Label(frame_ejes, text="Eje Y:").pack(side="left", padx=2)
        self.cbo_eje_y = ttk.Combobox(frame_ejes, state="readonly", width=11)
        self.cbo_eje_y.pack(side="left", padx=4)

        ttk.Button(frame_ejes, text="Actualizar Superficie", command=self.actualizar_graficos_superficie).pack(side="left", padx=4)
        ttk.Button(frame_ejes, text="Corte 2D con Bandas...", command=self.abrir_ventana_bandas_2d).pack(side="left", padx=4)
        ttk.Button(frame_ejes, text="Guardar Gráfico...", command=lambda: self.exportar_grafico_individual(self.fig_analisis, "Superficie_Doehlert")).pack(side="left", padx=4)
        ttk.Button(frame_ejes, text="Exportar Informe PDF...", command=self.exportar_informe_completo_pdf).pack(side="left", padx=6)

        paned = ttk.PanedWindow(self.tab_analisis, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        frame_reporte = ttk.LabelFrame(paned, text=" Resumen Estadístico y Óptimo ", padding=6)
        paned.add(frame_reporte, weight=1)

        self.txt_reporte = tk.Text(frame_reporte, wrap="word", font=self.fuente_mono, width=60)
        sb_txt = ttk.Scrollbar(frame_reporte, orient="vertical", command=self.txt_reporte.yview)
        self.txt_reporte.configure(yscrollcommand=sb_txt.set)
        sb_txt.pack(side="right", fill="y")
        self.txt_reporte.pack(fill="both", expand=True)

        frame_graficos = ttk.LabelFrame(paned, text=" Superficie de Respuesta 3D y Contornos 2D ", padding=6)
        paned.add(frame_graficos, weight=3)

        self.fig_analisis = Figure(figsize=(7, 5), dpi=95)
        self.ax_3d = self.fig_analisis.add_subplot(121, projection="3d")
        self.ax_contour = self.fig_analisis.add_subplot(122)

        self.canvas_analisis = FigureCanvasTkAgg(self.fig_analisis, master=frame_graficos)
        self.canvas_analisis.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(frame_graficos)
        toolbar_frame.pack(fill="x")
        self.toolbar_analisis = NavigationToolbar2Tk(self.canvas_analisis, toolbar_frame)
        self.toolbar_analisis.update()

    # ==============================================================
    # BARRA DE HERRAMIENTAS / MENÚ DE EXPORTACIÓN
    # ==============================================================
    def construir_menu_y_barra(self):
        """Menú simple Archivo/Ayuda y accesos directos a las tareas habituales."""
        menubar = tk.Menu(self)
        menu_archivo = tk.Menu(menubar, tearoff=0)
        menu_archivo.add_command(label="Importar datos experimentales (CSV)…", command=self.importar_archivo_csv_desde_menu)
        menu_archivo.add_command(label="Importar dataset para auditoría (CSV)…", command=self.importar_dataset_auditoria)
        menu_archivo.add_command(label="Importar auditoría anterior (JSON)…", command=self.importar_reporte_auditoria)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Abrir carpeta de ejemplos", command=lambda: self.abrir_carpeta("Examples"))
        menu_archivo.add_command(label="Abrir carpeta de exportaciones", command=lambda: self.abrir_carpeta("Exports"))
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.destroy)
        menubar.add_cascade(label="Archivo", menu=menu_archivo)

        menu_exp = tk.Menu(menubar, tearoff=0)
        menu_exp.add_command(label="Planilla del diseño (CSV)…", command=self.exportar_csv_plantilla)
        menu_exp.add_command(label="Reporte ejecutivo (MD/TXT)…", command=self.exportar_reporte_texto)
        menu_exp.add_command(label="Tablas (Excel/CSV)…", command=self.exportar_tablas)
        menu_exp.add_command(label="Gráfico actual (PNG/PDF/SVG)…", command=self.guardar_grafico_actual)
        menu_exp.add_separator()
        menu_exp.add_command(label="Informe completo (PDF)…", command=self.exportar_informe_completo_pdf)
        menubar.add_cascade(label="Exportar", menu=menu_exp)

        menu_language = tk.Menu(menubar, tearoff=0)
        menu_language.add_command(label="English", command=lambda: self.cambiar_idioma("en"))
        menu_language.add_command(label="Español", command=lambda: self.cambiar_idioma("es"))
        menubar.add_cascade(label="Language", menu=menu_language)

        menu_ayuda = tk.Menu(menubar, tearoff=0)
        menu_ayuda.add_command(label="Guía rápida de uso", command=self.mostrar_ayuda_rapida)
        menubar.add_cascade(label="Ayuda", menu=menu_ayuda)
        self.config(menu=menubar)

        barra = ttk.Frame(self, padding=(10, 5))
        barra.pack(fill="x")
        ttk.Button(barra, text="Importar datos (CSV)", command=self.importar_archivo_csv_desde_menu).pack(side="left", padx=3)
        ttk.Button(barra, text="Exportar informe PDF", command=self.exportar_informe_completo_pdf).pack(side="left", padx=3)
        ttk.Button(barra, text="Guía rápida", command=self.mostrar_ayuda_rapida).pack(side="left", padx=3)
        ttk.Label(barra, text="Los archivos generados se organizan en Exports por fecha.", foreground="#555555").pack(side="right", padx=8)
        ttk.Separator(self, orient="horizontal").pack(fill="x")

    def cambiar_idioma(self, language):
        """Persist the selected interface language and restart the application."""
        if language == get_language():
            return
        set_language(language)
        language_name = "English" if language == "en" else "Español"
        restart = messagebox.askyesno(
            "Language / Idioma",
            f"The interface language is now set to {language_name}.\n"
            "Restart Doehlert Suite now to apply the change? / "
            "¿Reiniciar ahora para aplicar el cambio?"
        )
        if restart:
            os.execl(sys.executable, sys.executable, os.path.abspath(__file__), *sys.argv[1:])

    def abrir_carpeta(self, nombre):
        ruta = os.path.join(self.carpeta_aplicacion(), nombre)
        os.makedirs(ruta, exist_ok=True)
        try:
            if sys.platform.startswith("linux"):
                import subprocess; subprocess.Popen(["xdg-open", ruta])
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", ruta])
            elif os.name == "nt":
                os.startfile(ruta)
        except Exception as exc:
            messagebox.showwarning("Abrir carpeta", f"No se pudo abrir automáticamente. La carpeta está en:\n{ruta}\n\n{exc}")

    # ==============================================================
    # PESTAÑA 3: ANOVA, LACK-OF-FIT Y COEFICIENTES
    # ==============================================================
    def construir_tab_modelo(self):
        """Construye la pestaña de inferencia del modelo: tabla ANOVA con descomposición del error residual (regresión, residuo, falta de ajuste, error puro, total), tabla de los diez coeficientes con error estándar, t, p-valor bilateral e IC 95 %, y un panel monoespaciado con la prueba de falta de ajuste. No ejecuta cálculos."""
        paned = ttk.PanedWindow(self.tab_modelo, orient="vertical")
        paned.pack(fill="both", expand=True, padx=12, pady=10)

        # --- Tabla ANOVA ---
        fr_anova = ttk.LabelFrame(paned, text=" Tabla ANOVA con descomposición del error residual ", padding=6)
        paned.add(fr_anova, weight=1)
        cols_a = (("fuente", "Fuente", 210, "w"), ("gl", "GL", 60, "center"),
                  ("ss", "Suma de cuadrados", 150, "e"), ("ms", "Cuadrado medio", 150, "e"),
                  ("f", "F", 110, "e"), ("p", "p-valor", 110, "e"))
        self.tree_anova = ttk.Treeview(fr_anova, columns=[c[0] for c in cols_a], show="headings", height=6, selectmode="browse")
        for cid, txt, w, anc in cols_a:
            self.tree_anova.heading(cid, text=txt)
            self.tree_anova.column(cid, width=w, anchor=anc)
        self.tree_anova.tag_configure("sub", foreground="#444444")
        self.tree_anova.pack(fill="both", expand=True)

        # --- Tabla de coeficientes ---
        fr_coef = ttk.LabelFrame(paned, text=" Coeficientes con inferencia (error estándar, t, p-valor bilateral, IC 95%) ", padding=6)
        paned.add(fr_coef, weight=2)
        cols_c = (("term", "Término", 190, "w"), ("b", "Coeficiente", 105, "e"), ("se", "Error estándar", 105, "e"),
                  ("t", "t", 80, "e"), ("p", "p-valor", 90, "e"), ("sig", "Sig.", 50, "center"),
                  ("lo", "IC 95% inferior", 115, "e"), ("hi", "IC 95% superior", 115, "e"))
        self.tree_coef = ttk.Treeview(fr_coef, columns=[c[0] for c in cols_c], show="headings", height=10, selectmode="browse")
        for cid, txt, w, anc in cols_c:
            self.tree_coef.heading(cid, text=txt)
            self.tree_coef.column(cid, width=w, anchor=anc)
        sb_c = ttk.Scrollbar(fr_coef, orient="vertical", command=self.tree_coef.yview)
        self.tree_coef.configure(yscrollcommand=sb_c.set)
        sb_c.pack(side="right", fill="y")
        self.tree_coef.pack(fill="both", expand=True)
        self.tree_coef.tag_configure("sig", foreground="#0b6623")
        self.tree_coef.tag_configure("ns", foreground="#7a7a7a")
        ttk.Label(fr_coef, text="Sig.:  *** p<0.001   ** p<0.01   * p<0.05   . p<0.1   ns p≥0.1   |   verde = significativo (p<0.05)",
                  foreground="#555555").pack(anchor="w", pady=(4, 0))

        # --- Falta de ajuste (texto monoespaciado) ---
        fr_lof = ttk.LabelFrame(paned, text=" Prueba de Falta de Ajuste (Lack of Fit) y error puro ", padding=6)
        paned.add(fr_lof, weight=1)
        self.txt_diag = tk.Text(fr_lof, wrap="word", font=self.fuente_mono, height=10)
        sb_diag = ttk.Scrollbar(fr_lof, orient="vertical", command=self.txt_diag.yview)
        self.txt_diag.configure(yscrollcommand=sb_diag.set)
        sb_diag.pack(side="right", fill="y")
        self.txt_diag.pack(fill="both", expand=True)

    # ==============================================================
    # PESTAÑA 4: DIAGNÓSTICO DE RESIDUOS
    # ==============================================================
    def construir_tab_diagnostico(self):
        """Construye la pestaña de diagnóstico de residuos: tabla corrida a corrida (valores reales, Y observada, Ŷ, residuo, leverage y residuo studentizado interno) y tres gráficos (residuos vs. ajustados, Q-Q normal y residuos vs. orden de ejecución). Al seleccionar una fila se resalta la corrida en los tres gráficos."""
        paned = ttk.PanedWindow(self.tab_diagnostico, orient="vertical")
        paned.pack(fill="both", expand=True, padx=12, pady=10)

        fr_tabla = ttk.LabelFrame(paned, text=" Diagnóstico corrida a corrida (selecciona una fila para resaltarla en los gráficos) ", padding=6)
        paned.add(fr_tabla, weight=2)
        self.lbl_leyenda_res = ttk.Label(fr_tabla, text="Sin modelo ajustado.", foreground="#555555")
        self.lbl_leyenda_res.pack(side="bottom", anchor="w", pady=(4, 0))
        self.tree_res = ttk.Treeview(fr_tabla, show="headings", selectmode="browse", height=8)
        sy = ttk.Scrollbar(fr_tabla, orient="vertical", command=self.tree_res.yview)
        sx = ttk.Scrollbar(fr_tabla, orient="horizontal", command=self.tree_res.xview)
        self.tree_res.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.tree_res.pack(fill="both", expand=True)
        self.tree_res.tag_configure("atipico", foreground="#b00020")
        self.tree_res.bind("<<TreeviewSelect>>", self._resaltar_corrida)

        fr_plots = ttk.LabelFrame(paned, text=" Validación de Supuestos ", padding=6)
        paned.add(fr_plots, weight=3)

        frame_top_diag = ttk.Frame(fr_plots)
        frame_top_diag.pack(fill="x", pady=(0, 4))
        ttk.Button(frame_top_diag, text="GENERAR LOG DE DEPURACIÓN...", command=self.exportar_log_debug).pack(side="left", padx=4)
        ttk.Button(frame_top_diag, text="Guardar Gráficos de Residuos...",
                   command=lambda: self.exportar_grafico_individual(self.fig_diag, "Diagnostico_Residuos")).pack(side="right", padx=4)

        self.fig_diag = Figure(figsize=(10, 3.7), dpi=95)
        self.ax_res_fit = self.fig_diag.add_subplot(131)
        self.ax_qq = self.fig_diag.add_subplot(132)
        self.ax_res_ord = self.fig_diag.add_subplot(133)

        self.canvas_diag = FigureCanvasTkAgg(self.fig_diag, master=fr_plots)
        self.canvas_diag.get_tk_widget().pack(fill="both", expand=True)

        toolbar_diag_frame = ttk.Frame(fr_plots)
        toolbar_diag_frame.pack(fill="x")
        self.toolbar_diag = NavigationToolbar2Tk(self.canvas_diag, toolbar_diag_frame)
        self.toolbar_diag.update()

    def _design_matrix(self, coded):
        return eng.design_matrix(coded)

    def _coded_from_real(self, values):
        """Transforma factores en unidades físicas a coordenadas codificadas mediante z=(x-centro)/escala. La transformación se aplica por columna y admite vectores o matrices NumPy compatibles. Los centros y escalas deben haberse configurado previamente y las escalas no deben ser cero."""
        values = np.asarray(values, dtype=float)
        return (values - self.factor_centers) / self.factor_scales

    def _real_from_coded(self, values):
        """Transforma coordenadas codificadas a unidades físicas mediante x=centro+escala*z. Es la transformación inversa de _coded_from_real y conserva la forma del arreglo de entrada."""
        values = np.asarray(values, dtype=float)
        return self.factor_centers + values * self.factor_scales

    def _configure_factor_transform(self, real_values, coded_values):
        self.factor_centers, self.factor_scales = eng.configure_factor_transform(real_values, coded_values)

    def _refresh_design_table(self):
        """Sincroniza el Treeview con los encabezados y filas de la matriz de diseño actualmente almacenada. Limpia filas anteriores, inserta las nuevas, habilita las acciones relacionadas y solicita la actualización del gráfico 3D."""
        self.tree_diseno["columns"] = self.headers_diseno
        for c in self.headers_diseno:
            self.tree_diseno.heading(c, text=c)
            self.tree_diseno.column(c, anchor="center", width=95)
        for item in self.tree_diseno.get_children():
            self.tree_diseno.delete(item)
        for row in self.matriz_generada:
            self.tree_diseno.insert("", "end", values=row)
        self.btn_exportar.config(state="normal")
        self.btn_simular.config(state="normal")
        self.btn_enviar_directo.config(state="normal")
        self._refresh_design_plot()

    def _refresh_design_plot(self):
        """Redibuja la geometría del diseño en unidades físicas. Distingue puntos periféricos y réplicas centrales, etiqueta las corridas y actualiza ejes, título y leyenda. Si todavía no hay matriz generada, termina sin modificar el gráfico."""
        if not self.matriz_generada:
            return
        real = np.asarray([[float(r[1]), float(r[2]), float(r[3])] for r in self.matriz_generada])
        self.ax_diseno_3d.clear()
        self.ax_diseno_3d.scatter(real[1:13, 0], real[1:13, 1], real[1:13, 2], color="#1f77b4", s=55, edgecolors="black", label="Puntos periféricos")
        centers = np.linalg.norm(np.asarray([[float(r[5]), float(r[6]), float(r[7])] for r in self.matriz_generada]), axis=1) < 1e-6
        self.ax_diseno_3d.scatter(real[centers, 0], real[centers, 1], real[centers, 2], color="#d62728", s=85, marker="*", edgecolors="black", label="Réplicas del centro")
        for i, point in enumerate(real):
            self.ax_diseno_3d.text(*point, f" #{i+1}", fontsize=8)
        labels = self.headers_diseno[1:4]
        self.ax_diseno_3d.set_xlabel(labels[0], labelpad=8)
        self.ax_diseno_3d.set_ylabel(labels[1], labelpad=8)
        self.ax_diseno_3d.set_zlabel(labels[2], labelpad=8)
        self.ax_diseno_3d.set_title("Distribución Esférica de Doehlert", fontsize=9)
        self.ax_diseno_3d.legend(loc="upper left", fontsize=8)
        self.fig_diseno.tight_layout()
        self.canvas_diseno.draw()

    def exportar_log_debug(self):
        """Abre un diálogo para exportar un informe de diagnóstico reproducible. Incluye hash SHA-256 del código fuente, versiones del entorno, configuración de factores, matrices y vectores estadísticos, reportes visibles, último traceback y eventos recientes. La exportación no altera el análisis."""
        ruta = self.dialogo_guardar("Log_Depuracion", ".txt",
            [("Log de depuración", "*.txt"), ("Todos los archivos", "*.*")], "Guardar log de depuración")
        if not ruta:
            return
        try:
            def array_text(name):
                """Rutina auxiliar interna de `array_text`. Procesa los datos y actualiza el estado de la interfaz según el contexto de la llamada."""
                value = getattr(self, name, None)
                return f"{name}:\n{np.array2string(np.asarray(value), precision=10, threshold=10000)}" if value is not None else f"{name}: None"
            packages = {}
            for package in ("numpy", "matplotlib", "scipy", "pandas"):
                try: packages[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError: packages[package] = "no instalado"
            source_path = os.path.abspath(__file__)
            try:
                with open(source_path, "rb") as source_file:
                    source_hash = hashlib.sha256(source_file.read()).hexdigest()
            except OSError:
                source_hash = "no disponible"
            try:
                with open(os.path.abspath(eng.__file__), "rb") as engine_file:
                    engine_hash = hashlib.sha256(engine_file.read()).hexdigest()
            except OSError:
                engine_hash = "no disponible"
            gui_config = []
            for index, entry in enumerate(getattr(self, "entries", []), start=1):
                gui_config.append({"factor": index, "nombre": entry["nombre"].get(),
                                   "minimo": entry["min"].get(), "maximo": entry["max"].get()})
            lines = [
                "DOEHLERT SUITE - LOG DE DEPURACIÓN", "="*72,
                f"Fecha: {datetime.datetime.now().astimezone().isoformat()}",
                f"Código fuente: {source_path}", f"SHA256 del código: {source_hash}",
                f"Motor: {os.path.abspath(eng.__file__)}", f"SHA256 del motor: {engine_hash}",
                f"Backend Matplotlib: {matplotlib.get_backend()}",
                f"Configuración visible: {json.dumps(gui_config, ensure_ascii=False)}",
                f"Réplicas centro configuradas: {self.spin_replicas.get()}",
                f"Python: {sys.version}", f"Ejecutable: {sys.executable}",
                f"Plataforma: {platform.platform()}", f"Sistema: {platform.system()} {platform.release()}",
                f"Directorio de trabajo: {os.getcwd()}", f"Archivo CSV cargado: {self.last_csv_path}",
                f"Versiones: {json.dumps(packages, ensure_ascii=False)}", "",
                "ESTADO DE LA APLICACIÓN", "-"*72,
                f"N corridas: {len(self.y_exp) if self.y_exp is not None else 0}",
                f"Nombres de factores: {self.nombres_factores}",
                f"Límites físicos: {self.limites_factores}",
                f"Centros físicos: {self.factor_centers}", f"Escalas físicas por unidad codificada: {self.factor_scales}",
                f"Encabezados: {self.headers_diseno}",
                f"Matriz visual: {json.dumps(self.matriz_generada, ensure_ascii=False, default=str)}", "",
                "ARRAYS NUMÉRICOS", "-"*72,
            ]
            for name in ("X_exp", "cod_exp", "y_exp", "betas", "cov_beta", "y_pred", "residuos", "residuos_std", "x_opt_cod", "x_opt", "ic_coords", "leverage"):
                lines.append(array_text(name))
            lines += ["", "REPORTE DE REGRESIÓN", "-"*72, self.txt_reporte.get("1.0", "end-1c"),
                      "", "REPORTE ANOVA / DIAGNÓSTICO", "-"*72, self.txt_diag.get("1.0", "end-1c"),
                      "", "ÚLTIMO ERROR", "-"*72, self.last_error or "Sin excepción registrada.",
                      "", "EVENTOS RECIENTES", "-"*72, "\n".join(self.debug_events)]
            with open(ruta, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("LOG generado", f"Log guardado correctamente en:\n{ruta}")
            self.log_event(f"Log de depuración exportado: {ruta}")
        except Exception as err:
            self.last_error = traceback.format_exc()
            messagebox.showerror("Error al generar LOG", str(err))

    # ==============================================================
    # PESTAÑA 5: AUDITORÍA, DOMINIO Y MÚLTIPLES RESPUESTAS
    # ==============================================================
    def construir_tab_auditoria(self):
        import queue
        self._audit_queue = queue.Queue()
        self._audit_obj = None
        self._audit_busy = False
        self._audit_previous = None
        top = ttk.Frame(self.tab_auditoria, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Comprueba la calidad del análisis y compara resultados anteriores",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 3))
        ttk.Label(top, text="Primero importa un CSV o ejecuta la auditoría completa. Los detalles aparecen en las pestañas inferiores.",
                  foreground="#555555", wraplength=1100).pack(anchor="w", pady=(0, 6))
        bar = ttk.Frame(top); bar.pack(fill="x")
        self.audit_quick_var = tk.BooleanVar(value=False)
        ttk.Button(bar, text="▶ Ejecutar auditoría completa", command=self.ejecutar_auditoria_integral).pack(side="left", padx=2)
        ttk.Button(bar, text="Importar dataset CSV…", command=self.importar_dataset_auditoria).pack(side="left", padx=2)
        ttk.Button(bar, text="Importar reporte JSON…", command=self.importar_reporte_auditoria).pack(side="left", padx=2)
        ttk.Button(bar, text="Exportar paquete completo…", command=self.exportar_paquete_auditoria).pack(side="left", padx=2)
        ttk.Checkbutton(bar, text="Modo rápido", variable=self.audit_quick_var).pack(side="left", padx=8)
        self.audit_status_var = tk.StringVar(value="Listo. Ejecute la auditoría para iniciar las 19 etapas.")
        ttk.Label(top, textvariable=self.audit_status_var).pack(anchor="w", pady=(5, 0))
        self.audit_progress = ttk.Progressbar(top, mode="determinate", maximum=100)
        self.audit_progress.pack(fill="x", pady=3)
        panes = ttk.PanedWindow(self.tab_auditoria, orient="horizontal"); panes.pack(fill="both", expand=True, padx=8, pady=5)
        left = ttk.LabelFrame(panes, text="Hallazgos · PASS / WARN / FAIL / SKIP", padding=5)
        right = ttk.Frame(panes)
        panes.add(left, weight=2); panes.add(right, weight=5)
        self.audit_tree = ttk.Treeview(left, columns=("status","section"), show="tree headings", height=18)
        self.audit_tree.heading("#0", text="Prueba"); self.audit_tree.heading("status", text="Estado")
        self.audit_tree.heading("section", text="Sección")
        self.audit_tree.column("#0", width=250); self.audit_tree.column("status", width=75, anchor="center")
        self.audit_tree.column("section", width=170)
        sy=ttk.Scrollbar(left,orient="vertical",command=self.audit_tree.yview); self.audit_tree.configure(yscrollcommand=sy.set)
        self.audit_tree.pack(side="left",fill="both",expand=True); sy.pack(side="right",fill="y")
        self.audit_tree.tag_configure("PASS", foreground="#16803c"); self.audit_tree.tag_configure("WARN", foreground="#a66a00")
        self.audit_tree.tag_configure("FAIL", foreground="#c62828"); self.audit_tree.bind("<<TreeviewSelect>>",self.mostrar_hallazgo_auditoria)
        tabs=ttk.Notebook(right); tabs.pack(fill="both",expand=True)
        detail_frame=ttk.Frame(tabs); graph_frame=ttk.Frame(tabs); compare_frame=ttk.Frame(tabs); raw_frame=ttk.Frame(tabs)
        tabs.add(detail_frame,text="Detalle y métricas"); tabs.add(graph_frame,text="Gráficas"); tabs.add(compare_frame,text="Regresiones"); tabs.add(raw_frame,text="Informe completo")
        self.audit_detail=tk.Text(detail_frame,wrap="word",font=self.fuente_mono); self.audit_detail.pack(fill="both",expand=True)
        self.audit_graph_frame=graph_frame; self.audit_canvas=None; self.audit_toolbar=None
        graph_split=ttk.PanedWindow(graph_frame,orient="horizontal"); graph_split.pack(fill="both",expand=True)
        self.audit_fig_tree=ttk.Treeview(graph_split,columns=("caption",),show="tree headings")
        self.audit_fig_tree.heading("#0",text="Gráfica"); self.audit_fig_tree.heading("caption",text="Descripción")
        self.audit_fig_tree.column("#0",width=150,minwidth=100,stretch=True); self.audit_fig_tree.column("caption",width=180,minwidth=120,stretch=True)
        self.audit_fig_tree.bind("<<TreeviewSelect>>",self.seleccionar_grafica_auditoria)
        self.audit_fig_host=ttk.Frame(graph_split); graph_split.add(self.audit_fig_tree,weight=1); graph_split.add(self.audit_fig_host,weight=4)
        self.audit_compare_tree=ttk.Treeview(compare_frame,columns=("prev","curr","change"),show="tree headings")
        for c,t in (("#0","Prueba"),("prev","Anterior"),("curr","Actual"),("change","Cambio")):
            self.audit_compare_tree.heading(c,text=t)
        self.audit_compare_tree.pack(fill="both",expand=True)
        self.audit_raw=tk.Text(raw_frame,wrap="none",font=self.fuente_mono); self.audit_raw.pack(fill="both",expand=True)
        self._audit_tabs=tabs
        self._audit_append("El auditor completo se ejecuta en un hilo separado. Los resultados, detalles, métricas, comparaciones y gráficas se muestran aquí; no se abre una segunda ventana.")

    def _audit_append(self, text):
        self.audit_detail.insert("end", tr(str(text))+"\n"); self.audit_detail.see("end")

    def _audit_module(self):
        import importlib.util
        path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"auditor","audit_doehlert.py")
        if not os.path.isfile(path): raise FileNotFoundError("No se encuentra auditor/audit_doehlert.py")
        spec=importlib.util.spec_from_file_location("doehlert_integrated_auditor",path)
        mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod
        spec.loader.exec_module(mod)
        return mod

    def ejecutar_auditoria_integral(self):
        import threading
        if self._audit_busy: return
        try:
            self._audit_mod=self._audit_module()
            self._audit_busy=True; self.audit_progress["value"]=0
            self.audit_status_var.set("Preparando auditoría…")
            target=os.path.abspath(__file__)
            data=getattr(self,"_audit_import_data",None)
            report=getattr(self,"_audit_import_report",None)
            quick=bool(self.audit_quick_var.get())
            def worker():
                try:
                    def progress(msg,frac): self._audit_queue.put(("progress",msg,frac))
                    a=self._audit_mod.Audit(target,quick=quick,progress=progress,import_data=data,import_report=report)
                    a.run()
                    self._audit_queue.put(("done",a,None))
                except Exception:
                    self._audit_queue.put(("error",traceback.format_exc(),None))
            threading.Thread(target=worker,daemon=True,name="DoehlertAudit").start()
            self.after(100,self._poll_auditoria)
        except Exception as e:
            self._audit_busy=False; messagebox.showerror("Auditoría",f"No se pudo iniciar:\n{e}")

    def _poll_auditoria(self):
        try:
            while True:
                kind,a,b=self._audit_queue.get_nowait()
                if kind=="progress":
                    self.audit_status_var.set(str(a)); self.audit_progress["value"]=float(b)*100
                elif kind=="error":
                    self._audit_busy=False; self.audit_status_var.set("Error durante la auditoría")
                    self._audit_append(a); messagebox.showerror("Auditoría",str(a))
                elif kind=="done":
                    self._audit_busy=False; self._audit_obj=a; self.audit_progress["value"]=100
                    c=a.counts()
                    self.audit_status_var.set(f"Finalizada · PASS {c['PASS']} · WARN {c['WARN']} · FAIL {c['FAIL']} · SKIP {c['SKIP']} · {a.meta.get('elapsed_s',0)} s")
                    self._render_auditoria()
        except Exception:
            pass
        if self._audit_busy: self.after(100,self._poll_auditoria)

    def _render_auditoria(self):
        a=self._audit_obj; self.audit_tree.delete(*self.audit_tree.get_children())
        for f in a.findings:
            self.audit_tree.insert("", "end", iid=str(f.id), text=f"{f.id} · {f.title}", values=(f.status,f.section),tags=(f.status,))
        self.audit_raw.delete("1.0","end")
        self.audit_raw.insert("end",tr(self._audit_mod.render_console(a,detail="all")))
        self.audit_fig_tree.delete(*self.audit_fig_tree.get_children())
        for fid,title,caption,fig in a.figures:
            self.audit_fig_tree.insert("", "end", iid=fid, text=title, values=(caption,))
        if a.figures:
            self.audit_fig_tree.selection_set(a.figures[0][0]); self.seleccionar_grafica_auditoria()
        self.audit_compare_tree.delete(*self.audit_compare_tree.get_children())
        if a.comparison:
            for r in a.comparison.get("rows",[]):
                self.audit_compare_tree.insert("", "end", text=f"{r['id']} · {r['title']}",
                    values=(r.get("prev") or "—",r.get("curr") or "—",r.get("change","")))
        self.audit_detail.delete("1.0","end")
        self._audit_append(f"AUDITORÍA COMPLETA\nArchivo: {a.meta['target']}\nSHA256: {a.meta['target_sha256']}\n")
        self._audit_append("Resumen: "+", ".join(f"{k}={v}" for k,v in a.counts().items()))
        if a.findings: self.audit_tree.selection_set(str(a.findings[0].id)); self.mostrar_hallazgo_auditoria()

    def mostrar_hallazgo_auditoria(self,event=None):
        if self._audit_obj is None:return
        sel=self.audit_tree.selection()
        if not sel:return
        fid=sel[0]; f=next((x for x in self._audit_obj.findings if x.id==fid),None)
        if f is None:return
        self.audit_detail.delete("1.0","end")
        self.audit_detail.insert("end",tr(f"{f.id} · {f.title}\nEstado: {f.status}\nSección: {f.section}\nResumen: {f.summary}\nTiempo: {f.elapsed:.3f} s\n\nDETALLES\n"+f.details+"\n\nMÉTRICAS\n"+json.dumps(f.metrics,ensure_ascii=False,indent=2,default=str)+"\n\nUbicación: "+f.where+"\n\nCorrección sugerida: "+f.fix))

    def seleccionar_grafica_auditoria(self,event=None):
        if self._audit_obj is None:return
        sel=self.audit_fig_tree.selection()
        if not sel:return
        item=next((x for x in self._audit_obj.figures if x[0]==sel[0]),None)
        if item:self._mostrar_figura_auditoria(item[3])

    def _mostrar_figura_auditoria(self,fig):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        if self.audit_canvas:
            self.audit_canvas.get_tk_widget().destroy()
        if self.audit_toolbar:self.audit_toolbar.destroy()
        self.audit_canvas=FigureCanvasTkAgg(fig,master=self.audit_fig_host)
        self.audit_canvas.draw(); self.audit_canvas.get_tk_widget().pack(fill="both",expand=True)
        self.audit_toolbar=NavigationToolbar2Tk(self.audit_canvas,self.audit_fig_host,pack_toolbar=False)
        self.audit_toolbar.update(); self.audit_toolbar.pack(fill="x")

    def importar_dataset_auditoria(self):
        p=filedialog.askopenfilename(initialdir=os.path.join(self.carpeta_aplicacion(), "Examples"),
            filetypes=[("Datos CSV", "*.csv"), ("Todos", "*.*")], title="Importar dataset para auditoría")
        if not p:return
        try:
            self._audit_mod=self._audit_module()
            ds=self._audit_mod.load_dataset_csv(p)
            self._audit_import_data=p
            self.audit_status_var.set(f"Dataset cargado: {len(ds['y'])} observaciones · ejecutar auditoría para analizarlo.")
            self._audit_append(f"Dataset importado: {p}\nN={len(ds['y'])}; notas={ds.get('notes',[])}")
        except Exception as e:messagebox.showerror("Importar dataset",str(e))

    def importar_reporte_auditoria(self):
        p=filedialog.askopenfilename(initialdir=self.carpeta_exportacion(),
            filetypes=[("Reportes de auditoría JSON", "*.json"), ("Todos", "*.*")], title="Importar auditoría anterior")
        if not p:return
        try:
            self._audit_mod=self._audit_module()
            self._audit_previous=self._audit_mod.load_report_json(p)
            self._audit_import_report=p
            self.audit_status_var.set("Reporte previo importado. La comparación se calculará en la próxima auditoría.")
            self._audit_append(f"Reporte anterior importado: {p}")
        except Exception as e:messagebox.showerror("Importar reporte",str(e))

    def exportar_paquete_auditoria(self):
        if self._audit_obj is None:
            messagebox.showwarning("Sin auditoría", "Primero ejecuta la auditoría completa."); return
        base = self.carpeta_exportacion()
        destino = os.path.join(base, self.nombre_exportacion("Auditoria_Completa", "").replace(".", "_"))
        n = 2
        while os.path.exists(destino):
            destino = os.path.join(base, self.nombre_exportacion(f"Auditoria_Completa_{n}", "").replace(".", "_"))
            n += 1
        try:
            os.makedirs(destino, exist_ok=True)
            res = self._audit_mod.export_all(self._audit_obj, destino, self._audit_mod.ALL_FORMATS,
                self._audit_mod.render_console(self._audit_obj, detail="all"))
            messagebox.showinfo("Exportación completa", f"El paquete se guardó automáticamente en:\n{destino}\n\n" +
                "\n".join(f"{k}: {v}" for k, v in res.items()))
        except Exception as e:
            messagebox.showerror("Exportación", f"{e}\n\n{traceback.format_exc()}")

    def cargar_csv_respuestas(self, ruta=None):
        """Importa un CSV de corridas y respuestas, tolerando BOM UTF-8, directiva sep= y delimitadores habituales. Detecta las columnas por nombre, valida números finitos y reconstruye coordenadas codificadas cuando faltan. Verifica la coherencia de las transformaciones, carga el estado y ejecuta el ajuste. La reconstrucción automática presupone el diseño Doehlert canónico estándar; para diseños modificados deben conservarse las columnas codificadas."""
        ruta = ruta or filedialog.askopenfilename(initialdir=os.path.join(self.carpeta_aplicacion(), "Examples"),
            filetypes=[("Datos experimentales CSV", "*.csv"), ("Todos los archivos", "*.*")], title="Importar datos experimentales")
        if not ruta: return
        self.last_csv_path = ruta
        self.log_event(f"Inicio de importación CSV: {ruta}")
        try:
            headers, X_arr, coded_arr, y_vals = eng.read_design_csv(ruta)   # lector compartido con la consola
            self._configure_factor_transform(X_arr, coded_arr)
            self.nombres_factores = headers[1:4]
            self.headers_diseno = ["Corrida", *self.nombres_factores, "Respuesta_Y", "x1_cod", "x2_cod", "x3_cod"]
            self.matriz_generada = [[i+1, *[round(float(v), 6) for v in X_arr[i]], f"{y_vals[i]:.8g}", *[round(float(v), 6) for v in coded_arr[i]]] for i in range(len(y_vals))]
            self._refresh_design_table()
            self.X_exp = X_arr
            self.y_exp = np.asarray(y_vals, dtype=float)
            self.cod_exp = coded_arr
            self.limites_factores = [(float(np.min(X_arr[:, i])), float(np.max(X_arr[:, i]))) for i in range(3)]
            self.ejecutar_ajuste_completo()
            self.cbo_eje_x["values"] = self.nombres_factores; self.cbo_eje_y["values"] = self.nombres_factores
            self.cbo_eje_x.current(0); self.cbo_eje_y.current(1)
            self.lbl_estado.config(text=f"Cargado: {len(y_vals)} corridas procesadas.", foreground="#006600")
            self.actualizar_graficos_superficie(); self.actualizar_graficos_diagnostico()
            self.last_csv_path = ruta
            self.notebook.select(self.tab_diseno)
            self.log_event(f"CSV importado y mostrado en pestaña 1: {ruta}; N={len(y_vals)}.")
        except Exception as err:
            self.last_error = traceback.format_exc(); self.log_event(f"Error importando CSV {ruta}: {err}")
            messagebox.showerror("Error al procesar CSV", f"{err}\n\nEl LOG de depuración conserva el traceback.")

    def ejecutar_ajuste_completo(self):
        """Ajusta el modelo cuadrático completo con el motor estadístico (doehlert_engine) y publica los resultados: coeficientes, covarianza, predicciones, residuos studentizados internos, leverage, ANOVA con error puro y falta de ajuste, análisis canónico (eigh de B), punto estacionario con IC por Método Delta y reportes de texto. Los widgets de tablas se refrescan solo si existen, de modo que el método también funciona sin interfaz (auditoría/pruebas). Lanza ModelError (ValueError) ante datos o diseños no válidos."""
        centros, escalas = getattr(self, "factor_centers", None), getattr(self, "factor_scales", None)
        if centros is None or escalas is None:          # sin estado previo: se derivan de los datos (validado)
            centros, escalas = eng.configure_factor_transform(self.X_exp, self.cod_exp)
            self.factor_centers, self.factor_scales = centros, escalas
        fit = fit_quadratic_model(self.cod_exp, self.y_exp, centros, escalas,
                                  limits_real=self.limites_factores, factor_names=self.nombres_factores,
                                  real=self.X_exp)
        self.fit = fit
        # Atributos de compatibilidad (superficies, bandas 2D, log de depuración)
        self.betas = fit.betas
        self.cov_beta = fit.cov_beta
        self.y_pred = fit.y_hat
        self.residuos = fit.residuals
        self.residuos_std = fit.student_resid
        self.leverage = fit.leverage
        self.t_crit = fit.t_crit
        self.s_res = fit.s_res
        can = fit.canonical
        if can.has_stationary:
            self.x_opt_cod, self.x_opt = can.x_cod, can.x_real
        else:
            self.x_opt_cod, self.x_opt = np.full(3, np.nan), np.full(3, np.nan)
        self.ic_coords = can.ic_coords

        self.txt_reporte.delete("1.0", "end")
        self.txt_reporte.insert("end", tr(eng.format_model_report(fit)))
        self.txt_diag.delete("1.0", "end")
        self.txt_diag.insert("end", tr(eng.format_lof_text(fit)))
        if hasattr(self, "tree_anova") and hasattr(self, "tree_res"):     # solo con interfaz gráfica real
            self._poblar_tablas_modelo()
            self._poblar_tabla_residuos()
        a = fit.anova
        self.log_event(f"Ajuste completado: N={fit.n}, rango={fit.rank}, R2={fit.r2:.8f}, condición={fit.condition:.6g}, "
                       f"LOF[{a.lof_status}] p={a.p_lof}, canónico={can.kind}.")

    def _poblar_tablas_modelo(self):
        """Vuelca en los Treeview de la pestaña 3 la tabla ANOVA y la tabla de coeficientes del ajuste actual, marca en verde los coeficientes significativos (p < 0.05) y escribe la prueba de falta de ajuste en el panel de texto."""
        """Rellena la pestaña 3 (ANOVA, coeficientes y notas de Lack of Fit)."""
        fit = self.fit
        self.tree_anova.delete(*self.tree_anova.get_children())
        self.tree_coef.delete(*self.tree_coef.get_children())
        for r in fit.anova.rows:
            self.tree_anova.insert("", "end", tags=("sub",) if r.level else (), values=(
                ("      " if r.level else "") + r.source,
                "" if r.df is None else r.df,
                "" if r.ss is None else eng.fmt_num(r.ss, 6),
                "" if r.ms is None else eng.fmt_num(r.ms, 6),
                "" if r.f is None else eng.fmt_num(r.f, 4),
                "" if r.p is None else eng.fmt_p(r.p)))
        for r in fit.coef_rows:
            significativo = np.isfinite(r.p) and r.p < 0.05
            self.tree_coef.insert("", "end", tags=("sig" if significativo else "ns",), values=(
                r.label, eng.fmt_num(r.estimate, 6), eng.fmt_num(r.se, 6), eng.fmt_num(r.t, 3),
                eng.fmt_p(r.p), r.sig, eng.fmt_num(r.ci_low, 6), eng.fmt_num(r.ci_high, 6)))

    def _poblar_tabla_residuos(self):
        """Reconstruye la tabla de diagnóstico por corrida de la pestaña 4 con los factores reales, Y observada, Ŷ, residuo ordinario, leverage y residuo studentizado interno; marca en rojo las corridas con |r_i| > 2 y actualiza la leyenda."""
        """Rellena la tabla de diagnóstico por corrida (pestaña 4)."""
        fit = self.fit
        cols = ["Corrida", *fit.factor_names, "Y_obs", "Ŷ_pred", "Residuo e_i", "Leverage h_ii", "Res. estud. r_i"]
        ids = [f"c{i}" for i in range(len(cols))]          # ids seguros (sin espacios)
        self.tree_res["columns"] = ids
        for cid, txt in zip(ids, cols):
            self.tree_res.heading(cid, text=txt)
            self.tree_res.column(cid, anchor="center", width=105 if cid != "c0" else 70)
        self.tree_res.delete(*self.tree_res.get_children())
        n_atipicos = 0
        for i in range(fit.n):
            r_i = fit.student_resid[i]
            atipico = bool(np.isfinite(r_i) and abs(r_i) > 2.0)
            n_atipicos += atipico
            self.tree_res.insert("", "end", iid=str(i), tags=("atipico",) if atipico else (), values=(
                i + 1, *[f"{v:.5g}" for v in fit.real[i]], f"{fit.y[i]:.5f}", f"{fit.y_hat[i]:.5f}",
                f"{fit.residuals[i]:+.5f}", f"{fit.leverage[i]:.4f}", eng.fmt_num(r_i, 4)))
        self.lbl_leyenda_res.config(text=(
            f"r_i = e_i / sqrt(MS_res·(1−h_ii))  |  Leverage medio = p/N = {eng.N_PARAMS / fit.n:.4f} (suma = {fit.leverage.sum():.2f})  |  "
            f"Filas en rojo: |r_i| > 2 ({n_atipicos} atípica(s))"))

    def evaluar_modelo(self, X1, X2, X3):
        """Predice Y recibiendo factores en unidades físicas."""
        coded = [(np.asarray(X1)-self.factor_centers[0])/self.factor_scales[0],
                 (np.asarray(X2)-self.factor_centers[1])/self.factor_scales[1],
                 (np.asarray(X3)-self.factor_centers[2])/self.factor_scales[2]]
        x1, x2, x3 = coded
        b = self.betas
        return (b[0]+b[1]*x1+b[2]*x2+b[3]*x3+b[4]*x1*x2+b[5]*x1*x3+b[6]*x2*x3+
                b[7]*x1**2+b[8]*x2**2+b[9]*x3**2)

    def _mascara_dominio_superficie(self, GX, GY, idx_x, idx_y, idx_fijo, val_fijo):
        """Máscara común de interpolación: esfera Doehlert + envolvente convexa + límites físicos."""
        try:
            from doehlert_domain import ExperimentalDomain
            dom=ExperimentalDomain(self.cod_exp,self.factor_centers,self.factor_scales,self.limites_factores)
            real=[None,None,None]
            real[idx_x]=GX; real[idx_y]=GY; real[idx_fijo]=val_fijo
            coded=[np.broadcast_to((real[j]-self.factor_centers[j])/self.factor_scales[j],GX.shape) for j in range(3)]
            points=np.stack(coded,axis=-1)
            return ~dom.mask(points)
        except Exception as exc:
            self.log_event(f"No se pudo calcular máscara geométrica de superficie: {exc}")
            # Fail closed: no dibujar regiones cuya pertenencia al dominio no pudo verificarse.
            return np.ones_like(GX,dtype=bool)

    def actualizar_graficos_superficie(self):
        """Actualiza los gráficos 3D y de contorno de la superficie estimada. Mantiene fijo el tercer factor en un valor representativo, genera una malla dentro de los límites físicos, evalúa el modelo y superpone el punto estacionario y sus intervalos cuando están disponibles."""
        if self.betas is None:
            return

        idx_x = self.cbo_eje_x.current()
        idx_y = self.cbo_eje_y.current()

        if idx_x == idx_y:
            messagebox.showwarning("Atención", "Selecciona dos factores distintos.")
            return

        idx_fijo = [i for i in [0, 1, 2] if i not in (idx_x, idx_y)][0]
        lim_fijo = self.limites_factores[idx_fijo]

        if not np.isnan(self.x_opt[idx_fijo]) and (lim_fijo[0] <= self.x_opt[idx_fijo] <= lim_fijo[1]):
            val_fijo = self.x_opt[idx_fijo]
            txt_fijo = f"{self.nombres_factores[idx_fijo]} fija en {val_fijo:.2f} (Óptimo)"
        else:
            val_fijo = (lim_fijo[0] + lim_fijo[1]) / 2.0
            txt_fijo = f"{self.nombres_factores[idx_fijo]} fija en {val_fijo:.2f} (Centro)"

        grid_x = np.linspace(self.limites_factores[idx_x][0], self.limites_factores[idx_x][1], 45)
        grid_y = np.linspace(self.limites_factores[idx_y][0], self.limites_factores[idx_y][1], 45)
        GX, GY = np.meshgrid(grid_x, grid_y)

        args = [None, None, None]
        args[idx_x] = GX
        args[idx_y] = GY
        args[idx_fijo] = val_fijo

        GZ = self.evaluar_modelo(args[0], args[1], args[2])
        fuera = self._mascara_dominio_superficie(GX,GY,idx_x,idx_y,idx_fijo,val_fijo)
        GZ = np.ma.masked_where(fuera,GZ)

        self.ax_3d.clear()
        self.ax_contour.clear()

        # Superficie 3D
        self.ax_3d.plot_surface(GX, GY, GZ, cmap="viridis", edgecolor="none", alpha=0.85)
        self.ax_3d.set_xlabel(self.nombres_factores[idx_x], labelpad=6)
        self.ax_3d.set_ylabel(self.nombres_factores[idx_y], labelpad=6)
        self.ax_3d.set_zlabel("Respuesta (Y)", labelpad=6)
        self.ax_3d.set_title("Superficie 3D\n" + txt_fijo, fontsize=9)

        if not np.isnan(self.x_opt[0]):
            opt_x = self.x_opt[idx_x]
            opt_y = self.x_opt[idx_y]
            opt_z = self.evaluar_modelo(self.x_opt[0], self.x_opt[1], self.x_opt[2])
            self.ax_3d.scatter([opt_x], [opt_y], [opt_z], color="red", s=80, marker="*")

        # Contornos 2D
        self.ax_contour.contourf(GX, GY, GZ, levels=14, cmap="viridis")
        lines = self.ax_contour.contour(GX, GY, GZ, levels=14, colors="black", linewidths=0.6)
        self.ax_contour.clabel(lines, inline=True, fontsize=8, fmt="%.1f")
        self.ax_contour.set_xlabel(self.nombres_factores[idx_x])
        self.ax_contour.set_ylabel(self.nombres_factores[idx_y])
        self.ax_contour.set_title("Contornos 2D\n" + txt_fijo, fontsize=9)

        # Proyección del óptimo con barras de incertidumbre IC 95%
        if not np.isnan(self.x_opt[0]):
            opt_x = self.x_opt[idx_x]
            opt_y = self.x_opt[idx_y]
            self.ax_contour.scatter([opt_x], [opt_y], color="red", s=90, marker="*", edgecolors="black", label="Óptimo")

            if self.ic_coords:
                err_x = self.t_crit * self.ic_coords[idx_x][2]
                err_y = self.t_crit * self.ic_coords[idx_y][2]
                if not np.isnan(err_x) and not np.isnan(err_y):
                    self.ax_contour.errorbar(
                        opt_x, opt_y,
                        xerr=err_x, yerr=err_y,
                        fmt="none", ecolor="red", elinewidth=1.2, capsize=4,
                        label="IC 95% Óptimo"
                    )

            self.ax_contour.legend(loc="upper right", fontsize=8)

        self.fig_analisis.tight_layout()
        self.canvas_analisis.draw()

    # ==============================================================
    # VENTANA DE CORTE 2D CON BANDAS IC / IP
    # ==============================================================
    def abrir_ventana_bandas_2d(self):
        """Abre ventana con el corte 2D de trayectoria, bandas IC e IP y guardado directo."""
        if self.betas is None or self.x_opt is None:
            messagebox.showwarning("Atención", "Primero debes ajustar el modelo y calcular el óptimo.")
            return

        win_corte = tk.Toplevel(self)
        win_corte.title("Corte de Trayectoria Óptima 2D (Bandas IC 95% e IP 95%)")
        win_corte.geometry("860x640")

        frame_ctrl = ttk.Frame(win_corte, padding=10)
        frame_ctrl.pack(fill="x")

        ttk.Label(frame_ctrl, text="Variable del eje X (recorrido):", font=("Segoe UI", 9, "bold")).pack(side="left", padx=5)
        cbo_var = ttk.Combobox(frame_ctrl, values=self.nombres_factores, state="readonly", width=16)
        cbo_var.current(0)
        cbo_var.pack(side="left", padx=6)

        fig_corte = Figure(figsize=(7, 5), dpi=100)
        ax = fig_corte.add_subplot(111)

        ttk.Button(
            frame_ctrl,
            text="Guardar Gráfico de Corte (PNG/PDF)...",
            command=lambda: self.exportar_grafico_individual(fig_corte, f"Corte_{cbo_var.get()}_Doehlert")
        ).pack(side="right", padx=6)

        canvas_corte = FigureCanvasTkAgg(fig_corte, master=win_corte)
        canvas_corte.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        toolbar = NavigationToolbar2Tk(canvas_corte, win_corte)
        toolbar.update()

        def trazar_corte(evt=None):
            """Callback interno que recalcula el corte de respuesta para el factor seleccionado. Sustituye valores fijos no válidos por el centro del dominio, recorre el rango del factor activo, calcula la media y sus bandas de confianza/predicción usando la covarianza del modelo, y redibuja el gráfico."""
            idx_activa = cbo_var.current()
            nom_activa = self.nombres_factores[idx_activa]

            x_fijos = list(self.x_opt)
            for j in range(3):
                if np.isnan(x_fijos[j]) or not (self.limites_factores[j][0] <= x_fijos[j] <= self.limites_factores[j][1]):
                    x_fijos[j] = (self.limites_factores[j][0] + self.limites_factores[j][1]) / 2.0

            lim_min, lim_max = self.limites_factores[idx_activa]
            t_vals = np.linspace(lim_min, lim_max, 150)

            y_pred_vals = []
            ic_low, ic_high = [], []
            ip_low, ip_high = [], []

            for val in t_vals:
                coords = list(x_fijos)
                coords[idx_activa] = val

                coded_coords = self._coded_from_real(coords)
                m_t = self._design_matrix(coded_coords.reshape(1, 3))[0]

                y_hat = float(m_t @ self.betas)
                var_mean = max(0.0, float(m_t @ self.cov_beta @ m_t.T))
                se_mean = math.sqrt(var_mean)
                se_pred = math.sqrt((self.s_res ** 2) + var_mean)

                y_pred_vals.append(y_hat)
                ic_low.append(y_hat - self.t_crit * se_mean)
                ic_high.append(y_hat + self.t_crit * se_mean)
                ip_low.append(y_hat - self.t_crit * se_pred)
                ip_high.append(y_hat + self.t_crit * se_pred)

            ax.clear()

            # 1. Banda de Predicción individual exterior
            ax.fill_between(t_vals, ip_low, ip_high, color="#b3cde3", alpha=0.45, label="IP 95% (Medición Individual)")

            # 2. Banda de Confianza de la media interior
            ax.fill_between(t_vals, ic_low, ic_high, color="#02818a", alpha=0.35, label="IC 95% (Media Estimada)")

            # 3. Línea central predicha
            ax.plot(t_vals, y_pred_vals, color="#014636", linewidth=2, label=r"Respuesta Predicha $\hat{y}$")

            # 4. Proyección del Óptimo puntual
            x_opt_act = self.x_opt[idx_activa]
            if lim_min <= x_opt_act <= lim_max:
                coded_opt = self._coded_from_real(x_fijos)
                m_opt = self._design_matrix(coded_opt.reshape(1, 3))[0]
                y_opt_val = float(m_opt @ self.betas)
                ax.plot([x_opt_act], [y_opt_val], marker="*", markersize=12, color="red", label=f"Óptimo ({x_opt_act:.2f})")
                ax.axvline(x_opt_act, color="red", linestyle=":", alpha=0.7)

            otros = [f"{self.nombres_factores[j]}={x_fijos[j]:.2f}" for j in range(3) if j != idx_activa]
            ax.set_title(f"Perfil de Sensibilidad y Bandas de Error: {nom_activa}\n(Fijando {', '.join(otros)})", fontsize=10)
            ax.set_xlabel(nom_activa, fontsize=9)
            ax.set_ylabel("Respuesta Estimada (Y)", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="best", fontsize=8)

            fig_corte.tight_layout()
            canvas_corte.draw()

        cbo_var.bind("<<ComboboxSelected>>", trazar_corte)
        ttk.Button(frame_ctrl, text="Actualizar Corte", command=trazar_corte).pack(side="left", padx=8)

        trazar_corte()

    # ==============================================================
    # GRÁFICOS DE DIAGNÓSTICO
    # ==============================================================
    def _trazar_diagnostico(self, ax_rf, ax_qq, ax_ord, fs=9):
        """Dibuja en tres ejes dados los gráficos de residuos studentizados internos: frente a valores ajustados, Q-Q normal con posiciones de Blom y frente al orden de ejecución. Es la rutina común de la pestaña 4 y del informe PDF. Devuelve un diccionario índice de corrida → punto del Q-Q, usado para el resaltado."""
        """Dibuja los 3 gráficos de residuos (reutilizado por la GUI y por el PDF)."""
        fit = self.fit
        r = fit.student_resid
        kw = dict(color="#1f77b4", edgecolor="black", s=45, alpha=0.85, zorder=3)

        # Gráfico 1: Residuos vs. Valores ajustados
        ax_rf.scatter(fit.y_hat, r, **kw)
        ax_rf.axhline(0, color="red", linestyle="--", linewidth=1.2)
        for lim in (2, -2):
            ax_rf.axhline(lim, color="gray", linestyle=":", linewidth=1)
        ax_rf.set_xlabel(r"Valores ajustados ($\hat{Y}$)", fontsize=fs)
        ax_rf.set_ylabel(r"Residuo estudentizado $r_i$", fontsize=fs)
        ax_rf.set_title("1. Residuos vs. Ajustados\n(homocedasticidad)", fontsize=fs)

        # Gráfico 2: Q-Q plot normal (posiciones de Blom)
        teoricos, ordenados, orden = eng.qq_points(r)
        kw2 = dict(kw, color="#2ca02c")
        ax_qq.scatter(teoricos, ordenados, **kw2)
        if len(teoricos):
            lo = min(teoricos[0], ordenados[0]); hi = max(teoricos[-1], ordenados[-1])
            ax_qq.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.2)
        ax_qq.set_xlabel("Cuantiles teóricos normales", fontsize=fs)
        ax_qq.set_ylabel("Residuos estudentizados ordenados", fontsize=fs)
        ax_qq.set_title("2. Gráfico de probabilidad normal\n(Q-Q plot)", fontsize=fs)

        # Gráfico 3: Residuos vs. orden de ejecución
        orden_ej = np.arange(1, fit.n + 1)
        ax_ord.plot(orden_ej, r, color="#999999", linewidth=0.8, zorder=2)
        ax_ord.scatter(orden_ej, r, **dict(kw, color="#ff7f0e"))
        ax_ord.axhline(0, color="red", linestyle="--", linewidth=1.2)
        for lim in (2, -2):
            ax_ord.axhline(lim, color="gray", linestyle=":", linewidth=1)
        ax_ord.set_xlabel("Orden de ejecución (nº de corrida)", fontsize=fs)
        ax_ord.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_ord.set_ylabel(r"Residuo estudentizado $r_i$", fontsize=fs)
        ax_ord.set_title("3. Residuos vs. Orden de ejecución\n(independencia / deriva)", fontsize=fs)

        for ax in (ax_rf, ax_qq, ax_ord):
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.tick_params(labelsize=fs - 1)
        return {int(orden[k]): (teoricos[k], ordenados[k]) for k in range(len(orden))}

    def actualizar_graficos_diagnostico(self):
        """Redibuja los tres gráficos de diagnóstico de residuos con el ajuste actual y crea los marcadores de resaltado de la corrida seleccionada. No hace nada si todavía no hay un modelo ajustado."""
        if self.fit is None:
            return
        for ax in (self.ax_res_fit, self.ax_qq, self.ax_res_ord):
            ax.clear()
        self._qq_pos = self._trazar_diagnostico(self.ax_res_fit, self.ax_qq, self.ax_res_ord)
        # Marcadores de resaltado (selección en la tabla)
        self._hl_artists = [ax.plot([], [], "o", ms=14, mfc="none", mec="red", mew=2.2, zorder=6)[0]
                            for ax in (self.ax_res_fit, self.ax_qq, self.ax_res_ord)]
        self.fig_diag.tight_layout()
        self.canvas_diag.draw()

    def _resaltar_corrida(self, event=None):
        """Callback de selección de la tabla de residuos: coloca un círculo rojo sobre la corrida elegida en los tres gráficos de diagnóstico."""
        sel = self.tree_res.selection()
        if not sel or self.fit is None or not self._hl_artists:
            return
        i = int(sel[0])
        r = self.fit.student_resid[i]
        pts = [(self.fit.y_hat[i], r), self._qq_pos.get(i), (i + 1, r)]
        for art, pt in zip(self._hl_artists, pts):
            if pt is None or not np.all(np.isfinite(pt)):
                art.set_data([], [])
            else:
                art.set_data([pt[0]], [pt[1]])
        self.canvas_diag.draw_idle()

    # ==============================================================
    # EXPORTACIONES (REPORTE, TABLAS, GRÁFICOS, PDF)
    # ==============================================================
    def _requiere_ajuste(self):
        """Comprueba que exista un modelo ajustado; si no, muestra una advertencia y devuelve False."""
        if self.fit is None:
            messagebox.showwarning("Atención", "Carga datos y ajusta el modelo antes de exportar.")
            return False
        return True

    def guardar_grafico_actual(self):
        """Exporta la figura de la pestaña visible (diseño 3D, superficies o residuos) mediante exportar_grafico_individual. En la pestaña de tablas, que no contiene gráficos, avisa al usuario."""
        """Guarda la figura de la pestaña visible (diseño 3D, superficies o residuos)."""
        idx = self.notebook.index(self.notebook.select())
        figuras = {0: (self.fig_diseno, "Diseno_Doehlert_3D"),
                   1: (self.fig_analisis, "Superficie_Doehlert"),
                   3: (self.fig_diag, "Diagnostico_Residuos")}
        if idx not in figuras:
            messagebox.showwarning("Sin gráfico", "La pestaña actual no contiene gráficos.\n"
                                   "Ve a la pestaña 1, 2 o 4 y vuelve a intentarlo.")
            return
        self.exportar_grafico_individual(*figuras[idx])

    def exportar_grafico_individual(self, figura, nombre_sugerido):
        """Solicita una ruta y exporta una figura Matplotlib a PNG (300 DPI), PDF o SVG según la extensión elegida. Comprueba que exista un modelo (o un diseño, para la figura 3D), rechaza extensiones no soportadas y comunica los fallos de escritura sin interrumpir la aplicación."""
        if (figura is self.fig_analisis or figura is self.fig_diag) and self.fit is None:
            messagebox.showwarning("Atención", "Carga datos y ajusta el modelo antes de exportar.")
            return
        if figura is self.fig_diseno and not self.matriz_generada:
            messagebox.showwarning("Atención", "Primero genera la matriz de diseño.")
            return
        ruta = self.dialogo_guardar(nombre_sugerido, ".png",
            [("Imagen PNG (300 DPI)", "*.png"), ("Documento PDF (vectorial)", "*.pdf"), ("Vectorial SVG", "*.svg")], "Guardar gráfico")
        if not ruta:
            return
        ext = os.path.splitext(ruta)[1].lower()
        if ext == "":
            ruta, ext = ruta + ".png", ".png"
        if ext not in (".png", ".pdf", ".svg"):
            messagebox.showwarning("Formato no soportado", f"La extensión «{ext}» no es válida. Usa .png, .pdf o .svg.")
            return
        try:
            figura.savefig(ruta, dpi=300, bbox_inches="tight")
            self.log_event(f"Gráfico exportado: {ruta}")
            messagebox.showinfo("Listo", f"Guardado en alta resolución:\n{ruta}")
        except Exception as err:
            self.last_error = traceback.format_exc()
            messagebox.showerror("Error", f"No se pudo guardar la imagen: {err}")

    def exportar_reporte_texto(self):
        """Guarda el reporte ejecutivo (timestamp, factores y niveles, coeficientes con p-valores, ANOVA con falta de ajuste, diagnóstico canónico y óptimo físico con IC 95 % por Método Delta) en Markdown o texto plano, con codificación UTF-8."""
        if not self._requiere_ajuste():
            return
        ruta = self.dialogo_guardar("Reporte_Ejecutivo", ".md",
            [("Markdown", "*.md"), ("Texto plano", "*.txt")], "Guardar reporte ejecutivo")
        if not ruta:
            return
        if not os.path.splitext(ruta)[1]:
            ruta += ".md"
        try:
            texto = tr(eng.build_markdown_report(self.fit))
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(texto)
            self.log_event(f"Reporte ejecutivo exportado: {ruta}")
            messagebox.showinfo("Reporte generado", f"Reporte guardado en:\n{ruta}")
        except Exception as err:
            self.last_error = traceback.format_exc()
            self.log_event(f"Error exportando reporte: {err}")
            messagebox.showerror("Error", f"No se pudo guardar el reporte: {err}")

    def localizar_archivo_exportado(self, ruta):
        """Translate textual cells in generated CSV/XLSX exports to the active UI language."""
        ext = os.path.splitext(ruta)[1].lower()
        if ext == ".csv":
            with open(ruta, "r", encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.reader(fh))
            with open(ruta, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerows([[tr(cell) for cell in row] for row in rows])
        elif ext == ".xlsx":
            try:
                from openpyxl import load_workbook
                wb = load_workbook(ruta)
                if get_language() == "en":
                    sheet_names = {"Resumen_Modelo": "Model_Summary", "Diagnostico_Residuos": "Residual_Diagnostics"}
                    for ws in wb.worksheets:
                        if ws.title in sheet_names:
                            ws.title = sheet_names[ws.title]
                for ws in wb.worksheets:
                    for row in ws.iter_rows():
                        for cell in row:
                            if isinstance(cell.value, str):
                                cell.value = tr(cell.value)
                wb.save(ruta)
            except ImportError:
                pass

    def exportar_tablas(self):
        """Exporta las tablas del modelo a un libro .xlsx con las hojas Resumen_Modelo, ANOVA_LackOfFit y Diagnostico_Residuos (si openpyxl está disponible) o a CSV UTF-8 delimitado por comas; el CSV genera además los archivos _coeficientes.csv y _anova.csv."""
        if not self._requiere_ajuste():
            return
        xlsx_ok = eng.openpyxl_available()
        tipos = ([("Libro de Excel (.xlsx)", "*.xlsx")] if xlsx_ok else []) + [("CSV UTF-8 (comas)", "*.csv")]
        ruta = self.dialogo_guardar("Tablas_Modelo", ".xlsx" if xlsx_ok else ".csv",
            tipos, "Exportar tablas del modelo")
        if not ruta:
            return
        ext = os.path.splitext(ruta)[1].lower()
        if ext not in (".xlsx", ".csv"):
            ext = ".xlsx" if xlsx_ok else ".csv"
            ruta += ext
        if ext == ".xlsx" and not xlsx_ok:
            messagebox.showwarning("openpyxl no disponible",
                                   "No se encontró «openpyxl»; se guardará en formato CSV UTF-8.\n"
                                   "(Para .xlsx instala: pip install openpyxl)")
            ruta, ext = os.path.splitext(ruta)[0] + ".csv", ".csv"
        try:
            if ext == ".xlsx":
                eng.write_xlsx(ruta, self.fit)
                self.localizar_archivo_exportado(ruta)
                detalle = "Sheets: Model_Summary, ANOVA_LackOfFit, Residual_Diagnostics." if get_language() == "en" else "Hojas: Resumen_Modelo, ANOVA_LackOfFit, Diagnostico_Residuos."
            else:
                escritos = eng.write_csv_bundle(ruta, self.fit)
                for archivo_generado in escritos:
                    self.localizar_archivo_exportado(archivo_generado)
                detalle = ("Generated files:\n" if get_language() == "en" else "Archivos generados:\n") + "\n".join(escritos)
            self.log_event(f"Tablas exportadas ({ext}): {ruta}")
            messagebox.showinfo("Exportado", f"Tablas guardadas en:\n{ruta}\n\n{detalle}")
        except PermissionError:
            messagebox.showerror("Error", f"Sin permiso de escritura (¿archivo abierto en otro programa?):\n{ruta}")
        except Exception as err:
            self.last_error = traceback.format_exc()
            self.log_event(f"Error exportando tablas: {err}")
            messagebox.showerror("Error", f"No se pudieron exportar las tablas: {err}")

    @staticmethod
    def _texto_ajustado(fig, texto, top, bottom, left=0.06):
        """Inserta texto monoespaciado en una página del PDF, reduciendo el tamaño de fuente según el número de líneas y el ancho de la línea más larga para que el bloque completo quepa en la página."""
        lineas = texto.split("\n")
        alto_pt = (top - bottom) * 11 * 72.0
        ancho_pt = (0.97 - left) * 8.5 * 72.0
        fs = min(8.0, alto_pt / (len(lineas) * 1.25), ancho_pt / (max(len(l) for l in lineas) * 0.62))
        fig.text(left, top, texto, fontsize=max(fs, 4.0), family="monospace", va="top")

    def exportar_informe_completo_pdf(self):
        """Genera un PDF de cuatro páginas con el ajuste actual: (1) resumen estadístico, diagnóstico canónico y óptimo; (2) superficie 3D y contornos; (3) ANOVA, falta de ajuste y coeficientes; (4) gráficos de residuos y tabla por corrida. No recalcula ni modifica el modelo."""
        if not self._requiere_ajuste():
            return
        ruta = self.dialogo_guardar("Informe_Completo", ".pdf",
            [("Documento PDF", "*.pdf")], "Guardar informe completo")
        if not ruta:
            return
        if not os.path.splitext(ruta)[1]:
            ruta += ".pdf"
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            fit = self.fit
            ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            sub = f"Variables: {', '.join(self.nombres_factores)} | Corridas: {fit.n} | {ts}"
            with PdfPages(ruta) as pdf:
                # Pág 1: resumen estadístico, diagnóstico canónico y óptimo
                f1 = Figure(figsize=(8.5, 11))
                f1.text(0.06, 0.95, tr("INFORME DE OPTIMIZACIÓN: DISEÑO DOEHLERT"), fontsize=14, weight="bold")
                f1.text(0.06, 0.93, sub, fontsize=8, color="#555555")
                self._texto_ajustado(f1, tr(eng.format_model_report(fit)), top=0.905, bottom=0.04)
                pdf.savefig(f1)

                # Pág 2: superficie 3D y contornos
                idx_x, idx_y = self.cbo_eje_x.current(), self.cbo_eje_y.current()
                if idx_x < 0 or idx_y < 0 or idx_x == idx_y:
                    idx_x, idx_y = 0, 1
                idx_fijo = [i for i in (0, 1, 2) if i not in (idx_x, idx_y)][0]
                lim_fijo = self.limites_factores[idx_fijo]
                xo = self.x_opt[idx_fijo]
                val_fijo = xo if (np.isfinite(xo) and lim_fijo[0] <= xo <= lim_fijo[1]) else (lim_fijo[0] + lim_fijo[1]) / 2.0
                gx = np.linspace(*self.limites_factores[idx_x], 45)
                gy = np.linspace(*self.limites_factores[idx_y], 45)
                GX, GY = np.meshgrid(gx, gy)
                args = [None, None, None]
                args[idx_x], args[idx_y], args[idx_fijo] = GX, GY, val_fijo
                GZ = self.evaluar_modelo(*args)
                fuera = self._mascara_dominio_superficie(GX,GY,idx_x,idx_y,idx_fijo,val_fijo)
                GZ = np.ma.masked_where(fuera,GZ)
                f2 = Figure(figsize=(8.5, 11))
                f2.text(0.06, 0.95, tr("SUPERFICIE DE RESPUESTA"), fontsize=14, weight="bold")
                f2.text(0.06, 0.93, f"{self.nombres_factores[idx_fijo]} fija en {val_fijo:.4g}", fontsize=8, color="#555555")
                ax3d = f2.add_axes([0.06, 0.50, 0.88, 0.40], projection="3d")
                ax3d.plot_surface(GX, GY, GZ, cmap="viridis", edgecolor="none", alpha=0.88)
                ax3d.set_xlabel(self.nombres_factores[idx_x], fontsize=8)
                ax3d.set_ylabel(self.nombres_factores[idx_y], fontsize=8)
                ax3d.set_zlabel("Respuesta (Y)", fontsize=8)
                ax3d.set_title("Superficie 3D", fontsize=9, weight="bold")
                axc = f2.add_axes([0.10, 0.07, 0.80, 0.35])
                cf = axc.contourf(GX, GY, GZ, levels=14, cmap="viridis")
                f2.colorbar(cf, ax=axc, fraction=0.04)
                axc.set_xlabel(self.nombres_factores[idx_x], fontsize=8)
                axc.set_ylabel(self.nombres_factores[idx_y], fontsize=8)
                axc.set_title("Contornos 2D", fontsize=9, weight="bold")
                if np.all(np.isfinite(self.x_opt)):
                    axc.plot([self.x_opt[idx_x]], [self.x_opt[idx_y]], "r*", ms=12, mec="black")
                pdf.savefig(f2)

                # Pág 3: ANOVA, falta de ajuste y coeficientes
                f3 = Figure(figsize=(8.5, 11))
                f3.text(0.06, 0.95, tr("ANOVA, FALTA DE AJUSTE Y COEFICIENTES"), fontsize=14, weight="bold")
                texto3 = tr("\n\n".join([eng.format_anova_text(fit), eng.format_lof_text(fit), eng.format_coef_text(fit)]))
                self._texto_ajustado(f3, texto3, top=0.92, bottom=0.04)
                pdf.savefig(f3)

                # Pág 4: diagnóstico de residuos (gráficos + tabla)
                f4 = Figure(figsize=(8.5, 11))
                f4.text(0.06, 0.95, tr("DIAGNÓSTICO DE RESIDUOS"), fontsize=14, weight="bold")
                a1 = f4.add_axes([0.09, 0.73, 0.38, 0.17])
                a2 = f4.add_axes([0.57, 0.73, 0.38, 0.17])
                a3 = f4.add_axes([0.09, 0.50, 0.86, 0.15])
                self._trazar_diagnostico(a1, a2, a3, fs=7)
                nom = [n[:9] for n in fit.factor_names]
                lineas = [f"{'Run':>3} {nom[0]:>10} {nom[1]:>10} {nom[2]:>10} {'Y_obs':>10} {'Y_pred':>10} {'e_i':>10} {'h_ii':>8} {'r_i':>8}"]
                for i in range(fit.n):
                    lineas.append(f"{i+1:>3} {fit.real[i][0]:>10.4f} {fit.real[i][1]:>10.4f} {fit.real[i][2]:>10.4f} "
                                  f"{fit.y[i]:>10.4f} {fit.y_hat[i]:>10.4f} {fit.residuals[i]:>10.4f} "
                                  f"{fit.leverage[i]:>8.4f} {eng.fmt_num(fit.student_resid[i], 3):>8}")
                self._texto_ajustado(f4, "\n".join(lineas), top=0.43, bottom=0.04)
                pdf.savefig(f4)

            self.log_event(f"Informe PDF exportado: {ruta}")
            messagebox.showinfo("Listo", f"Informe PDF generado en:\n{ruta}")
        except Exception as err:
            self.last_error = traceback.format_exc()
            self.log_event(f"Error exportando PDF: {err}")
            messagebox.showerror("Error", f"No se pudo exportar el PDF: {err}")


if __name__ == "__main__":
    app = DoehlertSuiteApp()
    if _ARGS is not None and _ARGS.archivo:
        app.after(300, lambda: app.cargar_csv_respuestas(_ARGS.archivo))
    app.mainloop()
