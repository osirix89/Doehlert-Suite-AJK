# SPDX-License-Identifier: GPL-3.0-only
"""Lightweight English/Spanish localization for Doehlert Suite.

English is the default language. Spanish keeps the original application wording.
The selected language is stored in the user's home directory, not in the repo.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

_CONFIG = Path.home() / '.doehlert_suite.json'
_LANGUAGE_CACHE = None
LANGUAGES = {'en': 'English', 'es': 'Español'}

# Source strings are Spanish; English translations are keyed by their exact source.
_EN = {
'Aplicación iniciada':'Application started',
'Resumen del modelo Doehlert':'Doehlert Model Summary','Parámetro':'Parameter','Valor':'Value',
'Coeficientes (variables codificadas)':'Coefficients (coded variables)','Análisis canónico':'Canonical Analysis',
'Eje':'Axis','Autovalor':'Eigenvalue','Generado':'Generated',
'rango=':'rank=','Corridas':'Runs','Matriz':'Matrix','Naturaleza':'Nature',
'Punto estacionario':'Stationary point','Respuesta':'Response','punto estacionario':'stationary point',
'¿Dentro del rango físico de cada factor?:':'Within the physical range of each factor?:',
'Fuente                  GL             SC             CM          F   p-valor':'Source                  df             SS             MS          F   p-value',
'Término                        Coef.          EE         t   p-valor  Sig   IC95% inf   IC95% sup':'Term                        Coefficient          SE         t   p-value  Sig   95% CI lower   95% CI upper',
'INFORME DE OPTIMIZACIÓN: DISEÑO DOEHLERT':'OPTIMIZATION REPORT: DOEHLERT DESIGN',
'SUPERFICIE DE RESPUESTA':'RESPONSE SURFACE',
'ANOVA, FALTA DE AJUSTE Y COEFICIENTES':'ANOVA, LACK OF FIT, AND COEFFICIENTS',
'DIAGNÓSTICO DE RESIDUOS':'RESIDUAL DIAGNOSTICS',
'=== REGRESIÓN POLINOMIAL (2º ORDEN) ===':'=== QUADRATIC POLYNOMIAL REGRESSION ===',
'Nº Corridas (N)':'Number of Runs (N)','Grados de Libertad':'Degrees of Freedom','Parámetros':'Parameters',
'Rango de matriz':'Matrix Rank','Condición (codificada)':'Condition Number (coded)',
'Error Estándar Residual':'Residual Standard Error','F regresión':'Regression F',
'Valor Crítico t':'Critical t Value','=== DIAGNÓSTICO CANÓNICO Y GEOMETRÍA ===':'=== CANONICAL ANALYSIS AND GEOMETRY ===',
'Matriz B (simétrica, 2º orden):':'B Matrix (symmetric, second order):','Autovalores (np.linalg.eigh, ascendentes):':'Eigenvalues (np.linalg.eigh, ascending):',
'Autovectores (columnas = ejes canónicos, vars. codificadas):':'Eigenvectors (columns = canonical axes, coded variables):',
'Naturaleza: MÁXIMO LOCAL ESTRICTO':'Nature: STRICT LOCAL MAXIMUM','Naturaleza: MÍNIMO LOCAL ESTRICTO':'Nature: STRICT LOCAL MINIMUM',
'PUNTO ESTACIONARIO x0':'STATIONARY POINT x0','Coordenadas codificadas:':'Coded coordinates:',
'Respuesta ŷ:':'Response ŷ:','IC 95% (Media):':'95% CI (Mean):','IP 95% (Indiv.):':'95% PI (Individual):',
'ÓPTIMO EN UNIDADES FÍSICAS (IC 95% por Método Delta):':'OPTIMUM IN PHYSICAL UNITS (95% CI by Delta Method):',
'EVALUACIÓN DEL DOMINIO EXPERIMENTAL:':'EXPERIMENTAL DOMAIN EVALUATION:',
'Radio codificado':'Coded radius','dominio Doehlert':'Doehlert domain',
'¿Estrictamente dentro del hiperelipsoide':'Strictly inside the hyperellipsoid',
'¿Dentro del rango físico de cada factor?:':'Within the physical range of each factor?:',
'=== TABLA ANOVA (con Falta de Ajuste) ===':'=== ANOVA TABLE (with Lack of Fit) ===',
'Fuente':'Source','p-valor':'p-value','Regresión (modelo)':'Regression (model)',
'Residuo':'Residual','Falta de ajuste':'Lack of fit','Error puro':'Pure error',
'(Ajuste perfecto: MS_res ≈ 0, F y t no definidos.)':'(Perfect fit: MS_res ≈ 0; F and t are undefined.)',
'=== DESCOMPOSICIÓN ANOVA DE RESIDUOS ===':'=== ANOVA RESIDUAL DECOMPOSITION ===',
'SS_Residual Total:':'Total Residual SS:','ERROR PURO (Pure Error):':'PURE ERROR:',
'FALTA DE AJUSTE (Lack-of-Fit):':'LACK OF FIT:','PRUEBA LOF (H0: el modelo no presenta falta de ajuste):':'LOF TEST (H0: the model has no lack of fit):',
'Estadístico F:':'F Statistic:','RESULTADO:':'RESULT:',
'NO SIGNIFICATIVO':'NOT SIGNIFICANT','SIGNIFICATIVO':'SIGNIFICANT',
'no se detecta falta de ajuste.':'no lack of fit was detected.',
'evidencia de falta de ajuste.':'evidence of lack of fit.',
'no computable.':'not computable.','centro':'center point','punto repetido':'repeated point',
'=== COEFICIENTES (variables codificadas) ===':'=== COEFFICIENTS (coded variables) ===',
'Término':'Term','Coef.':'Coefficient','EE':'SE','Sig':'Sig.','IC95% inf':'95% CI lower','IC95% sup':'95% CI upper',
'Intercepto':'Intercept','SÍ':'YES','NO':'NO',
'La matriz codificada debe tener N filas y 3 columnas, igual que la respuesta.':'The coded matrix must have N rows and 3 columns, matching the response.',
'Los datos contienen valores vacíos, NaN o infinitos.':'The data contain missing, NaN, or infinite values.',
'Se requieren más corridas que parámetros (10) para estimar el error residual.':'More runs than model parameters (10) are required to estimate residual error.',
'El CSV debe contener encabezado y al menos 13 corridas.':'The CSV must contain a header and at least 13 runs.',
'Faltan columnas: se esperan Corrida, 3 factores físicos y Respuesta_Y.':'Missing columns: expected Run, 3 physical factors, and Response_Y.',
'Se necesitan al menos 13 corridas válidas.':'At least 13 valid runs are required.',
"# Reporte Ejecutivo — Diseño Experimental Doehlert (3 factores)":"# Executive Report — Doehlert Experimental Design (3 Factors)",
'**Modelo:** polinomio cuadrático completo (10 parámetros)':'**Model:** full quadratic polynomial (10 parameters)',
'## 3. Coeficientes con inferencia estadística':'## 3. Coefficients and Statistical Inference',
'## 4. ANOVA con descomposición del error (Lack of Fit)':'## 4. ANOVA with Error Decomposition (Lack of Fit)',
'## 5. Diagnóstico canónico y punto estacionario':'## 5. Canonical Diagnostics and Stationary Point',
'*Reporte generado por la Suite Doehlert (motor estadístico nativo, sin SciPy).*':'*Report generated by Doehlert Suite (native statistical engine, without SciPy).*',
'Todos':'All',
'Listo. Ejecute la auditoría para iniciar las 19 etapas.':'Ready. Run the audit to start the 19 stages.',
'Los archivos generados se organizan en Exports por fecha.':'Generated files are organized by date under Exports.',
'X1 (7 Niveles):':'X1 (7 Levels):','X2 (5 Niveles):':'X2 (5 Levels):','X3 (3 Niveles):':'X3 (3 Levels):',
'Datos experimentales CSV':'Experimental Data CSV','Todos los archivos':'All files','Archivo CSV (decimal con punto)':'CSV File (decimal point)',
'Todos':'All',
'Diseño, Optimización y Diagnóstico':'Design, Optimization and Diagnostics',
'Respuesta Predicha $\\hat{y}$':'Predicted Response $\\hat{y}$',
'Sig.:  *** p<0.001   ** p<0.01   * p<0.05   . p<0.1   ns p≥0.1   |   verde = significativo (p<0.05)':'Sig.:  *** p<0.001   ** p<0.01   * p<0.05   . p<0.1   ns p≥0.1   |   green = significant (p<0.05)',
'IC 95% Óptimo':'95% CI Optimum',
'límite de tolerancia':'tolerance limit','radio unitario':'unit radius','tolerancia 1e-5':'tolerance 1e-5',
'unidades naturales':'natural units','codificadas (−1…+1)':'coded (−1…+1)','1/ε (límite)':'1/ε (limit)',
'umbral 1e-6':'threshold 1e-6','Delta (referencia)':'Delta (reference)','Monte Carlo (robusto)':'Monte Carlo (robust)',
'referencia':'reference','nominal 0.95':'nominal 0.95','α = 0.05':'α = 0.05','referencia (−H⁻¹b)':'reference (−H⁻¹b)',
'x* (β̂)':'x* (β̂)','IC 95 % Delta (ref)':'95% CI Delta (ref)','⬆ Exportar':'⬆ Export',
'Exportar TODO (carpeta)…':'Export ALL (folder)…','listo':'ready','Detalle':'Details','Comparación':'Comparison',
'Consola':'Console','Seleccionar el main.py a auditar':'Select the main.py file to audit',
'Importar dataset (CSV de la Suite)':'Import dataset (Suite CSV)','Importar reporte JSON previo':'Import previous JSON report',
'iniciando…':'starting…','▶ Ejecutar auditoría':'▶ Run Audit','📂 main.py…':'📂 main.py…',
'⬇ Dataset CSV…':'⬇ CSV Dataset…','⬇ Reporte JSON previo…':'⬇ Previous JSON Report…',
'modo rápido':'quick mode','Gráfica:':'Plot:','Carpeta de destino':'Destination folder',
'Cómo usar Doehlert Suite':'How to use Doehlert Suite',
'FLUJO RECOMENDADO':'RECOMMENDED WORKFLOW',
'Pestaña':'Tab', 'define los tres factores y genera el diseño.':'define the three factors and generate the design.',
'Exporta la planilla CSV y completa la columna Respuesta_Y en Excel.':'Export the CSV template and fill in the Response_Y column in your spreadsheet.',
'Menú Archivo > Importar datos (CSV), o pestaña 2 > Cargar CSV.':'File > Import experimental data (CSV), or Tab 2 > Load CSV.',
'Revisa superficies, ANOVA y diagnóstico en las pestañas 2, 3 y 4.':'Review response surfaces, ANOVA, and diagnostics in Tabs 2, 3, and 4.',
'Pestaña 5: ejecuta la auditoría y consulta sus hallazgos.':'Tab 5: run the audit and review its findings.',
'Los resultados se guardan por defecto en Exportaciones/AÑO-MES-DÍA.':'Results are saved by default under Exports/YYYY-MM-DD.',
'Los CSV de práctica están en la carpeta Examples.':'Sample CSV files are in the Examples folder.',
'Error interno':'Internal error',
'Se produjo un error. Genera el LOG de depuración para revisarlo.':'An error occurred. Generate the debug log to investigate it.',
'Configuración de Variables':'Factor Settings',
'Factor / Nivel':'Factor / Level','Nombre':'Name','Mínimo':'Minimum','Máximo':'Maximum',
'Réplicas del centro:':'Center-point replicates:',
'1. Calcular Matriz y Ver 3D':'1. Generate Design Matrix and View 3D',
'2. Simular Respuestas de Prueba':'2. Simulate Sample Responses',
'3. Enviar a Análisis (Pest. 2)':'3. Send to Analysis (Tab 2)',
'Exportar CSV...':'Export CSV...','Cargar CSV con Respuestas...':'Load Response CSV...',
'Eje X:':'X axis:','Eje Y:':'Y axis:','Actualizar Superficie':'Update Surface',
'Corte 2D con Bandas...':'2D Slice with Confidence Bands...','Guardar Gráfico...':'Save Plot...',
'Exportar Informe PDF...':'Export PDF Report...','Importar datos (CSV)':'Import Data (CSV)',
'Exportar informe PDF':'Export PDF Report','Guía rápida':'Quick Start',
'Los archivos generados se organizan en Exportaciones por fecha.':'Generated files are organized by date under Exports.',
'1. Diseño':'1. Design','2. Análisis y óptimo':'2. Analysis and Optimum',
'3. ANOVA y coeficientes':'3. ANOVA and Coefficients','4. Diagnóstico':'4. Diagnostics',
'5. Auditoría y varias respuestas':'5. Audit and Multiple Responses',
'Importar datos experimentales (CSV)…':'Import Experimental Data (CSV)…',
'Importar dataset para auditoría (CSV)…':'Import Dataset for Audit (CSV)…',
'Importar auditoría anterior (JSON)…':'Import Previous Audit (JSON)…',
'Abrir carpeta de ejemplos':'Open Examples Folder','Abrir carpeta de exportaciones':'Open Exports Folder',
'Salir':'Exit','Archivo':'File','Planilla del diseño (CSV)…':'Design Template (CSV)…',
'Reporte ejecutivo (MD/TXT)…':'Executive Report (MD/TXT)…','Tablas (Excel/CSV)…':'Tables (Excel/CSV)…',
'Gráfico actual (PNG/PDF/SVG)…':'Current Plot (PNG/PDF/SVG)…','Informe completo (PDF)…':'Full Report (PDF)…',
'Exportar':'Export','Guía rápida de uso':'Quick Start Guide','Ayuda':'Help',
'Matriz (Doble clic en Respuesta_Y para editar valor)':'Design Matrix (double-click Response_Y to edit)',
'Geometría Esférica de Doehlert (3D)':'Doehlert Spherical Geometry (3D)',
'Datos y Ajuste':'Data and Model Fit','Ningún dato procesado aún.':'No data have been processed yet.',
'Resumen Estadístico y Óptimo':'Statistical Summary and Optimum',
'Superficie de Respuesta 3D y Contornos 2D':'3D Response Surface and 2D Contours',
'Tabla ANOVA con descomposición del error residual':'ANOVA Table with Residual Error Decomposition',
'Coeficientes con inferencia (error estándar, t, p-valor bilateral, IC 95%)':'Coefficients and Inference (standard error, t, two-sided p-value, 95% CI)',
'Prueba de Falta de Ajuste (Lack of Fit) y error puro':'Lack-of-Fit Test and Pure Error',
'Diagnóstico corrida a corrida (selecciona una fila para resaltarla en los gráficos)':'Run-by-Run Diagnostics (select a row to highlight it in the plots)',
'Sin modelo ajustado.':'No model has been fitted.','Validación de Supuestos':'Assumption Checks',
'Hallazgos · PASS / WARN / FAIL / SKIP':'Findings · PASS / WARN / FAIL / SKIP',
'Prueba':'Test','Estado':'Status','Sección':'Section','Detalle y métricas':'Details and Metrics',
'Gráficas':'Plots','Regresiones':'Regressions','Informe completo':'Full Report',
'Gráfica':'Plot','Descripción':'Description','Óptimo':'Optimum',
'Puntos periféricos':'Peripheral Points','Réplicas del centro':'Center Replicates',
'IP 95% (Medición Individual)':'95% PI (Individual Prediction)',
'IC 95% (Media Estimada)':'95% CI (Estimated Mean)',
'Respuesta Predicha':'Predicted Response','Respuesta (Y)':'Response (Y)',
'Respuesta Estimada (Y)':'Estimated Response (Y)',
'Perfil de Sensibilidad y Bandas de Error:':'Sensitivity Profile and Error Bands:',
'Valores ajustados':'Fitted Values','Residuo estudentizado':'Studentized Residual',
'Cuantiles teóricos normales':'Theoretical Normal Quantiles',
'Residuos estudentizados ordenados':'Ordered Studentized Residuals',
'Orden de ejecución (nº de corrida)':'Run Order','Superficie 3D':'3D Surface','Contornos 2D':'2D Contours',
'Distribución Esférica de Doehlert':'Doehlert Spherical Distribution',
'Primero genera o carga una matriz.':'Generate or load a design matrix first.',
'Primero genera la matriz de diseño.':'Generate the design matrix first.',
'Primero debes ajustar el modelo y calcular el óptimo.':'Fit the model and calculate the optimum first.',
'Carga datos y ajusta el modelo antes de exportar.':'Load data and fit the model before exporting.',
'Selecciona dos factores distintos.':'Select two different factors.',
'Ingresa un número válido.':'Enter a valid number.',
'Las réplicas del centro deben ser un entero entre 1 y 10.':'Center-point replicates must be an integer between 1 and 10.',
'Error':'Error','Atención':'Warning','Sin datos':'No Data','Datos no válidos':'Invalid Data',
'Error al ajustar':'Fit Error','Simulación Exitosa':'Simulation Successful','Exportado':'Exported',
'Listo':'Done','Reporte generado':'Report Generated','Sin gráfico':'No Plot',
'Formato no soportado':'Unsupported Format','Auditoría':'Audit','Sin auditoría':'No Audit',
'Primero ejecuta la auditoría completa.':'Run the full audit first.',
'Preparando auditoría…':'Preparing audit…','Modo rápido':'Quick Mode',
'▶ Ejecutar auditoría completa':'▶ Run Full Audit','Importar dataset CSV…':'Import CSV Dataset…',
'Importar reporte JSON…':'Import JSON Report…','Exportar paquete completo…':'Export Full Package…',
'No se encuentra auditor/audit_doehlert.py':'auditor/audit_doehlert.py was not found',
'Importar datos experimentales':'Import Experimental Data','Importar dataset para auditoría':'Import Dataset for Audit',
'Importar auditoría anterior':'Import Previous Audit','Importar dataset':'Import Dataset',
'Importar reporte':'Import Report','Exportación completa':'Full Export',
'El paquete se guardó automáticamente en:':'The package was saved automatically to:',
'No se pudo iniciar:':'Could not start:','Error durante la auditoría':'Audit Error',
'No se pudo guardar:':'Could not save:','No se pudo guardar la imagen:':'Could not save the image:',
'No se pudo guardar el reporte:':'Could not save the report:',
'No se pudieron exportar las tablas:':'Could not export the tables:',
'No se pudo exportar el PDF:':'Could not export the PDF:',
'openpyxl no disponible':'openpyxl is not available',
'Archivo guardado con punto decimal:':'File saved using a decimal point:',
'Archivo CSV (decimal con punto)':'CSV File (decimal point)',
'Guardar planilla del diseño':'Save Design Template','Guardar log de depuración':'Save Debug Log',
'Guardar gráfico':'Save Plot','Guardar reporte ejecutivo':'Save Executive Report',
'Exportar tablas del modelo':'Export Model Tables','Guardar informe completo':'Save Full Report',
'No se pudo abrir automáticamente. La carpeta está en:':'Could not open automatically. The folder is located at:',
'Error al procesar CSV':'CSV Processing Error','Error importando CSV':'CSV Import Error',
'Error al generar LOG':'Debug Log Error','Log de depuración':'Debug Log',
'GENERAR LOG DE DEPURACIÓN...':'GENERATE DEBUG LOG...','Guardar Gráficos de Residuos...':'Save Residual Plots...',
'Comprueba la calidad del análisis y compara resultados anteriores':'Check analysis quality and compare previous results',
'Primero importa un CSV o ejecuta la auditoría completa. Los detalles aparecen en las pestañas inferiores.':'First import a CSV or run the full audit. Details are shown in the tabs below.',
'Variable del eje X (recorrido):':'X-axis variable (profile):',
'Guardar Gráfico de Corte (PNG/PDF)...':'Save Slice Plot (PNG/PDF)...','Actualizar Corte':'Update Slice',
'Exportado':'Exported','Reporte guardado en:':'Report saved to:',
'Informe PDF generado en:':'PDF report generated at:',
'Log guardado correctamente en:':'Log successfully saved to:',
'El LOG de depuración conserva el traceback.':'The debug log contains the traceback.',
'Los archivos generados se organizan en Exportaciones por fecha.':'Generated files are organized by date under Exports.',
}


def get_language() -> str:
    global _LANGUAGE_CACHE
    if _LANGUAGE_CACHE is not None:
        return _LANGUAGE_CACHE
    try:
        data = json.loads(_CONFIG.read_text(encoding='utf-8'))
        _LANGUAGE_CACHE = data.get('language', 'en') if data.get('language') in LANGUAGES else 'en'
    except Exception:
        _LANGUAGE_CACHE = 'en'
    return _LANGUAGE_CACHE


def set_language(language: str) -> None:
    global _LANGUAGE_CACHE
    if language not in LANGUAGES:
        raise ValueError(f'Unsupported language: {language}')
    _CONFIG.write_text(json.dumps({'language': language}, indent=2), encoding='utf-8')
    _LANGUAGE_CACHE = language


_ES = {
    'Language': 'Idioma', 'English': 'Inglés', 'Examples': 'Ejemplos',
    'Exports': 'Exportaciones', 'File': 'Archivo', 'Export': 'Exportar', 'Help': 'Ayuda',
    'Design': 'Diseño', 'Analysis and Optimum': 'Análisis y óptimo',
    'ANOVA and Coefficients': 'ANOVA y coeficientes', 'Diagnostics': 'Diagnóstico',
    'Audit and Multiple Responses': 'Auditoría y varias respuestas',
    'Home': 'Inicio', 'Back': 'Atrás', 'Forward': 'Adelante', 'Pan': 'Desplazar',
    'Zoom': 'Zoom', 'Subplots': 'Subgráficos', 'Save': 'Guardar',
    'Application started': 'Aplicación iniciada',
}

def tr(value):
    if not isinstance(value, str):
        return value
    if get_language() == 'es':
        return _ES.get(value, value)
    if value in _EN:
        return _EN[value]
    stripped = value.strip()
    if stripped in _EN:
        return value[:len(value)-len(value.lstrip())] + _EN[stripped] + value[len(value.rstrip()):]
    # Translate stable phrases embedded in dynamic messages and window titles.
    translated = value
    for source, target in sorted(_EN.items(), key=lambda item: len(item[0]), reverse=True):
        if len(source) >= 5 and source in translated:
            translated = translated.replace(source, target)
    for source, target in (('GL', 'df'), ('SC', 'SS'), ('CM', 'MS'), ('EE', 'SE'), ('SÍ', 'YES')):
        translated = re.sub(r'\b' + re.escape(source) + r'\b', target, translated)
    return translated


def install_tkinter_translations(tk, ttk, messagebox, filedialog):
    """Translate Tk widget labels, menus, dialogs and table headings."""
    def patch_kw_method(cls, method_name, keys):
        original = getattr(cls, method_name)
        def wrapped(self, *args, **kwargs):
            for key in keys:
                if key in kwargs:
                    kwargs[key] = tr(kwargs[key])
            return original(self, *args, **kwargs)
        setattr(cls, method_name, wrapped)

    for cls in (ttk.Label, ttk.Button, ttk.LabelFrame, ttk.Checkbutton, ttk.Radiobutton, ttk.Menubutton, tk.Label, tk.Button, tk.LabelFrame, tk.Checkbutton, tk.Radiobutton):
        original = cls.__init__
        def make_init(orig):
            def init(self, *args, **kwargs):
                if 'text' in kwargs:
                    kwargs['text'] = tr(kwargs['text'])
                if 'values' in kwargs:
                    kwargs['values'] = tuple(tr(v) for v in kwargs['values'])
                return orig(self, *args, **kwargs)
            return init
        cls.__init__ = make_init(original)

    original_combo_init = ttk.Combobox.__init__
    def combo_init(self, *args, **kwargs):
        if 'values' in kwargs:
            kwargs['values'] = tuple(tr(v) for v in kwargs['values'])
        return original_combo_init(self, *args, **kwargs)
    ttk.Combobox.__init__ = combo_init

    patch_kw_method(tk.Menu, 'add_command', ('label',))
    patch_kw_method(tk.Menu, 'add_cascade', ('label',))
    patch_kw_method(ttk.Notebook, 'add', ('text',))
    patch_kw_method(ttk.Treeview, 'heading', ('text',))
    patch_kw_method(ttk.Treeview, 'insert', ('text',))
    patch_kw_method(ttk.Treeview, 'column', ('text',))

    # StringVar is used for status labels and other dynamic captions.
    original_stringvar_init = tk.StringVar.__init__
    def stringvar_init(self, *args, **kwargs):
        if 'value' in kwargs:
            kwargs['value'] = tr(kwargs['value'])
        return original_stringvar_init(self, *args, **kwargs)
    tk.StringVar.__init__ = stringvar_init
    original_variable_set = tk.Variable.set
    def variable_set(self, value):
        return original_variable_set(self, tr(value))
    tk.Variable.set = variable_set

    # Text changed after widget creation (status labels, progress captions, etc.).
    original_configure = tk.Misc.configure
    def configure(self, cnf=None, **kwargs):
        if 'text' in kwargs:
            kwargs['text'] = tr(kwargs['text'])
        return original_configure(self, cnf, **kwargs)
    tk.Misc.configure = configure
    tk.Misc.config = configure

    for name in ('showinfo', 'showwarning', 'showerror', 'askyesno', 'askokcancel'):
        original = getattr(messagebox, name)
        def make_dialog(orig):
            def dialog(title=None, message=None, **kwargs):
                return orig(tr(title), tr(message), **kwargs)
            return dialog
        setattr(messagebox, name, make_dialog(original))

    # Matplotlib plot titles, axis labels, and legend labels follow the selected language.
    try:
        from matplotlib.axes import Axes
        for method_name in ('set_title', 'set_xlabel', 'set_ylabel', 'set_zlabel'):
            original = getattr(Axes, method_name, None)
            if original:
                def make_axes_method(orig):
                    def axes_method(self, label, *args, **kwargs):
                        return orig(self, tr(label), *args, **kwargs)
                    return axes_method
                setattr(Axes, method_name, make_axes_method(original))
        for method_name in ('plot', 'scatter', 'errorbar', 'fill_between'):
            original = getattr(Axes, method_name, None)
            if original:
                def make_plot_method(orig):
                    def plot_method(self, *args, **kwargs):
                        if 'label' in kwargs:
                            kwargs['label'] = tr(kwargs['label'])
                        return orig(self, *args, **kwargs)
                    return plot_method
                setattr(Axes, method_name, make_plot_method(original))
    except Exception:
        pass

    for name in ('askopenfilename', 'asksaveasfilename', 'askdirectory'):
        original = getattr(filedialog, name)
        def make_file_dialog(orig):
            def dialog(**kwargs):
                if 'title' in kwargs:
                    kwargs['title'] = tr(kwargs['title'])
                if 'filetypes' in kwargs:
                    kwargs['filetypes'] = [(tr(label), pattern) for label, pattern in kwargs['filetypes']]
                return orig(**kwargs)
            return dialog
        setattr(filedialog, name, make_file_dialog(original))
