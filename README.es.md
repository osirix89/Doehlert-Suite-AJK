# Doehlert Suite 0.1.0

**El idioma predeterminado de la aplicación es inglés. El español se puede seleccionar desde el menú _Language_.** La documentación principal en inglés está en [`README.md`](README.md); los manuales completos en español están en `docs/es/`.

Doehlert Suite es una aplicación de escritorio de código abierto, desarrollada en Python, para generar y analizar diseños experimentales Doehlert de tres factores. Ajusta modelos cuadráticos de superficie de respuesta, presenta ANOVA y diagnósticos, realiza optimización restringida, admite múltiples respuestas e incorpora una herramienta de auditoría.

> **Estado:** 0.1.0 es la primera versión pública. Los resultados son una ayuda para el análisis y no sustituyen la validación experimental, el conocimiento del dominio ni una revisión estadística independiente.

## Funcionalidades

- Generación y visualización de diseños Doehlert de tres factores.
- Importación de datos experimentales desde CSV.
- Ajuste de modelos cuadráticos completos de segundo orden.
- ANOVA, inferencia de coeficientes, falta de ajuste y diagnóstico de residuos.
- Búsqueda de máximos y mínimos restringidos al dominio experimental admitido.
- Importación y análisis de múltiples respuestas desde un CSV de formato ancho.
- Auditoría integrada y exportación de evidencias.
- Cambio de idioma de la interfaz entre inglés (predeterminado) y español.
- Datasets sintéticos y casos de validación en `Examples/`.

## Requisitos

- Python 3.10 o posterior.
- Tkinter. En algunas distribuciones Linux debe instalarse desde el gestor de paquetes (Debian/Ubuntu: `sudo apt install python3-tk`).
- NumPy, Matplotlib y SciPy (ver `requirements.txt`).
- `openpyxl` y `statsmodels` son dependencias opcionales para funciones auxiliares compatibles.

## Instalación

```bash
git clone https://github.com/TU-USUARIO/Doehlert-Suite.git
cd Doehlert-Suite
python -m venv .venv
```

Activá el entorno:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instalá y ejecutá:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Inicio rápido

1. Abrí **1. Diseño**, definí nombres y límites físicos de los tres factores y generá la matriz.
2. Exportá el CSV del diseño e ingresá las respuestas medidas en la columna correspondiente.
3. Usá **Archivo → Importar datos experimentales (CSV)** para cargar el archivo completo.
4. Revisá el modelo, óptimo, ANOVA, coeficientes y diagnósticos en las pestañas de análisis.
5. Utilizá la pestaña de auditoría y multirrespuesta cuando corresponda.
6. Exportá informes y tablas. Los nuevos archivos se agrupan en `Exports/AAAA-MM-DD/`.

Los archivos de `Examples/` son datasets sintéticos de demostración o validación. No representan mediciones experimentales reales.

## Selección de idioma

La interfaz inicia en inglés. Desde la interfaz inicial en inglés, elegí **Language → Español** para cambiar a español. La aplicación solicitará reiniciarse para reconstruir de forma consistente todas las ventanas y etiquetas. La selección se guarda en el directorio personal del usuario, fuera del repositorio. Para volver al inglés, elegí **Idioma → Inglés**.

## Manuales y documentación

- [Manual técnico y de usuario en español (TXT)](docs/es/TECHNICAL_MANUAL.txt)
- [Documentación técnica en español (PDF)](docs/Documentacion_Tecnica_ES.pdf)
- [Guía rápida en español](docs/es/QUICK_START.txt)
- [Technical and user manual in English (TXT)](docs/en/TECHNICAL_MANUAL.txt)
- [Technical documentation in English (PDF)](docs/Technical_Documentation_EN.pdf)
- [Quick start in English](docs/en/QUICK_START.txt)
- [Índice de documentación](docs/README.es.md)

## Pruebas automáticas

Desde la raíz del repositorio:

```bash
python -m unittest discover -s auditor/tests -v
```

Las pruebas cubren aspectos seleccionados del dominio, optimización, multirrespuesta y comparación con resultados publicados. Que todas pasen no demuestra que todos los datasets o flujos posibles estén libres de errores.

## Estructura del repositorio

```text
Doehlert-Suite/
├── main.py                     # Interfaz gráfica
├── i18n.py                     # Localización inglés/español
├── doehlert_engine.py          # Motor estadístico y ajuste
├── doehlert_domain.py          # Geometría del dominio experimental
├── doehlert_optimization.py    # Optimización restringida
├── doehlert_responses.py       # Manejo de CSV multirrespuesta
├── doehlert_publication.py     # Comparación con publicaciones
├── auditor/                    # Auditoría integrada y pruebas
├── Examples/                   # CSV de demostración y validación
├── docs/en/                    # Manuales en inglés
├── docs/es/                    # Manuales en español
├── requirements.txt
├── LICENSE                     # GPL-3.0-only
└── CHANGELOG.md
```

`Exports/` se crea al ejecutar la aplicación y no se versiona. Los temporales, cachés y preferencias locales quedan excluidos de Git.

## Alcance y limitaciones

- La implementación actual admite diseños de tres factores.
- La calidad del modelo depende de la cantidad, calidad y estructura de los datos ingresados.
- La optimización numérica depende del modelo ajustado y de las restricciones configuradas.
- Los ejemplos sintéticos no sustituyen la validación con datos experimentales propios.
- La versión 0.1.0 es una primera publicación. Al informar un error, incluí pasos para reproducirlo, sistema operativo, versión de Python y versiones de dependencias. No adjuntes datos confidenciales.

## Licencia

Doehlert Suite se distribuye bajo **GNU General Public License versión 3.0 únicamente (GPL-3.0-only)**. Podés usar, estudiar, modificar y redistribuir el programa respetando esa licencia. Las redistribuciones y obras derivadas cubiertas deben cumplir la GPL, incluidos sus requisitos sobre disponibilidad del código fuente. Consultá [`LICENSE`](LICENSE). Las dependencias de terceros conservan sus propias licencias.

## Contribuciones

Las contribuciones son bienvenidas. Abrí un issue describiendo el problema o enviá un pull request con una explicación breve y pruebas cuando corresponda. Mantené documentadas las fórmulas, los supuestos estadísticos y las restricciones del dominio.
