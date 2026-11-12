#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""
audit_doehlert.py  —  Auditor formal de la Suite Doehlert (main.py)
====================================================================

Valida, con NumPy/SciPy como "estándar dorado", la lógica numérica de la Suite
Doehlert (k = 3): funciones de distribución nativas, geometría del diseño,
mínimos cuadrados (OLS) de 2.º orden, análisis canónico + método Delta, ANOVA de
falta de ajuste (lack-of-fit) y entrada/salida de CSV.

El auditor NO abre la ventana de la Suite: importa main.py como módulo y ejecuta
sus métodos sobre una "app sin interfaz" (headless) con widgets simulados.

USO RÁPIDO
----------
    python3 audit_doehlert.py                    # audita, imprime checklist y exporta todo
    python3 audit_doehlert.py --gui              # abre el visor gráfico (Tkinter)
    python3 audit_doehlert.py --main ruta/main.py --out informe/
    python3 audit_doehlert.py --import-data mis_datos.csv        # audita TU dataset real
    python3 audit_doehlert.py --import-report auditoria_previa.json   # compara con una corrida previa
    python3 audit_doehlert.py --export html,pdf,json,csv --quick

SALIDAS (carpeta --out, por defecto ./audit_output_<fecha>/)
    audit_report.html   informe visual autocontenido (gráficos incrustados, filtros)
    audit_report.pdf    informe multipágina con todas las gráficas
    audit_report.json   resultados completos (re-importable con --import-report)
    audit_report.csv    checklist plano (una fila por hallazgo)
    audit_log.txt       salida de consola
    figures/*.png       cada gráfica por separado
    datasets/*.csv      datasets sintéticos en el formato de la Suite (importables en main.py)

CÓDIGO DE SALIDA: 0 si no hay [FAIL]; 1 si hay al menos un [FAIL].

Requisitos: numpy, scipy, matplotlib  (opcional: statsmodels; Tkinter para --gui).
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import csv
import datetime as dt
import hashlib
import html as html_lib
import importlib.util
import inspect
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
import textwrap
import time
import traceback
import types
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

try:  # SciPy es el "ground truth": sin él no hay auditoría posible
    from scipy import linalg as sla
    from scipy import special, stats
except ImportError:  # pragma: no cover
    sys.exit("[ERROR] Este auditor requiere SciPy (pip install scipy).")

try:  # statsmodels es opcional (segunda opinión independiente para OLS)
    import statsmodels.api as sm  # type: ignore
    HAVE_SM = True
except Exception:  # pragma: no cover
    sm = None
    HAVE_SM = False

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA = "doehlert-audit/1"            # versión del formato JSON (importable)
TOL_ABS = 1e-5                          # tolerancia absoluta (especificación)
TOL_REL = 1e-5                          # tolerancia relativa (especificación)
WARN_FACTOR = 100.0                     # error entre TOL y 100·TOL => [WARN]; más => [FAIL]
EPS = np.finfo(float).eps               # precisión de máquina (2.2e-16)
PRINT_TOL = 6e-5                        # los reportes de main.py imprimen 4 decimales => ±5e-5

STATUS_ORDER = ("FAIL", "WARN", "PASS", "SKIP")
COLORS = {"PASS": "#2e7d32", "FAIL": "#c62828", "WARN": "#ef6c00", "SKIP": "#757575"}
ANSI = {"PASS": "\033[92m", "FAIL": "\033[91m", "WARN": "\033[93m", "SKIP": "\033[90m", "END": "\033[0m"}

# Rangos de factores usados por los datasets sintéticos (Temperatura, Tiempo, pH)
DEFAULT_RANGES = [(50.0, 90.0), (10.0, 30.0), (3.0, 7.0)]
# Coeficientes publicados del algoritmo de Beasley-Springer-Moro (Moro, 1995)
BSM_A = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106046337]
BSM_B = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
BSM_C = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810338635,
         0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364,
         0.0000003960315187]
# Nombres de los 10 coeficientes en el orden de main.py: M = [1, x1, x2, x3, x1x2, x1x3, x2x3, x1², x2², x3²]
BETA_NAMES = ["b0", "b1", "b2", "b3", "b12", "b13", "b23", "b11", "b22", "b33"]


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES GENERALES
# ─────────────────────────────────────────────────────────────────────────────
def jsonable(o: Any) -> Any:
    """Convierte NumPy/NaN/inf a tipos JSON válidos (None para no finitos)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def fmt(x: Any, nd: int = 3) -> str:
    """Formato compacto para métricas (notación científica si hace falta)."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "nan" if math.isnan(xf) else ("inf" if xf > 0 else "-inf")
    if xf != 0 and (abs(xf) < 1e-3 or abs(xf) >= 1e6):
        return f"{xf:.{nd}e}"
    return f"{xf:.{nd + 1}g}" if abs(xf) < 1000 else f"{xf:.1f}"


def graded_status(err: float, tol: float, warn_factor: float = WARN_FACTOR) -> str:
    """PASS si err ≤ tol; WARN si err ≤ warn_factor·tol; FAIL si es mayor o no es finito."""
    if err is None or not math.isfinite(err):
        return "FAIL"
    if err <= tol:
        return "PASS"
    return "WARN" if err <= warn_factor * tol else "FAIL"


def worst(*statuses: str) -> str:
    """Estado más grave de una lista (FAIL > WARN > PASS > SKIP)."""
    for s in STATUS_ORDER:
        if s in statuses:
            return s
    return "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
#  MODELO DE DATOS DEL AUDITOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    """Un resultado de auditoría (una línea del checklist)."""
    id: str
    section: str
    title: str
    status: str                       # PASS | FAIL | WARN | SKIP
    summary: str = ""                 # una línea
    details: str = ""                 # texto multilínea (tablas, valores)
    metrics: Dict[str, Any] = field(default_factory=dict)
    where: str = ""                   # ubicación en el código auditado
    fix: str = ""                     # corrección sugerida
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return jsonable(self.__dict__.copy())


@dataclass
class RunResult:
    """Resultado de ejecutar ejecutar_ajuste_completo() sobre un dataset."""
    name: str
    X: np.ndarray
    y: np.ndarray
    cod: np.ndarray
    error: Optional[str] = None
    betas: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    s_res: float = float("nan")
    t_crit: float = float("nan")
    x_opt: Optional[np.ndarray] = None
    ic_coords: list = field(default_factory=list)
    y_pred: Optional[np.ndarray] = None
    report: str = ""
    diag: str = ""
    rep: Dict[str, Any] = field(default_factory=dict)      # resultados del reporte (pestaña 2): del objeto `fit` o parseados del texto
    lof: Dict[str, Any] = field(default_factory=dict)      # resultados del diagnóstico (pestaña 3)
    # Espacio en el que el programa ajusta el modelo:
    #   "natural": β en unidades físicas (versión antigua)  |  "coded": β en variables codificadas de Doehlert (motor nuevo)
    space: str = "natural"
    Z: Optional[np.ndarray] = None                         # matriz de factores sobre la que se ajustó el modelo
    centers: Optional[np.ndarray] = None                   # real = centro + escala · codificado
    scales: Optional[np.ndarray] = None
    fit: Any = None                                        # objeto ModelFit del motor nuevo (si existe)

    def to_real(self, x):
        """Convierte puntos del espacio del modelo a unidades físicas (identidad si el modelo es natural)."""
        x = np.asarray(x, float)
        return x if self.space == "natural" else self.centers + x * self.scales

    def to_model(self, x):
        """Convierte puntos físicos al espacio del modelo."""
        x = np.asarray(x, float)
        return x if self.space == "natural" else (x - self.centers) / self.scales

    def scale_vec(self):
        """Factor con que se amplifican las desviaciones del modelo al pasar a unidades físicas."""
        return np.ones(3) if self.space == "natural" else np.abs(self.scales)


# ─────────────────────────────────────────────────────────────────────────────
#  CARGA DE main.py SIN ABRIR TKINTER
# ─────────────────────────────────────────────────────────────────────────────
def _install_tk_stubs() -> None:
    """Si Tkinter no está disponible, instala módulos falsos para poder importar main.py."""
    tk = types.ModuleType("tkinter")

    class _Tk:  # clase base falsa para `class DoehlertSuiteApp(tk.Tk)`
        pass

    tk.Tk = _Tk
    tk.filedialog = types.ModuleType("tkinter.filedialog")
    tk.messagebox = types.ModuleType("tkinter.messagebox")
    tk.ttk = types.ModuleType("tkinter.ttk")
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.filedialog"] = tk.filedialog
    sys.modules["tkinter.messagebox"] = tk.messagebox
    sys.modules["tkinter.ttk"] = tk.ttk
    be = types.ModuleType("matplotlib.backends.backend_tkagg")
    be.FigureCanvasTkAgg = object
    be.NavigationToolbar2Tk = object
    sys.modules["matplotlib.backends.backend_tkagg"] = be


def find_default_main() -> Optional[str]:
    """Busca el main*.py más reciente (por fecha de modificación) junto al auditor o en el directorio actual."""
    cands: List[str] = []
    for base in {os.path.dirname(os.path.abspath(__file__)), os.getcwd()}:
        try:
            for fn in os.listdir(base):
                if re.fullmatch(r"main(_\w+)?\.py", fn) and "audit" not in fn:
                    cands.append(os.path.join(base, fn))
        except OSError:
            pass
    return max(cands, key=os.path.getmtime) if cands else None


def load_target(path: str):
    """Importa main.py como módulo (sin ejecutar el event loop: está bajo `if __name__ == '__main__'`)."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo a auditar: {path}")

    tdir = os.path.dirname(path)
    if tdir not in sys.path:
        sys.path.insert(0, tdir)                     # permite `import doehlert_engine` (motor separado)
    for name, m_ in list(sys.modules.items()):       # evita reutilizar una versión vieja del motor entre corridas
        f_ = getattr(m_, "__file__", None)
        if f_ and os.path.dirname(os.path.abspath(f_)) == tdir and name not in ("__main__",) and not name.startswith("audit_doehlert"):
            sys.modules.pop(name, None)

    def _exec():
        spec = importlib.util.spec_from_file_location("main_auditado", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["main_auditado"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    try:
        import tkinter  # noqa: F401  (si existe, se usa el real; solo se evita crear la ventana)
        return _exec()
    except Exception:
        _install_tk_stubs()           # sin Tk (servidor/CI) o fallo de backend: se simulan los módulos
        return _exec()


class _FakeText:
    """Imita tk.Text (solo delete/insert/get): captura el texto que main.py escribe en sus reportes."""
    def __init__(self) -> None:
        self.buf = ""

    def delete(self, *_a, **_k) -> None:
        self.buf = ""

    def insert(self, _pos, s, *_a, **_k) -> None:
        self.buf += s

    def get(self, *_a, **_k) -> str:
        return self.buf


class _FakeVar:
    """Imita Entry/Spinbox: solo .get()."""
    def __init__(self, v: Any) -> None:
        self.v = v

    def get(self) -> str:
        return str(self.v)


_WIDGETS = ["tree_diseno", "btn_exportar", "btn_simular", "btn_enviar_directo", "ax_diseno_3d", "fig_diseno",
            "canvas_diseno", "lbl_estado", "cbo_eje_x", "cbo_eje_y", "notebook", "tab_analisis"]


def make_headless_app(mod):
    """
    Crea una instancia de DoehlertSuiteApp SIN Tk: se copian sus métodos a una clase simple
    (evita tk.Tk.__init__ y su __getattr__) y se inyecta el estado inicial + widgets simulados.
    """
    cls = mod.DoehlertSuiteApp
    methods = {k: v for k, v in vars(cls).items() if inspect.isfunction(v) and k != "__init__"}
    Headless = type("HeadlessApp", (), methods)
    app = Headless()
    # Estado inicial: espejo de DoehlertSuiteApp.__init__
    app.matriz_generada, app.headers_diseno, app.factores_activos = [], [], []
    app.betas = app.cov_beta = app.X_exp = app.y_exp = app.y_pred = None
    app.residuos = app.residuos_std = app.x_opt = None
    app.cod_exp = None
    app.ic_coords, app.t_crit, app.s_res = [], 1.96, 1.0
    app.nombres_factores = ["X1", "X2", "X3"]
    app.limites_factores = [(0.0, 1.0)] * 3
    app.txt_reporte, app.txt_diag = _FakeText(), _FakeText()
    for w in _WIDGETS:
        setattr(app, w, MagicMock())
    # Los métodos de graficación no forman parte de la auditoría numérica
    app.actualizar_graficos_superficie = lambda *a, **k: None
    app.actualizar_graficos_diagnostico = lambda *a, **k: None
    return app


@contextlib.contextmanager
def patched_dialogs(mod, open_path: Optional[str] = None, save_path: Optional[str] = None):
    """Reemplaza filedialog/messagebox de main.py por versiones que devuelven rutas fijas y registran avisos."""
    log: List[Tuple[str, str, str]] = []
    mb, fd = getattr(mod, "messagebox"), getattr(mod, "filedialog")
    names = ["showinfo", "showerror", "showwarning"]
    saved = {n: getattr(mb, n, None) for n in names}
    saved_fd = {n: getattr(fd, n, None) for n in ("askopenfilename", "asksaveasfilename")}
    for n in names:
        setattr(mb, n, lambda title="", msg="", *a, _n=n, **k: log.append((_n, str(title), str(msg))))
    fd.askopenfilename = lambda *a, **k: open_path or ""
    fd.asksaveasfilename = lambda *a, **k: save_path or ""
    try:
        yield log
    finally:
        for n, f in saved.items():
            if f is not None:
                setattr(mb, n, f)
        for n, f in saved_fd.items():
            if f is not None:
                setattr(fd, n, f)


# ─────────────────────────────────────────────────────────────────────────────
#  REFERENCIA MATEMÁTICA INDEPENDIENTE  (el "estándar dorado" del auditor)
# ─────────────────────────────────────────────────────────────────────────────
# Modelo cuadrático completo de 3 factores:
#     f(x) = b0 + Σ b_i x_i + Σ_{i<j} b_ij x_i x_j + Σ b_ii x_i²  =  b0 + bᵀx + ½ xᵀ H x
# con el hessiano  H = [[2 b11, b12, b13], [b12, 2 b22, b23], [b13, b23, 2 b33]].
#   ∇f(x) = b + H x           =>  punto estacionario  x* = −H⁻¹ b      (¡sin factor ½!)
# (La forma equivalente x* = −½ B⁻¹ b es válida SOLO si B = H/2, es decir diag(b_ii) y b_ij/2 fuera de la diagonal.)
def design_matrix(X) -> np.ndarray:
    """Matriz de diseño M (N×10) en el orden de main.py: [1, x1, x2, x3, x1x2, x1x3, x2x3, x1², x2², x3²]."""
    X = np.atleast_2d(np.asarray(X, float))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([np.ones(len(X)), x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2])


def hessian_of(b: np.ndarray) -> np.ndarray:
    """Hessiano H(β) del polinomio cuadrático (constante: no depende de x)."""
    return np.array([[2 * b[7], b[4], b[5]],
                     [b[4], 2 * b[8], b[6]],
                     [b[5], b[6], 2 * b[9]]])


def poly_val(b: np.ndarray, x) -> float:
    """f(x) = m(x)ᵀ β."""
    return float(design_matrix(np.atleast_2d(x))[0] @ b)


def gradient_of(b: np.ndarray, x) -> np.ndarray:
    """∇f(x) = b_lin + H x  (derivada analítica)."""
    return np.asarray(b[1:4], float) + hessian_of(b) @ np.asarray(x, float)


def fd_gradient(b: np.ndarray, x, h: float = 1e-4) -> np.ndarray:
    """∇f por diferencias finitas centrales (verificación cruzada de la referencia): [f(x+h)−f(x−h)]/2h."""
    x = np.asarray(x, float)
    g = np.zeros(3)
    for i in range(3):
        step = h * max(1.0, abs(x[i]))
        e = np.zeros(3)
        e[i] = step
        g[i] = (poly_val(b, x + e) - poly_val(b, x - e)) / (2 * step)
    return g


def fd_hessian(b: np.ndarray, x, h: float = 1e-3) -> np.ndarray:
    """Hessiano numérico: derivada central del gradiente numérico."""
    x = np.asarray(x, float)
    H = np.zeros((3, 3))
    for j in range(3):
        step = h * max(1.0, abs(x[j]))
        e = np.zeros(3)
        e[j] = step
        H[:, j] = (fd_gradient(b, x + e, h) - fd_gradient(b, x - e, h)) / (2 * step)
    return 0.5 * (H + H.T)


def xstar_ref(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Punto estacionario de referencia: resuelve H x = −b_lin. Devuelve (x*, H); NaN si H es singular."""
    H = hessian_of(b)
    try:
        return np.linalg.solve(H, -np.asarray(b[1:4], float)), H
    except np.linalg.LinAlgError:
        return np.full(3, np.nan), H


def xstar_batch(B: np.ndarray) -> np.ndarray:
    """x*(β) vectorizado para n vectores β (n×10): H (n×3×3) y b (n×3) resueltos de una vez."""
    B = np.atleast_2d(B)
    n = len(B)
    H = np.zeros((n, 3, 3))
    H[:, 0, 0], H[:, 1, 1], H[:, 2, 2] = 2 * B[:, 7], 2 * B[:, 8], 2 * B[:, 9]
    H[:, 0, 1] = H[:, 1, 0] = B[:, 4]
    H[:, 0, 2] = H[:, 2, 0] = B[:, 5]
    H[:, 1, 2] = H[:, 2, 1] = B[:, 6]
    rhs = -B[:, 1:4][..., None]
    try:
        return np.linalg.solve(H, rhs)[..., 0]
    except np.linalg.LinAlgError:
        out = np.full((n, 3), np.nan)
        for k in range(n):
            try:
                out[k] = np.linalg.solve(H[k], rhs[k, :, 0])
            except np.linalg.LinAlgError:
                pass
        return out


def jacobian_analytic(b: np.ndarray) -> np.ndarray:
    """
    J = ∂x*/∂β (3×10) por el teorema de la función implícita.
    x* anula g(x,β) = H(β) x + b_lin(β)  =>  J = −H⁻¹ · ∂g/∂β   (evaluado en x*).
    Columnas de ∂g/∂β:  b0→0 | b_j (j=1..3)→e_j | b12→(x2,x1,0) | b13→(x3,0,x1) | b23→(0,x3,x2)
                        | b11→(2x1,0,0) | b22→(0,2x2,0) | b33→(0,0,2x3)
    """
    xs, H = xstar_ref(b)
    G = np.zeros((3, 10))
    G[:, 1:4] = np.eye(3)
    G[:, 4] = [xs[1], xs[0], 0.0]
    G[:, 5] = [xs[2], 0.0, xs[0]]
    G[:, 6] = [0.0, xs[2], xs[1]]
    G[0, 7], G[1, 8], G[2, 9] = 2 * xs[0], 2 * xs[1], 2 * xs[2]
    return -np.linalg.solve(H, G)


def jacobian_numeric(b: np.ndarray) -> np.ndarray:
    """J por diferencias finitas centrales de x*(β): columna k = [x*(β+h e_k) − x*(β−h e_k)]/2h."""
    J = np.zeros((3, 10))
    for k in range(10):
        h = 1e-6 * max(1.0, abs(b[k]))
        bp, bm = b.copy(), b.copy()
        bp[k] += h
        bm[k] -= h
        J[:, k] = (xstar_ref(bp)[0] - xstar_ref(bm)[0]) / (2 * h)
    return J


def ref_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """OLS de referencia por tres vías independientes: SciPy lstsq (SVD), QR y ecuaciones normales."""
    M = design_matrix(X)
    N, p = M.shape
    out: Dict[str, Any] = {"M": M, "N": N, "p": p, "df": N - p}
    out["beta"] = sla.lstsq(M, y)[0]                                   # SVD (gelsd)
    try:
        q, r = np.linalg.qr(M)
        out["beta_qr"] = sla.solve_triangular(r, q.T @ y)             # Householder-QR
    except Exception:
        out["beta_qr"] = np.full(p, np.nan)
    try:
        out["beta_ne"] = np.linalg.solve(M.T @ M, M.T @ y)            # (MᵀM)⁻¹ Mᵀ y
    except np.linalg.LinAlgError:
        out["beta_ne"] = np.full(p, np.nan)
    res = y - M @ out["beta"]
    out["resid"] = res
    out["ss_res"] = float(res @ res)
    out["ss_tot"] = float(((y - y.mean()) ** 2).sum())
    df = out["df"]
    out["s2"] = out["ss_res"] / df if df > 0 else float("nan")
    out["s"] = math.sqrt(out["s2"]) if df > 0 else float("nan")
    out["r2"] = 1 - out["ss_res"] / out["ss_tot"] if out["ss_tot"] > 0 else float("nan")
    out["r2a"] = (1 - (out["ss_res"] / df) / (out["ss_tot"] / (N - 1))) if df > 0 and out["ss_tot"] > 0 else float("nan")
    try:
        out["cov_inv"] = out["s2"] * np.linalg.inv(M.T @ M)           # s²·(MᵀM)⁻¹
    except np.linalg.LinAlgError:
        out["cov_inv"] = np.full((p, p), np.nan)
    out["cov_pinv"] = out["s2"] * np.linalg.pinv(M.T @ M)             # s²·pinv(MᵀM) (lo que usa main.py)
    out["cond_M"] = float(np.linalg.cond(M))
    out["cond_MtM"] = float(np.linalg.cond(M.T @ M))
    return out


def coded_reference(X: np.ndarray, y: np.ndarray, ranges: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Mismo modelo ajustado en variables codificadas (−1..+1): bien condicionado. Sirve para la invarianza de ŷ."""
    c = np.array([(a + b) / 2 for a, b in ranges])
    d = np.array([(b - a) / 2 for a, b in ranges])
    Xc = (X - c) / d
    Mc = design_matrix(Xc)
    beta = sla.lstsq(Mc, y)[0]
    return {"Xc": Xc, "Mc": Mc, "beta": beta, "yhat": Mc @ beta,
            "cond_M": float(np.linalg.cond(Mc)), "cond_MtM": float(np.linalg.cond(Mc.T @ Mc)),
            "cov_unscaled": np.linalg.inv(Mc.T @ Mc)}


def pure_error_ref(X: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """
    Error puro: agrupa corridas con factores idénticos (cualquier réplica, no solo el centro).
        SS_PE = Σ_g Σ_{i∈g} (y_i − ȳ_g)²      df_PE = Σ_g (n_g − 1)
    """
    Xr = np.round(np.asarray(X, float), 8)
    _, inv = np.unique(Xr, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    ss, df = 0.0, 0
    for g in np.unique(inv):
        yy = y[inv == g]
        if len(yy) > 1:
            ss += float(((yy - yy.mean()) ** 2).sum())
            df += len(yy) - 1
    return ss, df


def lof_reference(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """ANOVA de falta de ajuste de referencia (F de Fisher con p-valor de SciPy)."""
    rf = ref_fit(X, y)
    ss_pe, df_pe = pure_error_ref(X, y)
    ss_lof = max(0.0, rf["ss_res"] - ss_pe)
    df_lof = rf["df"] - df_pe
    ms_pe = ss_pe / df_pe if df_pe > 0 else float("nan")
    ms_lof = ss_lof / df_lof if df_lof > 0 else float("nan")
    ok = df_pe > 0 and df_lof > 0 and ms_pe > 0
    F = ms_lof / ms_pe if ok else float("nan")
    p = float(stats.f.sf(F, df_lof, df_pe)) if ok else float("nan")
    return {"ss_res": rf["ss_res"], "ss_pe": ss_pe, "df_pe": df_pe, "ss_lof": ss_lof, "df_lof": df_lof,
            "ms_pe": ms_pe, "ms_lof": ms_lof, "F": F, "p": p, "computable": bool(ok), "df_res": rf["df"]}


# ─────────────────────────────────────────────────────────────────────────────
#  PARSEO DE LOS REPORTES DE TEXTO DE main.py (pestañas 2 y 3)
# ─────────────────────────────────────────────────────────────────────────────
_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"


def _fl(s: Optional[str]) -> Optional[float]:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def parse_report(txt: str) -> Dict[str, Any]:
    """Extrae del reporte de la pestaña 2: N, gl, R², R²adj, s, t, naturaleza, ŷ(óptimo), IC e IP (± semiancho)."""
    out: Dict[str, Any] = {}
    m = re.search(r"Nº Corridas \(N\):\s*(\d+)\s*\|\s*Grados de Libertad:\s*(-?\d+)", txt)
    if m:
        out["N"], out["df"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"R²:\s*" + _NUM + r"\s*\|\s*R² Ajustado:\s*" + _NUM, txt)
    if m:
        out["r2"], out["r2a"] = _fl(m.group(1)), _fl(m.group(2))
    m = re.search(r"Error Estándar Residual \(s\):\s*" + _NUM, txt)
    if m:
        out["s"] = _fl(m.group(1))
    m = re.search(r"Valor Crítico t\(0\.025, df=(-?\d+)\):\s*" + _NUM, txt)
    if m:
        out["t_df"], out["t"] = int(m.group(1)), _fl(m.group(2))
    m = re.search(r"Naturaleza\s*:\s*(.+)", txt)
    if m:
        out["nature"] = m.group(1).strip()
    m = re.search(r"Respuesta ŷ\s*:\s*" + _NUM, txt)
    if m:
        out["y_opt"] = _fl(m.group(1))
    for key, label in (("ic", r"IC 95% \(Media\)"), ("ip", r"IP 95% \(Indiv\.\)")):
        m = re.search(label + r"\s*:\s*\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]\s*\(±" + _NUM + r"\)", txt)
        if m:
            out[key] = (_fl(m.group(1)), _fl(m.group(2)))
            out[key + "_half"] = _fl(m.group(3))
    return out


def parse_diag(txt: str) -> Dict[str, Any]:
    """Extrae del diagnóstico de la pestaña 3: SS_res, SS_PE, df_PE, SS_LOF, df_LOF, F, p-valor, veredicto."""
    out: Dict[str, Any] = {}
    pats = {"ss_res": r"SS_Residual Total\s*:\s*" + _NUM, "ss_pe": r"SS_PE\s*:\s*" + _NUM,
            "df_pe": r"df_PE\s*:\s*(-?\d+)", "ss_lof": r"SS_LOF\s*:\s*" + _NUM, "df_lof": r"df_LOF\s*:\s*(-?\d+)",
            "F": r"Estadístico F\s*:\s*" + _NUM, "p": r"p-valor\s*:\s*" + _NUM}
    for k, pat in pats.items():
        m = re.search(pat, txt)
        if m:
            out[k] = int(m.group(1)) if k.startswith("df_") else _fl(m.group(1))
    m = re.search(r"RESULTADO\s*:\s*(.+)", txt)
    if m:
        out["verdict"] = m.group(1).strip()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  DATASETS SINTÉTICOS CON PROPIEDADES CONOCIDAS
# ─────────────────────────────────────────────────────────────────────────────
# Verdad de terreno en coordenadas Doehlert p (las mismas de x1_cod, x2_cod, x3_cod):
#   f(p) = 85 − 12(p1−0.2)² − 15(p2+0.1)² − 9(p3−0.3)² + 4 p1 p2
# Gradiente nulo:  b + H p = 0  con  b = (4.8, −3.0, 5.4)  y  H = [[−24, 4, 0], [4, −30, 0], [0, 0, −18]]
TRUE_H = np.array([[-24.0, 4.0, 0.0], [4.0, -30.0, 0.0], [0.0, 0.0, -18.0]])
TRUE_B = np.array([4.8, -3.0, 5.4])
TRUE_P_STAR = np.linalg.solve(TRUE_H, -TRUE_B)           # óptimo exacto en coordenadas p (es un máximo: H definida negativa)


def truth_fn(p: np.ndarray, misfit: bool = False) -> np.ndarray:
    """Superficie verdadera. Con misfit=True se agregan términos cúbico y coseno que un modelo cuadrático NO puede ajustar."""
    p = np.atleast_2d(p)
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
    y = 85.0 - 12 * (p1 - 0.2) ** 2 - 15 * (p2 + 0.1) ** 2 - 9 * (p3 - 0.3) ** 2 + 4 * p1 * p2
    if misfit:
        y = y + 3.0 * (p1 ** 3 - 0.5 * p1) + 2.0 * np.cos(3.0 * p3)
    return y


def generate_design(a: "Audit", ranges, n_cent: Any = 3, names=("Temperatura", "Tiempo", "pH")):
    """Genera la matriz de diseño con el MÉTODO REAL de main.py (caja negra) usando una app headless."""
    app = make_headless_app(a.mod)
    app.entries = [{"nombre": _FakeVar(n), "min": _FakeVar(r[0]), "max": _FakeVar(r[1])} for n, r in zip(names, ranges)]
    app.spin_replicas = _FakeVar(n_cent)
    with patched_dialogs(a.mod) as log:
        app.generar_matriz_diseno()
    return app, log


def make_dataset(a: "Audit", name: str, ranges=None, n_cent: int = 3, sigma: float = 0.3, seed: int = 1,
                 misfit: bool = False, dup_rows: int = 0, note: str = "") -> Dict[str, Any]:
    """
    Construye un dataset sintético: diseño real de main.py + respuesta y = f(p) + ε, ε~N(0,σ²).
    p se recalcula desde los valores reales con la inversa exacta  p = (x − c)/Δ · α  para que
    y sea coherente con el X que recibe el ajuste (main.py redondea los reales a 3 decimales).
    """
    ranges = ranges or DEFAULT_RANGES
    app, _ = generate_design(a, ranges, n_cent)
    rows = app.matriz_generada
    if not rows:
        raise RuntimeError("generar_matriz_diseno() no produjo filas")
    X = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    cod = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])
    if dup_rows > 0:                                     # réplicas fuera del centro (corrida #4 repetida)
        X = np.vstack([X, np.repeat(X[[3]], dup_rows, axis=0)])
        cod = np.vstack([cod, np.repeat(cod[[3]], dup_rows, axis=0)])
    alpha = np.array(a.mod.ALPHA, float)
    c = np.array([(lo + hi) / 2 for lo, hi in ranges])
    d = np.array([(hi - lo) / 2 for lo, hi in ranges])
    P = (X - c) / d * alpha                              # coordenadas p exactas
    rng = np.random.default_rng(seed)
    y = truth_fn(P, misfit) + rng.normal(0.0, sigma, len(X))
    x_true = c + TRUE_P_STAR / alpha * d                 # óptimo verdadero en unidades reales
    return {"name": name, "note": note, "X": X, "y": y, "cod": cod, "ranges": list(ranges), "sigma": sigma,
            "misfit": misfit, "kind": "synthetic",
            "truth": {"p_star": TRUE_P_STAR, "x_opt": x_true,
                      "y_opt": float(truth_fn(TRUE_P_STAR[None, :], False)[0])}}


def collect_run(app, name: str, X: np.ndarray, y: np.ndarray, cod: np.ndarray, error: Optional[str] = None) -> RunResult:
    """Empaqueta en un RunResult lo que la app (headless) dejó tras ajustar: atributos, texto y objeto `fit` si existe."""
    rr = RunResult(name=name, X=X, y=y, cod=cod, error=error)
    rr.betas = None if getattr(app, "betas", None) is None else np.asarray(app.betas, float)
    rr.cov = None if getattr(app, "cov_beta", None) is None else np.asarray(app.cov_beta, float)
    rr.s_res, rr.t_crit = float(app.s_res), float(app.t_crit)
    rr.x_opt = None if getattr(app, "x_opt", None) is None else np.asarray(app.x_opt, float)
    rr.ic_coords = list(getattr(app, "ic_coords", []) or [])
    rr.y_pred = None if getattr(app, "y_pred", None) is None else np.asarray(app.y_pred, float)
    rr.report, rr.diag = app.txt_reporte.get(), app.txt_diag.get()
    rr.Z = X
    fit = getattr(app, "fit", None)
    if fit is not None and hasattr(fit, "betas") and hasattr(fit, "anova"):       # motor nuevo: resultados estructurados
        rr.space, rr.fit = "coded", fit
        rr.Z = np.asarray(fit.coded, float)
        rr.centers, rr.scales = np.asarray(fit.centers, float), np.asarray(fit.scales, float)
        rr.rep, rr.lof = fit_to_dicts(fit)
    else:                                                                           # versión antigua: se parsea el texto
        rr.rep, rr.lof = parse_report(rr.report), parse_diag(rr.diag)
    return rr


def run_fit(a: "Audit", ds: Dict[str, Any], cod_override: Optional[np.ndarray] = None, tag: str = "") -> RunResult:
    """Ejecuta el motor REAL (ejecutar_ajuste_completo) sobre un dataset y recoge atributos + texto parseado."""
    app = make_headless_app(a.mod)
    X, y = ds["X"].copy(), ds["y"].copy()
    cod = (cod_override if cod_override is not None else ds["cod"]).copy()
    app.X_exp, app.y_exp, app.cod_exp = X, y, cod
    app.limites_factores = [(float(X[:, i].min()), float(X[:, i].max())) for i in range(3)]
    err = None
    try:
        app.ejecutar_ajuste_completo()
    except Exception:
        err = traceback.format_exc(limit=3)
    return collect_run(app, ds["name"] + tag, X, y, cod, err)


def fit_to_dicts(fit) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Traduce el objeto ModelFit del motor nuevo a los mismos campos que se parseaban del texto."""
    an, ca = fit.anova, fit.canonical
    rep = {"N": fit.n, "df": an.df_res, "r2": fit.r2, "r2a": fit.r2_adj, "s": fit.s_res, "t_df": an.df_res, "t": fit.t_crit,
           "nature": ca.kind, "y_opt": ca.y_stationary if ca.has_stationary else None,
           "ic": tuple(ca.ic_mean), "ic_half": ca.delta_mean if ca.has_stationary else None,
           "ip": tuple(ca.ip_pred), "ip_half": ca.delta_pred if ca.has_stationary else None}
    lof = {"ss_res": an.ss_res, "ss_pe": an.ss_pe, "df_pe": an.df_pe, "ss_lof": an.ss_lof, "df_lof": an.df_lof,
           "F": an.f_lof, "p": an.p_lof, "verdict": an.lof_status, "ss_reg": an.ss_reg, "df_reg": an.df_reg}
    return rep, lof


# ─────────────────────────────────────────────────────────────────────────────
#  CLASE PRINCIPAL DEL AUDITOR
# ─────────────────────────────────────────────────────────────────────────────
class Audit:
    """Orquesta la auditoría: carga main.py, ejecuta los chequeos y acumula hallazgos, datos y gráficas."""

    def __init__(self, target: str, quick: bool = False, seed: int = 20260108,
                 progress: Optional[Callable[[str, float], None]] = None,
                 import_data: Optional[str] = None, import_report: Optional[str] = None) -> None:
        self.target = os.path.abspath(target)
        self.quick, self.seed = quick, seed
        self.progress = progress or (lambda msg, frac: None)
        self.import_data, self.import_report = import_data, import_report
        with open(self.target, "r", encoding="utf-8-sig") as f:
            self.src = f.read()
        self.tree = ast.parse(self.src)
        self.mod = load_target(self.target)
        # Fuentes auditables: main.py + los módulos .py hermanos que importó (p. ej. doehlert_engine.py)
        self.sources: List[Tuple[str, str, ast.AST]] = [(self.target, self.src, self.tree)]
        tdir = os.path.dirname(self.target)
        for m_ in list(sys.modules.values()):
            f_ = getattr(m_, "__file__", None)
            if f_ and f_.endswith(".py") and os.path.dirname(os.path.abspath(f_)) == tdir and os.path.abspath(f_) != self.target \
                    and not os.path.basename(f_).startswith("audit_doehlert"):
                try:
                    with open(f_, "r", encoding="utf-8-sig") as fh:
                        s_ = fh.read()
                    self.sources.append((os.path.abspath(f_), s_, ast.parse(s_)))
                except Exception:
                    pass
        self.findings: List[Finding] = []
        self.store: Dict[str, Any] = {}                    # datos numéricos para las gráficas
        self.datasets: List[Dict[str, Any]] = []
        self.runs: Dict[str, RunResult] = {}
        self.figures: List[Tuple[str, str, str, Figure]] = []   # (id, título, leyenda, figura)
        self.comparison: Optional[Dict[str, Any]] = None
        self.imported_info: Dict[str, Any] = {}
        self.t0 = time.time()
        self.meta = self._build_meta()

    # ---- metadatos reproducibles -------------------------------------------------
    def _build_meta(self) -> Dict[str, Any]:
        with open(self.target, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        return {"schema": SCHEMA, "target": self.target, "target_sha256": sha,
                "target_lines": self.src.count("\n") + 1, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__,
                "statsmodels": (sm.__version__ if HAVE_SM else None), "platform": platform.platform(),
                "quick": self.quick, "seed": self.seed}

    # ---- registro de hallazgos ---------------------------------------------------
    def add(self, id: str, section: str, title: str, status: str, summary: str = "", details: str = "",
            metrics: Optional[dict] = None, where: str = "", fix: str = "") -> Finding:
        f = Finding(id, section, title, status, summary, details, metrics or {}, where, fix)
        self.findings.append(f)
        return f

    def counts(self) -> Dict[str, int]:
        c = {s: 0 for s in STATUS_ORDER}
        for f in self.findings:
            c[f.status] = c.get(f.status, 0) + 1
        return c

    def by_section(self) -> Dict[str, List[Finding]]:
        out: Dict[str, List[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.section, []).append(f)
        return out

    def add_fig(self, fid: str, title: str, caption: str, fig: Optional[Figure]) -> None:
        if fig is not None:
            self.figures.append((fid, title, caption, fig))

    # ---- ejecución completa ------------------------------------------------------
    def run(self, only: Optional[List[str]] = None) -> "Audit":
        """Ejecuta todos los pasos (o solo los indicados por nombre de función en `only`, p. ej. ['chk_validation'])."""
        steps: List[Tuple[str, Callable[["Audit"], None]]] = [
            ("1.1 normal_ppf", chk_normal_ppf), ("1.2 betainc", chk_betainc), ("1.3 f_sf", chk_f_sf),
            ("1.4 t_ppf_975", chk_t_ppf), ("2.1 geometría Doehlert", chk_geometry),
            ("2.2 transformaciones", chk_transforms), ("2.3 datasets sintéticos", build_datasets),
            ("3.1 OLS", chk_ols), ("4.1 análisis canónico", chk_canonical), ("4.2 jacobiano", chk_jacobian),
            ("4.3 método Delta", chk_delta), ("4.4 cobertura extremo-a-extremo", chk_coverage),
            ("5.1 ANOVA lack-of-fit", chk_lof), ("6.1 caja blanca", chk_whitebox), ("7.1 E/S de CSV", chk_io),
            ("8.1 dataset importado", chk_imported), ("10.1 experimentos de validación", chk_validation),
            ("9.1 gráficas", build_figures), ("9.2 comparación con reporte previo", chk_compare)]
        if only:
            steps = [(lab, fn) for lab, fn in steps if fn.__name__ in only]
        n = len(steps)
        for i, (label, fn) in enumerate(steps):
            self.progress(label, i / n)
            t = time.time()
            before = len(self.findings)
            try:
                fn(self)
            except Exception:
                tb = traceback.format_exc(limit=4)
                self.add(f"ERR.{i}", "Auditor", f"Error interno en el paso «{label}»", "FAIL",
                         "El paso falló con una excepción (ver detalles).", tb)
            for f in self.findings[before:]:
                f.elapsed = round(time.time() - t, 3)
        self.progress("listo", 1.0)
        self.meta["elapsed_s"] = round(time.time() - self.t0, 2)
        return self


# ═════════════════════════════════════════════════════════════════════════════
#  CHEQUEOS  —  SECCIÓN 1: FUNCIONES ESTADÍSTICAS NATIVAS
# ═════════════════════════════════════════════════════════════════════════════
SEC_STAT = "1. Funciones estadísticas nativas"
SEC_GEO = "2. Geometría de Doehlert y transformaciones"
SEC_OLS = "3. Mínimos cuadrados (OLS)"
SEC_CAN = "4. Análisis canónico y método Delta"
SEC_LOF = "5. ANOVA y falta de ajuste"
SEC_WB = "6. Caja blanca (análisis estático)"
SEC_IO = "7. Entrada/salida de CSV"
SEC_IMP = "8. Dataset importado"
SEC_CMP = "9. Comparación con corrida previa"


class Sub:
    """Acumula sub-chequeos [PASS]/[WARN]/[FAIL] dentro de un hallazgo y calcula el estado global."""
    def __init__(self) -> None:
        self.items: List[Tuple[str, str, str]] = []

    def add(self, name: str, status: str, msg: str = "") -> None:
        self.items.append((name, status, msg))

    @property
    def status(self) -> str:
        return worst(*[s for _, s, _ in self.items]) if self.items else "SKIP"

    def text(self) -> str:
        return "\n".join(f"[{s}] {n}" + (f" — {m}" if m else "") for n, s, m in self.items)

    def failing(self) -> List[str]:
        return [n for n, s, _ in self.items if s == "FAIL"]


def short_fail(sub: "Sub", k: int = 2, w: int = 42) -> str:
    """Resumen corto de los sub-chequeos que fallan (para la línea de resumen del hallazgo)."""
    f = sub.failing()
    names = [n if len(n) <= w else n[: w - 1] + "…" for n in f[:k]]
    return f"{len(f)} de {len(sub.items)} sub-chequeos fallan: " + "; ".join(names) + (f"; +{len(f) - k} más" if len(f) > k else "")


def call_vec(fn: Callable, arr: np.ndarray) -> np.ndarray:
    """Evalúa fn sobre un arreglo; si fn no está vectorizada, la aplica elemento a elemento."""
    arr = np.asarray(arr, float)
    try:
        r = np.asarray(fn(arr), float)
        if r.shape == arr.shape:
            return r
    except Exception:
        pass
    return np.array([float(fn(float(v))) for v in arr.ravel()]).reshape(arr.shape)


def _subsample(*arrs: np.ndarray, n: int = 1500):
    idx = np.unique(np.linspace(0, len(arrs[0]) - 1, n).astype(int))
    return [np.asarray(x)[idx] for x in arrs]


# ---- 1.1  normal_ppf ---------------------------------------------------------------
def chk_normal_ppf(a: Audit) -> None:
    """
    Φ⁻¹(p) de main.normal_ppf (Beasley-Springer-Moro) vs scipy.stats.norm.ppf.
    Malla densa p ∈ [1e-6, 1−1e-6]: lineal + geométrica en ambas colas (donde BSM cambia de fórmula).
    Error absoluto  |q̂ − q|  y relativo  |q̂ − q| / |q|  (solo donde |q| > 1e-3).
    """
    m = a.mod
    n_lin, n_geo = (20000, 400) if a.quick else (200000, 2000)
    p = np.unique(np.concatenate([np.linspace(1e-6, 1 - 1e-6, n_lin), np.geomspace(1e-6, 0.5, n_geo),
                                  1 - np.geomspace(1e-6, 0.5, n_geo), [0.5, 0.025, 0.975, 0.08, 0.92]]))
    est, ref = call_vec(m.normal_ppf, p), stats.norm.ppf(p)
    err = np.abs(est - ref)
    rel = np.zeros_like(err)
    mk = np.abs(ref) > 1e-3
    rel[mk] = err[mk] / np.abs(ref[mk])
    central = np.abs(p - 0.5) < 0.42           # región central de BSM (racional en y=p−½); fuera: polinomio en log(−log p)
    emax, i = float(err.max()), int(err.argmax())
    n_bad = int((err > TOL_ABS).sum())
    st = graded_status(emax, TOL_ABS)
    rows = [("región central |p−½|<0.42", central), ("colas (|p−½|≥0.42)", ~central), ("toda la malla", np.ones_like(central))]
    det = ["Región".ljust(28) + "puntos".rjust(9) + "máx |error abs|".rjust(18) + "máx error rel".rjust(16)]
    for nm, msk in rows:
        det.append(nm.ljust(28) + f"{int(msk.sum()):9d}" + f"{err[msk].max():18.3e}" + f"{rel[msk].max():16.3e}")
    det.append(f"Peor punto: p = {p[i]:.6g}  →  est = {est[i]:.6f}  vs  ref = {ref[i]:.6f}")
    det.append(f"Puntos con error > {TOL_ABS:g}: {n_bad} de {len(p)} ({100 * n_bad / len(p):.2f} %)")
    a.store["normal_ppf"] = dict(zip(("p", "est", "ref", "err"), _subsample(p, est, ref, err)))
    a.add("1.1", SEC_STAT, "normal_ppf(p) vs scipy.stats.norm.ppf", st,
          f"máx |error| = {emax:.3e} (p = {p[i]:.3g}); {n_bad} puntos fuera de tolerancia",
          "\n".join(det), {"max_abs": emax, "max_rel": float(rel.max()), "n_bad": n_bad, "n": len(p),
                           "max_abs_central": float(err[central].max()), "max_abs_tails": float(err[~central].max())},
          "main.py · normal_ppf (cola del algoritmo BSM)",
          "En la cola, evaluar el polinomio de Horner desde el coeficiente de mayor orden: x = c[8]; for c_i in reversed(c[:8]): x = x*r + c_i")

    # Propiedades estructurales: simetría Φ⁻¹(p) = −Φ⁻¹(1−p) y monotonía estricta
    pp = np.linspace(1e-5, 0.5 - 1e-9, 5000)
    sym = float(np.max(np.abs(call_vec(m.normal_ppf, pp) + call_vec(m.normal_ppf, 1 - pp))))
    pm = np.linspace(1e-6, 1 - 1e-6, 20001)
    d = np.diff(call_vec(m.normal_ppf, pm))
    viol = int((d < 0).sum())
    st2 = "PASS" if (sym < 1e-7 and viol == 0) else "FAIL"
    a.add("1.1b", SEC_STAT, "normal_ppf: simetría Φ⁻¹(p) = −Φ⁻¹(1−p) y monotonía", st2,
          f"asimetría máx = {sym:.2e}; violaciones de monotonía = {viol}",
          "Una función cuantil debe ser creciente; si el polinomio de cola está mal, aparecen retrocesos.",
          {"asimetria": sym, "violaciones_monotonia": viol}, "main.py · normal_ppf")


# ---- 1.2  betainc_approx -----------------------------------------------------------
def chk_betainc(a: Audit) -> None:
    """
    I_x(a,b) regularizada (fracción continua de Lentz) vs scipy.special.betainc.
    Malla: a,b ∈ [0.1, 50], x desde 1e-12 hasta 1−1e-9. Se registran excepciones (RecursionError)
    y el comportamiento en las fronteras x = 0 y x = 1 (I_0 = 0, I_1 = 1).
    """
    m = a.mod
    A = B = [0.1, 0.5, 1, 2, 5, 10, 25, 50]
    X = [1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1 - 1e-9]
    errs = np.zeros((len(A), len(B)))
    rel_max, exc = 0.0, []
    for i, aa in enumerate(A):
        for j, bb in enumerate(B):
            for x in X:
                try:
                    est = float(m.betainc_approx(aa, bb, x))
                except RecursionError:
                    exc.append((aa, bb, x, "RecursionError"))
                    continue
                except Exception as e:
                    exc.append((aa, bb, x, type(e).__name__))
                    continue
                ref = float(special.betainc(aa, bb, x))
                e_abs = abs(est - ref)
                errs[i, j] = max(errs[i, j], e_abs)
                if ref > 1e-12:
                    rel_max = max(rel_max, e_abs / ref)
    emax = float(errs.max())
    front = []
    for aa, bb in [(0.5, 0.5), (2, 3), (10, 1)]:
        try:
            front.append((aa, bb, float(m.betainc_approx(aa, bb, 0.0)), float(m.betainc_approx(aa, bb, 1.0))))
        except Exception as e:
            front.append((aa, bb, type(e).__name__, type(e).__name__))
    front_ok = all(isinstance(f0, float) and abs(f0) < 1e-15 and abs(f1 - 1) < 1e-15 for _, _, f0, f1 in front)
    st = worst(graded_status(emax, TOL_ABS), "FAIL" if exc else "PASS", "PASS" if front_ok else "FAIL")
    det = [f"Malla: {len(A)}×{len(B)} pares (a,b) × {len(X)} valores de x = {len(A) * len(B) * len(X)} evaluaciones",
           f"Máx |error abs| = {emax:.3e}   máx error rel (ref>1e-12) = {rel_max:.3e}",
           f"Excepciones: {len(exc)}" + ("".join(f"\n   (a={e[0]}, b={e[1]}, x={e[2]}) → {e[3]}" for e in exc[:8]))]
    det.append("Fronteras: " + "; ".join(f"I_0({f[0]},{f[1]})={f[2]}, I_1={f[3]}" for f in front))
    a.store["betainc"] = {"A": A, "B": B, "err": errs}
    a.add("1.2", SEC_STAT, "betainc_approx(a,b,x) vs scipy.special.betainc", st,
          f"máx |error| = {emax:.2e}; excepciones = {len(exc)}; fronteras {'OK' if front_ok else 'ERROR'}",
          "\n".join(det), {"max_abs": emax, "max_rel": rel_max, "exceptions": len(exc), "frontier_ok": front_ok},
          "main.py · betainc_approx")

    # 1.2b  recursión en el punto de simetría x = (a+1)/(a+b+2) y convergencia con parámetros grandes
    sub = Sub()
    for aa, bb in [(1, 1), (2, 2), (0.5, 0.5), (10, 10), (50, 50)]:
        x = (aa + 1.0) / (aa + bb + 2.0)                # con a = b da exactamente 0.5
        try:
            est = float(m.betainc_approx(aa, bb, x))
            sub.add(f"x=(a+1)/(a+b+2), a=b={aa}", "PASS" if abs(est - special.betainc(aa, bb, x)) < TOL_ABS else "FAIL", f"I={est:.6f}")
        except RecursionError:
            sub.add(f"x=(a+1)/(a+b+2), a=b={aa}", "FAIL", "RecursionError: la rama simétrica se llama a sí misma indefinidamente")
        except Exception as e:
            sub.add(f"x=(a+1)/(a+b+2), a=b={aa}", "FAIL", type(e).__name__)
    worst_conv = 0.0
    for aa, bb in [(100, 100), (500, 300), (1000, 1000), (200, 5)]:
        for x in (0.3, 0.5, 0.7, aa / (aa + bb)):
            try:
                est = float(m.betainc_approx(aa, bb, x))
                worst_conv = max(worst_conv, abs(est - special.betainc(aa, bb, x)))
            except Exception:
                worst_conv = float("inf")
    sub.add("convergencia con a,b grandes (max_iter=200)", graded_status(worst_conv, TOL_ABS), f"máx |error| = {worst_conv:.2e}")
    a.add("1.2b", SEC_STAT, "betainc_approx: punto de simetría y convergencia (estabilidad numérica)", sub.status,
          f"{len(sub.failing())} sub-chequeos fallan", sub.text(), {"conv_err": worst_conv}, "main.py · betainc_approx (rama x ≥ (a+1)/(a+b+2))",
          "En la rama simétrica no volver a llamar a betainc_approx: calcular directamente la fracción continua con (b, a, 1−x).")


# ---- 1.3  f_sf ---------------------------------------------------------------------
def chk_f_sf(a: Audit) -> None:
    """
    Cola derecha de Fisher-Snedecor  S(F; d1, d2) = I_x(d2/2, d1/2),  x = d2/(d2 + d1 F)
    vs scipy.stats.f.sf. Incluye F = 1 con d1 = d2 (x = ½: caso simétrico) y F grande/pequeña.
    """
    m = a.mod
    Fs = np.concatenate([np.geomspace(1e-3, 1e3, 25), [1.0]])
    dfs = [1, 2, 3, 5, 10, 30, 100, 300]
    err = np.zeros((len(dfs), len(dfs)))
    exc, rel_max = [], 0.0
    for i, d1 in enumerate(dfs):
        for j, d2 in enumerate(dfs):
            for F in Fs:
                try:
                    est = float(m.f_sf(float(F), d1, d2))
                except RecursionError:
                    exc.append((F, d1, d2, "RecursionError"))
                    continue
                except Exception as e:
                    exc.append((F, d1, d2, type(e).__name__))
                    continue
                ref = float(stats.f.sf(F, d1, d2))
                err[i, j] = max(err[i, j], abs(est - ref))
                if ref > 1e-10:
                    rel_max = max(rel_max, abs(est - ref) / ref)
    emax = float(err.max())
    st = worst(graded_status(emax, TOL_ABS), "FAIL" if exc else "PASS")
    det = [f"{len(Fs) * len(dfs) ** 2} evaluaciones (F × df1 × df2)", f"Máx |error abs| = {emax:.3e}; máx error rel (ref>1e-10) = {rel_max:.3e}",
           f"Excepciones: {len(exc)}" + "".join(f"\n   F={e[0]:.3g}, df1={e[1]}, df2={e[2]} → {e[3]}" for e in exc[:8])]
    a.store["f_sf"] = {"dfs": dfs, "err": err}
    a.add("1.3", SEC_STAT, "f_sf(F, df1, df2) vs scipy.stats.f.sf", st, f"máx |error| = {emax:.2e}; excepciones = {len(exc)}",
          "\n".join(det), {"max_abs": emax, "max_rel": rel_max, "exceptions": len(exc)}, "main.py · f_sf")


# ---- 1.4  t_ppf_975 ----------------------------------------------------------------
def chk_t_ppf(a: Audit) -> None:
    """
    Cuantil t de Student  t_{0.975, ν}  vs scipy.stats.t.ppf(0.975, ν),  ν = 1…1000.
    Error relativo = |t̂ − t|/t. El intervalo de confianza es proporcional a t, de modo que el
    error relativo de t es el sesgo relativo del ancho de TODOS los intervalos al 95 %.
    """
    m = a.mod
    dfs = np.array(list(range(1, 31)) + [40, 50, 60, 100, 200, 500, 1000])
    est = np.array([float(m.t_ppf_975(int(d))) for d in dfs])
    ref = stats.t.ppf(0.975, dfs)
    rel = np.abs(est - ref) / ref
    emax = float(rel.max())
    st = graded_status(emax, TOL_REL)
    det = ["ν".rjust(5) + "t main".rjust(11) + "t scipy".rjust(11) + "error rel".rjust(12) + "  estado"]
    for d, e, r, rl in zip(dfs, est, ref, rel):
        if d <= 12 or d in (15, 20, 30, 60, 100, 1000):
            det.append(f"{d:5d}{e:11.4f}{r:11.4f}{rl:12.2e}  [{graded_status(rl, TOL_REL)}]")
    n_bad = int((rel > TOL_REL).sum())
    a.store["t_ppf"] = {"df": dfs, "est": est, "ref": ref, "rel": rel}
    a.add("1.4", SEC_STAT, "t_ppf_975(ν) vs scipy.stats.t.ppf(0.975, ν)", st,
          f"error rel máx = {emax:.2e} (ν = {int(dfs[rel.argmax()])}); {n_bad} de {len(dfs)} valores de ν fuera de tolerancia",
          "\n".join(det), {"max_rel": emax, "df_worst": int(dfs[rel.argmax()]),
                           "rel_df5": float(rel[list(dfs).index(5)]), "rel_df3": float(rel[list(dfs).index(3)])},
          "main.py · t_ppf_975 (serie de Cornish-Fisher truncada)",
          "Invertir la CDF t con bisección sobre betainc_approx: P(T>t) = ½·I_x(ν/2, ½) con x = ν/(ν+t²).")


# ═════════════════════════════════════════════════════════════════════════════
#  CHEQUEOS  —  SECCIÓN 2: GEOMETRÍA Y NORMALIZACIÓN DE DOEHLERT (k = 3)
# ═════════════════════════════════════════════════════════════════════════════
def chk_geometry(a: Audit) -> None:
    """
    Diseño de Doehlert k=3:  N = k²+k+1 = 13 puntos = centro + 12 vértices de un cuboctaedro de radio 1
    (empaquetamiento compacto de esferas). Propiedades verificables:
      · centro en el origen;  · los 12 periféricos a distancia 1 del centro (esfera unitaria);
      · cada vértice tiene exactamente 4 vecinos a distancia 1 (24 aristas);  · simetría central (p ∈ D ⇒ −p ∈ D);
      · niveles por eje: 7, 5 y 3 (en algún orden);  · ALPHA[j] = máx|p_j| (extremo de cada factor).
    """
    m = a.mod
    B = np.array(m.DOEHLERT_BASE, float)
    alpha = np.array(m.ALPHA, float)
    a.store["geometry"] = {"B": B}
    sub = Sub()
    sub.add("N = k²+k+1 = 13 puntos", "PASS" if B.shape == (13, 3) else "FAIL", f"forma {B.shape}")
    sub.add("punto 0 es el centro (0,0,0)", "PASS" if np.allclose(B[0], 0, atol=1e-15) else "FAIL", f"{B[0]}")
    per = B[1:]
    radii = np.linalg.norm(per, axis=1)
    sub.add("12 vértices sobre la esfera de radio 1", graded_status(float(np.abs(radii - 1).max()), 1e-12),
            f"máx |r−1| = {np.abs(radii - 1).max():.2e}")
    D = np.linalg.norm(per[:, None, :] - per[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    neigh = (np.abs(D - 1.0) < 1e-9).sum(axis=1)
    n_edges = int((np.abs(D - 1.0) < 1e-9).sum() // 2)
    dmin = float(D.min())
    sub.add("distancia mínima entre vértices = 1 (arista = radio)", graded_status(abs(dmin - 1), 1e-12), f"d_min = {dmin:.12f}")
    sub.add("cada vértice con 4 vecinos a distancia 1 (24 aristas)", "PASS" if (np.all(neigh == 4) and n_edges == 24) else "FAIL",
            f"vecinos por vértice = {sorted(set(neigh.tolist()))}, aristas = {n_edges}")
    sym_ok = all(any(np.allclose(-p, q, atol=1e-12) for q in per) for p in per)
    sub.add("simetría central (p ⇒ −p)", "PASS" if sym_ok else "FAIL")
    levels = [len(set(np.round(B[:, j], 9))) for j in range(3)]
    sub.add("niveles por eje = {7, 5, 3}", "PASS" if sorted(levels) == [3, 5, 7] else "FAIL", f"niveles reales por columna = {levels}")
    amax = np.abs(B).max(axis=0)
    sub.add("ALPHA[j] = máx|p_j| (el factor llega exactamente a sus extremos)", graded_status(float(np.abs(alpha - amax).max()), 1e-12),
            f"ALPHA = {np.round(alpha, 6).tolist()}  vs  máx|p_j| = {np.round(amax, 6).tolist()}")
    labels = {int(ax): int(n) for ax, n in re.findall(r"X(\d) \((\d+) Niveles\)", a.src)}
    if labels:
        ok = all(labels.get(j + 1) == levels[j] for j in range(3))
        sub.add("etiquetas de la interfaz coinciden con los niveles reales", "PASS" if ok else "FAIL",
                f"etiquetas = {[labels.get(j + 1) for j in range(3)]}  vs  reales = {levels}")
    else:
        sub.add("etiquetas de la interfaz", "SKIP", "no se encontraron etiquetas «Xn (k Niveles)» en el código")
    a.add("2.1", SEC_GEO, "DOEHLERT_BASE: geometría esférica (cuboctaedro) y niveles por factor", sub.status,
          f"{len(sub.failing())} de {len(sub.items)} propiedades fallan" if sub.failing() else "las 13 corridas cumplen la geometría de Doehlert",
          sub.text(), {"levels": levels, "alpha": alpha, "radii_max_dev": float(np.abs(radii - 1).max()), "edges": n_edges},
          "main.py · DOEHLERT_BASE, ALPHA y etiquetas de generar la interfaz",
          "Permutar columnas de DOEHLERT_BASE/ALPHA o corregir las etiquetas para que coincidan con los niveles reales.")


def chk_transforms(a: Audit) -> None:
    """
    Transformación real ↔ codificada (equivalente a _coded_from_real / _real_from_coded):
        real = c + (p/α)·Δ        p = (real − c)/Δ · α        c = (máx+mín)/2,  Δ = (máx−mín)/2
    Se ejecuta el método REAL generar_matriz_diseno() y se verifican: valores esperados, extremos (±Δ),
    biyección (p recuperado desde los reales publicados) y el manejo de réplicas del centro y de la coma decimal.
    """
    m = a.mod
    alpha = np.array(m.ALPHA, float)
    base = np.array(m.DOEHLERT_BASE, float)
    cases = [("rangos por defecto (50-90, 10-30, 3-7)", DEFAULT_RANGES, DEFAULT_RANGES, 3),
             ("rangos asimétricos y negativos", [(-5, 15), (0, 100), (2.5, 3.5)], [(-5, 15), (0, 100), (2.5, 3.5)], 5),
             ("rango estrecho (0.001-0.003 en X1)", [(0.001, 0.003), (100, 120), (6, 8)], [(0.001, 0.003), (100, 120), (6, 8)], 3),
             ("coma decimal en las entradas («50,5»)", [("50,5", "90,5"), ("10,25", "30,25"), ("3,5", "7,5")],
              [(50.5, 90.5), (10.25, 30.25), (3.5, 7.5)], 3),
             ("réplicas del centro = 1", DEFAULT_RANGES, DEFAULT_RANGES, 1), ("réplicas del centro = «abc» (inválido → 3)", DEFAULT_RANGES, DEFAULT_RANGES, "abc"),
             ("réplicas del centro = 0 (→ mínimo 1)", DEFAULT_RANGES, DEFAULT_RANGES, 0)]
    sub = Sub()
    table = []
    for name, inp, num, nc in cases:
        try:
            app, log = generate_design(a, inp, nc)
        except Exception as e:
            sub.add(name, "FAIL", f"excepción {type(e).__name__}: {e}")
            continue
        rows = app.matriz_generada
        n_exp = 13 + (3 if nc == "abc" else max(1, int(nc))) - 1
        invalid_in = nc == "abc" or nc == 0                 # entrada inválida: se acepta usar el valor por defecto O rechazarla con aviso
        if len(rows) == 0 and invalid_in and log:
            sub.add(name, "PASS", "entrada inválida rechazada con aviso: «" + log[0][2][:70] + "»")
            continue
        if len(rows) != n_exp:
            sub.add(name, "FAIL", f"filas = {len(rows)} (esperadas {n_exp}); avisos: {[l[2] for l in log][:1]}")
            continue
        corr = np.vstack([base, np.zeros((n_exp - 13, 3))])
        c = np.array([(lo + hi) / 2 for lo, hi in num])
        d = np.array([(hi - lo) / 2 for lo, hi in num])
        real_exp = c + corr / alpha * d
        real_main = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
        cod_main = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])
        e_real = float(np.abs(real_main - real_exp).max())
        p_back = (real_main - c) / d * alpha                    # p recuperado desde los reales publicados
        e_p = float(np.abs(p_back - corr).max())               # error de biyección en unidades p
        e_cod = float(np.abs(cod_main - corr).max())
        ext_ok = all(abs(real_main[:, j].max() - num[j][1]) <= 5.1e-4 and abs(real_main[:, j].min() - num[j][0]) <= 5.1e-4 for j in range(3))
        st = "PASS"
        why = []
        if e_p > 1e-3:
            st = "FAIL"; why.append(f"p recuperado se desvía {e_p:.3g} (>1e-3) por el redondeo a 3 decimales")
        if e_cod > 5.1e-4:
            st = "FAIL"; why.append(f"columnas x_cod difieren {e_cod:.2e} de DOEHLERT_BASE")
        if not ext_ok:
            st = "FAIL"; why.append("los extremos de algún factor no coinciden con mín/máx")
        sub.add(name, st, f"|real−esperado|={e_real:.1e}, |p−p̂|={e_p:.1e}" + (" · " + "; ".join(why) if why else ""))
        table.append((name, e_real, e_p, e_cod))
    names3 = ("_coded_from_real", "_real_from_coded", "_configure_factor_transform")
    absent = [n for n in names3 if not hasattr(m.DoehlertSuiteApp, n)]
    a.store["transforms"] = table
    a.add("2.2", SEC_GEO, "Transformación real ↔ codificada (generar_matriz_diseno) y biyección", sub.status,
          f"{len(sub.failing())} de {len(sub.items)} escenarios fallan" if sub.failing() else "transformación consistente y reversible",
          sub.text(), {"cases": len(sub.items), "failing": sub.failing()}, "main.py · generar_matriz_diseno: round(val_real, 3)",
          "No redondear los valores reales a 3 decimales (o redondear según Δ/1e4): con rangos estrechos se pierde la información del diseño.")
    if len(absent) == 3:
        a.add("2.2b", SEC_GEO, "Funciones de transformación citadas en la especificación", "SKIP",
              "no existen en esta versión de main.py: " + ", ".join(absent),
              "La transformación está implementada en línea dentro de generar_matriz_diseno(); se auditó ese método (hallazgo 2.2).")
        return
    # Las funciones existen: se prueban con el diseño por defecto  (real = c + s·z,  z = (real − c)/s,  s = Δ/α)
    sub2 = Sub()
    app, _ = generate_design(a, DEFAULT_RANGES, 3)
    rows = app.matriz_generada
    real = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    cod = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])
    c = np.array([(lo + hi) / 2 for lo, hi in DEFAULT_RANGES])
    s_ = np.array([(hi - lo) / 2 for lo, hi in DEFAULT_RANGES]) / alpha
    if hasattr(app, "_real_from_coded") and hasattr(app, "factor_centers"):
        e1 = float(np.abs(app._real_from_coded(cod) - (c + cod * s_)).max())
        sub2.add("_real_from_coded(z) = centro + escala·z", graded_status(e1, 1e-9), f"máx |error| = {e1:.1e}")
        e2 = float(np.abs(app._coded_from_real(real) - (real - c) / s_).max())
        sub2.add("_coded_from_real(x) = (x − centro)/escala", graded_status(e2, 1e-9), f"máx |error| = {e2:.1e}")
        e3 = float(np.abs(app._coded_from_real(app._real_from_coded(cod)) - cod).max())
        sub2.add("biyección: coded(real(z)) = z", graded_status(e3, 1e-12), f"máx |error| = {e3:.1e}")
        e4 = float(np.abs(app._real_from_coded(cod[0]) - c).max())
        sub2.add("admite un vector (no solo matrices)", graded_status(e4, 1e-9), f"centro → {np.round(app._real_from_coded(cod[0]), 6).tolist()}")
    if hasattr(app, "_configure_factor_transform"):
        app2 = make_headless_app(a.mod)
        app2._configure_factor_transform(real, cod)
        e5 = float(max(np.abs(np.asarray(app2.factor_centers) - c).max(), np.abs(np.asarray(app2.factor_scales) - s_).max() / s_.max()))
        sub2.add("_configure_factor_transform reconstruye centros y escalas", graded_status(e5, 1e-6), f"desviación = {e5:.1e}")
        try:
            bad = cod.copy()
            bad[:, 0] = 0.0                                   # factor sin variación: debe rechazarse con un error claro
            app2._configure_factor_transform(real, bad)
            sub2.add("rechaza columnas codificadas sin variación", "FAIL", "no lanzó error")
        except Exception as e_:
            sub2.add("rechaza columnas codificadas sin variación", "PASS", f"{type(e_).__name__}: {str(e_)[:60]}")
    a.add("2.2b", SEC_GEO, "Funciones _coded_from_real / _real_from_coded / _configure_factor_transform", sub2.status,
          "transformaciones exactas y reversibles" if sub2.status == "PASS" else f"{len(sub2.failing())} sub-chequeos fallan", sub2.text())


# ═════════════════════════════════════════════════════════════════════════════
#  DATASETS SINTÉTICOS  (generados con el método real de main.py)
# ═════════════════════════════════════════════════════════════════════════════
def _ranges_of(ds: Dict[str, Any]) -> List[Tuple[float, float]]:
    if ds.get("ranges"):
        return [tuple(r) for r in ds["ranges"]]
    X = ds["X"]
    return [(float(X[:, j].min()), float(X[:, j].max())) for j in range(3)]


def build_datasets(a: Audit) -> None:
    """
    Seis datasets con propiedades conocidas (óptimo verdadero, réplicas, falta de ajuste, mal condicionamiento):
      D1  óptimo conocido, ruido σ=0.3, 3 réplicas del centro       D4  sin réplicas (LOF no computable)
      D2  exacto, sin ruido (R² = 1, SS_PE = 0)                      D5  réplicas fuera del centro (rama np.unique)
      D3  con falta de ajuste (términos cúbico + coseno)             D6  rangos desplazados (matriz mal condicionada)
    + (opcional) el dataset real importado con --import-data.
    """
    specs = [("D1", "D1 · óptimo conocido (σ=0.3)", dict(sigma=0.3, n_cent=3, note="Respuesta cuadrática exacta + ruido gaussiano; óptimo verdadero conocido.")),
             ("D2", "D2 · exacto sin ruido", dict(sigma=0.0, n_cent=3, note="Sin ruido: debe dar R²=1, s≈0 y reproducir el óptimo verdadero.")),
             ("D3", "D3 · con falta de ajuste", dict(sigma=0.15, n_cent=3, misfit=True, note="Se agregan términos cúbico y coseno: la prueba LOF debe rechazar el modelo cuadrático.")),
             ("D4", "D4 · sin réplicas de centro", dict(sigma=0.3, n_cent=1, note="13 corridas: el error puro no es estimable.")),
             ("D5", "D5 · réplicas fuera del centro", dict(sigma=0.3, n_cent=1, dup_rows=2, note="La corrida #4 se repite 3 veces: ejercita la rama np.unique de main.py.")),
             ("D6", "D6 · rangos desplazados (mal condicionado)", dict(sigma=0.3, n_cent=3, ranges=[(1000.0, 1040.0), (100.0, 120.0), (6.0, 8.0)],
                                                                      note="Factores lejos del origen: cond(MᵀM) enorme en unidades naturales."))]
    for i, (name, label, kw) in enumerate(specs):
        ds = make_dataset(a, name, seed=a.seed + i, **kw)
        ds["label"] = label
        a.datasets.append(ds)
    if a.import_data:
        try:
            ds = load_dataset_csv(a.import_data)
            ds["name"] = "IMP"
            ds["label"] = "IMP · " + os.path.basename(a.import_data)
            a.datasets.append(ds)
            a.imported_info = {"path": a.import_data, "ok": True, "n": int(len(ds["y"])), "notes": ds.get("notes", [])}
        except Exception as e:
            a.imported_info = {"path": a.import_data, "ok": False, "error": f"{type(e).__name__}: {e}"}
    for ds in a.datasets:
        a.runs[ds["name"]] = run_fit(a, ds)
    # D1 exportado SIN columnas x_cod y leído con el cargador REAL de main: ¿cómo detecta luego las réplicas del centro?
    d1 = a.datasets[0]
    tmpd = tempfile.mkdtemp(prefix="audit_nocod_")
    p_nc = os.path.join(tmpd, "D1_sin_xcod.csv")
    with open(p_nc, "w", newline="", encoding="utf-8") as f:
        f.write("sep=,\n")
        w = csv.writer(f)
        w.writerow(["Corrida", "Temperatura", "Tiempo", "pH", "Respuesta_Y"])
        for i, (x, yv) in enumerate(zip(d1["X"], d1["y"]), start=1):
            w.writerow([i, *[f"{v:.4f}" for v in x], f"{yv:.4f}"])
    app_nc, _errs = _load_with_main(a, p_nc)
    cod_nc = app_nc.cod_exp if (app_nc.y_exp is not None and len(app_nc.y_exp) == len(d1["y"])) else np.zeros_like(d1["cod"])
    a.store["nocod_cod"] = np.asarray(cod_nc, float)
    a.runs["D1|sin_cod"] = run_fit(a, d1, cod_override=np.asarray(cod_nc, float), tag="|sin_cod")
    det = ["Dataset".ljust(7) + "N".rjust(3) + "  " + "rangos (min–max por factor)".ljust(46) + "óptimo verdadero (real)".ljust(30) + "ŷ*"]
    for ds in a.datasets:
        t = ds.get("truth")
        rg = " | ".join(f"{lo:g}–{hi:g}" for lo, hi in _ranges_of(ds))
        det.append(ds["name"].ljust(7) + f"{len(ds['y']):3d}  " + rg.ljust(46)
                   + ((str(np.round(t["x_opt"], 3).tolist()).ljust(30) + f"{t['y_opt']:.3f}") if t else "—"))
    det.append("")
    det += [f"{ds['label']}: {ds.get('note', '')}" for ds in a.datasets]
    a.add("2.3", SEC_GEO, "Datasets sintéticos de prueba (diseño generado por main.py)", "PASS",
          f"{len(a.datasets)} datasets; óptimo verdadero conocido (coordenadas p* = {np.round(TRUE_P_STAR, 4).tolist()})", "\n".join(det),
          {"n_datasets": len(a.datasets), "p_star": TRUE_P_STAR})


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3: AJUSTE POR MÍNIMOS CUADRADOS ORDINARIOS
# ═════════════════════════════════════════════════════════════════════════════
def chk_ols(a: Audit) -> None:
    """
    β̂ = (MᵀM)⁻¹Mᵀy;  Cov(β̂) = s²(MᵀM)⁻¹;  s² = SS_res/(N−10);  R² = 1−SS_res/SS_tot;  R²_adj = 1−MS_res/MS_tot.
    Referencias independientes: scipy.linalg.lstsq (SVD), QR, ecuaciones normales y (si existe) statsmodels.
    Invariancia: el apalancamiento h_ii = [M(MᵀM)⁻¹Mᵀ]_ii NO depende de la parametrización, así que
    diag(M·Cov(β̂)·Mᵀ)/s² calculado en unidades naturales debe igualar al calculado en unidades codificadas.
    """
    store: Dict[str, Any] = {}
    for ds in a.datasets:
        name = ds["name"]
        rr = a.runs[name]
        if rr.error or rr.betas is None:
            a.add(f"3.{name}", SEC_OLS, f"OLS · {ds['label']}", "FAIL", "ejecutar_ajuste_completo() lanzó una excepción", rr.error or "")
            continue
        try:
            X, y = ds["X"], ds["y"]
            Zm = rr.Z if rr.Z is not None else X                  # matriz sobre la que el programa ajustó (natural o codificada)
            rf = ref_fit(Zm, y)
            cr = coded_reference(X, y, _ranges_of(ds))            # parametrización alternativa (−1…+1) para las invarianzas
            M, N, b, C = rf["M"], rf["N"], rr.betas, rr.cov
            sub = Sub()
            scale = max(1.0, float(np.abs(rf["beta"]).max()))
            d_beta = float(np.abs(b - rf["beta"]).max() / scale)
            ok = np.allclose(b, rf["beta"], rtol=TOL_REL, atol=TOL_ABS)
            sub.add("β̂ vs scipy.linalg.lstsq (allclose rtol=atol=1e-5)", "PASS" if ok else graded_status(d_beta, TOL_REL), f"desviación relativa = {d_beta:.2e}")
            d_qr = float(np.nanmax(np.abs(rf["beta"] - rf["beta_qr"])) / scale)
            d_ne = float(np.nanmax(np.abs(rf["beta"] - rf["beta_ne"])) / scale)
            sub.add("coherencia de las referencias (SVD vs QR)", "PASS" if d_qr < 1e-6 else "WARN", f"SVD–QR = {d_qr:.1e}; SVD–ecuaciones normales = {d_ne:.1e} (las normales pierden ≈cond dígitos)")
            if HAVE_SM:
                res = sm.OLS(y, M).fit()
                d_sm = float(np.abs(b - res.params).max() / scale)
                sub.add("β̂ vs statsmodels.OLS", graded_status(d_sm, TOL_REL), f"desviación relativa = {d_sm:.2e}")
                se_sm, se_main = np.asarray(res.bse), np.sqrt(np.clip(np.diag(C), 0, None))
                d_se = float(np.max(np.abs(se_sm - se_main) / np.maximum(se_sm, 1e-300))) if rf["s"] > 1e-8 else 0.0
                sub.add("errores estándar de β̂ vs statsmodels", graded_status(d_se, 1e-4), f"desviación relativa máx = {d_se:.2e}")
            s2m = rr.s_res ** 2                                           # Cov/s² = (MᵀM)⁺ no depende del ruido: sirve aun con s≈0
            if s2m > 0:
                Cn, ref_c = C / s2m, np.linalg.pinv(M.T @ M)
                cs = float(np.abs(ref_c).max())
                d_cov = float(np.abs(Cn - ref_c).max() / cs) if cs > 0 else 0.0
                sub.add("Cov(β̂)/s² = pinv(MᵀM)", graded_status(d_cov, 1e-6), f"desviación relativa = {d_cov:.2e}")
                sym = float(np.abs(Cn - Cn.T).max() / cs) if cs > 0 else 0.0
                lam = np.linalg.eigvalsh(0.5 * (Cn + Cn.T))
                psd = sym < 1e-9 and lam.min() > -1e-8 * max(1e-300, lam.max())
                sub.add("Cov(β̂) simétrica y semidefinida positiva", "PASS" if psd else "FAIL", f"asimetría = {sym:.1e}; λ_min/λ_max = {lam.min() / max(1e-300, lam.max()):.2e}")
            else:
                d_cov = 0.0
                sub.add("Cov(β̂)", "SKIP", "s = 0")
            if rf["s"] > 1e-6:                                       # invariancia del apalancamiento
                h_nat = np.einsum("ij,jk,ik->i", M, C, M) / rr.s_res ** 2
                h_cod = np.einsum("ij,jk,ik->i", cr["Mc"], cr["cov_unscaled"], cr["Mc"])
                d_h = float(np.abs(h_nat - h_cod).max())
                sub.add("invariancia del apalancamiento h_ii (natural vs codificado)", "PASS" if d_h < 1e-6 else ("WARN" if d_h < 1e-2 else "FAIL"),
                        f"máx |h_nat − h_cod| = {d_h:.2e}  (Σh_ii = {h_nat.sum():.4f}, esperado 10)")
            else:
                d_h = float("nan")
                sub.add("invariancia del apalancamiento h_ii", "SKIP", "s≈0 (ajuste exacto)")
            sub.add("grados de libertad residuales df = N − 10", "PASS" if rr.rep.get("df") == N - 10 else "FAIL", f"reporte df = {rr.rep.get('df')}, esperado {N - 10}")
            d_s = abs(rr.s_res - rf["s"]) / max(rf["s"], 1e-300) if rf["s"] > 1e-8 else abs(rr.s_res - rf["s"])
            sub.add("error estándar residual s", graded_status(float(d_s), 1e-8), f"main = {rr.s_res:.6g}, ref = {rf['s']:.6g}")
            for key, lab in (("r2", "R²"), ("r2a", "R² ajustado")):
                v = rr.rep.get(key)
                sub.add(lab, "SKIP" if v is None else ("PASS" if abs(v - rf[key]) <= PRINT_TOL else "FAIL"),
                        f"reporte = {v}, ref = {rf[key]:.6f}")
            d_y = float(np.abs(rr.y_pred - M @ rf["beta"]).max()) if rr.y_pred is not None else float("nan")
            sub.add("valores ajustados ŷ = Mβ̂", graded_status(d_y / max(1.0, np.abs(y).max()), TOL_REL), f"máx |Δŷ| = {d_y:.2e}")
            d_inv = float(np.abs(rr.y_pred - cr["yhat"]).max() / max(1.0, np.abs(y).max())) if rr.y_pred is not None else float("nan")
            sub.add("ŷ invariante a la parametrización (natural vs codificado)", "PASS" if d_inv < 1e-8 else ("WARN" if d_inv < 1e-4 else "FAIL"), f"desviación relativa = {d_inv:.2e}")
            e = y - rr.y_pred
            if rf["s"] > 1e-8:
                orth = float(np.abs(M.T @ e).max() / (np.linalg.norm(M) * max(np.linalg.norm(e), 1e-300)))
                sub.add("ortogonalidad residual Mᵀe ≈ 0 (ecuaciones normales)", "PASS" if orth < 1e-8 else ("WARN" if orth < 1e-5 else "FAIL"), f"‖Mᵀe‖∞/(‖M‖‖e‖) = {orth:.2e}")
            else:
                sub.add("ortogonalidad residual Mᵀe ≈ 0", "SKIP", "residuos ≈ 0 (ajuste exacto): la razón carece de sentido")
            c_nat, c_cod = rf["cond_MtM"], cr["cond_MtM"]
            digits = max(0.0, math.log10(max(c_nat, 1.0)))
            st_c = "PASS" if c_nat < 1e8 else ("WARN" if c_nat < 1e14 else "FAIL")
            sub.add("condicionamiento de MᵀM en las unidades del ajuste (" + ("variables codificadas" if rr.space == "coded" else "unidades físicas") + ")", st_c,
                    f"cond(MᵀM) = {c_nat:.2e} (alternativa −1…+1: {c_cod:.1f}); " + (f"se pierden ≈{digits:.0f} de 16 dígitos" if digits < 16 else "numéricamente singular: se pierden los 16 dígitos"))
            a.add(f"3.{name}", SEC_OLS, f"OLS · {ds['label']}", sub.status,
                  (short_fail(sub) if sub.failing() else f"coeficientes y estadísticos correctos; cond(MᵀM) = {c_nat:.1e}"), sub.text(),
                  {"d_beta": d_beta, "d_cov": d_cov, "d_leverage": d_h, "cond_MtM_nat": c_nat, "cond_MtM_cod": c_cod, "cond_M_nat": rf["cond_M"], "N": N},
                  "main.py · ejecutar_ajuste_completo (ajuste en unidades naturales)",
                  "Ajustar en variables codificadas (x_cod) y des-codificar los coeficientes, o centrar/escalar las columnas antes de lstsq/pinv.")
            store[name] = {"d_beta": d_beta, "d_cov": d_cov, "cond_nat": c_nat, "cond_cod": c_cod,
                           "beta_main": b, "beta_ref": rf["beta"], "resid": e, "yhat": rr.y_pred}
        except Exception:
            a.add(f"3.{name}", SEC_OLS, f"OLS · {ds['label']}", "FAIL", "excepción durante la auditoría", traceback.format_exc(limit=3))
    a.store["ols"] = store


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4: ANÁLISIS CANÓNICO, JACOBIANO Y MÉTODO DELTA
# ═════════════════════════════════════════════════════════════════════════════
def nature_key(txt: str) -> str:
    """Normaliza el rótulo de naturaleza (p. ej. «MÁXIMO LOCAL ESTRICTO» y «MÁXIMO LOCAL» → MAX)."""
    t = txt.upper()
    return "MAX" if "MÁX" in t or "MAX" in t else ("MIN" if "MÍN" in t or "MIN" in t else ("SADDLE" if "SILLA" in t or "SADDLE" in t else "DEG"))


def classify_eig(lam: np.ndarray) -> str:
    """Naturaleza de la superficie según los autovalores del hessiano."""
    if np.all(lam < 0):
        return "MÁXIMO LOCAL"
    if np.all(lam > 0):
        return "MÍNIMO LOCAL"
    return "PUNTO DE SILLA"


def chk_canonical(a: Audit) -> None:
    """
    Punto estacionario  x* = −H⁻¹b  (equivale a  −½B⁻¹b  con B = H/2).  Verificaciones:
      (i) x*_main = x*_ref;  (ii) ∇f(x*_main) ≈ 0;  (iii) naturaleza por autovalores de H;
      (iv) ŷ(x*) y los intervalos del reporte son coherentes con las fórmulas  ŷ ± t·√(mᵀCov m)  (IC)  y  ŷ ± t·√(s²+mᵀCov m)  (IP).
    La referencia se auto-valida contra gradiente y hessiano por diferencias finitas.
    """
    store: Dict[str, Any] = {}
    for ds in a.datasets:
        name = ds["name"]
        rr = a.runs[name]
        if rr.error or rr.betas is None or rr.x_opt is None:
            a.add(f"4.{name}", SEC_CAN, f"Óptimo canónico · {ds['label']}", "FAIL", "sin resultados del motor", rr.error or "")
            continue
        try:
            b, x_m = rr.betas, rr.x_opt                                # x_m: óptimo reportado (unidades físicas)
            x_rm, H = xstar_ref(b)                                      # referencia en el espacio del modelo
            x_r = rr.to_real(x_rm)                                      # …y en unidades físicas
            x_mm = rr.to_model(x_m)                                     # óptimo reportado llevado al espacio del modelo
            rng_ = _ranges_of(ds)
            delta = np.array([(hi - lo) / 2 for lo, hi in rng_])
            sub = Sub()
            x0 = (rr.Z if rr.Z is not None else ds["X"]).mean(axis=0)  # auto-validación de la referencia
            g_err = float(np.abs(gradient_of(b, x0) - fd_gradient(b, x0)).max() / max(1.0, np.abs(gradient_of(b, x0)).max()))
            h_err = float(np.abs(hessian_of(b) - fd_hessian(b, x0)).max() / max(1.0, np.abs(hessian_of(b)).max()))
            sub.add("referencia: ∇f y H analíticos = diferencias finitas", "PASS" if (g_err < 1e-5 and h_err < 1e-3) else "WARN", f"error ∇ = {g_err:.1e}, error H = {h_err:.1e}")
            if not np.all(np.isfinite(x_r)):
                sub.add("hessiano invertible", "WARN", "H singular: no hay punto estacionario único")
            ok = np.allclose(x_m, x_r, rtol=TOL_REL, atol=TOL_ABS)
            ratio = (x_m / x_r)[np.isfinite(x_m / x_r)]
            sub.add("x*_main = −H⁻¹b (allclose rtol=atol=1e-5)", "PASS" if ok else "FAIL",
                    f"main = {np.round(x_m, 4).tolist()} | ref = {np.round(x_r, 4).tolist()}" + (f" | cociente main/ref = {np.round(ratio, 3).tolist()}" if not ok else ""))
            g = gradient_of(b, x_mm)
            gscale = float(np.linalg.norm(b[1:4]) + np.linalg.norm(H @ x_mm) + 1e-300)
            g_rel = float(np.linalg.norm(g) / gscale)
            sub.add("∇f(x*_main) ≈ 0", "PASS" if g_rel < 1e-6 else "FAIL", f"‖∇f‖ = {np.linalg.norm(g):.3e} (relativo {g_rel:.2e})")
            lam = np.linalg.eigvalsh(H)
            nat_ref = classify_eig(lam)
            nat_main = rr.rep.get("nature")
            sub.add("naturaleza de la superficie (autovalores de H)", "SKIP" if nat_main is None else ("PASS" if nature_key(nat_main) == nature_key(nat_ref) else "FAIL"),
                    f"main = {nat_main}, ref = {nat_ref}, λ(H) = {np.round(lam, 3).tolist()}")
            m_x = design_matrix(x_mm)[0]
            y_opt = float(m_x @ b)
            if rr.rep.get("y_opt") is not None and np.isfinite(y_opt):
                sub.add("ŷ(x*) del reporte = m(x*)ᵀβ̂ (consistencia interna)", "PASS" if abs(rr.rep["y_opt"] - y_opt) <= PRINT_TOL else "FAIL",
                        f"reporte = {rr.rep['y_opt']:.4f}, polinomio = {y_opt:.4f}")
            var_m = max(0.0, float(m_x @ rr.cov @ m_x))
            for key, lab, v in (("ic_half", "IC 95 % de la media", var_m), ("ip_half", "IP 95 % individual", rr.s_res ** 2 + var_m)):
                got = rr.rep.get(key)
                if got is None or not np.isfinite(v):
                    continue
                exp_ = rr.t_crit * math.sqrt(v)
                sub.add(f"{lab}: semiancho = t·√(…) (consistencia interna)", "PASS" if abs(got - exp_) <= 6e-4 else "FAIL", f"reporte = ±{got:.3f}, fórmula = ±{exp_:.3f}")
            t = ds.get("truth")
            if t is not None and np.all(np.isfinite(x_r)):
                e_m = float(np.max(np.abs(x_m - t["x_opt"]) / delta))
                e_r = float(np.max(np.abs(x_r - t["x_opt"]) / delta))
                sub.add("distancia al óptimo verdadero (en unidades de semirrango Δ)", "PASS" if e_m <= max(3 * e_r, 0.02) else "FAIL",
                        f"main = {e_m:.3f}Δ, referencia = {e_r:.3f}Δ  (verdadero {np.round(t['x_opt'], 3).tolist()})")
            a.add(f"4.{name}", SEC_CAN, f"Óptimo canónico · {ds['label']}", sub.status,
                  (f"{len(sub.failing())} sub-chequeos fallan" if sub.failing() else "punto estacionario y naturaleza correctos"), sub.text(),
                  {"x_main": x_m, "x_ref": x_r, "grad_rel": g_rel, "eig": lam, "nature_ref": nat_ref},
                  "main.py · ejecutar_ajuste_completo (B_mat, x_opt = −0.5·B⁻¹·b)",
                  "B_mat ya es el hessiano (diagonal 2·b_ii): usar x_opt = −(B_inv @ b_vec). (Con B = diag(b_ii), b_ij/2 sí corresponde −0.5·B⁻¹·b.)")
            store[name] = {"x_main": x_m, "x_ref": x_r, "x_true": (t or {}).get("x_opt"), "delta": delta, "grad_rel": g_rel,
                           "ranges": rng_, "label": ds["label"]}
        except Exception:
            a.add(f"4.{name}", SEC_CAN, f"Óptimo canónico · {ds['label']}", "FAIL", "excepción durante la auditoría", traceback.format_exc(limit=3))
    a.store["canonical"] = store


def chk_jacobian(a: Audit) -> None:
    """
    calcular_jacobiano_optimo(betas, B_inv, x_opt) debe devolver J = ∂x*/∂β (3×10).
    Se compara columna a columna con (a) la derivación implícita analítica J = −H⁻¹·∂g/∂β y
    (b) la diferencia finita central de x*(β). Se agrupan las columnas: b0 | lineales | interacciones | cuadráticos.
    """
    m = a.mod
    groups = [("b0", [0]), ("lineales b1–b3", [1, 2, 3]), ("interacciones b12,b13,b23", [4, 5, 6]), ("cuadráticos b11,b22,b33", [7, 8, 9])]
    for ds in [a.datasets[0], a.datasets[2]]:
        name = ds["name"]
        rr = a.runs[name]
        if rr.betas is None:
            continue
        b = rr.betas
        xs, H = xstar_ref(b)
        if not np.all(np.isfinite(xs)):
            a.add(f"4.J.{name}", SEC_CAN, f"Jacobiano del óptimo · {ds['label']}", "SKIP", "hessiano singular")
            continue
        try:
            J_main = np.asarray(m.calcular_jacobiano_optimo(b.copy(), np.linalg.inv(H), xs.copy()), float)
        except Exception as e:
            a.add(f"4.J.{name}", SEC_CAN, f"Jacobiano del óptimo · {ds['label']}", "FAIL", f"la función lanzó {type(e).__name__}: {e}")
            continue
        J_an, J_num = jacobian_analytic(b), jacobian_numeric(b)
        sc = max(1e-12, float(np.abs(J_an).max()))
        sub = Sub()
        d_ref = float(np.abs(J_an - J_num).max() / sc)
        sub.add("referencia: J analítica = J por diferencias finitas", "PASS" if d_ref < 1e-4 else "WARN", f"desviación relativa = {d_ref:.1e}")
        bad_cols: List[int] = []
        for gname, cols in groups:
            d_an = float(np.abs(J_main[:, cols] - J_an[:, cols]).max() / sc)
            d_nu = float(np.abs(J_main[:, cols] - J_num[:, cols]).max() / sc)
            st = "PASS" if d_an <= 1e-5 else ("WARN" if d_an <= 1e-3 else "FAIL")
            if st != "PASS":
                bad_cols += cols
            sub.add(f"columnas {gname}", st, f"vs analítica: {d_an:.2e}; vs numérica: {d_nu:.2e}")
        if np.allclose(J_main, J_an, rtol=TOL_REL, atol=1e-9 * sc):
            sub.add("allclose(J_main, J_analítica; rtol=1e-5)", "PASS")
        else:
            sub.add("allclose(J_main, J_analítica; rtol=1e-5)", "FAIL", "columnas con discrepancia: " + ", ".join(BETA_NAMES[k] for k in sorted(set(bad_cols))))
        if name == "D1":
            a.store["jacobian"] = {"main": J_main, "analytic": J_an, "numeric": J_num}
        a.add(f"4.J.{name}", SEC_CAN, f"Jacobiano ∂x*/∂β · {ds['label']}", sub.status,
              ("discrepancia en " + ", ".join(BETA_NAMES[k] for k in sorted(set(bad_cols)))) if bad_cols else "las 30 derivadas parciales coinciden con la referencia",
              sub.text(), {"cols_bad": [BETA_NAMES[k] for k in sorted(set(bad_cols))], "max_rel_dev": float(np.abs(J_main - J_an).max() / sc)},
              "main.py · calcular_jacobiano_optimo (bloque lineal J[:, 1:4])",
              "Con x* = −H⁻¹b la derivada respecto de b_j es −H⁻¹[:, j]: quitar el factor 0.5 del bloque lineal.")


def chk_delta(a: Audit) -> None:
    """
    Método Delta:  Cov(x*) ≈ J·Cov(β̂)·Jᵀ;  IC 95 % de cada coordenada = x*_i ± t_{0.975,ν}·√Cov(x*)_ii.
    Se compara el error estándar reportado por main (ic_coords) con la referencia y se valida la
    aproximación lineal con simulación Monte Carlo: β ~ N(β̂, Cov) → x*(β) (resuelto en lote).
    """
    n_mc = 4000 if a.quick else 20000
    for ds in [a.datasets[0], a.datasets[2]]:
        name = ds["name"]
        rr = a.runs[name]
        if rr.betas is None or rr.cov is None:
            continue
        b, C = rr.betas, 0.5 * (rr.cov + rr.cov.T)
        xs, _ = xstar_ref(b)
        if not np.all(np.isfinite(xs)):
            continue
        J = jacobian_analytic(b)
        Sx = J @ C @ J.T
        sv = rr.scale_vec()                                               # modelo codificado → SE en unidades físicas
        se_ref = np.sqrt(np.clip(np.diag(Sx), 0, None)) * sv
        df = len(ds["y"]) - 10
        t_ex = float(stats.t.ppf(0.975, df)) if df > 0 else float("nan")
        sub = Sub()
        if len(rr.ic_coords) == 3 and all(np.isfinite(c[2]) for c in rr.ic_coords):
            se_m = np.array([c[2] for c in rr.ic_coords], float)
            d_se = float(np.max(np.abs(se_m - se_ref) / np.maximum(se_ref, 1e-300)))
            sub.add("error estándar de x* (main) vs J·Cov·Jᵀ de referencia", "PASS" if d_se < 1e-3 else "FAIL",
                    f"main = {np.round(se_m, 4).tolist()} | ref = {np.round(se_ref, 4).tolist()} | desviación relativa máx = {d_se:.2e}")
            hw_m = np.array([(c[1] - c[0]) / 2 for c in rr.ic_coords])
            sub.add("semiancho del IC 95 % de x* = t_exacto·SE_ref", "PASS" if np.allclose(hw_m, t_ex * se_ref, rtol=1e-3, atol=1e-6) else "FAIL",
                    f"main = {np.round(hw_m, 4).tolist()} | ref = {np.round(t_ex * se_ref, 4).tolist()}")
        else:
            sub.add("IC de coordenadas del reporte", "FAIL", "ic_coords vacío o no finito")
        rng = np.random.default_rng(a.seed + 7)
        try:
            draws = rng.multivariate_normal(b, C, size=n_mc, method="eigh")
            xs_mc = xstar_batch(draws)
            xs_mc = xs_mc[np.all(np.isfinite(xs_mc), axis=1)]
            xs_model = xs.copy()
            xs_mc_model = xs_mc.copy()
            xs_mc = rr.to_real(xs_mc)                                      # simulaciones en unidades físicas
            sd_mc = np.std(xs_mc, axis=0)
            q75, q25 = np.percentile(xs_mc, [75, 25], axis=0)
            sd_rob = (q75 - q25) / 1.349                           # SD robusta (IQR): resiste colas pesadas del cociente
            ratio = sd_rob / np.maximum(se_ref, 1e-300)
            worst_r = float(np.abs(ratio - 1).max())
            st = "PASS" if worst_r < 0.15 else ("WARN" if worst_r < 0.35 else "FAIL")
            sub.add(f"validez de la aproximación lineal (Monte Carlo, n={len(xs_mc)})", st,
                    f"SE_MC(robusto) / SE_Delta = {np.round(ratio, 3).tolist()} (ideal 1)")
            xs_r = rr.to_real(xs)
            lo, hi = xs_r - t_ex * se_ref, xs_r + t_ex * se_ref
            cover = float(np.mean(np.all((xs_mc >= lo) & (xs_mc <= hi), axis=1)))
            sub.add("cobertura del rectángulo IC (referencia) sobre la simulación", "PASS" if cover > 0.88 else "WARN",
                    f"{100 * cover:.1f} % de las simulaciones caen dentro del IC conjunto (el IC marginal individual cubre ≈ 95 %)")
            if name == "D1":
                a.store["delta"] = {"xs": xs_r, "samples": xs_mc[:3000], "se_ref": se_ref, "se_main": (se_m if 'se_m' in locals() else None),
                                    "t": t_ex, "ratio": ratio}
        except Exception as e:
            sub.add("simulación Monte Carlo", "WARN", f"no se pudo ejecutar: {type(e).__name__}: {e}")
        a.add(f"4.Δ.{name}", SEC_CAN, f"Método Delta (J·Cov·Jᵀ) e IC del óptimo · {ds['label']}", sub.status,
              (f"{len(sub.failing())} sub-chequeos fallan" if sub.failing() else "propagación de incertidumbre correcta"), sub.text(),
              {"se_ref": se_ref, "t_exact": t_ex}, "main.py · método Delta en ejecutar_ajuste_completo")


def chk_coverage(a: Audit) -> None:
    """
    Prueba extremo-a-extremo (Monte Carlo): se simulan R datasets a partir de la verdad conocida y se mide, para cada factor,
    la frecuencia con que el IC 95 % que REPORTA main contiene al óptimo verdadero (cobertura nominal ≈ 0.95), junto con
    el sesgo normalizado (x̂ − x_true)/Δ. Se compara con el mismo procedimiento implementado con la referencia correcta.
    """
    d1 = a.datasets[0]
    rr0 = a.runs["D1"]
    Z0 = rr0.Z if rr0.Z is not None else d1["X"]
    X, cod = d1["X"], d1["cod"]
    alpha = np.array(a.mod.ALPHA, float)
    rng_ = _ranges_of(d1)
    c = np.array([(lo + hi) / 2 for lo, hi in rng_])
    delta = np.array([(hi - lo) / 2 for lo, hi in rng_])
    mu = truth_fn((X - c) / delta * alpha)
    x_true = d1["truth"]["x_opt"]
    R = 120 if a.quick else 400
    rs = np.random.default_rng(a.seed + 999)
    hit_m, hit_r = np.zeros(3), np.zeros(3)
    xm_all, xr_all = [], []
    for _ in range(R):
        yr = mu + rs.normal(0.0, d1["sigma"], len(mu))
        rr = run_fit(a, {"name": "cov", "X": X, "y": yr, "cod": cod})
        rf = ref_fit(Z0, yr)
        xr = rr0.to_real(xstar_ref(rf["beta"])[0])
        J = jacobian_analytic(rf["beta"])
        se = np.sqrt(np.clip(np.diag(J @ rf["cov_inv"] @ J.T), 0, None)) * rr0.scale_vec()
        t = stats.t.ppf(0.975, rf["df"])
        hit_r += (np.abs(xr - x_true) <= t * se)
        xr_all.append(xr)
        if rr.x_opt is not None and len(rr.ic_coords) == 3:
            lo = np.array([ic[0] for ic in rr.ic_coords])
            hi = np.array([ic[1] for ic in rr.ic_coords])
            hit_m += ((x_true >= lo) & (x_true <= hi))
            xm_all.append(rr.x_opt)
    cov_m, cov_r = hit_m / R, hit_r / R
    xm_all, xr_all = np.array(xm_all), np.array(xr_all)
    bias_m = (xm_all.mean(axis=0) - x_true) / delta if len(xm_all) else np.full(3, np.nan)
    bias_r = (xr_all.mean(axis=0) - x_true) / delta
    if cov_m.min() >= 0.90 and np.abs(cov_m - cov_r).max() <= 0.05:
        st = "PASS"
    elif cov_m.min() >= 0.80:
        st = "WARN"
    else:
        st = "FAIL"
    det = [f"R = {R} datasets simulados (σ = {d1['sigma']}), óptimo verdadero = {np.round(x_true, 3).tolist()}",
           f"Cobertura IC 95 % reportado por main : {np.round(cov_m, 3).tolist()}",
           f"Cobertura con la referencia correcta  : {np.round(cov_r, 3).tolist()}  (nominal 0.95)",
           f"Sesgo normalizado (x̂−x_true)/Δ main  : {np.round(bias_m, 3).tolist()}",
           f"Sesgo normalizado, referencia          : {np.round(bias_r, 3).tolist()}"]
    a.store["coverage"] = {"cov_main": cov_m, "cov_ref": cov_r, "bias_main": bias_m, "bias_ref": bias_r, "R": R}
    a.add("4.C", SEC_CAN, "Cobertura extremo-a-extremo del IC 95 % del óptimo (Monte Carlo)", st,
          f"cobertura main = {np.round(cov_m, 2).tolist()} vs referencia = {np.round(cov_r, 2).tolist()} (nominal 0.95)", "\n".join(det),
          {"coverage_main": cov_m, "coverage_ref": cov_r, "bias_main": bias_m, "bias_ref": bias_r, "R": R}, "main.py · óptimo + IC (método Delta)")


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 5: ANOVA Y FALTA DE AJUSTE (LACK-OF-FIT)
# ═════════════════════════════════════════════════════════════════════════════
def _cmp_lof(sub: Sub, d: Dict[str, Any], lr: Dict[str, Any]) -> None:
    """Compara los campos del diagnóstico de main (4 decimales impresos) con la referencia."""
    def num(key: str, label: str, ref: float) -> None:
        v = d.get(key)
        if v is None:
            sub.add(label, "FAIL", "el campo no aparece en el diagnóstico")
        else:
            sub.add(label, "PASS" if abs(v - ref) <= PRINT_TOL + 1e-9 * abs(ref) else "FAIL", f"main = {v}, ref = {ref:.4f}")
    num("ss_res", "SS_residual total", lr["ss_res"])
    num("ss_pe", "SS_PE (error puro) = Σ(y−ȳ_grupo)²", lr["ss_pe"])
    sub.add("df_PE = Σ(n_g − 1)", "PASS" if d.get("df_pe") == lr["df_pe"] else "FAIL", f"main = {d.get('df_pe')}, ref = {lr['df_pe']}")
    num("ss_lof", "SS_LOF = SS_res − SS_PE", lr["ss_lof"])
    sub.add("df_LOF = (N−10) − df_PE", "PASS" if d.get("df_lof") == lr["df_lof"] else "FAIL", f"main = {d.get('df_lof')}, ref = {lr['df_lof']}")
    f_main = d.get("F")
    main_ok = f_main is not None and np.isfinite(f_main)
    if lr["computable"]:
        if not main_ok:
            sub.add("estadístico F = MS_LOF/MS_PE y p-valor", "FAIL", "main declara «no computable» pero la referencia sí lo es")
        else:
            okF = abs(f_main - lr["F"]) <= PRINT_TOL + 1e-4 * abs(lr["F"])
            okp = d.get("p") is not None and abs(d["p"] - lr["p"]) <= PRINT_TOL
            sub.add("estadístico F = MS_LOF/MS_PE", "PASS" if okF else "FAIL", f"main = {f_main}, ref = {lr['F']:.4f}")
            sub.add("p-valor = P(F_{df_LOF, df_PE} > F)", "PASS" if okp else "FAIL", f"main = {d.get('p')}, ref = {lr['p']:.4f}")
    else:
        sub.add("F y p no computables (df_PE=0, df_LOF≤0 o MS_PE=0)", "PASS" if not main_ok else "FAIL",
                "ambos coinciden en «no computable»" if not main_ok else f"main reporta F = {f_main} donde la referencia es indefinida")


def chk_lof(a: Audit) -> None:
    """
    Descomposición  SS_res = SS_PE + SS_LOF;  F = (SS_LOF/df_LOF)/(SS_PE/df_PE) ~ F(df_LOF, df_PE) bajo H0 (el modelo cuadrático basta).
    La referencia agrupa TODAS las réplicas exactas (no solo el centro); main usa el centro (si hay ≥2) o np.unique.
    """
    store: Dict[str, Any] = {}
    for ds in a.datasets:
        name = ds["name"]
        rr = a.runs[name]
        if rr.error or not rr.diag:
            a.add(f"5.{name}", SEC_LOF, f"Lack-of-Fit · {ds['label']}", "FAIL", "sin diagnóstico", rr.error or "")
            continue
        lr = lof_reference(ds["X"], ds["y"])
        sub = Sub()
        _cmp_lof(sub, rr.lof, lr)
        if ds["name"] == "D3" and lr["computable"]:       # el dataset con falta de ajuste debe ser rechazado por la prueba
            sub.add("detección de falta de ajuste conocida (D3)", "PASS" if lr["p"] < 0.05 else "WARN", f"p_ref = {lr['p']:.2e} (se espera < 0.05)")
        a.add(f"5.{name}", SEC_LOF, f"Lack-of-Fit · {ds['label']}", sub.status,
              (short_fail(sub) if sub.failing() else
               (f"F = {lr['F']:.3f}, p = {lr['p']:.4f}" if lr["computable"] else "no computable (coincide con la referencia)")),
              sub.text(), {"ref": lr, "main": rr.lof}, "main.py · ejecutar_ajuste_completo (bloque Lack-of-Fit)")
        store[name] = {"ref": lr, "main": rr.lof}
    a.store["lof"] = store
    # Escenario: CSV sin columnas codificadas -> main asume x_cod = 0 en todas las corridas
    rr = a.runs.get("D1|sin_cod")
    if rr is not None and rr.diag:
        lr = lof_reference(a.datasets[0]["X"], a.datasets[0]["y"])
        sub = Sub()
        _cmp_lof(sub, rr.lof, lr)
        a.add("5.cod", SEC_LOF, "Lack-of-Fit tras importar un CSV sin columnas x_cod (D1)", sub.status,
              ("las réplicas se detectan mal: " + short_fail(sub, k=3, w=30)) if sub.failing() else "robusto ante la ausencia de x_cod",
              sub.text() + "\n\nSi el CSV no trae x_cod, el cargador puede dejarlas en 0 en todas las filas: "
              "entonces ejecutar_ajuste_completo interpreta TODAS las corridas como puntos centrales (SS_PE sobre las N corridas, df_LOF negativo).",
              {"main": rr.lof, "ref": lr}, "main.py · cargar_csv_respuestas (rama else de c1,c2,c3) + ejecutar_ajuste_completo",
              "Si faltan x_cod, calcularlas desde los reales: p = (x − c)/Δ, o detectar réplicas por filas idénticas de X (np.unique).")


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 6: CAJA BLANCA (ANÁLISIS ESTÁTICO DEL CÓDIGO + PRUEBAS DE ROBUSTEZ)
# ═════════════════════════════════════════════════════════════════════════════
def _find_func(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _stores(nodes) -> set:
    return {n.id for s in nodes for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _loads(nodes) -> set:
    return {n.id for s in nodes for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def unbound_candidates(func: ast.FunctionDef) -> List[Tuple[int, List[str]]]:
    """
    Nombres asignados SOLO dentro de un `try` (y no antes ni en el `except`) y usados después del `try`:
    si el `try` falla antes de asignarlos, el uso posterior lanza UnboundLocalError.
    """
    out = []
    args = {x.arg for x in func.args.args}
    for i, stmt in enumerate(func.body):
        if isinstance(stmt, ast.Try):
            if stmt.handlers and all(h.body and isinstance(h.body[-1], (ast.Return, ast.Raise)) for h in stmt.handlers):
                continue                       # si el except termina con return/raise, no hay camino hacia el uso posterior
            before = _stores(func.body[:i]) | args
            in_try = _stores(stmt.body)
            in_exc = _stores(list(stmt.handlers)) | _stores(stmt.finalbody)
            after = _loads(func.body[i + 1:])
            risky = sorted((in_try - before - in_exc) & after)
            if risky:
                out.append((stmt.lineno, risky))
    return out


@contextlib.contextmanager
def _inv_raises():
    """Inyecta una falla: np.linalg.inv lanza LinAlgError (matriz singular) mientras dura el contexto."""
    orig = np.linalg.inv

    def boom(*_a, **_k):
        raise np.linalg.LinAlgError("Singular matrix (falla inyectada por el auditor)")
    np.linalg.inv = boom
    try:
        yield
    finally:
        np.linalg.inv = orig


def chk_whitebox(a: Audit) -> None:
    """Análisis estático (AST/regex) + pruebas dinámicas de robustez sobre el código de main.py."""
    src = a.src

    def find_any(name: str):
        """Busca la función en main.py y en los módulos hermanos (p. ej. doehlert_engine.py)."""
        for _p, s_, t_ in a.sources:
            n_ = _find_func(t_, name)
            if n_ is not None:
                return n_, s_
        return None, ""
    f_ppf, s_ppf = find_any("normal_ppf")
    f_fit, s_fit = find_any("ejecutar_ajuste_completo")
    f_jac, s_jac = find_any("calcular_jacobiano_optimo")
    f_can, s_can = find_any("canonical_analysis")
    seg = lambda n, s_: (ast.get_source_segment(s_, n) or "") if n is not None else ""

    # 6.1 Constantes publicadas del algoritmo BSM
    consts: Dict[str, Any] = {}
    if f_ppf is not None:
        for st in ast.walk(f_ppf):
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and st.targets[0].id in ("a", "b", "c"):
                try:
                    consts[st.targets[0].id] = ast.literal_eval(st.value)
                except Exception:
                    pass
    if len(consts) == 3:
        diffs = []
        for nm, pub in (("a", BSM_A), ("b", BSM_B), ("c", BSM_C)):
            got = consts[nm]
            if len(got) != len(pub):
                diffs.append((nm, -1, float("inf")))
                continue
            diffs += [(nm, k, abs(g - p)) for k, (g, p) in enumerate(zip(got, pub)) if g != p]
        mx = max((d[2] for d in diffs), default=0.0)
        st = "PASS" if mx == 0 else ("WARN" if mx < 1e-6 else "FAIL")
        a.add("6.1", SEC_WB, "Coeficientes de Beasley-Springer-Moro vs valores publicados", st,
              "idénticos a los publicados" if mx == 0 else f"{len(diffs)} constante(s) difieren (máx {mx:.2e})",
              "\n".join(f"{nm}[{k}] difiere en {d:.3e}" for nm, k, d in diffs) or "a, b y c coinciden dígito a dígito con Moro (1995).",
              {"max_diff": mx, "n_diff": len(diffs)}, "main.py · normal_ppf", "Usar c[5] = 0.0003951896511919.")
    else:
        a.add("6.1", SEC_WB, "Coeficientes de Beasley-Springer-Moro vs valores publicados", "SKIP", "no se pudieron extraer las listas a, b, c por AST")

    # 6.2 Orden de evaluación del polinomio (Horner)
    s = seg(f_ppf, s_ppf)
    if re.search(r"x\s*=\s*c\[0\]", s) and re.search(r"for\s+\w+\s+in\s+c\[1:\]", s):
        a.add("6.2", SEC_WB, "Horner de la cola en normal_ppf: orden de coeficientes", "FAIL",
              "el bucle parte de c[0] y avanza hacia c[8]: Horner exige comenzar por el de mayor grado (c[8])",
              "Patrón detectado: x = c[0]; for c_val in c[1:]: x = x*r + c_val  ⇒ evalúa Σ c[8−k] r^k en lugar de Σ c[k] r^k.\n"
              "La prueba numérica 1.1 lo confirma (errores de varias unidades en las colas).", {}, "main.py · normal_ppf (bucle de Horner)",
              "x = c[8]; for c_val in reversed(c[:8]): x = x*r + c_val")
    elif re.search(r"x\s*=\s*c\[8\]|reversed\(c", s):
        a.add("6.2", SEC_WB, "Horner de la cola en normal_ppf: orden de coeficientes", "PASS", "Horner desde el coeficiente de mayor grado")
    else:
        a.add("6.2", SEC_WB, "Horner de la cola en normal_ppf: orden de coeficientes", "SKIP", "patrón no reconocido; ver la prueba numérica 1.1")

    # 6.3 Coherencia hessiano / factor del punto estacionario / jacobiano
    sf, sj = seg(f_fit, s_fit), seg(f_jac, s_jac)
    sc_ = seg(f_can, s_can)
    d_main = 2.0 if re.search(r"2(\.0)?\s*\*\s*betas\[7\]", sf) else (1.0 if re.search(r"\[\s*betas\[7\]\s*,", sf) else None)
    f_x = 0.5 if re.search(r"x_opt\s*=\s*-\s*0\.5\s*\*", sf) else (1.0 if re.search(r"x_opt\s*=\s*-\s*\(?\s*B_inv", sf) else None)
    f_j = 0.5 if re.search(r"-\s*0\.5\s*\*\s*B_inv\[", sj) else (1.0 if re.search(r"=\s*-\s*B_inv\[", sj) else None)
    new_arch = (re.search(r"x_s\s*=\s*-\s*0\.5\s*\*\s*np\.linalg\.solve\(\s*B\b", sc_) and re.search(r"inv\(\s*2(\.0)?\s*\*\s*B\s*\)", sc_)
                and re.search(r"J\[:,\s*1\s*\+\s*j\]\s*=\s*-\s*H_inv\[:,\s*j\]", sj))
    if None in (d_main, f_x, f_j) and new_arch:
        a.add("6.3", SEC_WB, "Coherencia B ↔ x* = −½B⁻¹b ↔ jacobiano (convención del motor nuevo)", "PASS",
              "B estándar (b_ii, b_ij/2): x* = −½B⁻¹b, y el jacobiano usa H⁻¹ = (2B)⁻¹ con J[:,1+j] = −H⁻¹[:,j]",
              "Convención consistente: B = H/2  ⇒  −½B⁻¹b = −H⁻¹b. Se confirma además dinámicamente en 4.* y 4.J.*.", {}, "doehlert_engine.py · canonical_analysis / calcular_jacobiano_optimo")
    elif None in (d_main, f_x, f_j):
        a.add("6.3", SEC_WB, "Coherencia B_mat ↔ x* = −f·B⁻¹b ↔ jacobiano", "SKIP", "patrones no reconocidos; ver 4.* y 4.J.*")
    else:
        need = d_main / 2.0                         # si B_mat = d·B_std, el factor correcto es d/2
        ok = (f_x == need) and (f_j == need)
        a.add("6.3", SEC_WB, "Coherencia B_mat ↔ x* = −f·B⁻¹b ↔ jacobiano", "PASS" if ok else "FAIL",
              "consistentes" if ok else f"B_mat usa diagonal {d_main:g}·b_ii ⇒ f correcto = {need:g}; el código aplica f = {f_x:g} en x_opt y f = {f_j:g} en el jacobiano",
              f"Si B_mat = [[{d_main:g}·b11, b12, b13], …] entonces ∇f = b + B_mat·x ⇒ x* = −B_mat⁻¹·b·({need:g}·2/{d_main:g}).\n"
              "Con la matriz B estándar (b_ii, b_ij/2) la fórmula x* = −½B⁻¹b es correcta; mezclar ambas convenciones duplica el error.",
              {"diag_factor": d_main, "x_factor": f_x, "jac_factor": f_j}, "main.py · ejecutar_ajuste_completo y calcular_jacobiano_optimo",
              "Dejar B_mat como hessiano y usar x_opt = −(B_inv @ b_vec) y J[:, 1+j] = −B_inv[:, j].")

    # 6.4 Variables posiblemente sin asignar tras un try/except (en el motor y en la interfaz)
    cand_all: List[Tuple[str, int, List[str]]] = []
    for fname in ("ejecutar_ajuste_completo", "canonical_analysis", "fit_quadratic_model", "compute_anova", "cargar_csv_respuestas"):
        node_, _s = find_any(fname)
        if node_ is not None:
            cand_all += [(fname, ln, ns) for ln, ns in unbound_candidates(node_)]
    names = sorted({n for _f, _l, ns in cand_all for n in ns})
    a.add("6.4", SEC_WB, "Variables asignadas solo dentro de try y usadas después (UnboundLocalError)", "FAIL" if names else "PASS",
          ("posibles NameError/UnboundLocalError: " + ", ".join(names)) if names else "todas las variables usadas tras un try están definidas",
          "\n".join(f"{fn}: try en la línea {ln}: {', '.join(ns)}" for fn, ln, ns in cand_all), {"names": names},
          "bloques try del óptimo", "Inicializar esas variables en NaN antes del try: delta_mean = delta_pred = float('nan').")

    # 6.5 Robustez dinámica: matriz B singular (np.linalg.inv lanza LinAlgError)
    with _inv_raises():
        rr = run_fit(a, a.datasets[0], tag="|singular")
    if rr.error:
        last = rr.error.strip().splitlines()[-1]
        a.add("6.5", SEC_WB, "Robustez ante hessiano singular (np.linalg.inv → LinAlgError inyectado)", "FAIL",
              f"el motor se cae: {last}", rr.error, {}, "main.py · bloque except Exception del óptimo",
              "Inicializar delta_mean/delta_pred y completar el reporte con NaN cuando falle la inversión.")
    else:
        a.add("6.5", SEC_WB, "Robustez ante hessiano singular (np.linalg.inv → LinAlgError inyectado)", "PASS",
              "el motor degrada con gracia (óptimo NaN, reporte completo)")

    # 6.6 Otras observaciones de calidad numérica
    obs: List[str] = []
    for pth, s_, _t in a.sources:
        for m_ in re.finditer(r"np\.random\.seed\(", s_):
            obs.append(f"{os.path.basename(pth)}, línea {s_[:m_.start()].count(chr(10)) + 1}: np.random.seed() modifica el generador global (usar np.random.default_rng(42)).")
    if f_fit is not None:
        sf = seg(f_fit, s_fit)
        n_exc = [h.lineno for h in ast.walk(f_fit) if isinstance(h, ast.ExceptHandler) and isinstance(h.type, ast.Name) and h.type.id == "Exception"]
        if n_exc:
            obs.append(f"línea(s) {n_exc}: `except Exception` silencia errores numéricos del óptimo (se convierten en NaN sin aviso).")
        if re.search(r"np\.linalg\.eig\(", sf):
            obs.append("autovalores con np.linalg.eig sobre una matriz simétrica: usar eigvalsh (reales y más estable).")
        if re.search(r"residuos\s*/\s*s_res", sf):
            obs.append("los residuos «estandarizados» e/s no corrigen por el apalancamiento: los studentizados son e/(s·√(1−h_ii)).")
        if re.search(r"pinv\(", sf):
            obs.append("Cov(β̂) usa pinv(MᵀM): con cond(MᵀM) > 1/ε puede truncar valores singulares sin aviso (ver 3.*).")
    f_gen = _find_func(a.tree, "generar_matriz_diseno")
    if f_gen is not None and re.search(r"round\([^)]*,\s*3\)", seg(f_gen, src)):
        obs.append("generar_matriz_diseno redondea los valores reales a 3 decimales (ver 2.2).")
    a.add("6.6", SEC_WB, "Otras observaciones de calidad numérica y mantenibilidad", "WARN" if obs else "PASS",
          f"{len(obs)} observación(es)" if obs else "sin observaciones", "\n".join("• " + o for o in obs))


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 7: ENTRADA / SALIDA DE CSV
# ═════════════════════════════════════════════════════════════════════════════
def csv_text(rows, headers, sep=",", dec=".", sepline=False, crlf=False, blanks=False, drop_cod=False) -> str:
    """Construye el texto de un CSV de la Suite con variantes (separador, decimal, saltos de línea, líneas vacías)."""
    hdr = headers[:5] if drop_cod else headers
    lines: List[str] = []
    if sepline:
        lines.append("sep=" + sep)
    lines.append(sep.join(hdr))
    if blanks:
        lines.append("")
    for k, r in enumerate(rows):
        cells = [str(r[0])]
        for i in range(1, len(r)):
            v = r[i]
            cells.append(f"{float(v):.4f}" if (i in (1, 2, 3, 5, 6, 7) and v != "") else str(v))
        if drop_cod:
            cells = cells[:5]
        line = sep.join(cells)
        lines.append(line.replace(".", dec) if dec != "." else line)
        if blanks and k % 4 == 3:
            lines.append("")
    return ("\r\n" if crlf else "\n").join(lines) + ("\r\n" if crlf else "\n")


def _load_with_main(a: Audit, path: str):
    """Ejecuta cargar_csv_respuestas() de main sobre `path`; devuelve (app, avisos_de_error)."""
    app = make_headless_app(a.mod)
    with patched_dialogs(a.mod, open_path=path) as log:
        try:
            app.cargar_csv_respuestas()
        except Exception as e:                      # una excepción no capturada por main es en sí un defecto
            log.append(("exception", type(e).__name__, str(e)))
    return app, [m for k, _t, m in log if k in ("showerror", "exception")]


def chk_io(a: Audit) -> None:
    """
    Importación/exportación de main: (a) ida y vuelta exportar → importar, (b) formatos de CSV habituales
    (Excel en español con «;» y coma decimal, BOM, CRLF, líneas vacías, sin x_cod), (c) datos incompletos
    deben producir un error claro, y (d) los datasets sintéticos que exporta este auditor deben cargar en main.
    """
    tmp = tempfile.mkdtemp(prefix="audit_io_")
    ds = a.datasets[0]
    app, _ = generate_design(a, DEFAULT_RANGES, 3)
    rows = [list(r) for r in app.matriz_generada]
    for r, yv in zip(rows, ds["y"]):
        r[4] = f"{yv:.2f}"
    app.matriz_generada = rows
    X0 = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
    y0 = np.array([float(r[4]) for r in rows])
    c0 = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])

    # (a) ida y vuelta con el exportador real
    sub = Sub()
    p_exp = os.path.join(tmp, "export_propio.csv")
    with patched_dialogs(a.mod, save_path=p_exp) as log:
        try:
            app.exportar_csv_plantilla()
        except Exception as e:
            log.append(("exception", type(e).__name__, str(e)))
    if not os.path.isfile(p_exp):
        sub.add("exportar_csv_plantilla escribe el archivo", "FAIL", "no se creó el archivo; avisos: " + str([m for _k, _t, m in log][:1]))
    else:
        with open(p_exp, "r", encoding="utf-8-sig") as f:
            txt = f.read().splitlines()
        sub.add("archivo exportado con punto decimal y coma como separador", "PASS" if (len(txt) >= 2 and "," in txt[-1] and ";" not in txt[-1]) else "FAIL", f"1.ª línea: «{txt[0]}»")
        app2, errs = _load_with_main(a, p_exp)
        n_ok = app2.y_exp is not None and len(app2.y_exp) == len(rows)
        if n_ok:
            e_x = float(np.abs(app2.X_exp - X0).max()); e_y = float(np.abs(app2.y_exp - y0).max()); e_c = float(np.abs(app2.cod_exp - c0).max())
            sub.add("ida y vuelta exportar → importar", "PASS" if max(e_x, e_y, e_c) < 6e-5 else "FAIL", f"máx|ΔX|={e_x:.1e}, máx|Δy|={e_y:.1e}, máx|Δx_cod|={e_c:.1e}")
        else:
            sub.add("ida y vuelta exportar → importar", "FAIL", "main no puede importar su propio archivo exportado: " + str(errs[:1]))
    a.add("7.1", SEC_IO, "Exportación → importación de la planilla (ida y vuelta)", sub.status,
          "la Suite reimporta lo que exporta sin pérdida" if sub.status == "PASS" else "falla la ida y vuelta", sub.text(), {}, "main.py · exportar_csv_plantilla / cargar_csv_respuestas")

    # (b) formatos de CSV
    hdr = list(app.headers_diseno)
    cases = [("formato propio (sep=, · UTF-8)", dict(sepline=True), "utf-8", "load", True),
             ("formato anterior (sin sep=, · UTF-8 con BOM)", dict(), "utf-8-sig", "load", True),
             ("saltos de línea CRLF (Windows)", dict(crlf=True), "utf-8-sig", "load", True),
             ("líneas en blanco intercaladas", dict(blanks=True), "utf-8", "load", True),
             ("Excel en español: «;» y coma decimal", dict(sep=";", dec=","), "utf-8-sig", "load", False),
             ("sin columnas x_cod", dict(drop_cod=True), "utf-8", "load", True),
             ("menos de 13 corridas", dict(), "utf-8", "error", True),
             ("una respuesta vacía", dict(), "utf-8", "error", False)]
    sub = Sub()
    table = []
    for name, kw, enc, expect, required in cases:
        rws = [list(r) for r in rows]
        if name.startswith("menos"):
            rws = rws[:8]
        if name.startswith("una respuesta"):
            rws[5][4] = ""
        path = os.path.join(tmp, re.sub(r"\W+", "_", name) + ".csv")
        with open(path, "w", encoding=enc, newline="") as f:
            f.write(csv_text(rws, hdr, **kw))
        app2, errs = _load_with_main(a, path)
        loaded = app2.y_exp is not None and len(app2.y_exp) == len(rws)
        if expect == "load":
            ok = loaded
            if loaded:
                ok = np.allclose(app2.X_exp, X0, atol=1e-3) and np.allclose(app2.y_exp, y0, atol=1e-3)
            msg = "cargado y coincide" if ok else ("error: " + (errs[0][:90] if errs else "no se cargó"))
            if ok and kw.get("drop_cod") and np.allclose(app2.cod_exp, 0):
                sub.add(name + " → x_cod resultante", "WARN", "x_cod queda en 0 para todas las corridas (todas pasan a ser «centro»; ver 5.cod)")
        else:
            ok = not loaded and bool(errs)
            msg = "rechazado con aviso" if ok else ("cargó silenciosamente datos incompletos" if loaded else "falló sin mensaje")
        st = "PASS" if ok else ("FAIL" if required else "WARN")
        sub.add(name, st, msg)
        table.append((name, st))
    a.store["io_cases"] = table
    a.add("7.2", SEC_IO, "Importación de CSV: formatos habituales y datos incompletos", sub.status,
          f"{sum(1 for _, s in table if s == 'PASS')} de {len(table)} casos correctos", sub.text(), {"cases": table},
          "main.py · cargar_csv_respuestas",
          "Detectar el separador (';' si la cabecera lo contiene y no hay ','), aceptar coma decimal y avisar si falta alguna respuesta.")

    # (d) datasets sintéticos del auditor deben ser importables en main
    sub = Sub()
    for d in a.datasets:
        if d["kind"] != "synthetic":
            continue
        p = os.path.join(tmp, f"{d['name']}.csv")
        write_dataset_csv(d, p)
        app2, errs = _load_with_main(a, p)
        ok = app2.y_exp is not None and len(app2.y_exp) == len(d["y"]) and np.allclose(app2.y_exp, d["y"], atol=1e-4) and np.allclose(app2.X_exp, d["X"], atol=1e-3)
        sub.add(f"{d['name']} exportado → importado por main", "PASS" if ok else "FAIL", "ok" if ok else str(errs[:1]))
    a.add("7.3", SEC_IO, "Los datasets sintéticos exportados por el auditor se importan en main.py", sub.status,
          "todos los CSV de datasets/ cargan correctamente" if sub.status == "PASS" else "algún dataset no carga", sub.text())


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 8: DATASET REAL IMPORTADO (--import-data)
# ═════════════════════════════════════════════════════════════════════════════
def chk_imported(a: Audit) -> None:
    """Audita un CSV real del usuario: ¿lo carga main?, ¿es el diseño estimable?, y (vía los pasos 3-5) todos los resultados."""
    if not a.import_data:
        return
    info = a.imported_info
    if not info.get("ok"):
        a.add("8.1", SEC_IMP, "Importación del dataset", "FAIL", "no se pudo interpretar el CSV", info.get("error", ""))
        return
    ds = next(d for d in a.datasets if d["name"] == "IMP")
    sub = Sub()
    app2, errs = _load_with_main(a, a.import_data)
    if app2.y_exp is not None and len(app2.y_exp) == len(ds["y"]):
        ok = np.allclose(app2.X_exp, ds["X"], atol=1e-6) and np.allclose(app2.y_exp, ds["y"], atol=1e-9)
        sub.add("el cargador de main.py lee este archivo", "PASS" if ok else "FAIL", "datos idénticos a los del lector del auditor" if ok else "main lee valores distintos")
    else:
        sub.add("el cargador de main.py lee este archivo", "FAIL", "main NO puede cargarlo: " + (errs[0][:120] if errs else "sin mensaje") + " (el auditor lo lee con su lector tolerante)")
    X, y = ds["X"], ds["y"]
    M = design_matrix(X)
    rank = int(np.linalg.matrix_rank(M))
    sub.add("N ≥ 13 corridas", "PASS" if len(y) >= 13 else "FAIL", f"N = {len(y)}")
    sub.add("rango de la matriz de diseño = 10 (modelo estimable)", "PASS" if rank == 10 else "FAIL", f"rango = {rank}")
    ss_pe, df_pe = pure_error_ref(X, y)
    sub.add("réplicas disponibles para el error puro", "PASS" if df_pe > 0 else "WARN", f"df_PE = {df_pe}")
    sub.add("respuesta no constante", "PASS" if np.ptp(y) > 0 else "FAIL", f"rango de Y = [{y.min():.4g}, {y.max():.4g}]")
    for note in ds.get("notes", []):
        sub.add("nota del lector", "WARN", note)
    a.add("8.1", SEC_IMP, f"Dataset importado: {os.path.basename(a.import_data)}", sub.status,
          f"N = {len(y)}; rangos = {[tuple(round(v, 4) for v in r) for r in _ranges_of(ds)]}", sub.text() +
          "\n\nLos hallazgos 3.IMP, 4.IMP, 4.J/4.Δ y 5.IMP aplican la misma batería a estos datos.", {"N": len(y), "rank": rank})


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 9: COMPARACIÓN CON UNA CORRIDA PREVIA (--import-report)
# ═════════════════════════════════════════════════════════════════════════════
def chk_compare(a: Audit) -> None:
    """Compara los hallazgos actuales con un reporte JSON previo: corregidos, regresiones, nuevos y retirados."""
    if not a.import_report:
        return
    prev = load_report_json(a.import_report)
    cmp = compare_reports(prev, a)
    a.comparison = cmp
    s = cmp["summary"]
    st = "FAIL" if s["regressed"] else ("PASS" if (s["fixed"] or not (s["changed"] or s["new"] or s["removed"])) else "WARN")
    a.add("9.1", SEC_CMP, f"Comparación con {os.path.basename(a.import_report)}", st,
          f"{s['fixed']} corregidos · {s['regressed']} regresiones · {s['new']} nuevos · {s['removed']} retirados · {s['same']} sin cambios",
          "\n".join(f"{r['change']:>10}  {r['id']:<8} {r['prev'] or '—':>5} → {r['curr'] or '—':<5} {r['title'][:70]}" for r in cmp["rows"] if r["change"] != "igual"),
          s)


# ═════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 10: EXPERIMENTOS DE VALIDACIÓN CON VALORES CONOCIDOS
# ═════════════════════════════════════════════════════════════════════════════
# Cuatro experimentos completos (factores, réplicas y TODAS las respuestas Y) cuyos resultados correctos se conocen
# de antemano. Se ejecutan por el MISMO camino que sigue una persona (generar matriz → escribir Y → «Enviar a análisis»)
# y se comparan, magnitud por magnitud, con una referencia independiente (NumPy/SciPy).
SEC_VAL = "10. Validación con experimentos de valores conocidos"
VAL_NAMES = ("Temperatura", "Tiempo", "pH")
VAL_TOL_REL, VAL_TOL_ABS = 1e-6, 1e-8        # tolerancia: |obtenido − esperado| ≤ 1e-8 + 1e-6·|esperado|


def _truth_E1(p: np.ndarray) -> np.ndarray:
    """E1 (verificable a mano): y = 80 + 3·p2 − 12·p1² − 12·p2² − 9·p3². Con los valores de Doehlert, todas las y son decimales exactos."""
    p = np.atleast_2d(p)
    return 80.0 + 3.0 * p[:, 1] - 12.0 * p[:, 0] ** 2 - 12.0 * p[:, 1] ** 2 - 9.0 * p[:, 2] ** 2


def reference_results(Z: np.ndarray, y: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> Dict[str, Any]:
    """
    Resultados correctos de un experimento, calculados con la referencia independiente, en el espacio de
    variables Z (codificadas o físicas). real = centers + scales·x  (identidad si centers=0, scales=1).
    """
    rf = ref_fit(Z, y)
    lr = lof_reference(Z, y)
    b = rf["beta"]
    xs, H = xstar_ref(b)
    t = float(stats.t.ppf(0.975, rf["df"]))
    out: Dict[str, Any] = {"betas": b, "se_betas": np.sqrt(np.clip(np.diag(rf["cov_inv"]), 0, None)), "n": rf["N"], "df": rf["df"],
                           "r2": rf["r2"], "r2a": rf["r2a"], "s": rf["s"], "ss_tot": rf["ss_tot"], "ss_res": rf["ss_res"],
                           "ss_reg": rf["ss_tot"] - rf["ss_res"], "ss_pe": lr["ss_pe"], "df_pe": lr["df_pe"], "ss_lof": lr["ss_lof"],
                           "df_lof": lr["df_lof"], "F": lr["F"] if lr["computable"] else None, "p": lr["p"] if lr["computable"] else None,
                           "lof_computable": lr["computable"], "t": t, "leverage": np.einsum("ij,jk,ik->i", rf["M"], np.linalg.inv(rf["M"].T @ rf["M"]), rf["M"])}
    lam = np.linalg.eigvalsh(H) / 2.0                           # autovalores de B = H/2 (la matriz de segundo orden estándar)
    out["eig_B"] = np.sort(lam)
    out["nature"] = classify_eig(lam)
    if np.all(np.isfinite(xs)):
        m = design_matrix(xs)[0]
        var = max(0.0, float(m @ rf["cov_inv"] @ m))
        J = jacobian_analytic(b)
        se_x = np.sqrt(np.clip(np.diag(J @ rf["cov_inv"] @ J.T), 0, None))
        out.update({"x_opt_model": xs, "x_opt_real": centers + scales * xs, "y_opt": float(m @ b), "ic_half": t * math.sqrt(var),
                    "ip_half": t * math.sqrt(rf["s2"] + var), "se_x_real": np.abs(scales) * se_x})
    return out


def build_validation_experiments(a: Audit) -> List[Dict[str, Any]]:
    """Construye los experimentos E1–E4: factores, réplicas, valores de Y a ingresar y resultados esperados."""
    alpha = np.array(a.mod.ALPHA, float)
    specs = [
        ("E1", "Exacto y verificable a mano", 3, _truth_E1, 2,
         "Superficie y = 80 + 3·p2 − 12·p1² − 12·p2² − 9·p3² sin ruido. Todas las respuestas son decimales exactos. Las 2 réplicas del centro "
         "difieren en +0,2 y −0,2 (suma cero): no alteran los coeficientes, solo aportan el error puro. El resultado correcto se puede calcular a mano."),
        ("E2", "Completo (los 10 coeficientes distintos de cero)", 3, lambda p: truth_fn(p, False), 6,
         "y = 85 − 12(p1−0,2)² − 15(p2+0,1)² − 9(p3−0,3)² + 4·p1·p2, con réplicas del centro de ±0,2 (suma cero). "
         "Ejercita linealidad, cuadráticos e interacciones: los coeficientes recuperados deben ser los verdaderos."),
        ("E3", "Con falta de ajuste", 3, lambda p: truth_fn(p, True), 6,
         "E2 más un término cúbico y un coseno que un modelo cuadrático no puede ajustar: la prueba de falta de ajuste debe rechazar el modelo (p < 0,05)."),
        ("E4", "Mínimo: 13 corridas (1 centro)", 1, lambda p: truth_fn(p, False), 6,
         "Datos de E2 con una sola corrida central (13 corridas, el mínimo). El programa debe ajustar el modelo, pero declarar que la falta de ajuste no es computable."),
    ]
    exps: List[Dict[str, Any]] = []
    for eid, title, n_cent, fn, dec, desc in specs:
        app, _ = generate_design(a, DEFAULT_RANGES, n_cent)
        rows = app.matriz_generada
        X = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
        cod = np.array([[float(r[5]), float(r[6]), float(r[7])] for r in rows])
        c = np.array([(lo + hi) / 2 for lo, hi in DEFAULT_RANGES])
        d = np.array([(hi - lo) / 2 for lo, hi in DEFAULT_RANGES])
        P = (X - c) / d * alpha                                    # coordenadas p exactas de cada corrida
        y = np.asarray(fn(P), float)
        center_rows = [i for i in range(len(P)) if np.allclose(P[i], 0.0, atol=1e-9)]
        if len(center_rows) == 3:                                  # réplicas: perturbaciones de suma cero (0, +0.2, −0.2)
            for k, delta in zip(center_rows, (0.0, 0.2, -0.2)):
                y[k] += delta
        y = np.round(y, dec)                                       # exactamente lo que la persona escribe
        scales = d / alpha
        exp = {"id": eid, "title": title, "desc": desc, "n_cent": n_cent, "names": list(VAL_NAMES), "ranges": list(DEFAULT_RANGES),
               "dec": dec, "X": X, "cod": cod, "y": y, "centers": c, "scales": scales, "N": len(y),
               "expected": {"coded": reference_results(cod, y, c, scales), "natural": reference_results(X, y, np.zeros(3), np.ones(3))}}
        if eid == "E1":                                            # resultado exacto «a mano»
            hand = {"betas": [80, 0, 3, 0, 0, 0, 0, -12, -12, -9], "ss_res": 0.08, "df_res": 5, "s": math.sqrt(0.08 / 5), "ss_pe": 0.08, "df_pe": 2,
                    "ss_lof": 0.0, "df_lof": 3, "F": 0.0, "p": 1.0, "x_opt_model": [0.0, 0.125, 0.0], "y_opt": 80.1875,
                    "x_opt_real": (c + scales * np.array([0.0, 0.125, 0.0])).tolist()}
            exp["hand"] = hand
        exps.append(exp)
    return exps


def run_validation_experiment(a: Audit, exp: Dict[str, Any]) -> Tuple[RunResult, List[str]]:
    """Reproduce el camino de una persona: generar la matriz, escribir Y como texto en la tabla y pulsar «Enviar a análisis»."""
    app, _ = generate_design(a, exp["ranges"], exp["n_cent"], tuple(exp["names"]))
    for row, yv in zip(app.matriz_generada, exp["y"]):
        row[4] = f"{yv:.{exp['dec']}f}"                          # lo que se tipea en la celda Respuesta_Y
    with patched_dialogs(a.mod) as log:
        try:
            app.enviar_datos_a_analisis()
        except Exception as e:
            log.append(("exception", type(e).__name__, str(e)))
    msgs = [m for k, _t, m in log if k in ("showerror", "showwarning", "exception")]
    X = getattr(app, "X_exp", None)
    if X is None:
        rr = RunResult(name=exp["id"], X=exp["X"], y=exp["y"], cod=exp["cod"], error="; ".join(msgs) or "el programa no cargó los datos")
        return rr, msgs
    return collect_run(app, exp["id"], np.asarray(X, float), np.asarray(app.y_exp, float), np.asarray(app.cod_exp, float)), msgs


def compare_validation(exp: Dict[str, Any], rr: RunResult) -> List[Dict[str, Any]]:
    """Compara, magnitud por magnitud, lo que obtuvo el programa con el resultado correcto. Devuelve filas para tabla y guía."""
    E = exp["expected"][rr.space]
    rows: List[Dict[str, Any]] = []

    def put(group: str, name: str, where: str, exp_v: Any, got: Any, tol_abs: float = VAL_TOL_ABS, tol_rel: float = VAL_TOL_REL) -> None:
        """Una magnitud: estado PASS/FAIL según |obtenido − esperado| ≤ tol_abs + tol_rel·|esperado|."""
        if exp_v is None or (isinstance(exp_v, float) and not math.isfinite(exp_v)):
            ok = got is None or (isinstance(got, float) and not math.isfinite(got))
            rows.append({"group": group, "name": name, "where": where, "exp": None, "got": got, "tol": None, "diff": None, "status": "PASS" if ok else "FAIL"})
            return
        if got is None or (isinstance(got, float) and not math.isfinite(got)):
            rows.append({"group": group, "name": name, "where": where, "exp": exp_v, "got": None, "tol": tol_abs + tol_rel * abs(exp_v), "diff": None, "status": "FAIL"})
            return
        tol = tol_abs + tol_rel * abs(exp_v)
        diff = abs(float(got) - float(exp_v))
        rows.append({"group": group, "name": name, "where": where, "exp": float(exp_v), "got": float(got), "tol": tol, "diff": diff, "status": "PASS" if diff <= tol else "FAIL"})

    legacy = rr.fit is None                    # versión antigua: R², SS, F, p, ŷ e intervalos se leen del texto impreso (3-4 decimales)
    pr_abs, pr_rel = (6e-4, 1e-4) if legacy else (VAL_TOL_ABS, VAL_TOL_REL)
    nm = exp["names"]
    labels = [f"b0 (intercepto)", f"b1 ({nm[0]})", f"b2 ({nm[1]})", f"b3 ({nm[2]})", f"b12 ({nm[0]}·{nm[1]})", f"b13 ({nm[0]}·{nm[2]})",
              f"b23 ({nm[1]}·{nm[2]})", f"b11 ({nm[0]}²)", f"b22 ({nm[1]}²)", f"b33 ({nm[2]}²)"]
    put("Datos", "N.º de corridas", "Pestaña 1 / 3", E["n"], rr.rep.get("N", len(rr.y)), 0, 0)
    put("Datos", "Grados de libertad residuales", "Pestaña 3 (ANOVA)", E["df"], rr.rep.get("df"), 0, 0)
    if rr.betas is not None:
        for k in range(10):
            put("Coeficientes", labels[k], "Pestaña 3 (Coeficientes)", E["betas"][k], rr.betas[k], 1e-6, VAL_TOL_REL)
        if rr.cov is not None:
            for k in range(10):
                put("Errores estándar", "SE de " + labels[k].split(" ")[0], "Pestaña 3 (Coeficientes)", E["se_betas"][k],
                    math.sqrt(max(0.0, float(rr.cov[k, k]))), 1e-6, VAL_TOL_REL)
    put("Ajuste", "R²", "Pestaña 2/3 (resumen)", E["r2"], rr.rep.get("r2"), pr_abs, pr_rel)
    put("Ajuste", "R² ajustado", "Pestaña 2/3 (resumen)", E["r2a"], rr.rep.get("r2a"), pr_abs, pr_rel)
    put("Ajuste", "s (error estándar residual)", "Pestaña 3 (ANOVA)", E["s"], rr.s_res)
    L = rr.lof
    if "ss_reg" in L:
        put("ANOVA", "SS regresión", "Pestaña 3 (ANOVA)", E["ss_reg"], L.get("ss_reg"))
    put("ANOVA", "SS residual", "Pestaña 3 (ANOVA)", E["ss_res"], L.get("ss_res"), pr_abs, pr_rel)
    put("ANOVA", "SS error puro (SS_PE)", "Pestaña 3 (Lack of Fit)", E["ss_pe"], L.get("ss_pe"), pr_abs, pr_rel)
    put("ANOVA", "gl error puro", "Pestaña 3 (Lack of Fit)", E["df_pe"], L.get("df_pe"), 0, 0)
    put("ANOVA", "SS falta de ajuste (SS_LOF)", "Pestaña 3 (Lack of Fit)", E["ss_lof"], L.get("ss_lof"), pr_abs, pr_rel)
    put("ANOVA", "gl falta de ajuste", "Pestaña 3 (Lack of Fit)", E["df_lof"], L.get("df_lof"), 0, 0)
    put("ANOVA", "F de falta de ajuste", "Pestaña 3 (Lack of Fit)", E["F"], L.get("F"), max(1e-6, pr_abs), max(VAL_TOL_REL, pr_rel))
    put("ANOVA", "p-valor de falta de ajuste", "Pestaña 3 (Lack of Fit)", E["p"], L.get("p"), max(1e-6, pr_abs), 0)
    if "x_opt_real" in E and rr.x_opt is not None:
        for j in range(3):
            put("Óptimo", f"Óptimo de {nm[j]} (unidades físicas)", "Pestaña 2 (reporte canónico)", E["x_opt_real"][j], rr.x_opt[j], 1e-6, VAL_TOL_REL)
        put("Óptimo", "Respuesta predicha en el óptimo", "Pestaña 2 (reporte canónico)", E["y_opt"], rr.rep.get("y_opt"), max(1e-6, pr_abs), max(VAL_TOL_REL, pr_rel))
        put("Óptimo", "Semiancho IC 95 % de la media (±)", "Pestaña 2 (reporte canónico)", E["ic_half"], rr.rep.get("ic_half"), max(1e-6, pr_abs), max(1e-5, pr_rel))
        put("Óptimo", "Semiancho IP 95 % individual (±)", "Pestaña 2 (reporte canónico)", E["ip_half"], rr.rep.get("ip_half"), max(1e-6, pr_abs), max(1e-5, pr_rel))
        if len(rr.ic_coords) == 3:
            for j in range(3):
                put("Óptimo", f"Error estándar de {nm[j]}* (método Delta)", "Pestaña 2 (reporte canónico)", E["se_x_real"][j], rr.ic_coords[j][2], 1e-6, 1e-5)
        nat = rr.rep.get("nature")
        rows.append({"group": "Óptimo", "name": "Naturaleza de la superficie (autovalores de B)", "where": "Pestaña 2 (reporte canónico)", "exp": E["nature"], "got": nat,
                     "tol": None, "diff": None, "status": "SKIP" if nat is None else ("PASS" if nature_key(nat) == nature_key(E["nature"]) else "FAIL")})
    h = exp.get("hand")
    if h is not None and rr.space == "coded" and rr.betas is not None:
        # Valor TEÓRICO exacto: difiere del valor «esperado» de arriba solo por el redondeo a 6 decimales de x_cod en la tabla (≲ 1e-4)
        for k in range(10):
            put("A mano (valor teórico exacto)", "Coeficiente " + labels[k].split(" ")[0], "Pestaña 3 (Coeficientes)", float(h["betas"][k]), rr.betas[k], 2e-4, 0)
        put("A mano (valor teórico exacto)", "s = √(0,08/5)", "Pestaña 3 (ANOVA)", h["s"], rr.s_res, 1e-4, 0)
        put("A mano (valor teórico exacto)", "SS error puro = 0,2² + 0,2²", "Pestaña 3 (Lack of Fit)", h["ss_pe"], L.get("ss_pe"), 1e-6, 0)
        put("A mano (valor teórico exacto)", "p-valor de falta de ajuste = 1", "Pestaña 3 (Lack of Fit)", h["p"], L.get("p"), 1e-4, 0)
        if rr.x_opt is not None:
            for j in range(3):
                put("A mano (valor teórico exacto)", f"Óptimo de {nm[j]}", "Pestaña 2 (reporte canónico)", h["x_opt_real"][j], rr.x_opt[j], 1e-3, 0)
            put("A mano (valor teórico exacto)", "ŷ en el óptimo = 80,1875", "Pestaña 2 (reporte canónico)", h["y_opt"], rr.rep.get("y_opt"), 1e-3, 0)
    if rr.fit is not None:
        ev = np.sort(np.asarray(rr.fit.canonical.eigenvalues, float))
        for j in range(3):
            put("Óptimo", f"Autovalor λ{j + 1} de B", "Pestaña 2 (reporte canónico)", E["eig_B"][j], ev[j], 1e-8, VAL_TOL_REL)
        lev = np.asarray(rr.fit.leverage, float)
        put("Diagnóstico", "Suma de leverages (= 10 parámetros)", "Pestaña 4 (residuos)", 10.0, float(lev.sum()), 1e-8, 1e-8)
        put("Diagnóstico", "Leverage máximo h_ii", "Pestaña 4 (residuos)", float(E["leverage"].max()), float(lev.max()), 1e-8, VAL_TOL_REL)
    return rows


def chk_validation(a: Audit) -> None:
    """Ejecuta E1–E4 por el camino de la interfaz y compara con los resultados correctos (hallazgos 10.E1 … 10.E4)."""
    exps = build_validation_experiments(a)
    res = []
    for exp in exps:
        rr, msgs = run_validation_experiment(a, exp)
        if rr.error or rr.betas is None:
            a.add(f"10.{exp['id']}", SEC_VAL, f"{exp['id']} · {exp['title']}", "FAIL",
                  "el programa no pudo analizar el experimento: " + (rr.error or "")[:160], "\n".join(msgs) + "\n" + (rr.error or ""))
            res.append({"exp": exp, "rr": rr, "rows": []})
            continue
        rows = compare_validation(exp, rr)
        n_ok = sum(1 for r in rows if r["status"] == "PASS")
        n_bad = [r for r in rows if r["status"] == "FAIL"]
        st = "FAIL" if n_bad else "PASS"
        det = [f"Modelo del programa ajustado en {'variables codificadas' if rr.space == 'coded' else 'unidades físicas'}; N = {exp['N']} corridas; "
               f"tolerancia: |obtenido − esperado| ≤ {VAL_TOL_ABS:g} + {VAL_TOL_REL:g}·|esperado|", ""]
        det.append("Magnitud".ljust(48) + "esperado".rjust(26) + "obtenido".rjust(26) + "  estado")
        for r in rows:
            f1 = lambda v: "—" if v is None else (v if isinstance(v, str) else f"{v:.8g}")
            det.append(r["name"][:47].ljust(48) + f1(r["exp"]).rjust(26) + f1(r["got"]).rjust(26) + f"  [{r['status']}]")
        a.add(f"10.{exp['id']}", SEC_VAL, f"{exp['id']} · {exp['title']}", st,
              (f"{len(n_bad)} de {len(rows)} magnitudes difieren: " + "; ".join(r["name"] for r in n_bad[:3])) if n_bad else f"las {len(rows)} magnitudes coinciden con el resultado correcto",
              "\n".join(det), {"n_values": exp["N"], "n_ok": n_ok, "n_fail": len(n_bad)}, "main.py / doehlert_engine.py · flujo completo (generar → ingresar Y → enviar a análisis)")
        res.append({"exp": exp, "rr": rr, "rows": rows})
    a.store["validation"] = res


# ---------- Exportación del paquete de validación: CSV importables + guía HTML + JSON ----------
def _fnum(v: Any, nd: int = 8) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{float(v):.{nd}g}"


def export_validation(a: Audit, outdir: str) -> List[str]:
    """
    Escribe el paquete de validación:
      · E?_*.csv        planilla con los valores de Y (se importa en main.py con «Cargar CSV con Respuestas…» o con doehlert_engine.py)
      · guia_validacion.html   guía paso a paso (qué escribir) + resultados esperados con tolerancias + resultado de la corrida automática
      · valores_esperados.json  los mismos valores esperados en formato máquina
    """
    os.makedirs(outdir, exist_ok=True)
    if "validation" not in a.store:
        chk_validation(a)
    res = a.store["validation"]
    e = html_lib.escape
    paths: List[str] = []
    fname = {"E1": "E1_exacto_a_mano", "E2": "E2_completo", "E3": "E3_falta_de_ajuste", "E4": "E4_minimo_13_corridas"}
    for r in res:
        ex = r["exp"]
        p = os.path.join(outdir, fname[ex["id"]] + ".csv")
        write_dataset_csv({"X": ex["X"], "y": ex["y"], "cod": ex["cod"]}, p, names=tuple(ex["names"]), dec=6)
        paths.append(p)
    out = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Guía de validación · Suite Doehlert</title><style>{_HTML_CSS}"
           "table.v td.num{text-align:right;font-family:ui-monospace,Consolas,monospace}td.PASS{color:var(--pass);font-weight:700}td.FAIL{color:var(--fail);font-weight:700}"
           "td.SKIP{color:var(--skip)}.box{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 16px;margin:12px 0}"
           "input.chk{transform:scale(1.3)}@media print{.noprint{display:none}}</style></head><body><main>",
           "<h1>Guía de validación · Suite Doehlert</h1>",
           f"<div class='meta'>Programa auditado: <b>{e(os.path.basename(a.meta['target']))}</b> (sha256 {a.meta['target_sha256'][:12]}…) · generada el {e(a.meta['created'])}. "
           "Los valores «esperados» se calcularon con una referencia independiente (NumPy/SciPy). La columna «Programa» muestra lo que devolvió el programa en la corrida automática.</div>",
           "<h2>¿Por qué hay que ingresar 15 valores?</h2><div class='box'>"
           "<p>El modelo cuadrático de 3 factores tiene <b>10 coeficientes</b> (constante, 3 lineales, 3 interacciones y 3 cuadráticos). Cada coeficiente se estima a partir de las "
           "<b>respuestas medidas</b> (Y) en cada corrida: los factores por sí solos no contienen información sobre la respuesta, por eso el programa no puede inventarlas.</p>"
           "<ul><li>El diseño de Doehlert tiene <b>13 corridas canónicas</b>: es el mínimo (E4). Con menos de 13 el programa rechaza el análisis.</li>"
           "<li>A esas 13 se suman las <b>réplicas del centro</b> (el programa permite de 1 a 10). Con 3 réplicas hay 13 + 2 = <b>15 corridas</b>: ese es el valor por defecto.</li>"
           "<li>Las réplicas del centro miden el <b>error puro</b> y permiten la prueba de <b>falta de ajuste</b>; hacen falta al menos 2 corridas centrales (14 valores) para que sea computable.</li>"
           "<li>Para no tipear, se puede <b>importar el CSV</b> de cada experimento (Pestaña 2 → «Cargar CSV con Respuestas…») o analizarlo por consola: "
           "<code>python3 doehlert_engine.py E2_completo.csv</code>.</li></ul></div>",
           "<h2>Cómo validar (dos caminos)</h2><div class='box'><p><b>A. Importar (sin tipear):</b> Pestaña 2 → «Cargar CSV con Respuestas…» → elegir el CSV del experimento → comparar con las tablas de resultados esperados.</p>"
           "<p><b>B. Manual (lo que haría una persona):</b> Pestaña 1 → escribir nombres y límites de los factores y las réplicas del centro → «1. Calcular Matriz y Ver 3D» → "
           "doble clic en cada celda de <i>Respuesta_Y</i> y escribir el valor de la tabla → «3. Enviar a Análisis (Pest. 2)» → revisar las pestañas 2, 3 y 4.</p></div>"]
    for r in res:
        ex, rr, rows = r["exp"], r["rr"], r["rows"]
        out.append(f"<h2>{e(ex['id'])} · {e(ex['title'])}</h2><div class='box'><p>{e(ex['desc'])}</p>")
        out.append("<table class='v'><tr><th>Factor</th><th>Mínimo</th><th>Máximo</th></tr>" +
                   "".join(f"<tr><td>{e(n)}</td><td>{lo:g}</td><td>{hi:g}</td></tr>" for n, (lo, hi) in zip(ex["names"], ex["ranges"])) +
                   f"</table><p>Réplicas del centro: <b>{ex['n_cent']}</b> → <b>{ex['N']} corridas</b> (hay que ingresar <b>{ex['N']} valores</b> de Y).</p></div>")
        out.append("<h3>Valores que hay que ingresar</h3><table class='v'><tr><th>Corrida</th>" + "".join(f"<th>{e(n)}</th>" for n in ex["names"]) +
                   "<th>Respuesta_Y a escribir</th><th class='noprint'>✔</th></tr>")
        for i in range(ex["N"]):
            out.append(f"<tr><td>{i + 1}</td>" + "".join(f"<td class='num'>{ex['X'][i, j]:.6g}</td>" for j in range(3)) +
                       f"<td class='num'><b>{ex['y'][i]:.{ex['dec']}f}</b></td><td class='noprint'><input class='chk' type='checkbox'></td></tr>")
        out.append("</table><p style='color:var(--mut);font-size:.88rem'>Los factores deben coincidir con los que muestra la Pestaña 1 después de «Calcular Matriz» (tolerancia 1e-5).</p>")
        if rows:
            out.append(f"<h3>Resultados esperados (el modelo del programa se ajusta en {'variables codificadas de Doehlert' if rr.space == 'coded' else 'unidades físicas'})</h3>"
                       "<table class='v'><tr><th>Grupo</th><th>Magnitud</th><th>Dónde verlo</th><th>Esperado</th><th>Tolerancia ±</th><th>Programa</th><th>Auto</th><th class='noprint'>Tu valor</th></tr>")
            for q in rows:
                out.append(f"<tr><td>{e(q['group'])}</td><td>{e(q['name'])}</td><td>{e(q['where'])}</td><td class='num'>{_fnum(q['exp'])}</td>"
                           f"<td class='num'>{_fnum(q['tol'], 2)}</td><td class='num'>{_fnum(q['got'])}</td><td class='{q['status']}'>{q['status']}</td>"
                           "<td class='noprint'><input type='text' size='12'></td></tr>")
            out.append("</table>")
        if ex["id"] == "E1" and "hand" in ex:
            h = ex["hand"]
            out.append("<h3>Verificación a mano (E1)</h3><div class='box'><p>Como las respuestas son exactas, el ajuste devuelve <b>exactamente</b> los coeficientes de la superficie: "
                       "b0 = 80, b2 = 3, b11 = −12, b22 = −12, b33 = −9 y el resto = 0. "
                       "(El programa muestra, por ejemplo, b11 = −12,000011: la tabla redondea x_cod a 6 decimales, y esa diferencia de 1e-5 es esperable; "
                       "las filas «A mano» usan una tolerancia de 2e-4 y las demás filas esperan el valor exacto de la matriz que el programa realmente usa.)</p>"
                       "<p>Los residuos solo provienen de las réplicas del centro (80,2 − 80 = +0,2 y 79,8 − 80 = −0,2): SS_res = 0,2² + 0,2² = <b>0,08</b> con 5 gl, "
                       f"s² = 0,08/5 = 0,016 y s = <b>{h['s']:.6f}</b>. Ese mismo valor es el error puro: SS_PE = 0,08 con 2 gl. La falta de ajuste es nula: SS_LOF = 0, F = 0, p = 1.</p>"
                       f"<p>Óptimo: ∂y/∂p2 = 3 − 24·p2 = 0 ⇒ p2* = 0,125 (p1* = p3* = 0) y ŷ* = 80 + 3·0,125 − 12·0,125² = <b>{h['y_opt']}</b>. "
                       f"En unidades físicas: {', '.join(f'{n} = {v:.6g}' for n, v in zip(ex['names'], h['x_opt_real']))}. B = diag(−12, −12, −9): todos los autovalores negativos ⇒ <b>máximo</b>.</p></div>")
    out.append("</main></body></html>")
    pg = os.path.join(outdir, "guia_validacion.html")
    with open(pg, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    paths.append(pg)
    pj = os.path.join(outdir, "valores_esperados.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(jsonable({r["exp"]["id"]: {"titulo": r["exp"]["title"], "factores": dict(zip(r["exp"]["names"], r["exp"]["ranges"])), "replicas_centro": r["exp"]["n_cent"],
                                              "Y": r["exp"]["y"], "esperado_codificado": r["exp"]["expected"]["coded"], "esperado_fisico": r["exp"]["expected"]["natural"],
                                              "a_mano": r["exp"].get("hand")} for r in res}), f, ensure_ascii=False, indent=2)
    paths.append(pj)
    return paths


def fig_validation(a: Audit) -> Optional[Figure]:
    """Desviación normalizada |obtenido − esperado| / tolerancia de cada magnitud (≤ 1 ⇒ PASS), por experimento."""
    V = a.store.get("validation")
    if not V:
        return None
    fig = _fig(12, 4.2)
    ax = fig.add_subplot(121)
    for k, r in enumerate(V):
        vals = [(q["diff"] / q["tol"]) if (q["diff"] is not None and q["tol"]) else 0.0 for q in r["rows"]]
        ax.semilogy(np.full(len(vals), k) + np.linspace(-0.25, 0.25, max(1, len(vals)))[:len(vals)], np.maximum(vals, 1e-6), ".", ms=5,
                    color=COLORS["PASS"] if all(q["status"] != "FAIL" for q in r["rows"]) else COLORS["FAIL"])
    ax.axhline(1.0, color="#ef6c00", ls="--", label="límite de tolerancia")
    ax.set_xticks(range(len(V)))
    ax.set_xticklabels([r["exp"]["id"] for r in V])
    ax.legend(fontsize=8)
    _style(ax, "Desviación / tolerancia por magnitud", "experimento", "|obtenido − esperado| / tolerancia")
    ax2 = fig.add_subplot(122)
    ok = [sum(1 for q in r["rows"] if q["status"] == "PASS") for r in V]
    bad = [sum(1 for q in r["rows"] if q["status"] == "FAIL") for r in V]
    x = np.arange(len(V))
    ax2.bar(x, ok, color=COLORS["PASS"], label="PASS")
    ax2.bar(x, bad, bottom=ok, color=COLORS["FAIL"], label="FAIL")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["exp"]["id"] for r in V])
    ax2.legend(fontsize=8)
    _style(ax2, "Magnitudes verificadas por experimento", "experimento", "cantidad")
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS  (objetos Figure independientes de pyplot: sirven para HTML, PDF, PNG y la GUI)
# ═════════════════════════════════════════════════════════════════════════════
from matplotlib.colors import SymLogNorm


def _fig(w: float = 11.0, h: float = 4.2) -> Figure:
    return Figure(figsize=(w, h), dpi=100, facecolor="white")


def _style(ax, title: str = "", xl: str = "", yl: str = "") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xl, fontsize=8.5)
    ax.set_ylabel(yl, fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)


def fig_dashboard(a: Audit) -> Figure:
    """Tablero resumen: hallazgos por sección y estado + dona con los totales."""
    secs = a.by_section()
    names = list(secs.keys())
    fig = _fig(11, 4.6)
    ax = fig.add_subplot(121)
    left = np.zeros(len(names))
    for st in ("PASS", "WARN", "FAIL", "SKIP"):
        v = np.array([sum(1 for f in secs[s] if f.status == st) for s in names], float)
        ax.barh(range(len(names)), v, left=left, color=COLORS[st], label=st, edgecolor="white")
        left += v
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.split(". ", 1)[-1][:34] for n in names], fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Hallazgos por sección", "cantidad")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax2 = fig.add_subplot(122)
    c = a.counts()
    vals = [c[s] for s in ("PASS", "WARN", "FAIL", "SKIP")]
    if sum(vals) > 0:
        ax2.pie(vals, colors=[COLORS[s] for s in ("PASS", "WARN", "FAIL", "SKIP")], startangle=90, counterclock=False,
                wedgeprops=dict(width=0.38, edgecolor="white"), labels=[f"{s} {n}" if n else "" for s, n in zip(("PASS", "WARN", "FAIL", "SKIP"), vals)],
                textprops=dict(fontsize=9))
    ax2.set_title("Resultado global", fontsize=10, fontweight="bold")
    ax2.text(0, 0, f"{c['FAIL']} FAIL\n{c['WARN']} WARN\n{c['PASS']} PASS", ha="center", va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_normal(a: Audit) -> Optional[Figure]:
    """Φ⁻¹: paridad main vs SciPy y error absoluto en escala logit (colas amplificadas)."""
    d = a.store.get("normal_ppf")
    if not d:
        return None
    fig = _fig(11, 4.2)
    ax = fig.add_subplot(121)
    ax.plot(d["ref"], d["ref"], color="#2e7d32", lw=1.5, label="SciPy (exacto)")
    ax.scatter(d["ref"], d["est"], s=6, color="#c62828", alpha=0.7, label="main.normal_ppf")
    _style(ax, "Paridad: cuantil estimado vs exacto", "Φ⁻¹(p) SciPy", "Φ⁻¹(p) main")
    ax.legend(fontsize=8)
    ax2 = fig.add_subplot(122)
    ax2.semilogy(d["p"], np.maximum(d["err"], 1e-17), ".", ms=3, color="#1565c0")
    ax2.axhline(TOL_ABS, color="#ef6c00", ls="--", lw=1.2, label=f"tolerancia {TOL_ABS:g}")
    ax2.set_xscale("logit")
    ax2.legend(fontsize=8)
    _style(ax2, "Error absoluto vs p (escala logit)", "p", "|error|")
    fig.tight_layout()
    return fig


def fig_special(a: Audit) -> Optional[Figure]:
    """Mapas de calor del error de betainc y f_sf y curva del error relativo de t_ppf_975 vs ν."""
    bi, fs, tp = a.store.get("betainc"), a.store.get("f_sf"), a.store.get("t_ppf")
    if not (bi and fs and tp):
        return None
    fig = _fig(12, 4.0)
    for k, (d, lab, ticks) in enumerate(((bi, "betainc: log10 máx |error|", None), (fs, "f_sf: log10 máx |error|", None))):
        ax = fig.add_subplot(1, 3, k + 1)
        Z = np.log10(np.maximum(d["err"], 1e-17))
        im = ax.imshow(Z, cmap="RdYlGn_r", vmin=-16, vmax=0, aspect="auto")
        xs = d["B"] if "B" in d else d["dfs"]
        ys = d["A"] if "A" in d else d["dfs"]
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=7)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{v:g}" for v in ys], fontsize=7)
        _style(ax, lab, "b" if "B" in d else "df2", "a" if "A" in d else "df1")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(1, 3, 3)
    ax.loglog(tp["df"], np.maximum(tp["rel"], 1e-17), "o-", ms=4, color="#c62828")
    ax.axhline(TOL_REL, color="#ef6c00", ls="--", label=f"tolerancia {TOL_REL:g}")
    ax.legend(fontsize=8)
    _style(ax, "t_ppf_975: error relativo vs ν", "grados de libertad ν", "|t̂ − t|/t")
    fig.tight_layout()
    return fig


def fig_geometry(a: Audit) -> Optional[Figure]:
    """Geometría 3D del diseño (aristas a distancia 1), radios al centro y matriz de distancias."""
    g = a.store.get("geometry")
    if not g:
        return None
    B = g["B"]
    fig = _fig(12, 4.2)
    ax = fig.add_subplot(131, projection="3d")
    D = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=2)
    for i in range(1, len(B)):
        for j in range(i + 1, len(B)):
            if abs(D[i, j] - 1.0) < 1e-9:
                ax.plot(*zip(B[i], B[j]), color="#90a4ae", lw=0.8)
    ax.scatter(B[1:, 0], B[1:, 1], B[1:, 2], s=35, color="#1565c0", edgecolor="k")
    ax.scatter(*B[0], s=90, marker="*", color="#c62828", edgecolor="k")
    ax.set_title("Cuboctaedro de Doehlert (k = 3)", fontsize=10, fontweight="bold")
    ax.set_xlabel("p1", fontsize=8); ax.set_ylabel("p2", fontsize=8); ax.set_zlabel("p3", fontsize=8)
    ax2 = fig.add_subplot(132)
    r = np.linalg.norm(B, axis=1)
    ax2.bar(range(len(r)), r, color=["#c62828"] + ["#1565c0"] * (len(r) - 1))
    ax2.axhline(1.0, color="#2e7d32", ls="--", label="radio unitario")
    ax2.legend(fontsize=8)
    _style(ax2, "Distancia de cada punto al centro", "corrida", "‖p‖")
    ax3 = fig.add_subplot(133)
    im = ax3.imshow(D, cmap="viridis")
    _style(ax3, "Matriz de distancias entre puntos", "corrida", "corrida")
    ax3.grid(False)
    fig.colorbar(im, ax=ax3, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_ols(a: Audit) -> Optional[Figure]:
    """OLS: desviación de β̂ vs SciPy, condicionamiento natural vs codificado y residuos de D1."""
    o = a.store.get("ols")
    if not o:
        return None
    names = list(o.keys())
    fig = _fig(12, 4.2)
    ax = fig.add_subplot(131)
    ax.bar(names, np.log10(np.maximum([o[n]["d_beta"] for n in names], 1e-17)), color="#1565c0")
    ax.axhline(np.log10(TOL_REL), color="#ef6c00", ls="--", label="tolerancia 1e-5")
    ax.legend(fontsize=8)
    _style(ax, "β̂ main vs scipy.linalg.lstsq", "dataset", "log10 desviación relativa")
    ax2 = fig.add_subplot(132)
    x = np.arange(len(names))
    ax2.bar(x - 0.2, [o[n]["cond_nat"] for n in names], 0.4, color="#c62828", label="unidades naturales")
    ax2.bar(x + 0.2, [o[n]["cond_cod"] for n in names], 0.4, color="#2e7d32", label="codificadas (−1…+1)")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.axhline(1 / EPS, color="k", ls=":", lw=1, label="1/ε (límite)")
    ax2.legend(fontsize=7)
    _style(ax2, "Número de condición cond(MᵀM)", "dataset", "cond")
    ax3 = fig.add_subplot(133)
    k = names[0]
    ax3.scatter(o[k]["yhat"], o[k]["resid"], color="#1565c0", edgecolor="k", s=30)
    ax3.axhline(0, color="k", lw=1)
    _style(ax3, f"Residuos vs ajustados ({k})", "ŷ", "residuo")
    fig.tight_layout()
    return fig


def fig_optimum(a: Audit) -> Optional[Figure]:
    """Posición del óptimo reportado por main vs referencia, en unidades de semirrango respecto del óptimo verdadero."""
    c = a.store.get("canonical")
    if not c:
        return None
    names = [n for n in c if c[n]["x_true"] is not None]
    if not names:
        return None
    fig = _fig(12, 4.0)
    for j in range(3):
        ax = fig.add_subplot(1, 4, j + 1)
        xm = [(c[n]["x_main"][j] - c[n]["x_true"][j]) / c[n]["delta"][j] for n in names]
        xr = [(c[n]["x_ref"][j] - c[n]["x_true"][j]) / c[n]["delta"][j] for n in names]
        ax.axhline(0, color="k", lw=1)
        ax.plot(names, xr, "o", color="#2e7d32", ms=8, label="referencia (−H⁻¹b)")
        ax.plot(names, xm, "s", color="#c62828", ms=7, label="main.py")
        _style(ax, f"Factor X{j + 1}", "dataset", "(x̂ − x_verdadero)/Δ")
        if j == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(144)
    allnames = list(c.keys())
    ax.bar(allnames, np.log10(np.maximum([c[n]["grad_rel"] for n in allnames], 1e-17)), color="#6a1b9a")
    ax.axhline(-6, color="#ef6c00", ls="--", label="umbral 1e-6")
    ax.legend(fontsize=8)
    _style(ax, "‖∇f(x*_main)‖ relativo", "dataset", "log10")
    fig.tight_layout()
    return fig


def fig_jacobian(a: Audit) -> Optional[Figure]:
    """Jacobiano ∂x*/∂β (3×10): el de main, la referencia analítica y la diferencia absoluta."""
    j = a.store.get("jacobian")
    if not j:
        return None
    fig = _fig(12, 3.6)
    vmax = float(np.abs(j["analytic"]).max())
    for k, (key, lab) in enumerate((("main", "main.py"), ("analytic", "referencia analítica"))):
        ax = fig.add_subplot(1, 3, k + 1)
        # escala simétrica-logarítmica: las derivadas de b11 (enormes en unidades naturales) no ocultan al resto
        im = ax.imshow(j[key], cmap="coolwarm", norm=SymLogNorm(linthresh=max(vmax * 1e-5, 1e-12), vmin=-vmax, vmax=vmax), aspect="auto")
        ax.set_xticks(range(10)); ax.set_xticklabels(BETA_NAMES, fontsize=7, rotation=45)
        ax.set_yticks(range(3)); ax.set_yticklabels(["x1*", "x2*", "x3*"], fontsize=8)
        _style(ax, f"J = ∂x*/∂β · {lab}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(133)
    diff = np.abs(j["main"] - j["analytic"])
    im = ax.imshow(np.log10(np.maximum(diff, 1e-17)), cmap="RdYlGn_r", vmin=-14, vmax=1, aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels(BETA_NAMES, fontsize=7, rotation=45)
    ax.set_yticks(range(3)); ax.set_yticklabels(["x1*", "x2*", "x3*"], fontsize=8)
    _style(ax, "log10 |J_main − J_ref|")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_delta(a: Audit) -> Optional[Figure]:
    """Método Delta: nube Monte Carlo de x*(β) y comparación de errores estándar (main, Delta, Monte Carlo)."""
    d = a.store.get("delta")
    if not d:
        return None
    S = d["samples"]
    fig = _fig(12, 4.0)
    for k, (i, j) in enumerate(((0, 1), (0, 2))):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(S[:, i], S[:, j], s=3, alpha=0.25, color="#1565c0")
        ax.plot(d["xs"][i], d["xs"][j], "r*", ms=14, label="x* (β̂)")
        ax.errorbar(d["xs"][i], d["xs"][j], xerr=d["t"] * d["se_ref"][i], yerr=d["t"] * d["se_ref"][j], color="#c62828", capsize=4, label="IC 95 % Delta (ref)")
        _style(ax, f"Monte Carlo: X{i + 1} vs X{j + 1}", f"x{i + 1}*", f"x{j + 1}*")
        ax.legend(fontsize=7)
    ax = fig.add_subplot(133)
    x = np.arange(3)
    ax.bar(x - 0.27, d["se_ref"], 0.27, color="#2e7d32", label="Delta (referencia)")
    ax.bar(x, d["se_ref"] * d["ratio"], 0.27, color="#1565c0", label="Monte Carlo (robusto)")
    if d.get("se_main") is not None:
        ax.bar(x + 0.27, d["se_main"], 0.27, color="#c62828", label="main.py")
    ax.set_xticks(x); ax.set_xticklabels(["X1*", "X2*", "X3*"])
    ax.legend(fontsize=7)
    _style(ax, "Error estándar de x*", "", "SE")
    fig.tight_layout()
    return fig


def fig_coverage(a: Audit) -> Optional[Figure]:
    """Cobertura empírica del IC 95 % del óptimo y sesgo normalizado (main vs referencia)."""
    c = a.store.get("coverage")
    if not c:
        return None
    fig = _fig(11, 3.8)
    x = np.arange(3)
    ax = fig.add_subplot(121)
    ax.bar(x - 0.2, c["cov_main"], 0.4, color="#c62828", label="main.py")
    ax.bar(x + 0.2, c["cov_ref"], 0.4, color="#2e7d32", label="referencia")
    ax.axhline(0.95, color="k", ls="--", lw=1, label="nominal 0.95")
    ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(["X1*", "X2*", "X3*"])
    ax.legend(fontsize=8)
    _style(ax, f"Cobertura empírica del IC 95 % (R = {c['R']})", "", "proporción")
    ax2 = fig.add_subplot(122)
    ax2.bar(x - 0.2, c["bias_main"], 0.4, color="#c62828", label="main.py")
    ax2.bar(x + 0.2, c["bias_ref"], 0.4, color="#2e7d32", label="referencia")
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(["X1*", "X2*", "X3*"])
    ax2.legend(fontsize=8)
    _style(ax2, "Sesgo del óptimo estimado", "", "(x̂ − x_verdadero)/Δ")
    fig.tight_layout()
    return fig


def fig_lof(a: Audit) -> Optional[Figure]:
    """Falta de ajuste: SS_PE, SS_LOF y p-valor de main vs referencia, por dataset."""
    L = a.store.get("lof")
    if not L:
        return None
    names = list(L.keys())
    fig = _fig(12, 3.8)
    x = np.arange(len(names))
    for k, (key, lab) in enumerate((("ss_pe", "SS_PE (error puro)"), ("ss_lof", "SS_LOF"))):
        ax = fig.add_subplot(1, 3, k + 1)
        ref = [L[n]["ref"][key] for n in names]
        mn = [np.nan if L[n]["main"].get(key) is None else L[n]["main"][key] for n in names]
        ax.bar(x - 0.2, ref, 0.4, color="#2e7d32", label="referencia")
        ax.bar(x + 0.2, mn, 0.4, color="#c62828", label="main.py")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.legend(fontsize=7)
        _style(ax, lab, "dataset", "valor")
    ax = fig.add_subplot(133)
    pr = [L[n]["ref"]["p"] for n in names]
    pm = [np.nan if (L[n]["main"].get("p") is None) else L[n]["main"]["p"] for n in names]
    ax.plot(x, pr, "o", color="#2e7d32", ms=9, label="referencia")
    ax.plot(x, pm, "s", color="#c62828", ms=6, label="main.py")
    ax.axhline(0.05, color="#ef6c00", ls="--", label="α = 0.05")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(fontsize=7)
    _style(ax, "p-valor de falta de ajuste", "dataset", "p")
    fig.tight_layout()
    return fig


def build_figures(a: Audit) -> None:
    """Construye todas las gráficas; si una falla se registra como [WARN] sin detener la auditoría."""
    specs = [("dashboard", "Tablero de resultados", "Cantidad de hallazgos por sección y estado.", fig_dashboard),
             ("normal_ppf", "normal_ppf vs SciPy", "Paridad y error absoluto de Φ⁻¹(p); las colas muestran el defecto del polinomio de Horner.", fig_normal),
             ("especiales", "betainc, f_sf y t_ppf_975", "Error de las funciones especiales; el error de t_ppf_975 con pocos grados de libertad sesga todos los IC.", fig_special),
             ("geometria", "Geometría de Doehlert", "Cuboctaedro de 12 vértices a radio 1 más el centro.", fig_geometry),
             ("ols", "Mínimos cuadrados", "Desviación de β̂, condicionamiento de MᵀM y residuos.", fig_ols),
             ("optimo", "Punto estacionario", "Desvío del óptimo estimado respecto del verdadero, en unidades de semirrango.", fig_optimum),
             ("jacobiano", "Jacobiano ∂x*/∂β", "Comparación entre el jacobiano de main.py y la derivación analítica.", fig_jacobian),
             ("delta", "Método Delta y Monte Carlo", "Validez de la propagación de incertidumbre al óptimo.", fig_delta),
             ("cobertura", "Cobertura del IC 95 %", "Prueba extremo-a-extremo con datasets simulados.", fig_coverage),
             ("lof", "Falta de ajuste (ANOVA)", "SS_PE, SS_LOF y p-valor: main vs referencia.", fig_lof),
             ("validacion", "Experimentos de validación", "Desviación de cada magnitud respecto del resultado correcto (≤ 1 ⇒ dentro de la tolerancia).", fig_validation)]
    for fid, title, cap, fn in specs:
        try:
            a.add_fig(fid, title, cap, fn(a))
        except Exception:
            a.add(f"9.F.{fid}", SEC_CMP, f"Gráfica «{title}» no disponible", "WARN", "falló la generación de la figura", traceback.format_exc(limit=3))


# ═════════════════════════════════════════════════════════════════════════════
#  IMPORTADORES
# ═════════════════════════════════════════════════════════════════════════════
def load_dataset_csv(path: str) -> Dict[str, Any]:
    """
    Lector TOLERANTE de planillas de la Suite: acepta `sep=,`, separador «,» «;» o tabulador, decimal con coma,
    BOM, CRLF y líneas vacías. Columnas: Corrida | f1 | f2 | f3 | Respuesta_Y | x1_cod | x2_cod | x3_cod (las x_cod son opcionales).
    """
    with open(path, "rb") as fb:
        blob = fb.read()
    try:
        raw = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = blob.decode("latin-1")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    notes: List[str] = []
    if lines and lines[0].lower().startswith("sep="):
        lines = lines[1:]
    if len(lines) < 2:
        raise ValueError("el archivo no tiene datos")
    head = lines[0]
    sep = ";" if (";" in head and "," not in head) else ("\t" if ("\t" in head and "," not in head) else ",")
    table = list(csv.reader(lines, delimiter=sep))
    header = [h.strip() for h in table[0]]
    low = [h.lower() for h in header]
    y_idx = next((i for i, h in enumerate(low) if h in ("respuesta_y", "respuesta", "y", "response", "resultado")), None)
    if y_idx is None:
        y_idx = 4
        notes.append("no se encontró la columna de respuesta por nombre; se asumió la 5.ª columna")
    X, y, cod, skipped = [], [], [], 0
    has_cod = False
    for row in table[1:]:
        try:
            num = lambda s: float(s.strip().replace(",", "."))
            xs = [num(row[1]), num(row[2]), num(row[3])]
            yv = num(row[y_idx])
        except (ValueError, IndexError):
            skipped += 1
            continue
        X.append(xs)
        y.append(yv)
        try:
            cod.append([num(row[y_idx + 1]), num(row[y_idx + 2]), num(row[y_idx + 3])])
            has_cod = True
        except (ValueError, IndexError):
            cod.append(None)
    if len(y) == 0:
        raise ValueError("no se pudo leer ninguna fila numérica")
    if skipped:
        notes.append(f"{skipped} fila(s) descartadas por no ser numéricas o estar incompletas")
    X, y = np.array(X, float), np.array(y, float)
    ranges = [(float(X[:, j].min()), float(X[:, j].max())) for j in range(3)]
    if has_cod and all(c is not None for c in cod):
        codarr = np.array(cod, float)
    else:
        c0 = np.array([(lo + hi) / 2 for lo, hi in ranges])
        d0 = np.array([max((hi - lo) / 2, 1e-300) for lo, hi in ranges])
        codarr = (X - c0) / d0
        notes.append("el archivo no trae x_cod completas: el auditor las deriva como (x−c)/Δ (main.py usaría ceros)")
    return {"name": "IMP", "label": "IMP", "X": X, "y": y, "cod": codarr, "ranges": ranges, "kind": "imported",
            "notes": notes, "names": header[1:4], "note": f"{len(y)} corridas importadas de {os.path.basename(path)}"}


def write_dataset_csv(ds: Dict[str, Any], path: str, names=("Temperatura", "Tiempo", "pH"), dec: int = 4) -> None:
    """Escribe un dataset en el MISMO formato que exporta main.py (sep=, / coma / punto decimal / 4 decimales)."""
    hdr = ["Corrida", *names, "Respuesta_Y", "x1_cod", "x2_cod", "x3_cod"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("sep=,\n")
        w = csv.writer(f, delimiter=",")
        w.writerow(hdr)
        for i, (x, yv, c) in enumerate(zip(ds["X"], ds["y"], ds["cod"]), start=1):
            w.writerow([i, *[f"{v:.{dec}f}" for v in x], f"{yv:.{dec}f}", *[f"{v:.{dec}f}" for v in c]])


def load_report_json(path: str) -> Dict[str, Any]:
    """Importa un reporte JSON generado por este auditor (para comparar corridas)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not str(data.get("meta", {}).get("schema", "")).startswith("doehlert-audit"):
        raise ValueError("el archivo no es un reporte de audit_doehlert.py (falta meta.schema)")
    return data


def compare_reports(prev: Dict[str, Any], a: Audit) -> Dict[str, Any]:
    """Tabla de cambios por id de hallazgo: corregido / regresión / cambió / nuevo / retirado / igual."""
    pm = {f["id"]: f for f in prev.get("findings", []) if not str(f["id"]).startswith("9.")}
    cm = {f.id: f for f in a.findings if not f.id.startswith("9.")}
    ids = list(cm.keys()) + [i for i in pm if i not in cm]
    rows, s = [], {"fixed": 0, "regressed": 0, "changed": 0, "new": 0, "removed": 0, "same": 0}
    for i in ids:
        p, c = pm.get(i), cm.get(i)
        if p and c:
            ps, cs = p["status"], c.status
            if ps == cs:
                ch, key = "igual", "same"
            elif ps == "FAIL" and cs in ("PASS", "WARN"):
                ch, key = "corregido", "fixed"
            elif cs == "FAIL":
                ch, key = "regresión", "regressed"
            else:
                ch, key = "cambió", "changed"
            rows.append({"id": i, "title": c.title, "prev": ps, "curr": cs, "change": ch})
        elif c:
            ch, key = "nuevo", "new"
            rows.append({"id": i, "title": c.title, "prev": None, "curr": c.status, "change": ch})
        else:
            ch, key = "retirado", "removed"
            rows.append({"id": i, "title": p["title"], "prev": p["status"], "curr": None, "change": ch})
        s[key] += 1
    return {"previous": prev.get("meta", {}), "rows": rows, "summary": s}


# ═════════════════════════════════════════════════════════════════════════════
#  SALIDA DE CONSOLA
# ═════════════════════════════════════════════════════════════════════════════
def render_console(a: Audit, color: bool = False, detail: str = "problems") -> str:
    """Checklist estructurado [PASS]/[FAIL]/[WARN]. detail: 'none' | 'problems' (detalles de FAIL/WARN) | 'all'."""
    def tag(st: str) -> str:
        return f"{ANSI[st]}[{st}]{ANSI['END']}" if color else f"[{st}]"
    L: List[str] = []
    bar = "═" * 100
    L += [bar, " AUDITORÍA FORMAL · Suite Doehlert (k = 3)", bar,
          f" Archivo auditado : {a.meta['target']}  ({a.meta['target_lines']} líneas, sha256 {a.meta['target_sha256'][:12]}…)",
          f" Entorno          : Python {a.meta['python']} · NumPy {a.meta['numpy']} · SciPy {a.meta['scipy']} · Matplotlib {a.meta['matplotlib']}"
          f" · statsmodels {a.meta['statsmodels'] or 'no instalado'}",
          f" Tolerancias      : atol = {TOL_ABS:g}, rtol = {TOL_REL:g}; [WARN] hasta {WARN_FACTOR:g}× la tolerancia; reportes impresos ±{PRINT_TOL:g}",
          f" Modo             : {'rápido' if a.quick else 'completo'} · semilla {a.seed}", ""]
    for sec, fs in a.by_section().items():
        L.append(f"── {sec} " + "─" * max(0, 96 - len(sec)))
        for f in fs:
            L.append(f" {tag(f.status)} {f.id:<7} {f.title}")
            if f.summary:
                L.append(f"          ↳ {f.summary}")
            if detail == "all" or (detail == "problems" and f.status in ("FAIL", "WARN")):
                for ln in f.details.splitlines():
                    if detail == "all" or ln.startswith("[FAIL]") or ln.startswith("[WARN]") or ln.startswith("•"):
                        L.append("             " + ln)
                if f.fix and f.status in ("FAIL", "WARN"):
                    L.append(f"          ✎ corrección sugerida: {f.fix}")
        L.append("")
    c = a.counts()
    L += [bar, f" RESUMEN: {tag('PASS')} {c['PASS']}   {tag('WARN')} {c['WARN']}   {tag('FAIL')} {c['FAIL']}   {tag('SKIP')} {c['SKIP']}   "
                f"(total {len(a.findings)} · {a.meta.get('elapsed_s', 0)} s)", bar]
    fails = [f for f in a.findings if f.status == "FAIL"]
    if fails:
        L += ["", " HALLAZGOS QUE REQUIEREN ATENCIÓN (FAIL):"]
        for f in fails:
            L.append(f"  • {f.id:<7} {f.title}" + (f"   [{f.where}]" if f.where else ""))
    if a.comparison:
        s = a.comparison["summary"]
        L += ["", f" COMPARACIÓN CON CORRIDA PREVIA: {s['fixed']} corregidos · {s['regressed']} regresiones · {s['new']} nuevos · {s['removed']} retirados"]
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════════
#  EXPORTADORES
# ═════════════════════════════════════════════════════════════════════════════
def fig_png_b64(fig: Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def report_dict(a: Audit) -> Dict[str, Any]:
    """Estructura serializable del reporte (formato re-importable con --import-report)."""
    ds_info = [{"name": d["name"], "label": d.get("label", d["name"]), "N": int(len(d["y"])), "kind": d["kind"],
                "ranges": _ranges_of(d), "truth": d.get("truth")} for d in a.datasets]
    return jsonable({"meta": a.meta, "summary": a.counts(), "findings": [f.to_dict() for f in a.findings],
                     "datasets": ds_info, "comparison": a.comparison, "imported": a.imported_info})


def export_json(a: Audit, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict(a), f, ensure_ascii=False, indent=2)
    return path


def export_csv(a: Audit, path: str) -> str:
    """Checklist plano (una fila por hallazgo), con BOM y «;»-free: coma + punto decimal."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "seccion", "titulo", "estado", "resumen", "ubicacion", "correccion_sugerida", "metricas_json"])
        for fd in a.findings:
            w.writerow([fd.id, fd.section, fd.title, fd.status, fd.summary, fd.where, fd.fix, json.dumps(jsonable(fd.metrics), ensure_ascii=False)])
    return path


def export_png(a: Audit, outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for fid, _t, _c, fig in a.figures:
        p = os.path.join(outdir, f"{fid}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        paths.append(p)
    return paths


def export_datasets(a: Audit, outdir: str) -> List[str]:
    """Datasets sintéticos en formato de la Suite: se pueden cargar en main.py con «Cargar CSV»."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for d in a.datasets:
        p = os.path.join(outdir, f"{d['name']}.csv")
        write_dataset_csv(d, p, names=tuple(d.get("names") or ("Temperatura", "Tiempo", "pH")))
        paths.append(p)
    return paths


def _checklist_pages(a: Audit, per_page: int = 46) -> List[Figure]:
    """Páginas de texto con el checklist completo (para el PDF)."""
    rows: List[Tuple[str, str, str]] = []
    for sec, fs in a.by_section().items():
        rows.append(("SEC", sec, ""))
        for f in fs:
            rows.append((f.status, f"{f.id:<7} {f.title}", f.summary))
    pages, cur = [], []
    for r in rows:
        cur.append(r)
        if len(cur) >= per_page:
            pages.append(cur)
            cur = []
    if cur:
        pages.append(cur)
    figs = []
    for k, pg in enumerate(pages):
        fig = Figure(figsize=(8.5, 11), dpi=100, facecolor="white")
        fig.text(0.05, 0.97, f"Checklist de auditoría · Suite Doehlert  (pág. {k + 1}/{len(pages)})", fontsize=12, fontweight="bold", va="top")
        y = 0.94
        for st, title, summ in pg:
            if st == "SEC":
                fig.text(0.05, y, title, fontsize=9.5, fontweight="bold", va="top", color="#1a237e")
                y -= 0.019
            else:
                fig.text(0.05, y, f"[{st}]", fontsize=8, va="top", family="monospace", color=COLORS[st], fontweight="bold")
                fig.text(0.13, y, title[:78], fontsize=8, va="top")
                y -= 0.0125
                if summ:
                    fig.text(0.15, y, "↳ " + summ[:100], fontsize=7, va="top", color="#555555")
                    y -= 0.0165
        figs.append(fig)
    return figs


def export_pdf(a: Audit, path: str) -> str:
    """PDF multipágina: portada con metadatos, checklist y todas las gráficas."""
    with PdfPages(path) as pdf:
        cover = Figure(figsize=(8.5, 11), dpi=100, facecolor="white")
        c = a.counts()
        cover.text(0.5, 0.88, "Auditoría formal", ha="center", fontsize=24, fontweight="bold")
        cover.text(0.5, 0.84, "Suite Doehlert (k = 3)", ha="center", fontsize=16)
        info = [f"Archivo: {os.path.basename(a.meta['target'])}  ({a.meta['target_lines']} líneas)", f"sha256: {a.meta['target_sha256'][:32]}…",
                f"Fecha: {a.meta['created']}", f"Python {a.meta['python']} · NumPy {a.meta['numpy']} · SciPy {a.meta['scipy']}",
                f"Modo: {'rápido' if a.quick else 'completo'} · semilla {a.seed}"]
        for i, t in enumerate(info):
            cover.text(0.5, 0.77 - 0.03 * i, t, ha="center", fontsize=10)
        for i, st in enumerate(("PASS", "WARN", "FAIL", "SKIP")):
            cover.text(0.2 + 0.2 * i, 0.55, str(c[st]), ha="center", fontsize=34, fontweight="bold", color=COLORS[st])
            cover.text(0.2 + 0.2 * i, 0.52, st, ha="center", fontsize=11, color=COLORS[st])
        fails = [f for f in a.findings if f.status == "FAIL"]
        cover.text(0.08, 0.44, "Hallazgos [FAIL]:", fontsize=11, fontweight="bold")
        for i, f in enumerate(fails[:16]):
            cover.text(0.08, 0.41 - 0.022 * i, f"• {f.id}  {f.title[:88]}", fontsize=8)
        pdf.savefig(cover)
        for fg in _checklist_pages(a):
            pdf.savefig(fg)
        for _fid, title, cap, fig in a.figures:
            fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
            pdf.savefig(fig, bbox_inches="tight")
            fig.suptitle("")
    return path


_HTML_CSS = """
:root{--bg:#ffffff;--fg:#1d2430;--mut:#5b6573;--card:#f5f7fa;--bd:#dde3ea;--pass:#2e7d32;--fail:#c62828;--warn:#ef6c00;--skip:#757575}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--fg:#e6e9ee;--mut:#9aa5b4;--card:#1b212b;--bd:#2d3643}}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);margin:0;line-height:1.45}
main{max-width:1180px;margin:0 auto;padding:24px 18px 60px}h1{margin:.2em 0}h2{margin-top:2em;border-bottom:2px solid var(--bd);padding-bottom:.2em}
.meta{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 16px;font-size:.88rem;color:var(--mut)}
.chips{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.chip{border-radius:12px;padding:10px 20px;color:#fff;font-weight:700;font-size:1.05rem;min-width:110px;text-align:center}
.chip small{display:block;font-weight:500;opacity:.9}.PASS{background:var(--pass)}.FAIL{background:var(--fail)}.WARN{background:var(--warn)}.SKIP{background:var(--skip)}
.filters button{border:1px solid var(--bd);background:var(--card);color:var(--fg);border-radius:20px;padding:6px 14px;margin-right:6px;cursor:pointer;font-size:.9rem}
.filters button.on{background:var(--fg);color:var(--bg)}details.f{border:1px solid var(--bd);border-radius:10px;margin:8px 0;background:var(--card)}
details.f summary{cursor:pointer;padding:9px 12px;list-style:none;display:flex;gap:10px;align-items:baseline}details.f summary::-webkit-details-marker{display:none}
.badge{font-family:ui-monospace,Consolas,monospace;font-size:.78rem;font-weight:700;color:#fff;border-radius:6px;padding:2px 8px;white-space:nowrap}
.fid{font-family:ui-monospace,Consolas,monospace;color:var(--mut);font-size:.85rem;min-width:52px}.ttl{font-weight:600}.sum{color:var(--mut);font-size:.88rem;display:block}
.body{padding:2px 14px 12px}pre{background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px;overflow:auto;font-size:.8rem;white-space:pre-wrap}
.fix{border-left:4px solid var(--warn);padding:6px 10px;background:var(--bg);margin:8px 0;font-size:.88rem}.where{color:var(--mut);font-size:.82rem}
figure{margin:18px 0;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px}figure img{max-width:100%;height:auto;background:#fff;border-radius:6px}
figcaption{color:var(--mut);font-size:.88rem;margin-top:6px}table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{border:1px solid var(--bd);padding:5px 8px;text-align:left}
"""

_HTML_JS = """
document.querySelectorAll('.filters button').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 const f=b.dataset.f;document.querySelectorAll('details.f').forEach(d=>{d.style.display=(f==='ALL'||d.dataset.s===f)?'':'none'});
 document.querySelectorAll('h2.sec').forEach(h=>{let n=h.nextElementSibling,vis=false;while(n&&!n.matches('h2')){if(n.matches('details.f')&&n.style.display!=='none')vis=true;n=n.nextElementSibling}h.style.display=vis?'':'none'})}));
document.getElementById('openfails')?.addEventListener('click',()=>document.querySelectorAll('details.f[data-s=FAIL]').forEach(d=>d.open=true));
"""


def export_html(a: Audit, path: str) -> str:
    """Informe HTML autocontenido: resumen, filtros por estado, hallazgos desplegables, gráficas incrustadas y comparación."""
    e = html_lib.escape
    c = a.counts()
    out: List[str] = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                      f"<title>Auditoría Suite Doehlert</title><style>{_HTML_CSS}</style></head><body><main>",
                      "<h1>Auditoría formal · Suite Doehlert (k = 3)</h1>",
                      f"<div class='meta'><b>Archivo:</b> {e(a.meta['target'])} ({a.meta['target_lines']} líneas) · <b>sha256:</b> {a.meta['target_sha256'][:16]}… · "
                      f"<b>fecha:</b> {e(a.meta['created'])}<br><b>Entorno:</b> Python {a.meta['python']} · NumPy {a.meta['numpy']} · SciPy {a.meta['scipy']} · "
                      f"Matplotlib {a.meta['matplotlib']} · statsmodels {a.meta['statsmodels'] or 'no instalado'} · modo {'rápido' if a.quick else 'completo'} · semilla {a.seed}<br>"
                      f"<b>Tolerancias:</b> atol = {TOL_ABS:g}, rtol = {TOL_REL:g}; [WARN] hasta {WARN_FACTOR:g}× la tolerancia.</div>",
                      "<div class='chips'>" + "".join(f"<div class='chip {s}'>{c[s]}<small>{s}</small></div>" for s in ("PASS", "WARN", "FAIL", "SKIP")) + "</div>",
                      "<div class='filters'><button class='on' data-f='ALL'>Todos</button>" + "".join(f"<button data-f='{s}'>{s}</button>" for s in ("FAIL", "WARN", "PASS", "SKIP")) +
                      " <button id='openfails'>Desplegar los FAIL</button></div>"]
    fig_map = {fid: (title, cap, fig) for fid, title, cap, fig in a.figures}
    if "dashboard" in fig_map:
        t, cap, fg = fig_map["dashboard"]
        out.append(f"<figure><img alt='{e(t)}' src='data:image/png;base64,{fig_png_b64(fg)}'><figcaption>{e(cap)}</figcaption></figure>")
    for sec, fs in a.by_section().items():
        out.append(f"<h2 class='sec'>{e(sec)}</h2>")
        for f in fs:
            body = ""
            if f.details:
                body += f"<pre>{e(f.details)}</pre>"
            if f.where:
                body += f"<div class='where'>📍 {e(f.where)}</div>"
            if f.fix:
                body += f"<div class='fix'><b>Corrección sugerida:</b> {e(f.fix)}</div>"
            out.append(f"<details class='f' data-s='{f.status}'><summary><span class='badge {f.status}'>{f.status}</span><span class='fid'>{e(f.id)}</span>"
                       f"<span><span class='ttl'>{e(f.title)}</span><span class='sum'>{e(f.summary)}</span></span></summary><div class='body'>{body or '<i>sin detalles</i>'}</div></details>")
    out.append("<h2>Gráficas</h2>")
    for fid, title, cap, fig in a.figures:
        if fid == "dashboard":
            continue
        out.append(f"<figure><b>{e(title)}</b><br><img alt='{e(title)}' src='data:image/png;base64,{fig_png_b64(fig)}'><figcaption>{e(cap)}</figcaption></figure>")
    if a.comparison:
        out.append("<h2>Comparación con la corrida previa</h2><table><tr><th>Cambio</th><th>ID</th><th>Antes</th><th>Ahora</th><th>Hallazgo</th></tr>")
        for r in a.comparison["rows"]:
            if r["change"] != "igual":
                out.append(f"<tr><td>{e(r['change'])}</td><td>{e(str(r['id']))}</td><td>{r['prev'] or '—'}</td><td>{r['curr'] or '—'}</td><td>{e(r['title'])}</td></tr>")
        out.append("</table>")
    out.append(f"<script>{_HTML_JS}</script></main></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def export_all(a: Audit, outdir: str, formats: List[str], console_text: str = "") -> Dict[str, Any]:
    """Exporta los formatos pedidos a `outdir`. formats ⊂ {html, pdf, json, csv, png, datasets, log}."""
    os.makedirs(outdir, exist_ok=True)
    res: Dict[str, Any] = {}
    fm = set(formats)
    if "json" in fm:
        res["json"] = export_json(a, os.path.join(outdir, "audit_report.json"))
    if "csv" in fm:
        res["csv"] = export_csv(a, os.path.join(outdir, "audit_report.csv"))
    if "html" in fm:
        res["html"] = export_html(a, os.path.join(outdir, "audit_report.html"))
    if "pdf" in fm:
        res["pdf"] = export_pdf(a, os.path.join(outdir, "audit_report.pdf"))
    if "png" in fm:
        res["png"] = export_png(a, os.path.join(outdir, "figures"))
    if "datasets" in fm:
        res["datasets"] = export_datasets(a, os.path.join(outdir, "datasets"))
    if "validation" in fm:
        res["validation"] = export_validation(a, os.path.join(outdir, "validation"))
    if "log" in fm:
        p = os.path.join(outdir, "audit_log.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(console_text or render_console(a, color=False, detail="all"))
        res["log"] = p
    return res


ALL_FORMATS = ["html", "pdf", "json", "csv", "png", "datasets", "validation", "log"]


# ═════════════════════════════════════════════════════════════════════════════
#  INTERFAZ GRÁFICA  (python3 audit_doehlert.py --gui)
# ═════════════════════════════════════════════════════════════════════════════
def launch_gui(target: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> int:
    """
    Visor interactivo: ejecuta la auditoría en un hilo, muestra el checklist con colores, el detalle de cada
    hallazgo, todas las gráficas (con barra de zoom) y permite IMPORTAR (main.py, dataset CSV, reporte JSON previo)
    y EXPORTAR (HTML, PDF, JSON, CSV, PNG, datasets, todo).
    """
    try:
        import queue
        import threading
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    except Exception as e:                                       # pragma: no cover
        print(f"[ERROR] La interfaz gráfica requiere Tkinter: {e}\nUse el modo consola (sin --gui).")
        return 2

    ICON = {"PASS": "✔ PASS", "FAIL": "✘ FAIL", "WARN": "▲ WARN", "SKIP": "○ SKIP"}

    class AuditorGUI(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Auditor formal · Suite Doehlert")
            self.geometry("1320x820")
            self.minsize(1050, 640)
            self.target = target or find_default_main() or ""
            self.import_data = getattr(args, "import_data", None)
            self.import_report = getattr(args, "import_report", None)
            self.quick = tk.BooleanVar(value=bool(getattr(args, "quick", False)))
            self.audit: Optional[Audit] = None
            self.q: "queue.Queue" = queue.Queue()
            self.canvas: Any = None
            self.toolbar: Any = None
            self._build()
            self.after(150, self._poll)
            if self.target and os.path.isfile(self.target):
                self.after(300, self.run_audit)

        # ---- construcción de la ventana -----------------------------------------
        def _build(self) -> None:
            bar = ttk.Frame(self, padding=(8, 6))
            bar.pack(fill="x")
            ttk.Button(bar, text="▶ Ejecutar auditoría", command=self.run_audit).pack(side="left", padx=2)
            ttk.Button(bar, text="📂 main.py…", command=self.pick_main).pack(side="left", padx=2)
            ttk.Button(bar, text="⬇ Dataset CSV…", command=self.pick_dataset).pack(side="left", padx=2)
            ttk.Button(bar, text="⬇ Reporte JSON previo…", command=self.pick_report).pack(side="left", padx=2)
            mb = ttk.Menubutton(bar, text="⬆ Exportar")
            menu = tk.Menu(mb, tearoff=0)
            for lab, fmt in (("Informe HTML…", "html"), ("Informe PDF…", "pdf"), ("JSON (re-importable)…", "json"), ("Checklist CSV…", "csv"),
                             ("Gráficas PNG (carpeta)…", "png"), ("Datasets sintéticos CSV (carpeta)…", "datasets"),
                             ("Experimentos de validación: guía + CSV (carpeta)…", "validation")):
                menu.add_command(label=lab, command=lambda f=fmt: self.export(f))
            menu.add_separator()
            menu.add_command(label="Exportar TODO (carpeta)…", command=lambda: self.export("all"))
            mb["menu"] = menu
            mb.pack(side="left", padx=2)
            ttk.Checkbutton(bar, text="modo rápido", variable=self.quick).pack(side="left", padx=10)
            self.lbl_target = ttk.Label(bar, text="", foreground="#555")
            self.lbl_target.pack(side="left", padx=10)
            pb = ttk.Frame(self, padding=(8, 0))
            pb.pack(fill="x")
            self.prog = ttk.Progressbar(pb, mode="determinate", maximum=100)
            self.prog.pack(side="left", fill="x", expand=True)
            self.lbl_status = ttk.Label(pb, text="listo", width=44, anchor="e")
            self.lbl_status.pack(side="left", padx=8)
            self.lbl_counts = ttk.Label(self, text="", font=("TkDefaultFont", 11, "bold"), padding=(10, 4))
            self.lbl_counts.pack(fill="x")
            pw = ttk.PanedWindow(self, orient="horizontal")
            pw.pack(fill="both", expand=True, padx=6, pady=4)
            left = ttk.Frame(pw)
            pw.add(left, weight=2)
            fl = ttk.Frame(left)
            fl.pack(fill="x")
            self.cbo_filter = ttk.Combobox(fl, values=["Todos", "FAIL", "WARN", "PASS", "SKIP"], state="readonly", width=8)
            self.cbo_filter.current(0)
            self.cbo_filter.pack(side="left", padx=2, pady=2)
            self.cbo_filter.bind("<<ComboboxSelected>>", lambda e: self._fill_tree())
            self.var_search = tk.StringVar()
            ent = ttk.Entry(fl, textvariable=self.var_search)
            ent.pack(side="left", fill="x", expand=True, padx=2)
            ent.bind("<KeyRelease>", lambda e: self._fill_tree())
            self.tree = ttk.Treeview(left, columns=("st", "id", "title"), show="headings", selectmode="browse")
            for c, t, w in (("st", "Estado", 80), ("id", "ID", 60), ("title", "Hallazgo", 420)):
                self.tree.heading(c, text=t)
                self.tree.column(c, width=w, anchor="w")
            for st in STATUS_ORDER:
                self.tree.tag_configure(st, foreground=COLORS[st])
            sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            self.tree.pack(fill="both", expand=True)
            self.tree.bind("<<TreeviewSelect>>", self._on_select)
            right = ttk.Notebook(pw)
            self.nb = right
            pw.add(right, weight=3)
            self.tab_det = ttk.Frame(right)
            self.tab_fig = ttk.Frame(right)
            self.tab_cmp = ttk.Frame(right)
            self.tab_sum = ttk.Frame(right)
            right.add(self.tab_det, text=" Detalle ")
            right.add(self.tab_fig, text=" Gráficas ")
            right.add(self.tab_cmp, text=" Comparación ")
            right.add(self.tab_sum, text=" Consola ")
            self.txt = tk.Text(self.tab_det, wrap="word", font=("Consolas", 10), padx=8, pady=8)
            self.txt.pack(fill="both", expand=True)
            for st in STATUS_ORDER:
                self.txt.tag_configure(st, foreground=COLORS[st], font=("TkDefaultFont", 12, "bold"))
            self.txt.tag_configure("h", font=("TkDefaultFont", 12, "bold"))
            self.txt.tag_configure("fix", foreground="#b25e00")
            ff = ttk.Frame(self.tab_fig)
            ff.pack(fill="x")
            ttk.Label(ff, text="Gráfica:").pack(side="left", padx=4)
            self.cbo_fig = ttk.Combobox(ff, state="readonly", width=48)
            self.cbo_fig.pack(side="left", padx=4, pady=4)
            self.cbo_fig.bind("<<ComboboxSelected>>", lambda e: self._show_fig())
            self.fig_holder = ttk.Frame(self.tab_fig)
            self.fig_holder.pack(fill="both", expand=True)
            self.tree_cmp = ttk.Treeview(self.tab_cmp, columns=("ch", "id", "prev", "curr", "title"), show="headings")
            for c, t, w in (("ch", "Cambio", 90), ("id", "ID", 60), ("prev", "Antes", 60), ("curr", "Ahora", 60), ("title", "Hallazgo", 420)):
                self.tree_cmp.heading(c, text=t)
                self.tree_cmp.column(c, width=w)
            self.tree_cmp.pack(fill="both", expand=True)
            self.txt_sum = tk.Text(self.tab_sum, wrap="none", font=("Consolas", 9))
            self.txt_sum.pack(fill="both", expand=True)
            self._refresh_target()

        def _refresh_target(self) -> None:
            extra = ""
            if self.import_data:
                extra += f" · dataset: {os.path.basename(self.import_data)}"
            if self.import_report:
                extra += f" · previo: {os.path.basename(self.import_report)}"
            self.lbl_target.config(text=f"{os.path.basename(self.target) or '(sin main.py)'}{extra}")

        # ---- importar ------------------------------------------------------------
        def pick_main(self) -> None:
            p = filedialog.askopenfilename(title="Seleccionar el main.py a auditar", filetypes=[("Python", "*.py"), ("Todos", "*.*")])
            if p:
                self.target = p
                self._refresh_target()
                self.run_audit()

        def pick_dataset(self) -> None:
            p = filedialog.askopenfilename(title="Importar dataset (CSV de la Suite)", filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
            if p:
                self.import_data = p
                self._refresh_target()
                self.run_audit()

        def pick_report(self) -> None:
            p = filedialog.askopenfilename(title="Importar reporte JSON previo", filetypes=[("JSON", "*.json"), ("Todos", "*.*")])
            if p:
                self.import_report = p
                self._refresh_target()
                if self.audit is not None:
                    self.run_audit()

        # ---- ejecución en hilo ---------------------------------------------------
        def run_audit(self) -> None:
            if not (self.target and os.path.isfile(self.target)):
                messagebox.showwarning("Auditor", "Seleccione primero el archivo main.py a auditar.")
                return
            self.prog["value"] = 0
            self.lbl_status.config(text="iniciando…")
            q = self.q
            tgt, qk, idat, irep = self.target, bool(self.quick.get()), self.import_data, self.import_report

            def work() -> None:
                try:
                    a = Audit(tgt, quick=qk, import_data=idat, import_report=irep, progress=lambda m, f: q.put(("p", m, f)))
                    a.run()
                    q.put(("done", a))
                except Exception:
                    q.put(("err", traceback.format_exc()))
            threading.Thread(target=work, daemon=True).start()

        def _poll(self) -> None:
            try:
                while True:
                    msg = self.q.get_nowait()
                    if msg[0] == "p":
                        self.prog["value"] = msg[2] * 100
                        self.lbl_status.config(text=msg[1])
                    elif msg[0] == "done":
                        self.audit = msg[1]
                        self.prog["value"] = 100
                        self.lbl_status.config(text=f"completado en {self.audit.meta.get('elapsed_s', 0)} s")
                        self._on_done()
                    elif msg[0] == "err":
                        self.lbl_status.config(text="error")
                        messagebox.showerror("Auditor", msg[1][-1500:])
            except queue.Empty:
                pass
            self.after(120, self._poll)

        def _on_done(self) -> None:
            a = self.audit
            assert a is not None
            c = a.counts()
            self.lbl_counts.config(text=f"  ✔ {c['PASS']} PASS    ▲ {c['WARN']} WARN    ✘ {c['FAIL']} FAIL    ○ {c['SKIP']} SKIP    ({len(a.findings)} hallazgos)",
                                   foreground=COLORS["FAIL"] if c["FAIL"] else COLORS["PASS"])
            self._fill_tree()
            self.cbo_fig["values"] = [f"{t}" for _i, t, _c, _f in a.figures]
            if a.figures:
                self.cbo_fig.current(0)
                self._show_fig()
            self.tree_cmp.delete(*self.tree_cmp.get_children())
            if a.comparison:                                   # primero los cambios; luego los hallazgos sin cambios
                for r in sorted(a.comparison["rows"], key=lambda r: r["change"] == "igual"):
                    self.tree_cmp.insert("", "end", values=(r["change"], r["id"], r["prev"] or "—", r["curr"] or "—", r["title"]))
            self.txt_sum.delete("1.0", "end")
            self.txt_sum.insert("1.0", render_console(a, color=False, detail="problems"))

        def _fill_tree(self) -> None:
            if self.audit is None:
                return
            flt, q = self.cbo_filter.get(), self.var_search.get().lower().strip()
            self.tree.delete(*self.tree.get_children())
            for i, f in enumerate(self.audit.findings):
                if flt != "Todos" and f.status != flt:
                    continue
                if q and q not in (f.id + " " + f.title + " " + f.summary).lower():
                    continue
                self.tree.insert("", "end", iid=str(i), values=(ICON[f.status], f.id, f.title), tags=(f.status,))

        def _on_select(self, _e=None) -> None:
            sel = self.tree.selection()
            if not sel or self.audit is None:
                return
            f = self.audit.findings[int(sel[0])]
            self.txt.config(state="normal")
            self.txt.delete("1.0", "end")
            self.txt.insert("end", f"{ICON[f.status]}  ", f.status)
            self.txt.insert("end", f"{f.id} · {f.title}\n\n", "h")
            self.txt.insert("end", f"{f.summary}\n\n")
            if f.details:
                self.txt.insert("end", f.details + "\n\n")
            if f.where:
                self.txt.insert("end", f"📍 {f.where}\n")
            if f.fix:
                self.txt.insert("end", f"\n✎ Corrección sugerida: {f.fix}\n", "fix")
            self.txt.config(state="disabled")

        def _show_fig(self) -> None:
            if self.audit is None or not self.audit.figures:
                return
            idx = self.cbo_fig.current()
            fig = self.audit.figures[idx][3]
            for w in self.fig_holder.winfo_children():
                w.destroy()
            self.canvas = FigureCanvasTkAgg(fig, master=self.fig_holder)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_holder)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.canvas.draw()

        # ---- exportar --------------------------------------------------------------
        def export(self, fmt: str) -> None:
            a = self.audit
            if a is None:
                messagebox.showinfo("Auditor", "Ejecute primero la auditoría.")
                return
            try:
                if fmt in ("html", "pdf", "json", "csv"):
                    p = filedialog.asksaveasfilename(defaultextension="." + fmt, initialfile=f"audit_report.{fmt}", filetypes=[(fmt.upper(), "*." + fmt)])
                    if not p:
                        return
                    {"html": export_html, "pdf": export_pdf, "json": export_json, "csv": export_csv}[fmt](a, p)
                    messagebox.showinfo("Exportado", f"Guardado en:\n{p}")
                else:
                    d = filedialog.askdirectory(title="Carpeta de destino")
                    if not d:
                        return
                    if fmt == "all":
                        res = export_all(a, d, ALL_FORMATS, render_console(a, False, "all"))
                        messagebox.showinfo("Exportado", "Archivos generados en:\n" + d + "\n\n" + "\n".join(sorted(res.keys())))
                    elif fmt == "png":
                        export_png(a, os.path.join(d, "figures"))
                        messagebox.showinfo("Exportado", f"Gráficas guardadas en {os.path.join(d, 'figures')}")
                    elif fmt == "validation":
                        export_validation(a, os.path.join(d, "validation"))
                        messagebox.showinfo("Exportado", f"Guía y CSV de validación en {os.path.join(d, 'validation')}\n(abrí guia_validacion.html)")
                    else:
                        export_datasets(a, os.path.join(d, "datasets"))
                        messagebox.showinfo("Exportado", f"Datasets guardados en {os.path.join(d, 'datasets')}\n(se pueden cargar en main.py con «Cargar CSV»)")
            except Exception as e:
                messagebox.showerror("Error al exportar", f"{type(e).__name__}: {e}")

    AuditorGUI().mainloop()
    return 0


# ═════════════════════════════════════════════════════════════════════════════
#  LÍNEA DE COMANDOS
# ═════════════════════════════════════════════════════════════════════════════
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="audit_doehlert.py", description="Auditor formal de la Suite Doehlert (main.py).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=textwrap.dedent("""\
                                 ejemplos:
                                   python3 audit_doehlert.py
                                   python3 audit_doehlert.py --main main_1.py --out informe --quick
                                   python3 audit_doehlert.py --import-data mis_datos.csv --import-report audit_previo.json
                                   python3 audit_doehlert.py --gui"""))
    ap.add_argument("--main", help="ruta del main.py a auditar (por defecto: el main*.py más reciente junto al auditor)")
    ap.add_argument("--out", help="carpeta de salida (por defecto ./audit_output_<fecha>)")
    ap.add_argument("--export", default="all", help=f"formatos a exportar: all | none | lista separada por comas de {','.join(ALL_FORMATS)}")
    ap.add_argument("--import-data", help="CSV real a auditar con la misma batería (formato de la Suite; tolera «;» y coma decimal)")
    ap.add_argument("--import-report", help="reporte JSON de una corrida previa para comparar (corregidos / regresiones)")
    ap.add_argument("--gui", action="store_true", help="abre el visor gráfico (Tkinter)")
    ap.add_argument("--open", action="store_true", help="abre el informe HTML en el navegador al terminar")
    ap.add_argument("--quick", action="store_true", help="modo rápido (mallas y simulaciones más pequeñas)")
    ap.add_argument("--validate", action="store_true", help="solo los experimentos de validación (E1–E4): rápido; exporta la guía y los CSV con los valores a ingresar")
    ap.add_argument("--seed", type=int, default=20260108, help="semilla de los datasets sintéticos")
    ap.add_argument("--detail", choices=["none", "problems", "all"], default="problems", help="detalle en consola (por defecto: solo FAIL/WARN)")
    ap.add_argument("--no-color", action="store_true", help="sin colores ANSI")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    target = args.main or find_default_main()
    if args.gui:
        return launch_gui(target, args)
    if not target:
        print("[ERROR] No se encontró main.py. Indique la ruta con --main.", file=sys.stderr)
        return 2

    def progress(msg: str, frac: float) -> None:
        sys.stderr.write(f"\r[{int(frac * 100):3d}%] {msg:<44}")
        sys.stderr.flush()

    try:
        a = Audit(target, quick=args.quick, seed=args.seed, progress=progress, import_data=args.import_data, import_report=args.import_report)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {target}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    a.run(only=["chk_validation", "build_figures"] if args.validate else None)
    sys.stderr.write("\r" + " " * 60 + "\r")
    color = sys.stdout.isatty() and not args.no_color
    print(render_console(a, color=color, detail=args.detail))
    if args.export.lower() != "none":
        fmts = ALL_FORMATS if args.export.lower() == "all" else [x.strip() for x in args.export.split(",") if x.strip()]
        if args.validate and args.export.lower() == "all":
            fmts = ["html", "json", "csv", "validation", "log"]
        bad = [x for x in fmts if x not in ALL_FORMATS]
        if bad:
            print(f"[ERROR] formatos desconocidos: {bad}", file=sys.stderr)
            return 2
        out = args.out or os.path.join(os.getcwd(), "audit_output_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        res = export_all(a, out, fmts, render_console(a, color=False, detail="all"))
        print(f"\n Archivos exportados en: {os.path.abspath(out)}")
        for k, v in res.items():
            print(f"   · {k:<8} {v if isinstance(v, str) else str(len(v)) + ' archivos'}")
        if args.open and "html" in res:
            webbrowser.open("file://" + os.path.abspath(res["html"]))
    return 1 if a.counts()["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
