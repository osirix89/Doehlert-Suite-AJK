#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""
doehlert_engine.py
==================
Motor matemático/estadístico de la Suite Doehlert (3 factores, modelo cuadrático
completo de 10 parámetros).

Restricciones de arquitectura
-----------------------------
* Cero dependencias de SciPy: las distribuciones t de Student y F de Fisher se
  implementan con la beta incompleta regularizada (fracción continua de Lentz).
* Sin Tkinter ni Matplotlib: este módulo es 100 % lógica, por lo que puede
  importarse y probarse sin entorno gráfico.
* Dependencia opcional: ``openpyxl`` (solo para exportar .xlsx).
* Uso por consola (sin ventana):  python3 doehlert_engine.py --help

Orden interno de los parámetros (columnas de la matriz de diseño)
-----------------------------------------------------------------
    [ b0, b1, b2, b3, b12, b13, b23, b11, b22, b33 ]
Para mostrar tablas se usa el orden DISPLAY_ORDER:
    [ b0, b1, b2, b3, b11, b22, b33, b12, b13, b23 ]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ==============================================================
# GEOMETRÍA CANÓNICA DE DOEHLERT (k = 3)
# ==============================================================
# X1: 7 niveles (0, ±sqrt3/6, ±1/sqrt3, ±sqrt3/2)
# X2: 5 niveles (0, ±0.5, ±1)
# X3: 3 niveles (0, ±sqrt(2/3))
# Los 12 puntos periféricos están sobre la esfera de radio 1 (en variables
# codificadas canónicas), de modo que el dominio experimental es ||x|| <= 1.
SQRT3 = math.sqrt(3.0)
SQRT_2_3 = math.sqrt(2.0 / 3.0)

DOEHLERT_BASE = [
    (0.0, 0.0, 0.0),                          # Centro inicial
    (0.0, 1.0, 0.0),                          # X2
    (0.0, -1.0, 0.0),
    (SQRT3 / 2.0, 0.5, 0.0),                  # X1, X2
    (-SQRT3 / 2.0, -0.5, 0.0),
    (-SQRT3 / 2.0, 0.5, 0.0),
    (SQRT3 / 2.0, -0.5, 0.0),
    (SQRT3 / 6.0, 0.5, SQRT_2_3),             # X1, X2, X3
    (-SQRT3 / 6.0, -0.5, -SQRT_2_3),
    (-SQRT3 / 6.0, 0.5, -SQRT_2_3),
    (SQRT3 / 6.0, -0.5, SQRT_2_3),
    (1.0 / SQRT3, 0.0, -SQRT_2_3),            # X1, X3
    (-1.0 / SQRT3, 0.0, SQRT_2_3),
]
ALPHA = [SQRT3 / 2.0, 1.0, SQRT_2_3]

N_PARAMS = 10
PARAM_KEYS = ("b0", "b1", "b2", "b3", "b12", "b13", "b23", "b11", "b22", "b33")
DISPLAY_ORDER = (0, 1, 2, 3, 7, 8, 9, 4, 5, 6)   # b0 b1 b2 b3 b11 b22 b33 b12 b13 b23

NAN = float("nan")


class ModelError(ValueError):
    """Error de datos/modelo con mensaje apto para mostrar al usuario."""


# ==============================================================
# FUNCIONES ESTADÍSTICAS NATIVAS (SIN SCIPY)
# ==============================================================
def normal_ppf(p):
    """Cuantil normal estándar inverso (aproximación Beasley-Springer-Moro)."""
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("normal_ppf requiere probabilidades estrictamente entre 0 y 1.")
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106046337]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810338635, 0.0038405729373609, 0.0003951896511919,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187,
    ]
    flat_p = p.reshape(-1)
    flat_res = np.zeros_like(flat_p)
    for i, val in enumerate(flat_p):
        y = val - 0.5
        if abs(y) < 0.42:
            r = y * y
            flat_res[i] = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0)
        else:
            tail = val if y < 0 else 1.0 - val
            r = math.log(-math.log(tail))
            x = c[-1]
            for c_val in reversed(c[:-1]):
                x = x * r + c_val
            flat_res[i] = -x if y < 0 else x
    return flat_res.reshape(p.shape)


def _beta_continued_fraction(a, b, x, max_iter=300):
    """Fracción continua de Lentz para la beta incompleta regularizada."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def betainc_approx(a, b, x, max_iter=300):
    """Beta incompleta regularizada I_x(a, b)."""
    a, b, x = float(a), float(b), float(x)
    if a <= 0 or b <= 0:
        raise ValueError("Los parámetros a y b deben ser positivos.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(log_bt)
    threshold = (a + 1.0) / (a + b + 2.0)
    if x <= threshold:
        result = bt * _beta_continued_fraction(a, b, x, max_iter) / a
    else:
        result = 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x, max_iter) / b
    return min(1.0, max(0.0, result))


def f_sf(f_val, df1, df2):
    """Cola derecha P(F > f) de la distribución F(df1, df2)."""
    if df1 <= 0 or df2 <= 0 or not np.isfinite(f_val):
        return NAN
    if f_val <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_val)
    return betainc_approx(df2 / 2.0, df1 / 2.0, x)


def f_cdf(f_val, df1, df2):
    """Función de distribución acumulada de F(df1, df2)."""
    sf = f_sf(f_val, df1, df2)
    return NAN if math.isnan(sf) else 1.0 - sf


def t_two_sided_p(t_val, df):
    """p-valor bilateral P(|T| > |t|) de la t de Student con df grados de libertad."""
    if df <= 0 or not np.isfinite(t_val):
        return NAN
    if t_val == 0:
        return 1.0
    x = df / (df + t_val * t_val)
    return betainc_approx(df / 2.0, 0.5, x)


def t_cdf(t_val, df):
    """Función de distribución acumulada de la t de Student."""
    p2 = t_two_sided_p(t_val, df)
    if math.isnan(p2):
        return NAN
    return 1.0 - 0.5 * p2 if t_val > 0 else 0.5 * p2


def t_ppf_975(df):
    """Cuantil t(0.975, df) por inversión numérica (bisección) de la beta incompleta."""
    if int(df) != df or df <= 0:
        raise ValueError("Los grados de libertad deben ser un entero positivo.")
    df = int(df)
    target = 0.05
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = betainc_approx(df / 2.0, 0.5, mid)
        if val < target:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2.0
    return math.sqrt(df * (1.0 - x) / x)


def significance_label(p):
    """Código de significancia estilo R."""
    if p is None or not np.isfinite(p):
        return "n/d"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return "ns"


# ==============================================================
# FORMATO
# ==============================================================
def fmt_num(x, nd=6):
    """Formatea un número tolerando None / NaN / inf."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/d"
    if math.isnan(x):
        return "n/d"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{x:.{nd}f}"


def _wrap(text, first="  ", rest="  ", width=84):
    """Envuelve texto largo (mantiene los paneles monoespaciados legibles)."""
    return textwrap.fill(text, width=width, initial_indent=first, subsequent_indent=rest)


def fmt_p(p):
    """Formatea un p-valor."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/d"
    if not np.isfinite(p):
        return "n/d"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


# ==============================================================
# MODELO: MATRIZ DE DISEÑO Y TRANSFORMACIONES
# ==============================================================
def design_matrix(coded):
    """Matriz X (N x 10) del modelo cuadrático completo en variables codificadas."""
    coded = np.atleast_2d(np.asarray(coded, dtype=float))
    x1, x2, x3 = coded[:, 0], coded[:, 1], coded[:, 2]
    return np.column_stack([np.ones(len(coded)), x1, x2, x3,
                            x1 * x2, x1 * x3, x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2])


def configure_factor_transform(real_values, coded_values):
    """
    Recupera centros y escalas físicas (real = centro + escala * codificado)
    desde el rango físico de cada factor y la geometría canónica (extremos ±ALPHA).
    Devuelve (centros, escalas) como arrays de longitud 3.
    """
    real_values = np.asarray(real_values, dtype=float)
    coded_values = np.asarray(coded_values, dtype=float)
    centers, scales = [], []
    for j in range(3):
        c = coded_values[:, j]
        r = real_values[:, j]
        if len(np.unique(np.round(c, 8))) < 2:
            raise ModelError(f"No hay variación suficiente en x{j+1}_cod para reconstruir la escala.")
        center = (float(np.min(r)) + float(np.max(r))) / 2.0
        scale = (float(np.max(r)) - float(np.min(r))) / (2.0 * ALPHA[j])
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            raise ModelError(f"Transformación de escala inválida para el factor {j+1}.")
        err = float(np.max(np.abs(center + scale * c - r)))
        tolerance = max(0.01, 2e-3 * max(1.0, float(np.ptp(r))))
        if err > tolerance:
            raise ModelError(f"Las columnas físicas y codificadas del factor {j+1} no son "
                             f"compatibles (error máximo {err:.5g}).")
        centers.append(center)
        scales.append(scale)
    return np.asarray(centers, dtype=float), np.asarray(scales, dtype=float)


def factor_levels(real_values, decimals=4):
    """Lista de niveles distintos (ordenados) de cada factor físico."""
    real_values = np.asarray(real_values, dtype=float)
    return [np.unique(np.round(real_values[:, j], decimals)) for j in range(real_values.shape[1])]


def calcular_jacobiano_optimo(betas, H_inv, x_s):
    """
    Jacobiano (3 x 10) del punto estacionario x_s respecto de los 10 coeficientes.
    `betas` se conserva por compatibilidad de firma (el Jacobiano no depende de él);
    `H_inv` es la inversa de la Hessiana completa H = 2B.

    Con H = 2B (Hessiano de la superficie) y b = (b1, b2, b3):
        gradiente nulo:  b + H x_s = 0   =>   x_s = -H^{-1} b
        diferencial:     dx_s = -H^{-1} (db + dH x_s)
    Orden de columnas: [b0, b1, b2, b3, b12, b13, b23, b11, b22, b33].
    """
    J = np.zeros((3, N_PARAMS))
    for j in range(3):
        J[:, 1 + j] = -H_inv[:, j]
    derivadas_H = {
        4: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float),   # b12
        5: np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float),   # b13
        6: np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float),   # b23
        7: np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),   # b11
        8: np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=float),   # b22
        9: np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=float),   # b33
    }
    for idx, dH in derivadas_H.items():
        J[:, idx] = -(H_inv @ dH @ x_s)
    return J


# ==============================================================
# ESTRUCTURAS DE RESULTADOS
# ==============================================================
@dataclass
class CoefRow:
    key: str
    label: str
    estimate: float
    se: float
    t: float
    p: float
    sig: str
    ci_low: float
    ci_high: float


@dataclass
class AnovaRow:
    source: str
    df: Optional[int]
    ss: Optional[float]
    ms: Optional[float]
    f: Optional[float]
    p: Optional[float]
    level: int = 0


@dataclass
class AnovaResult:
    rows: list
    ss_reg: float
    df_reg: int
    ms_reg: float
    f_reg: float
    p_reg: float
    ss_res: float
    df_res: int
    ms_res: float
    ss_pe: float
    df_pe: int
    ms_pe: float
    ss_lof: float
    df_lof: int
    ms_lof: float
    f_lof: float
    p_lof: float
    ss_tot: float
    df_tot: int
    lof_status: str          # ok | sin_replicas | sin_gl_lof | error_puro_nulo
    lof_message: str
    groups: list             # grupos de réplicas (dict)
    perfect_fit: bool


@dataclass
class CanonicalResult:
    B: np.ndarray
    b_lin: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kind: str
    justification: str
    axis_notes: list
    has_stationary: bool
    x_cod: Optional[np.ndarray] = None
    x_real: Optional[np.ndarray] = None
    y_stationary: float = NAN
    radius: float = NAN
    inside_ellipsoid: bool = False
    inside_box: bool = False
    ic_coords: list = field(default_factory=lambda: [(NAN, NAN, NAN)] * 3)   # (lo, hi, se_real)
    ic_mean: tuple = (NAN, NAN)
    ip_pred: tuple = (NAN, NAN)
    delta_mean: float = NAN
    delta_pred: float = NAN
    note: str = ""


@dataclass
class ModelFit:
    n: int
    factor_names: list
    centers: np.ndarray
    scales: np.ndarray
    limits_real: list
    coded: np.ndarray
    real: np.ndarray
    y: np.ndarray
    M: np.ndarray
    betas: np.ndarray
    XtX_inv: np.ndarray
    cov_beta: np.ndarray
    y_hat: np.ndarray
    residuals: np.ndarray
    leverage: np.ndarray
    student_resid: np.ndarray
    rank: int
    condition: float
    r2: float
    r2_adj: float
    s_res: float
    t_crit: float
    coef_rows: list
    anova: AnovaResult
    canonical: CanonicalResult


# ==============================================================
# ANOVA CON DESCOMPOSICIÓN DEL ERROR (LACK OF FIT)
# ==============================================================
def compute_anova(y, coded, y_hat, n_params=N_PARAMS):
    """
    SS_tot = SS_reg + SS_res;  SS_res = SS_pe + SS_lof.
    Error puro: réplicas = corridas con coordenadas codificadas idénticas
    (típicamente el punto central (0,0,0)).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    ybar = float(np.mean(y))
    e = y - y_hat
    ss_tot = float(np.sum((y - ybar) ** 2))
    ss_reg = float(np.sum((y_hat - ybar) ** 2))
    ss_res = float(np.sum(e ** 2))
    df_reg, df_res, df_tot = n_params - 1, n - n_params, n - 1
    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res
    scale_y = max(1.0, float(np.mean(y ** 2)))
    perfect = ms_res <= 1e-24 * scale_y
    if perfect:
        f_reg = p_reg = NAN
    else:
        f_reg = ms_reg / ms_res
        p_reg = f_sf(f_reg, df_reg, df_res)

    # --- Error puro: agrupar todas las combinaciones codificadas repetidas ---
    groups_idx = {}
    for i, row in enumerate(np.asarray(coded, dtype=float)):
        groups_idx.setdefault(tuple(float(v) for v in np.round(row, 8)), []).append(i)
    ss_pe, df_pe, groups = 0.0, 0, []
    for key, idx in groups_idx.items():
        if len(idx) > 1:
            vals = y[idx]
            ss_g = float(np.sum((vals - np.mean(vals)) ** 2))
            ss_pe += ss_g
            df_pe += len(idx) - 1
            groups.append({"coords": key, "n": len(idx), "mean": float(np.mean(vals)), "ss": ss_g,
                           "runs": [i + 1 for i in idx], "is_center": all(abs(c) < 1e-8 for c in key)})
    df_lof = df_res - df_pe
    ss_lof = max(0.0, ss_res - ss_pe)
    ms_pe = ss_pe / df_pe if df_pe > 0 else NAN
    ms_lof = ss_lof / df_lof if df_lof > 0 else NAN
    pe_zero = df_pe > 0 and ss_pe <= 1e-18 * max(1.0, float(np.sum(y ** 2)))

    f_lof = p_lof = NAN
    if df_pe == 0:
        status = "sin_replicas"
        msg = ("No hay réplicas (corridas con coordenadas idénticas): no se puede estimar el "
               "error puro ni probar la falta de ajuste.")
    elif df_lof <= 0:
        status = "sin_gl_lof"
        msg = ("No quedan grados de libertad para la falta de ajuste (df_lof ≤ 0); "
               "la prueba no es computable.")
    elif pe_zero:
        status = "error_puro_nulo"
        msg = ("ERROR PURO NULO (SS_pe ≈ 0): las réplicas coinciden exactamente (datos determinísticos "
               "o sin dispersión de réplica). MS_pe = 0, por lo que F_lof = MS_lof / 0 no está definido "
               "y no se reporta p-valor. La prueba de falta de ajuste no es aplicable.")
    else:
        status = "ok"
        f_lof = ms_lof / ms_pe
        p_lof = f_sf(f_lof, df_lof, df_pe)
        if p_lof > 0.05:
            msg = (f"NO SIGNIFICATIVA (p = {fmt_p(p_lof)} > 0.05): no se detecta falta de ajuste; "
                   "el modelo cuadrático es adecuado frente al error puro.")
        else:
            msg = (f"SIGNIFICATIVA (p = {fmt_p(p_lof)} ≤ 0.05): evidencia de falta de ajuste; "
                   "el modelo cuadrático podría ser insuficiente (considerar términos adicionales o transformar la respuesta).")

    rows = [
        AnovaRow("Regresión (modelo)", df_reg, ss_reg, ms_reg, f_reg, p_reg, 0),
        AnovaRow("Residuo", df_res, ss_res, ms_res, None, None, 0),
        AnovaRow("Falta de ajuste", df_lof, ss_lof, ms_lof, f_lof, p_lof, 1),
        AnovaRow("Error puro", df_pe, ss_pe, ms_pe, None, None, 1),
        AnovaRow("Total", df_tot, ss_tot, None, None, None, 0),
    ]
    return AnovaResult(rows, ss_reg, df_reg, ms_reg, f_reg, p_reg, ss_res, df_res, ms_res,
                       ss_pe, df_pe, ms_pe, ss_lof, df_lof, ms_lof, f_lof, p_lof, ss_tot, df_tot,
                       status, msg, groups, perfect)


# ==============================================================
# ANÁLISIS CANÓNICO Y PUNTO ESTACIONARIO
# ==============================================================
def build_B_matrix(betas):
    """Matriz simétrica B de coeficientes de segundo orden."""
    b12, b13, b23 = betas[4], betas[5], betas[6]
    b11, b22, b33 = betas[7], betas[8], betas[9]
    return np.array([[b11, 0.5 * b12, 0.5 * b13],
                     [0.5 * b12, b22, 0.5 * b23],
                     [0.5 * b13, 0.5 * b23, b33]], dtype=float)


def canonical_analysis(betas, cov_beta, s_res, t_crit, centers, scales, limits_real, factor_names):
    """Análisis canónico: eigh(B), clasificación geométrica y punto estacionario con IC (Delta)."""
    B = build_B_matrix(betas)
    b_lin = np.asarray(betas[1:4], dtype=float)
    evals, evecs = np.linalg.eigh(B)
    for j in range(3):                        # signo reproducible: mayor componente positiva
        if evecs[np.argmax(np.abs(evecs[:, j])), j] < 0:
            evecs[:, j] *= -1.0
    # Umbral relativo a los autovalores y absoluto respecto de la escala de β (respuesta casi plana)
    tol = max(1e-9 * float(np.max(np.abs(evals))),
              100.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(betas, ord=np.inf))), 1e-300)
    n_pos, n_neg = int(np.sum(evals > tol)), int(np.sum(evals < -tol))
    n_zero = 3 - n_pos - n_neg

    axis_notes = []
    for i in range(3):
        v = evecs[:, i]
        dom = int(np.argmax(np.abs(v)))
        share = 100.0 * v[dom] ** 2
        if evals[i] > tol:
            sentido = "Ŷ AUMENTA (ascenso)"
        elif evals[i] < -tol:
            sentido = "Ŷ DISMINUYE (descenso)"
        else:
            sentido = "Ŷ casi constante (curvatura ≈ 0)"
        axis_notes.append(f"Eje w{i+1}: λ{i+1} = {evals[i]:+.5f} → al alejarse de x0 sobre v{i+1}, {sentido}; "
                          f"dominado por {factor_names[dom]} ({share:.0f}% de la norma)")

    if n_zero > 0:
        kind = "SUPERFICIE DEGENERADA (cresta/valle estacionario)"
        just = (f"Hay {n_zero} autovalor(es) ≈ 0 (|λ| ≤ {tol:.2e}): B es singular, no existe un punto estacionario "
                "único. La respuesta es casi constante a lo largo de los autovectores asociados (cresta o valle).")
    elif n_neg == 3:
        kind = "MÁXIMO LOCAL ESTRICTO"
        just = ("Los tres autovalores de B son negativos (B definida negativa): Ŷ decrece en toda dirección al "
                "alejarse del punto estacionario ⇒ máximo local estricto.")
    elif n_pos == 3:
        kind = "MÍNIMO LOCAL ESTRICTO"
        just = ("Los tres autovalores de B son positivos (B definida positiva): Ŷ crece en toda dirección al "
                "alejarse del punto estacionario ⇒ mínimo local estricto.")
    else:
        kind = "PUNTO DE SILLA (SADDLE POINT)"
        asc = [f"v{i+1}" for i in range(3) if evals[i] > tol]
        desc = [f"v{i+1}" for i in range(3) if evals[i] < -tol]
        just = (f"Autovalores de signos mixtos ({n_pos} positivo(s), {n_neg} negativo(s)): B es indefinida ⇒ punto de "
                f"silla. Ascenso a lo largo de {', '.join(asc)} (λ > 0) y descenso a lo largo de {', '.join(desc)} (λ < 0).")

    res = CanonicalResult(B=B, b_lin=b_lin, eigenvalues=evals, eigenvectors=evecs, kind=kind,
                          justification=just, axis_notes=axis_notes, has_stationary=False)
    if n_zero > 0:
        res.note = "Sin punto estacionario único: no se calculan coordenadas óptimas."
        return res
    try:
        x_s = -0.5 * np.linalg.solve(B, b_lin)             # x_s = -1/2 B^{-1} b
        H_inv = np.linalg.inv(2.0 * B)
    except np.linalg.LinAlgError:
        res.note = "B singular: no se pudo resolver el punto estacionario."
        return res
    if not np.all(np.isfinite(x_s)):
        res.note = "El punto estacionario no es finito."
        return res

    res.has_stationary = True
    res.x_cod = x_s
    res.x_real = centers + scales * x_s
    m_s = design_matrix(x_s.reshape(1, 3))[0]
    res.y_stationary = float(m_s @ betas)
    res.radius = float(np.linalg.norm(x_s))
    res.inside_ellipsoid = res.radius < 1.0                  # dominio Doehlert: ||x|| <= 1
    res.inside_box = all(limits_real[i][0] <= res.x_real[i] <= limits_real[i][1] for i in range(3))

    var_mean = max(0.0, float(m_s @ cov_beta @ m_s))
    se_mean = math.sqrt(var_mean)
    res.delta_mean = t_crit * se_mean
    res.ic_mean = (res.y_stationary - res.delta_mean, res.y_stationary + res.delta_mean)
    res.delta_pred = t_crit * math.sqrt(s_res ** 2 + var_mean)
    res.ip_pred = (res.y_stationary - res.delta_pred, res.y_stationary + res.delta_pred)

    J = calcular_jacobiano_optimo(betas, H_inv, x_s)         # Método Delta
    cov_x = J @ cov_beta @ J.T
    ic = []
    for i in range(3):
        se_real = abs(scales[i]) * math.sqrt(max(0.0, float(cov_x[i, i])))
        ic.append((res.x_real[i] - t_crit * se_real, res.x_real[i] + t_crit * se_real, se_real))
    res.ic_coords = ic
    return res


# ==============================================================
# AJUSTE COMPLETO
# ==============================================================
def fit_quadratic_model(coded, y, centers, scales, limits_real=None, factor_names=None, real=None):
    """Ajusta el modelo cuadrático completo y calcula toda la inferencia estadística."""
    coded = np.asarray(coded, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    centers = np.asarray(centers, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if coded.ndim != 2 or coded.shape[1] != 3 or coded.shape[0] != y.shape[0]:
        raise ModelError("La matriz codificada debe tener N filas y 3 columnas, igual que la respuesta.")
    if not (np.all(np.isfinite(coded)) and np.all(np.isfinite(y))):
        raise ModelError("Los datos contienen valores vacíos, NaN o infinitos.")
    n = len(y)
    if n < N_PARAMS + 1:
        raise ModelError("Se requieren más corridas que parámetros (10) para estimar el error residual.")
    names = list(factor_names) if factor_names else ["X1", "X2", "X3"]
    real = (centers + coded * scales) if real is None else np.asarray(real, dtype=float)
    if limits_real is None:
        limits_real = [(float(real[:, j].min()), float(real[:, j].max())) for j in range(3)]

    M = design_matrix(coded)
    rank = int(np.linalg.matrix_rank(M))
    if rank < N_PARAMS:
        raise ModelError(f"Diseño no identificable (matriz singular): rango {rank}/10. "
                         "No se puede ajustar un modelo cuadrático completo.")
    try:
        M_pinv = np.linalg.pinv(M, rcond=1e-12)
    except np.linalg.LinAlgError:
        raise ModelError("No se pudo factorizar la matriz de diseño (singular).")
    XtX_inv = M_pinv @ M_pinv.T                       # (M'M)^-1 sin formar M'M
    betas = M_pinv @ y
    y_hat = M @ betas
    resid = y - y_hat
    df_res = n - N_PARAMS
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ms_res = ss_res / df_res
    s_res = math.sqrt(ms_res)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NAN
    r2_adj = 1.0 - ms_res / (ss_tot / (n - 1)) if ss_tot > 0 else NAN
    cov_beta = ms_res * XtX_inv
    t_crit = t_ppf_975(df_res)
    perfect = ms_res <= 1e-24 * max(1.0, float(np.mean(y ** 2)))

    # Leverage (diagonal de la hat matrix) y residuo estudentizado interno
    leverage = np.clip(np.einsum("ij,ji->i", M, M_pinv), 0.0, 1.0)     # diag(M M⁺) = diag(hat)
    student = np.full(n, NAN)
    if not perfect:
        for i in range(n):
            denom = ms_res * (1.0 - leverage[i])
            if denom > 1e-14:
                student[i] = resid[i] / math.sqrt(denom)

    # Tabla de coeficientes con inferencia
    labels = {
        "b0": "b0 (Intercepto)",
        "b1": f"b1 ({names[0]})", "b2": f"b2 ({names[1]})", "b3": f"b3 ({names[2]})",
        "b11": f"b11 ({names[0]}²)", "b22": f"b22 ({names[1]}²)", "b33": f"b33 ({names[2]}²)",
        "b12": f"b12 ({names[0]}·{names[1]})", "b13": f"b13 ({names[0]}·{names[2]})",
        "b23": f"b23 ({names[1]}·{names[2]})",
    }
    coef_rows = []
    for idx in DISPLAY_ORDER:
        key = PARAM_KEYS[idx]
        se = math.sqrt(max(0.0, float(cov_beta[idx, idx])))
        if perfect or se <= 0:
            t_i = p_i = NAN
            ci = (float(betas[idx]), float(betas[idx])) if perfect else (NAN, NAN)
        else:
            t_i = float(betas[idx]) / se
            p_i = t_two_sided_p(t_i, df_res)
            ci = (float(betas[idx]) - t_crit * se, float(betas[idx]) + t_crit * se)
        coef_rows.append(CoefRow(key, labels[key], float(betas[idx]), se, t_i, p_i,
                                 significance_label(p_i), ci[0], ci[1]))

    anova = compute_anova(y, coded, y_hat)
    canon = canonical_analysis(betas, cov_beta, s_res, t_crit, centers, scales, limits_real, names)
    return ModelFit(n=n, factor_names=names, centers=centers, scales=scales, limits_real=limits_real,
                    coded=coded, real=real, y=y, M=M, betas=betas, XtX_inv=XtX_inv, cov_beta=cov_beta,
                    y_hat=y_hat, residuals=resid, leverage=leverage, student_resid=student,
                    rank=rank, condition=float(np.linalg.cond(M)), r2=r2, r2_adj=r2_adj, s_res=s_res,
                    t_crit=t_crit, coef_rows=coef_rows, anova=anova, canonical=canon)


def qq_points(values):
    """Cuantiles teóricos normales (posiciones de Blom) vs. valores ordenados. Ignora NaN."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    idx = np.where(ok)[0]
    order = idx[np.argsort(values[idx])]
    m = len(order)
    if m == 0:
        return np.array([]), np.array([]), order
    prob = (np.arange(1, m + 1) - 0.375) / (m + 0.25)
    return normal_ppf(prob), values[order], order


# ==============================================================
# TABLAS PARA GUI / EXPORTACIÓN
# ==============================================================
def residual_table(fit):
    """(encabezados, filas) del diagnóstico corrida a corrida."""
    header = ["Corrida", "x1_cod", "x2_cod", "x3_cod", *fit.factor_names,
              "Y_obs", "Y_pred", "Residuo", "Leverage", "Res_estud"]
    rows = []
    for i in range(fit.n):
        rows.append([i + 1, *[float(v) for v in fit.coded[i]], *[float(v) for v in fit.real[i]],
                     float(fit.y[i]), float(fit.y_hat[i]), float(fit.residuals[i]),
                     float(fit.leverage[i]), float(fit.student_resid[i])])
    return header, rows


def coef_table(fit):
    header = ["Término", "Coeficiente", "Error estándar", "t", "p-valor", "Sig.", "IC95% inf", "IC95% sup"]
    rows = [[r.label, r.estimate, r.se, r.t, r.p, r.sig, r.ci_low, r.ci_high] for r in fit.coef_rows]
    return header, rows


def anova_table(fit):
    header = ["Fuente", "GL", "SC", "CM", "F", "p-valor"]
    rows = []
    for r in fit.anova.rows:
        rows.append([("    " if r.level else "") + r.source, r.df, r.ss, r.ms, r.f, r.p])
    return header, rows


def summary_pairs(fit):
    a = fit.anova
    return [("N corridas", fit.n), ("Parámetros", N_PARAMS), ("GL residuales", a.df_res),
            ("Rango de X", fit.rank), ("Número de condición", fit.condition),
            ("R²", fit.r2), ("R² ajustado", fit.r2_adj), ("Error estándar residual s", fit.s_res),
            ("F de regresión", a.f_reg), ("p-valor (regresión)", a.p_reg),
            ("t crítico (0.975, df_res)", fit.t_crit)]


# ==============================================================
# TEXTO FORMATEADO (panel monoespaciado)
# ==============================================================
def format_summary_text(fit):
    a = fit.anova
    return "\n".join([
        "=== REGRESIÓN POLINOMIAL (2º ORDEN) ===",
        f"Nº Corridas (N): {fit.n} | Grados de Libertad: {a.df_res} | Parámetros: 10",
        f"Rango de matriz: {fit.rank}/10 | Condición (codificada): {fit.condition:.5g}",
        f"R²: {fmt_num(fit.r2)}  |  R² Ajustado: {fmt_num(fit.r2_adj)}",
        f"Error Estándar Residual (s): {fmt_num(fit.s_res)}",
        f"F regresión (9, {a.df_res}) = {fmt_num(a.f_reg, 4)} | p = {fmt_p(a.p_reg)}",
        f"Valor Crítico t(0.975, df={a.df_res}): {fmt_num(fit.t_crit)}",
    ])


def format_canonical_text(fit):
    c = fit.canonical
    names = fit.factor_names
    L = ["=== DIAGNÓSTICO CANÓNICO Y GEOMETRÍA ===", "Matriz B (simétrica, 2º orden):"]
    for i in range(3):
        L.append("  [" + "  ".join(f"{c.B[i, j]:+10.5f}" for j in range(3)) + " ]")
    L.append("Autovalores (np.linalg.eigh, ascendentes):")
    L.append("  " + "   ".join(f"λ{i+1}={c.eigenvalues[i]:+.5f}" for i in range(3)))
    L.append("Autovectores (columnas = ejes canónicos, vars. codificadas):")
    L.append("  " + " " * 10 + "".join(f"{'v' + str(j+1):>10}" for j in range(3)))
    for i in range(3):
        L.append(f"  {names[i][:9]:<10}" + "".join(f"{c.eigenvectors[i, j]:>10.5f}" for j in range(3)))
    L += ["", f"Naturaleza: {c.kind}", _wrap(c.justification)]
    L += [_wrap(n, first="  • ", rest="    ") for n in c.axis_notes]
    L += ["", "PUNTO ESTACIONARIO x0 = -½ B⁻¹ b:"]
    if not c.has_stationary:
        L.append(f"  {c.note}")
        return "\n".join(L)
    L.append("  Coordenadas codificadas: (" + ", ".join(f"{v:.5f}" for v in c.x_cod) + ")")
    L.append(f"  Respuesta ŷ: {fmt_num(c.y_stationary)}   (Ŷ en x0)")
    L.append(f"  IC 95% (Media): [{fmt_num(c.ic_mean[0], 4)}, {fmt_num(c.ic_mean[1], 4)}] (±{fmt_num(c.delta_mean, 4)})")
    L.append(f"  IP 95% (Indiv.): [{fmt_num(c.ip_pred[0], 4)}, {fmt_num(c.ip_pred[1], 4)}] (±{fmt_num(c.delta_pred, 4)})")
    L += ["", "ÓPTIMO EN UNIDADES FÍSICAS (IC 95% por Método Delta):"]
    for i, name in enumerate(names):
        lo, hi, _ = c.ic_coords[i]
        L.append(f"  {name}: {fmt_num(c.x_real[i])}  (IC [{fmt_num(lo)}, {fmt_num(hi)}])")
    L += ["", "EVALUACIÓN DEL DOMINIO EXPERIMENTAL:",
          f"  Radio codificado ||x0|| = {c.radius:.5f}  (dominio Doehlert: ||x|| ≤ 1)",
          f"  ¿Estrictamente dentro del hiperelipsoide (||x0|| < 1)?: {'SÍ' if c.inside_ellipsoid else 'NO'}",
          f"  ¿Dentro del rango físico de cada factor?: {'SÍ' if c.inside_box else 'NO'}"]
    if not c.inside_ellipsoid:
        L.append("  Nota: el punto estacionario queda fuera del dominio; es una extrapolación y no")
        L.append("  debe interpretarse como óptimo operativo sin experimentos de confirmación.")
    if "SILLA" in c.kind:
        L.append("  Nota: en un punto de silla Ŷ(x0) no es máximo ni mínimo; usar análisis de cresta.")
    return "\n".join(L)


def format_model_report(fit):
    """Texto del panel 'Resumen Estadístico y Óptimo'."""
    return format_summary_text(fit) + "\n\n" + format_canonical_text(fit)


def format_anova_text(fit):
    a = fit.anova
    L = ["=== TABLA ANOVA (con Falta de Ajuste) ===",
         f"{'Fuente':<22}{'GL':>4}{'SC':>15}{'CM':>15}{'F':>11}{'p-valor':>10}"]
    for r in a.rows:
        L.append(f"{('  ' if r.level else '') + r.source:<22}{'' if r.df is None else r.df:>4}"
                 f"{fmt_num(r.ss, 5) if r.ss is not None else '':>15}"
                 f"{fmt_num(r.ms, 5) if r.ms is not None else '':>15}"
                 f"{fmt_num(r.f, 4) if r.f is not None else '':>11}"
                 f"{fmt_p(r.p) if r.p is not None else '':>10}")
    if a.perfect_fit:
        L.append("(Ajuste perfecto: MS_res ≈ 0, F y t no definidos.)")
    return "\n".join(L)


def format_lof_text(fit):
    """Descomposición del error residual y prueba de falta de ajuste (etiquetas SS_PE, df_LOF… como en main__4_)."""
    a = fit.anova
    L = ["=== DESCOMPOSICIÓN ANOVA DE RESIDUOS ===",
         f"N={fit.n} | p=10 | rango={fit.rank}",
         f"SS_Residual Total: {fmt_num(a.ss_res, 8)} (df={a.df_res})", "",
         "ERROR PURO (Pure Error):",
         f"  SS_PE: {fmt_num(a.ss_pe, 8)}", f"  df_PE: {a.df_pe}", f"  MS_PE: {fmt_num(a.ms_pe, 8)}"]
    for g in a.groups:
        etiqueta = "centro" if g["is_center"] else "punto repetido"
        L.append(f"    · {etiqueta} {tuple(round(c, 4) for c in g['coords'])}: n={g['n']}, "
                 f"corridas {g['runs']}, media={g['mean']:.5f}")
    L += ["", "FALTA DE AJUSTE (Lack-of-Fit):",
          f"  SS_LOF: {fmt_num(a.ss_lof, 8)}", f"  df_LOF: {a.df_lof}", f"  MS_LOF: {fmt_num(a.ms_lof, 8)}", "",
          "PRUEBA LOF (H0: el modelo no presenta falta de ajuste):"]
    if a.lof_status == "ok":
        L += [f"  Estadístico F: {fmt_num(a.f_lof, 8)}   (F = MS_LOF / MS_PE; gl {a.df_lof}, {a.df_pe})",
              f"  p-valor: {fmt_num(a.p_lof, 8)}"]
        L.append("  RESULTADO: " + ("NO SIGNIFICATIVO (p > 0.05): no se detecta falta de ajuste." if a.p_lof > 0.05
                                     else "SIGNIFICATIVO (p <= 0.05): evidencia de falta de ajuste."))
    else:
        L.append("  RESULTADO: no computable.")
    L.append(_wrap(a.lof_message))
    return "\n".join(L)


def format_coef_text(fit):
    L = ["=== COEFICIENTES (variables codificadas) ===",
         f"{'Término':<24}{'Coef.':>12}{'EE':>12}{'t':>10}{'p-valor':>10}{'Sig':>5}{'IC95% inf':>12}{'IC95% sup':>12}"]
    for r in fit.coef_rows:
        L.append(f"{r.label[:23]:<24}{fmt_num(r.estimate, 5):>12}{fmt_num(r.se, 5):>12}{fmt_num(r.t, 3):>10}"
                 f"{fmt_p(r.p):>10}{r.sig:>5}{fmt_num(r.ci_low, 5):>12}{fmt_num(r.ci_high, 5):>12}")
    L.append("Sig.: *** p<0.001 | ** p<0.01 | * p<0.05 | . p<0.1 | ns p≥0.1")
    return "\n".join(L)


# ==============================================================
# REPORTE EJECUTIVO (Markdown / texto)
# ==============================================================
def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown_report(fit, timestamp=None):
    ts = timestamp or datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    a, c = fit.anova, fit.canonical
    levels = factor_levels(fit.real)
    L = ["# Reporte Ejecutivo — Diseño Experimental Doehlert (3 factores)", "",
         f"**Generado:** {ts}  ", f"**Corridas:** {fit.n}  ", "**Modelo:** polinomio cuadrático completo (10 parámetros)", "",
         "## 1. Factores y niveles", ""]
    rows = []
    for j, name in enumerate(fit.factor_names):
        lv = ", ".join(f"{v:g}" for v in levels[j])
        rows.append([name, f"{fit.limits_real[j][0]:g}", f"{fit.limits_real[j][1]:g}", len(levels[j]), lv])
    L += [_md_table(["Factor", "Mínimo", "Máximo", "Nº niveles", "Niveles"], rows), "",
          "## 2. Resumen del modelo", ""]
    L += [f"- N = {fit.n}; GL residual = {a.df_res}; rango(X) = {fit.rank}/10; condición = {fit.condition:.5g}",
          f"- R² = {fmt_num(fit.r2)}; R² ajustado = {fmt_num(fit.r2_adj)}; s = {fmt_num(fit.s_res)}",
          f"- F de regresión (9, {a.df_res}) = {fmt_num(a.f_reg, 4)}; p = {fmt_p(a.p_reg)}", "",
          "## 3. Coeficientes con inferencia estadística", ""]
    rows = [[r.label, fmt_num(r.estimate, 5), fmt_num(r.se, 5), fmt_num(r.t, 3), fmt_p(r.p), r.sig,
             fmt_num(r.ci_low, 5), fmt_num(r.ci_high, 5)] for r in fit.coef_rows]
    L += [_md_table(["Término", "Coef.", "EE", "t", "p-valor", "Sig.", "IC95% inf", "IC95% sup"], rows),
          "", "Significancia: `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.1, `ns` p≥0.1. "
          f"IC 95% con t(0.975, {a.df_res}) = {fmt_num(fit.t_crit, 5)}.", "",
          "## 4. ANOVA con descomposición del error (Lack of Fit)", ""]
    rows = [[("&nbsp;&nbsp;" if r.level else "") + r.source, "" if r.df is None else r.df,
             "" if r.ss is None else fmt_num(r.ss, 5), "" if r.ms is None else fmt_num(r.ms, 5),
             "" if r.f is None else fmt_num(r.f, 4), "" if r.p is None else fmt_p(r.p)] for r in a.rows]
    L += [_md_table(["Fuente", "GL", "SC", "CM", "F", "p-valor"], rows), "",
          f"**Prueba de falta de ajuste:** {a.lof_message}", ""]
    for g in a.groups:
        L.append(f"- Réplicas {'en el centro' if g['is_center'] else 'en el punto'} "
                 f"{tuple(round(v, 4) for v in g['coords'])}: n = {g['n']}, corridas {g['runs']}.")
    L += ["", "## 5. Diagnóstico canónico y punto estacionario", "", "```", format_canonical_text(fit), "```", ""]
    if c.has_stationary:
        rows = [[fit.factor_names[i], fmt_num(c.x_cod[i], 5), fmt_num(c.x_real[i], 5),
                 fmt_num(c.ic_coords[i][0], 5), fmt_num(c.ic_coords[i][1], 5)] for i in range(3)]
        L += ["### Óptimo físico (IC 95%, Método Delta)", "",
              _md_table(["Factor", "Codificado", "Físico", "IC95% inf", "IC95% sup"], rows), "",
              f"- Ŷ(x0) = {fmt_num(c.y_stationary)}; IC 95% media = [{fmt_num(c.ic_mean[0], 4)}, {fmt_num(c.ic_mean[1], 4)}]",
              f"- Dentro del hiperelipsoide (||x0|| = {c.radius:.4f} < 1): **{'SÍ' if c.inside_ellipsoid else 'NO'}**; "
              f"dentro de rangos físicos: **{'SÍ' if c.inside_box else 'NO'}**", ""]
    L += ["---", "*Reporte generado por la Suite Doehlert (motor estadístico nativo, sin SciPy).*"]
    return "\n".join(L)


# ==============================================================
# EXPORTACIÓN CSV / XLSX
# ==============================================================
def openpyxl_available():
    return importlib.util.find_spec("openpyxl") is not None


def _cell(v):
    """Convierte a tipo seguro para CSV/Excel (NaN -> 'n/d')."""
    if v is None:
        return ""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v):
            return "n/d"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return v


def write_csv_bundle(path, fit):
    """
    Guarda la matriz completa de diagnóstico en `path` (UTF-8, comas, punto decimal) y,
    junto a ella, `<base>_coeficientes.csv` y `<base>_anova.csv`. Devuelve la lista de rutas.
    """
    base, _ = os.path.splitext(path)
    written = []

    def dump(p, header, rows):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(header)
            for r in rows:
                w.writerow([_cell(v) for v in r])
        written.append(p)

    dump(path, *residual_table(fit))
    dump(base + "_coeficientes.csv", *coef_table(fit))
    dump(base + "_anova.csv", *anova_table(fit))
    return written


def write_xlsx(path, fit):
    """Libro .xlsx con hojas Resumen_Modelo, ANOVA_LackOfFit y Diagnostico_Residuos (requiere openpyxl)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1F4E78")
    a, c = fit.anova, fit.canonical

    def put_table(ws, r0, header, rows):
        for j, h in enumerate(header, start=1):
            cell = ws.cell(row=r0, column=j, value=h)
            cell.font, cell.fill = head_font, head_fill
            cell.alignment = Alignment(horizontal="center")
        for i, row in enumerate(rows, start=1):
            for j, v in enumerate(row, start=1):
                ws.cell(row=r0 + i, column=j, value=_cell(v))
        return r0 + len(rows) + 2

    def title(ws, r, text):
        ws.cell(row=r, column=1, value=text).font = Font(bold=True, size=12)
        return r + 1

    def autosize(ws):
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)

    wb = Workbook()
    ws = wb.active
    ws.title = "Resumen_Modelo"
    r = title(ws, 1, "Resumen del modelo Doehlert")
    r = put_table(ws, r, ["Parámetro", "Valor"],
                  [("Generado", datetime.datetime.now().astimezone().isoformat(timespec="seconds"))] + summary_pairs(fit))
    r = title(ws, r, "Coeficientes (variables codificadas)")
    r = put_table(ws, r, *coef_table(fit))
    r = title(ws, r, "Análisis canónico")
    r = put_table(ws, r, ["Eje", "Autovalor", *[f"v ({n})" for n in fit.factor_names]],
                  [[f"λ{i+1}", float(c.eigenvalues[i]), *[float(v) for v in c.eigenvectors[:, i]]] for i in range(3)])
    pairs = [("Naturaleza", c.kind), ("Justificación", c.justification)]
    if c.has_stationary:
        pairs += [(f"x0 codificado {fit.factor_names[i]}", float(c.x_cod[i])) for i in range(3)]
        pairs += [(f"x0 físico {fit.factor_names[i]}", float(c.x_real[i])) for i in range(3)]
        pairs += [(f"IC95% Delta {fit.factor_names[i]}", f"[{c.ic_coords[i][0]:.6g}, {c.ic_coords[i][1]:.6g}]") for i in range(3)]
        pairs += [("Ŷ(x0)", c.y_stationary), ("IC95% media inf", c.ic_mean[0]), ("IC95% media sup", c.ic_mean[1]),
                  ("Radio ||x0||", c.radius), ("Dentro del hiperelipsoide", "SÍ" if c.inside_ellipsoid else "NO"),
                  ("Dentro de rangos físicos", "SÍ" if c.inside_box else "NO")]
    else:
        pairs.append(("Punto estacionario", c.note))
    put_table(ws, r, ["Concepto", "Valor"], pairs)
    autosize(ws)
    ws.column_dimensions["A"].width = 34

    ws2 = wb.create_sheet("ANOVA_LackOfFit")
    r = title(ws2, 1, "Tabla ANOVA con Falta de Ajuste")
    r = put_table(ws2, r, *anova_table(fit))
    ws2.cell(row=r, column=1, value="Prueba LOF:").font = Font(bold=True)
    ws2.cell(row=r, column=2, value=a.lof_message)
    r += 2
    r = title(ws2, r, "Grupos de réplicas (error puro)")
    put_table(ws2, r, ["Coordenadas codificadas", "n", "Media Y", "SS", "Corridas"],
              [[str(tuple(round(v, 4) for v in g["coords"])), g["n"], g["mean"], g["ss"], str(g["runs"])]
               for g in a.groups])
    autosize(ws2)
    ws2.column_dimensions["B"].width = 14

    ws3 = wb.create_sheet("Diagnostico_Residuos")
    put_table(ws3, 1, *residual_table(fit))
    autosize(ws3)
    wb.save(path)


# ==============================================================
# LECTURA DE CSV DE DISEÑO (compartida por la GUI y la consola)
# ==============================================================
def read_design_csv(path):
    """
    Lee un CSV con columnas: Corrida, Factor1, Factor2, Factor3, Respuesta_Y [, x1_cod, x2_cod, x3_cod].
    Acepta directiva 'sep=', separadores , ; tabulador y coma decimal. Si faltan las columnas
    codificadas, las reconstruye con la geometría canónica (extremos ±ALPHA).
    Devuelve (encabezados, real[N,3], codificado[N,3], y[N]).
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    delim = None
    if lines and lines[0].strip().lower().startswith("sep="):
        delim = lines.pop(0).strip()[4:5] or ","
    text = "\n".join(lines)
    if delim is None:
        try:
            delim = csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
        except csv.Error:
            delim = ","
    rows = list(csv.reader(text.splitlines(), delimiter=delim))
    if len(rows) < 14:
        raise ModelError("El CSV debe contener encabezado y al menos 13 corridas.")
    headers = [h.strip() for h in rows[0]]
    if len(headers) < 5:
        raise ModelError("Faltan columnas: se esperan Corrida, 3 factores físicos y Respuesta_Y.")
    norm = [h.lower() for h in headers]

    def find(cands):
        return next((norm.index(c) for c in cands if c in norm), None)

    y_idx = find(["respuesta_y", "respuesta", "y", "response", "resultado"])
    if y_idx is None:
        y_idx = 4
    coded_idx = [find([f"x{i}_cod", f"x{i} cod", f"x{i}_coded"]) for i in (1, 2, 3)]
    have_coded = all(i is not None for i in coded_idx)
    X, Y, C = [], [], []
    for line_no, row in enumerate(rows[1:], start=2):
        if not row or all(not c.strip() for c in row):
            continue

        def num(i, label):
            try:
                return float(row[i].strip().replace(",", "."))
            except (ValueError, IndexError):
                raise ModelError(f"Fila {line_no}: valor inválido o vacío en {label}.")

        Y.append(num(y_idx, "Respuesta_Y"))
        X.append([num(i, f"factor {headers[i]}") for i in (1, 2, 3)])
        if have_coded:
            C.append([num(i, f"x{k+1}_cod") for k, i in enumerate(coded_idx)])
    if len(Y) < 13:
        raise ModelError("Se necesitan al menos 13 corridas válidas.")
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if have_coded:
        C = np.asarray(C, dtype=float)
    else:
        cols = []
        for j in range(3):
            lo, hi = float(X[:, j].min()), float(X[:, j].max())
            if hi - lo <= 0:
                raise ModelError(f"El factor {headers[1 + j]} no varía: no se puede reconstruir la codificación.")
            cols.append((X[:, j] - (lo + hi) / 2.0) / ((hi - lo) / 2.0 / ALPHA[j]))
        C = np.column_stack(cols)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(C)) and np.all(np.isfinite(Y))):
        raise ModelError("El CSV contiene NaN o valores infinitos.")
    return headers, X, C, Y


# ==============================================================
# INTERFAZ DE LÍNEA DE COMANDOS (sin ventana)
# ==============================================================
def build_cli_parser():
    p = argparse.ArgumentParser(
        prog="doehlert_engine.py",
        description=("Análisis Doehlert de 3 factores por consola: ajusta el modelo cuadrático completo y "
                     "muestra ANOVA con Falta de Ajuste, coeficientes con inferencia, diagnóstico canónico "
                     "y punto estacionario. No requiere ventana, Tkinter ni SciPy."),
        epilog=("Formato del CSV: Corrida, Factor1, Factor2, Factor3, Respuesta_Y [, x1_cod, x2_cod, x3_cod]. "
                "Es el mismo archivo que exporta la GUI (main.py). Códigos de salida: 0 = éxito, 2 = datos inválidos."))
    p.add_argument("csv", metavar="ARCHIVO.csv", help="CSV con las 13 o más corridas y la respuesta")
    p.add_argument("--reporte", metavar="RUTA", help="guarda el reporte ejecutivo (.md o .txt)")
    p.add_argument("--tablas", metavar="RUTA",
                   help="guarda las tablas: .xlsx (requiere openpyxl) o .csv (genera además _coeficientes.csv y _anova.csv)")
    p.add_argument("-q", "--silencioso", action="store_true", help="no imprime el análisis en pantalla")
    return p


def cli_main(argv=None):
    args = build_cli_parser().parse_args(argv)
    try:
        headers, real, coded, y = read_design_csv(args.csv)
        centers, scales = configure_factor_transform(real, coded)
        fit = fit_quadratic_model(coded, y, centers, scales, factor_names=headers[1:4], real=real)
    except (ModelError, OSError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    if not args.silencioso:
        print("\n\n".join([format_summary_text(fit), format_anova_text(fit), format_lof_text(fit),
                            format_coef_text(fit), format_canonical_text(fit)]))
    try:
        if args.reporte:
            with open(args.reporte, "w", encoding="utf-8") as f:
                f.write(build_markdown_report(fit))
            print(f"Reporte guardado en: {args.reporte}", file=sys.stderr)
        if args.tablas:
            ruta = args.tablas
            if ruta.lower().endswith(".xlsx") and not openpyxl_available():
                ruta = os.path.splitext(ruta)[0] + ".csv"
                print("openpyxl no está instalado: se guarda en CSV en su lugar.", file=sys.stderr)
            escritos = [None]
            if ruta.lower().endswith(".xlsx"):
                write_xlsx(ruta, fit)
                escritos = [ruta]
            else:
                escritos = write_csv_bundle(ruta, fit)
            print("Tablas guardadas en: " + ", ".join(escritos), file=sys.stderr)
    except OSError as err:
        print(f"Error al escribir archivos: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
