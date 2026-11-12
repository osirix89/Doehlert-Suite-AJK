# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""Comparación transparente de un ajuste con coeficientes/métricas publicados.
No reconstruye ni inventa respuestas experimentales ausentes.
"""
from __future__ import annotations
import numpy as np
from doehlert_engine import PARAM_KEYS

def compare_published(fit, published_coefficients, published_metrics=None, atol=1e-6, rtol=1e-4):
    """Devuelve diferencias por coeficiente y métrica; ausencias quedan como NOT_AVAILABLE.

    published_coefficients: dict con claves PARAM_KEYS o vector de longitud 10.
    published_metrics: dict opcional, p.ej. {'r2': .85, 'r2_adj': .81}.
    """
    if isinstance(published_coefficients, dict):
        missing=[k for k in PARAM_KEYS if k not in published_coefficients]
        if missing: raise ValueError('Faltan coeficientes publicados: '+', '.join(missing))
        pub=np.asarray([published_coefficients[k] for k in PARAM_KEYS],float)
    else: pub=np.asarray(published_coefficients,dtype=float)
    if pub.shape!=(10,) or not np.all(np.isfinite(pub)):
        raise ValueError('Se requieren diez coeficientes publicados finitos en el orden PARAM_KEYS.')
    coef=[]
    for k,a,b in zip(PARAM_KEYS,pub,fit.betas):
        diff=float(b-a); ok=bool(np.isclose(b,a,atol=atol,rtol=rtol))
        coef.append({'parameter':k,'published':float(a),'calculated':float(b),'difference':diff,'status':'MATCH' if ok else 'DIFF'})
    metrics={}
    actual={'r2':fit.r2,'r2_adj':fit.r2_adj,'s_res':fit.s_res,
            'f_model':fit.anova.f_reg,'p_model':fit.anova.p_reg,
            'f_lof':fit.anova.f_lof,'p_lof':fit.anova.p_lof}
    for key,val in actual.items():
        if published_metrics is None or key not in published_metrics or published_metrics[key] is None:
            metrics[key]={'published':None,'calculated':None if not np.isfinite(val) else float(val),'status':'NOT_AVAILABLE'}
        else:
            p=float(published_metrics[key]); ok=bool(np.isclose(val,p,atol=atol,rtol=rtol))
            metrics[key]={'published':p,'calculated':float(val),'difference':float(val-p),'status':'MATCH' if ok else 'DIFF'}
    return {'coefficient_order':list(PARAM_KEYS),'coefficients':coef,'metrics':metrics,
            'reproduction_warning':'Los coeficientes publicados no permiten reconstruir de forma única las respuestas originales ni reproducir R²/ANOVA sin la matriz de respuestas y el diseño exactos.'}
