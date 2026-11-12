# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""Optimización cuadrática restringida al dominio experimental admisible."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from doehlert_engine import design_matrix, build_B_matrix

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

@dataclass
class Optimum:
    sense: str
    x_coded: list
    x_real: list
    prediction: float
    ci_mean: tuple
    pi_prediction: tuple
    success: bool
    message: str
    location: str
    active_hull_facets: int
    objective_evaluations: int

    def to_dict(self):
        return asdict(self)

def predict(betas, x):
    return float(design_matrix(np.asarray(x, dtype=float).reshape(1,3))[0] @ np.asarray(betas, dtype=float))

def optimize_quadratic(fit, domain, sense='max', starts=40, seed=1729):
    if minimize is None:
        raise RuntimeError('La optimización restringida requiere SciPy. Instala scipy>=1.9.')
    sense = sense.lower()
    if sense not in ('max','min'):
        raise ValueError("sense debe ser 'max' o 'min'.")
    beta = np.asarray(fit.betas, dtype=float)
    sign = -1.0 if sense == 'max' else 1.0
    B = build_B_matrix(beta)
    b = beta[1:4]
    def objective(x): return sign * predict(beta, x)
    def jac(x): return sign * (b + 2.0 * B @ x)
    constraints = [
        {'type':'ineq', 'fun':domain.hull_inequalities},
        {'type':'ineq', 'fun':lambda x: domain.sphere_radius**2 - float(np.dot(x,x)),
         'jac':lambda x: -2.0*np.asarray(x, dtype=float)},
    ]
    # Los límites físicos son restricciones lineales en coordenadas codificadas.
    if domain.limits_real is not None:
        for j,(lo,hi) in enumerate(domain.limits_real):
            constraints.append({'type':'ineq','fun':lambda x,j=j,lo=lo: domain.centers[j]+domain.scales[j]*x[j]-lo})
            constraints.append({'type':'ineq','fun':lambda x,j=j,hi=hi: hi-domain.centers[j]-domain.scales[j]*x[j]})
    rng=np.random.default_rng(seed)
    candidates=[np.zeros(3), np.mean(domain.points,axis=0), *domain.points]
    # Puntos aleatorios dentro de la caja del diseño, filtrados por todas las restricciones.
    lo=np.min(domain.points,axis=0); hi=np.max(domain.points,axis=0)
    for p in rng.uniform(lo,hi,size=(max(0,starts),3)):
        if domain.feasible(p): candidates.append(p)
    best=None; total_nfev=0
    for x0 in candidates:
        if not domain.feasible(x0):
            # Proyecta el inicio al punto observado más cercano, que siempre es factible.
            x0=domain.points[int(np.argmin(np.linalg.norm(domain.points-x0,axis=1)))]
        res=minimize(objective, x0, jac=jac, method='SLSQP', constraints=constraints,
                     options={'ftol':1e-11,'maxiter':1000,'disp':False})
        total_nfev += int(getattr(res,'nfev',0))
        if res.x is not None and np.all(np.isfinite(res.x)) and domain.feasible(res.x):
            if best is None or res.fun < best.fun: best=res
    if best is None:
        raise RuntimeError('No se encontró un óptimo factible. Verifica las restricciones y la geometría.')
    x=np.asarray(best.x,dtype=float); chk=domain.check(x); yhat=predict(beta,x)
    m=design_matrix(x.reshape(1,3))[0]
    var_mean=max(0.0,float(m @ fit.cov_beta @ m))
    se_mean=var_mean**0.5
    delta_mean=float(fit.t_crit)*se_mean
    delta_pred=float(fit.t_crit)*(fit.s_res**2+var_mean)**0.5
    return Optimum(sense, x.tolist(), domain.to_real(x).tolist(), yhat,
                   (yhat-delta_mean,yhat+delta_mean),(yhat-delta_pred,yhat+delta_pred),
                   bool(best.success),str(best.message),chk.location,chk.active_hull_facets,total_nfev)
