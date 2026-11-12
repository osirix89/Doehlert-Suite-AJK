# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
"""Validación geométrica independiente del dominio experimental Doehlert (3D).

Distingue esfera teórica, envolvente convexa observada, límites físicos y
su intersección. Requiere numpy; scipy se usa para construir la envolvente.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - dependencia opcional con mensaje explícito
    ConvexHull = None
    QhullError = Exception

@dataclass
class DomainCheck:
    inside_sphere: bool
    sphere_radius: float
    inside_hull: bool
    inside_physical: bool
    inside_admissible: bool
    hull_max_violation: float
    physical_violations: tuple
    active_hull_facets: int
    location: str

class ExperimentalDomain:
    def __init__(self, coded_points, centers=None, scales=None, limits_real=None, sphere_radius=1.0, tolerance=1e-8):
        pts = np.asarray(coded_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4 or not np.all(np.isfinite(pts)):
            raise ValueError('La geometría requiere al menos cuatro puntos 3D finitos.')
        self.points = pts
        self.centers = np.zeros(3) if centers is None else np.asarray(centers, dtype=float)
        self.scales = np.ones(3) if scales is None else np.asarray(scales, dtype=float)
        if self.centers.shape != (3,) or self.scales.shape != (3,) or np.any(self.scales == 0):
            raise ValueError('Centros y escalas deben contener tres valores; las escalas no pueden ser cero.')
        self.limits_real = limits_real
        self.sphere_radius = float(sphere_radius)
        self.tol = float(tolerance)
        if self.sphere_radius <= 0 or self.tol <= 0:
            raise ValueError('Radio y tolerancia deben ser positivos.')
        if ConvexHull is None:
            raise RuntimeError('La envolvente convexa requiere SciPy. Instala scipy>=1.9.')
        try:
            self.hull = ConvexHull(pts)
        except QhullError as exc:
            raise ValueError('No se puede construir la envolvente: los puntos no abarcan un volumen 3D.') from exc

    def to_real(self, coded):
        return self.centers + self.scales * np.asarray(coded, dtype=float)

    def check(self, point):
        x = np.asarray(point, dtype=float)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            raise ValueError('El punto a validar debe contener tres coordenadas finitas.')
        radius = float(np.linalg.norm(x))
        vals = self.hull.equations[:, :3] @ x + self.hull.equations[:, 3]
        max_v = float(np.max(vals))
        inside_hull = max_v <= self.tol
        real = self.to_real(x)
        violations = []
        if self.limits_real is None:
            inside_physical = True
        else:
            for j, (lo, hi) in enumerate(self.limits_real):
                if real[j] < lo - self.tol:
                    violations.append(f'Factor {j+1}: {real[j]:.8g} < mínimo {lo:.8g}')
                if real[j] > hi + self.tol:
                    violations.append(f'Factor {j+1}: {real[j]:.8g} > máximo {hi:.8g}')
            inside_physical = not violations
        inside_sphere = radius <= self.sphere_radius + self.tol
        active = int(np.sum(np.abs(vals) <= self.tol * 10))
        if not inside_hull or not inside_sphere or not inside_physical:
            location = 'FUERA DEL DOMINIO ADMISIBLE'
        elif active >= 3:
            location = 'VÉRTICE / INTERSECCIÓN DE CARAS'
        elif active == 2:
            location = 'ARISTA / INTERSECCIÓN DE CARAS'
        elif active == 1 or abs(radius-self.sphere_radius) <= self.tol*10:
            location = 'CARA / FRONTERA'
        else:
            location = 'INTERIOR'
        return DomainCheck(inside_sphere, radius, inside_hull, inside_physical,
                           inside_sphere and inside_hull and inside_physical,
                           max_v, tuple(violations), active, location)

    def hull_inequalities(self, x):
        """SciPy SLSQP convention: returned values must be >= 0."""
        return -(self.hull.equations[:, :3] @ np.asarray(x) + self.hull.equations[:, 3])

    def feasible(self, x):
        c = self.check(x)
        return c.inside_admissible

    def mask(self, coded_grid, include_sphere=True):
        """Máscara booleana para arrays (...,3), consistente con check()."""
        arr = np.asarray(coded_grid, dtype=float)
        flat = arr.reshape(-1, 3)
        hull_ok = np.all(flat @ self.hull.equations[:, :3].T + self.hull.equations[:, 3] <= self.tol, axis=1)
        ok = hull_ok
        if include_sphere:
            ok &= np.linalg.norm(flat, axis=1) <= self.sphere_radius + self.tol
        if self.limits_real is not None:
            real = self.to_real(flat)
            for j, (lo, hi) in enumerate(self.limits_real):
                ok &= (real[:, j] >= lo-self.tol) & (real[:, j] <= hi+self.tol)
        return ok.reshape(arr.shape[:-1])
