# SPDX-License-Identifier: GPL-3.0-only
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from doehlert_engine import DOEHLERT_BASE, design_matrix, fit_quadratic_model
from doehlert_domain import ExperimentalDomain
from doehlert_optimization import optimize_quadratic

class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x=np.asarray(DOEHLERT_BASE,dtype=float)
        # Respuesta conocida: máximo en el centro; mínimo en la frontera.
        y=10.0-2*x[:,0]**2-3*x[:,1]**2-4*x[:,2]**2
        cls.fit=fit_quadratic_model(x,y,np.zeros(3),np.ones(3),limits_real=[(-1,1)]*3,real=x)
        cls.domain=ExperimentalDomain(x,np.zeros(3),np.ones(3),[(-1,1)]*3)
    def test_restricted_maximum_at_center(self):
        r=optimize_quadratic(self.fit,self.domain,'max',starts=10)
        self.assertTrue(self.domain.feasible(r.x_coded))
        self.assertAlmostEqual(r.prediction,10.0,places=6)
        self.assertLess(np.linalg.norm(r.x_coded),1e-5)
    def test_restricted_minimum_is_feasible_boundary(self):
        r=optimize_quadratic(self.fit,self.domain,'min',starts=10)
        self.assertTrue(self.domain.feasible(r.x_coded))
        self.assertGreaterEqual(np.linalg.norm(r.x_coded),0.99)
        self.assertLess(r.prediction,9.0)

if __name__=='__main__': unittest.main()
