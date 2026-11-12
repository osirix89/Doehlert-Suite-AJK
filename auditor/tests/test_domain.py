# SPDX-License-Identifier: GPL-3.0-only
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from doehlert_engine import DOEHLERT_BASE
from doehlert_domain import ExperimentalDomain

class DomainTests(unittest.TestCase):
    def setUp(self): self.domain=ExperimentalDomain(np.asarray(DOEHLERT_BASE))
    def test_center_is_inside_all_domains(self):
        c=self.domain.check([0,0,0])
        self.assertTrue(c.inside_sphere); self.assertTrue(c.inside_hull); self.assertTrue(c.inside_admissible)
    def test_outside_sphere(self): self.assertFalse(self.domain.check([1.2,0,0]).inside_sphere)
    def test_inside_sphere_but_outside_hull(self):
        # Direction toward a cube corner lies outside the convex hull of the Doehlert design.
        p=np.array([0.51394508,0.01313996,0.55244270])
        c=self.domain.check(p)
        self.assertTrue(c.inside_sphere)
        self.assertFalse(c.inside_hull)
    def test_mask_matches_scalar_checks(self):
        pts=np.array([[0,0,0],[1,0,0],[.58,.58,.58]])
        self.assertEqual(self.domain.mask(pts).tolist(),[self.domain.feasible(p) for p in pts])

if __name__=='__main__': unittest.main()
