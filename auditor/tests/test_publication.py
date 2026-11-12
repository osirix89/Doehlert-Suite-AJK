# SPDX-License-Identifier: GPL-3.0-only
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from doehlert_engine import DOEHLERT_BASE, fit_quadratic_model, PARAM_KEYS
from doehlert_publication import compare_published

class PublicationTests(unittest.TestCase):
    def test_comparison_marks_unavailable_metrics(self):
        x=np.asarray(DOEHLERT_BASE); y=10-2*x[:,0]**2-3*x[:,1]**2-4*x[:,2]**2
        fit=fit_quadratic_model(x,y,np.zeros(3),np.ones(3),limits_real=[(-1,1)]*3,real=x)
        result=compare_published(fit,fit.betas,published_metrics={'r2':1.0})
        self.assertTrue(all(c['status']=='MATCH' for c in result['coefficients']))
        self.assertEqual(result['metrics']['r2']['status'],'MATCH')
        self.assertEqual(result['metrics']['p_lof']['status'],'NOT_AVAILABLE')
        self.assertIn('no permiten reconstruir',result['reproduction_warning'])

if __name__=='__main__': unittest.main()
