# SPDX-License-Identifier: GPL-3.0-only
import sys, unittest, tempfile, csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from doehlert_engine import DOEHLERT_BASE
from doehlert_responses import read_multiresponse_csv

class MultiResponseTests(unittest.TestCase):
    def test_wide_csv_two_responses(self):
        pts=np.asarray(DOEHLERT_BASE); path=Path(tempfile.mktemp(suffix='.csv'))
        try:
            with path.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['Corrida','X1','X2','X3','Y1','Y2','x1_cod','x2_cod','x3_cod'])
                for i,p in enumerate(pts): w.writerow([i+1,*p,10-p@p,20+2*p[0]-p[1],*p])
            d=read_multiresponse_csv(path)
            self.assertEqual(set(d.responses),{'Y1','Y2'})
            self.assertEqual(d.real.shape,(13,3)); self.assertEqual(d.coded.shape,(13,3))
            self.assertEqual(len(d.responses['Y1']),13)
        finally: path.unlink(missing_ok=True)

if __name__=='__main__': unittest.main()
