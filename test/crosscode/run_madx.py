"""MAD-X twiss reference: tunes, chromaticity, alfa, synch_1..5, and the Taylor
coefficients of Q(deltap) from a deltap scan.

    python run_madx.py "BEAM, PARTICLE=ELECTRON, ENERGY=3.0;" e

Note: for a beam with beta0 < 1 the MAD-X summary DQ1/DQ2 and the table DX are
derivatives with respect to PT (TrackPad's `wrt=:deltae`); the deltap-scan fit
below is the deltaP derivative (TrackPad's default).
"""
import sys, json
import numpy as np
from cpymad.madx import Madx
from lattice import madx_text

beam, tag = sys.argv[1], sys.argv[2]
m = Madx(stdout=False)
m.input(madx_text(beam))
m.command.twiss(chrom=True, sequence='RING')
tw = m.table.twiss; s = m.table.summ
out = dict(q1=s.q1[0], q2=s.q2[0], dq1_pt=s.dq1[0], dq2_pt=s.dq2[0],
           alfa=s.alfa[0], gammatr=s.gammatr[0], length=s.length[0],
           betx0=tw.betx[0], bety0=tw.bety[0], dx0_pt=tw.dx[0], dpx0_pt=tw.dpx[0],
           synch=[s.synch_1[0], s.synch_2[0], s.synch_3[0], s.synch_4[0], s.synch_5[0]])
ds = np.array([-2e-3, -1e-3, -5e-4, 0, 5e-4, 1e-3, 2e-3]); q1 = []; q2 = []
for d in ds:
    m.input(f'twiss, sequence=RING, deltap={d};')
    q1.append(m.table.summ.q1[0]); q2.append(m.table.summ.q2[0])
c1 = np.polyfit(ds, q1, 3); c2 = np.polyfit(ds, q2, 3)
out.update(dq1=c1[2], dq2=c2[2], chrom2x_taylor=c1[1], chrom2y_taylor=c2[1])
print(json.dumps(out, indent=1))
json.dump(out, open(f"madx_{tag}.json", "w"))
