"""MAD-X and PTC reference for the combined-function ring of
`test/verify_curved_multipole.jl` ("Combined-function ring against MAD-X and
PTC"): 24 cells of gradient dipoles at rho = 11.46 m, 3 GeV electrons.

    python run_combined_function.py

The point of this ring is the (1+x/rho) factor on the *gradient* of a bend.
PTC's `exact` flag switches both that and the kinetic curvature term, so the
two PTC rows bracket the effect:

  exact=false  the fully expanded model -- straight multipole kick in a bent
               frame, which is what AT, JuTrack and TrackPad before 0.2.0 do
  exact=true   the curved-frame multipole and the exact body; TrackPad's
               `ExactSBend` reproduces this, and MAD-X `twiss, chrom` agrees

TrackPad's `SBend` sits between them: it takes the curved-frame field but keeps
the expanded kinetic term, so it matches neither row exactly and is pinned as a
regression instead.
"""
import math
from cpymad.madx import Madx

ANG = 2 * math.pi / 48
SRC = f"""
beam, particle=electron, energy=3.0;
bf: sbend, l=1.5, angle={ANG!r}, k1=0.42, e1=0, e2=0;
bd: sbend, l=1.5, angle={ANG!r}, k1=-0.52, e1=0, e2=0;
dr: drift, l=0.6;
cell: line=(bf, dr, bd, dr);
ring: line=(24*cell);
use, sequence=ring;
"""

m = Madx(stdout=False)
m.input(SRC)
m.command.twiss(chrom=True)
s, t = m.table.summ, m.table.twiss
print("MAD-X 5.09.03  twiss, chrom")
print(f"  length = {s.length[0]:.9f}   alfa = {s.alfa[0]:.10g}")
print(f"  q1     = {s.q1[0]:.9f}       q2   = {s.q2[0]:.9f}")
print(f"  dq1    = {s.dq1[0]:.7f}      dq2  = {s.dq2[0]:.7f}")
print(f"  betx0  = {t.betx[0]:.9f}     dx0  = {t.dx[0]:.9f}")

for exact in ("true", "false"):
    m.input(f"""
    ptc_create_universe;
    ptc_create_layout, model=1, method=6, nst=20, exact={exact};
    ptc_twiss, closed_orbit, icase=5, no=3, summary_table=ps;
    ptc_end;
    """)
    p = m.table.ps
    print(f"PTC exact={exact}")
    print(f"  q1     = {p.q1[0]:.9f}       q2   = {p.q2[0]:.9f}")
    print(f"  dq1    = {p.dq1[0]:.7f}      dq2  = {p.dq2[0]:.7f}")
    print(f"  alpha_c= {p.alpha_c[0]:.10g}")
