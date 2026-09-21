"""PTC reference: chromaticity, alpha_c, second-order chromaticity and
anharmonicities (exact=true, drift-kick-drift, method 6, 20 steps).

    python run_ptc.py "BEAM, PARTICLE=ELECTRON, ENERGY=3.0;" e
"""
import sys, json
from cpymad.madx import Madx
from lattice import madx_text

beam, tag = sys.argv[1], sys.argv[2]
m = Madx(stdout=False)
m.input(madx_text(beam))
m.input("""
ptc_create_universe;
ptc_create_layout, model=1, method=6, nst=20, exact=true;
ptc_twiss, closed_orbit, icase=5, no=3, summary_table=ptc_twiss_summary;
select_ptc_normal, q1=0, q2=0;
select_ptc_normal, dq1=1, dq2=1;
select_ptc_normal, dq1=2, dq2=2;
select_ptc_normal, anhx=1,0,0; select_ptc_normal, anhx=0,1,0;
select_ptc_normal, anhy=1,0,0; select_ptc_normal, anhy=0,1,0;
ptc_normal, closed_orbit, normal, icase=5, no=4;
ptc_end;
""")
s = m.table.ptc_twiss_summary
out = {k: float(s[k][0]) for k in ("length", "alpha_c", "eta_c", "gamma_tr", "q1", "q2", "dq1", "dq2")}
nt = m.table.normal_results
# PTC: DQ of order 2 is d2Q/ddelta2 (twice the Taylor coefficient); ANH are dQ/d(2J).
for name, o1, o2, o3, v in zip(nt.name, nt.order1, nt.order2, nt.order3, nt.value):
    out[f"{name.lower()}_{int(o1)}{int(o2)}{int(o3)}"] = float(v)
out["dQx_dJx"] = 2 * out["anhx_100"]; out["dQx_dJy"] = 2 * out["anhx_010"]
out["dQy_dJx"] = 2 * out["anhy_100"]; out["dQy_dJy"] = 2 * out["anhy_010"]
print(json.dumps(out, indent=1))
json.dump(out, open(f"ptc_{tag}.json", "w"))
